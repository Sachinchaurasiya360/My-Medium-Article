# Voice Agents Deep Dive  Part 17: Production Infrastructure  Scaling, Monitoring, and Reliability

---

**Series:** Building Voice Agents  A Developer's Deep Dive from Audio Fundamentals to Production
**Part:** 17 of 19 (Production Voice Systems)
**Audience:** Developers with Python experience who want to build voice-powered AI agents from the ground up
**Reading time:** ~50 minutes

---

In Part 16, we optimized our voice agent from 3 seconds of latency down to under 800 milliseconds using streaming ASR, streaming LLM responses, speculative TTS, and filler word injection. Our agent feels fast.

Now we need it to stay fast when 100, 500, or 10,000 people call simultaneously. Part 17 is about the infrastructure that makes voice agents production-grade: scaling, monitoring, cost control, and high availability.

> **The production reality**: A voice agent that works for 1 user is a demo. An agent that works for 1,000 concurrent users reliably is a product. The gap between them is infrastructure.

---

## Section 1: The Unique Challenges of Scaling Voice

Voice agents are fundamentally different from stateless web services, and those differences make scaling harder:

```mermaid
graph TD
    subgraph "Web APIs (Stateless)"
        R1[Request] --> S1[Server A]
        R2[Request] --> S2[Server B]
        R3[Request] --> S3[Server C]
        S1 & S2 & S3 --> DB[(Database)]
    end

    subgraph "Voice Agents (Stateful)"
        C1[Call 1\nWebSocket] --> W1[Worker A\nSession state in memory\nAudio buffers\nConversation history]
        C2[Call 2\nWebSocket] --> W1
        C3[Call 3\nWebSocket] --> W2[Worker B]
        W1 & W2 --> Redis[(Redis\nSession State)]
        W1 & W2 --> PG[(PostgreSQL\nCall History)]
    end
```

**The key constraints:**

| Challenge | Why It's Hard | Solution |
|-----------|---------------|----------|
| WebSocket statefulness | Can't load-balance across servers | Sticky sessions |
| Audio buffer state | In-memory, can't lose mid-call | Redis for session state |
| Real-time constraints | GC pauses cause audio glitches | Worker isolation, tuning |
| Concurrent call capacity | Each call uses CPU/memory | Horizontal scaling + HPA |
| Model loading time | Cold-starting ASR/TTS is slow | Pre-warmed worker pools |

---

## Section 2: Session State Externalization

The foundation of scalable voice: move all call state out of worker memory into Redis.

```python
"""
session_state.py  External session state management for voice agents.
"""
import redis.asyncio as aioredis
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
import asyncio


@dataclass
class CallSession:
    """Complete serializable state for one voice call."""
    session_id: str
    user_id: Optional[str]
    phone_number: Optional[str]
    worker_id: str                    # Which worker owns this session
    call_start_time: float
    language: str = "en-US"
    turn_count: int = 0
    conversation_history: list[dict] = field(default_factory=list)
    collected_slots: dict[str, Any] = field(default_factory=dict)
    call_intent: Optional[str] = None
    last_activity: float = field(default_factory=time.time)
    status: str = "active"           # "active", "hold", "transfer", "ended"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CallSession":
        return cls(**data)


class RedisSessionStore:
    """
    Redis-backed session store for voice agent call state.
    Enables session recovery if a worker crashes.
    """

    SESSION_TTL_SECONDS = 3600  # 1 hour (for abandoned calls)

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self._redis: Optional[aioredis.Redis] = None

    async def _get_redis(self) -> aioredis.Redis:
        if not self._redis:
            self._redis = await aioredis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
        return self._redis

    def _key(self, session_id: str) -> str:
        return f"voice:session:{session_id}"

    async def create(self, session: CallSession) -> None:
        """Create a new session in Redis."""
        r = await self._get_redis()
        await r.setex(
            self._key(session.session_id),
            self.SESSION_TTL_SECONDS,
            json.dumps(session.to_dict()),
        )

    async def get(self, session_id: str) -> Optional[CallSession]:
        """Retrieve a session by ID."""
        r = await self._get_redis()
        data = await r.get(self._key(session_id))
        if not data:
            return None
        return CallSession.from_dict(json.loads(data))

    async def update(self, session: CallSession) -> None:
        """Update session state and refresh TTL."""
        session.last_activity = time.time()
        r = await self._get_redis()
        await r.setex(
            self._key(session.session_id),
            self.SESSION_TTL_SECONDS,
            json.dumps(session.to_dict()),
        )

    async def delete(self, session_id: str) -> None:
        """Delete a session (call ended)."""
        r = await self._get_redis()
        await r.delete(self._key(session_id))

    async def list_active_sessions(self, worker_id: str) -> list[str]:
        """List all sessions owned by a specific worker."""
        r = await self._get_redis()
        session_ids = []
        async for key in r.scan_iter(f"voice:session:*"):
            data = await r.get(key)
            if data:
                session_data = json.loads(data)
                if session_data.get("worker_id") == worker_id:
                    session_ids.append(session_data["session_id"])
        return session_ids

    async def transfer_sessions(self, from_worker: str, to_worker: str) -> int:
        """Transfer all sessions from one worker to another (for failover)."""
        r = await self._get_redis()
        transferred = 0
        async for key in r.scan_iter(f"voice:session:*"):
            data = await r.get(key)
            if data:
                session_data = json.loads(data)
                if session_data.get("worker_id") == from_worker:
                    session_data["worker_id"] = to_worker
                    await r.setex(key, self.SESSION_TTL_SECONDS, json.dumps(session_data))
                    transferred += 1
        return transferred


class VoiceAgentWorkerPool:
    """
    Manages a pool of voice agent workers.
    Tracks capacity and routes new calls to available workers.
    """

    MAX_CALLS_PER_WORKER = 50  # Tune based on CPU/memory

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._redis: Optional[aioredis.Redis] = None
        self.worker_id = self._generate_worker_id()

    def _generate_worker_id(self) -> str:
        import socket
        import os
        hostname = socket.gethostname()
        pid = os.getpid()
        return f"{hostname}-{pid}"

    async def _get_redis(self) -> aioredis.Redis:
        if not self._redis:
            self._redis = await aioredis.from_url(self.redis_url, decode_responses=True)
        return self._redis

    async def register_worker(self) -> None:
        """Register this worker in Redis with its capacity."""
        r = await self._get_redis()
        await r.hset(
            "voice:workers",
            self.worker_id,
            json.dumps({
                "worker_id": self.worker_id,
                "active_calls": 0,
                "max_calls": self.MAX_CALLS_PER_WORKER,
                "started_at": time.time(),
                "last_heartbeat": time.time(),
            })
        )
        print(f"Worker registered: {self.worker_id}")

    async def heartbeat(self) -> None:
        """Update this worker's heartbeat in Redis."""
        r = await self._get_redis()
        data = await r.hget("voice:workers", self.worker_id)
        if data:
            worker_data = json.loads(data)
            worker_data["last_heartbeat"] = time.time()
            await r.hset("voice:workers", self.worker_id, json.dumps(worker_data))

    async def increment_calls(self) -> None:
        """Increment active call count for this worker."""
        r = await self._get_redis()
        data = await r.hget("voice:workers", self.worker_id)
        if data:
            worker_data = json.loads(data)
            worker_data["active_calls"] = worker_data.get("active_calls", 0) + 1
            await r.hset("voice:workers", self.worker_id, json.dumps(worker_data))

    async def decrement_calls(self) -> None:
        """Decrement active call count when a call ends."""
        r = await self._get_redis()
        data = await r.hget("voice:workers", self.worker_id)
        if data:
            worker_data = json.loads(data)
            worker_data["active_calls"] = max(0, worker_data.get("active_calls", 1) - 1)
            await r.hset("voice:workers", self.worker_id, json.dumps(worker_data))

    async def get_all_workers(self) -> list[dict]:
        """Get all registered workers and their status."""
        r = await self._get_redis()
        workers_data = await r.hgetall("voice:workers")
        workers = [json.loads(v) for v in workers_data.values()]

        # Filter out dead workers (no heartbeat in 30 seconds)
        now = time.time()
        alive_workers = [
            w for w in workers
            if now - w.get("last_heartbeat", 0) < 30
        ]
        return alive_workers

    async def run_heartbeat_loop(self, interval: float = 10.0) -> None:
        """Run continuous heartbeat loop."""
        await self.register_worker()
        while True:
            await self.heartbeat()
            await asyncio.sleep(interval)
```

---

## Section 3: Infrastructure Architecture

### Docker Compose  Development Stack

```yaml
# docker-compose.yml  Full voice agent development stack
version: '3.8'

services:
  voice-gateway:
    build: .
    command: python -m voice_agent.gateway
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/voice_agent
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DEEPGRAM_API_KEY=${DEEPGRAM_API_KEY}
    depends_on:
      - redis
      - postgres
    volumes:
      - .:/app
    restart: unless-stopped

  voice-worker:
    build: .
    command: python -m voice_agent.worker
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/voice_agent
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DEEPGRAM_API_KEY=${DEEPGRAM_API_KEY}
    depends_on:
      - redis
      - postgres
    deploy:
      replicas: 3  # Start with 3 workers
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=voice_agent
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
  prometheus_data:
  grafana_data:
```

### Kubernetes  Production Deployment

```yaml
# kubernetes/voice-agent-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: voice-worker
  namespace: voice-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: voice-worker
  template:
    metadata:
      labels:
        app: voice-worker
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8001"
        prometheus.io/path: "/metrics"
    spec:
      containers:
        - name: voice-worker
          image: your-registry/voice-agent:latest
          command: ["python", "-m", "voice_agent.worker"]
          ports:
            - containerPort: 8001
              name: metrics
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "2000m"
              memory: "2Gi"
          env:
            - name: REDIS_URL
              valueFrom:
                secretKeyRef:
                  name: voice-agent-secrets
                  key: redis-url
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: voice-agent-secrets
                  key: openai-api-key
          livenessProbe:
            httpGet:
              path: /health
              port: 8001
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 8001
            initialDelaySeconds: 10
            periodSeconds: 5
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: voice-worker-hpa
  namespace: voice-agent
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: voice-worker
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Pods
      pods:
        metric:
          name: voice_active_calls_per_pod
        target:
          type: AverageValue
          averageValue: "30"  # Scale when avg > 30 calls/pod
---
apiVersion: v1
kind: Service
metadata:
  name: voice-gateway
  namespace: voice-agent
spec:
  type: LoadBalancer
  selector:
    app: voice-gateway
  ports:
    - port: 443
      targetPort: 8000
      protocol: TCP
  sessionAffinity: ClientIP         # Sticky sessions for WebSocket
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 3600          # 1-hour session stickiness
```

---

## Section 4: Prometheus Monitoring

```python
"""
monitoring.py  Prometheus metrics for voice agent infrastructure.
"""
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import asyncio
from functools import wraps
from typing import Callable, Any


class VoiceAgentMonitor:
    """
    Prometheus metrics for voice agent monitoring.
    Exposes metrics at /metrics for Prometheus scraping.
    """

    def __init__(self, port: int = 8001):
        self.port = port

        # Gauge metrics (current state)
        self.active_calls = Gauge(
            "voice_active_calls",
            "Number of currently active voice calls",
            ["worker_id"],
        )

        self.worker_capacity = Gauge(
            "voice_worker_capacity_ratio",
            "Ratio of active calls to max capacity",
            ["worker_id"],
        )

        # Counter metrics (cumulative)
        self.calls_total = Counter(
            "voice_calls_total",
            "Total number of calls handled",
            ["status", "language"],  # status: completed, abandoned, escalated
        )

        self.asr_errors = Counter(
            "voice_asr_errors_total",
            "Total ASR failures",
            ["engine", "reason"],
        )

        self.llm_errors = Counter(
            "voice_llm_errors_total",
            "Total LLM failures",
            ["model", "reason"],
        )

        self.escalations = Counter(
            "voice_escalations_total",
            "Total calls escalated to human agents",
            ["reason"],
        )

        # Histogram metrics (distributions)
        self.asr_latency = Histogram(
            "voice_asr_latency_seconds",
            "ASR transcription latency",
            ["engine"],
            buckets=[0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0],
        )

        self.llm_latency = Histogram(
            "voice_llm_latency_seconds",
            "LLM response latency (time to first token)",
            ["model"],
            buckets=[0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0],
        )

        self.tts_latency = Histogram(
            "voice_tts_latency_seconds",
            "TTS synthesis latency (time to first audio byte)",
            ["engine"],
            buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
        )

        self.e2e_latency = Histogram(
            "voice_e2e_latency_seconds",
            "End-to-end latency (end of user speech to first audio byte)",
            buckets=[0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0],
        )

        self.call_duration = Histogram(
            "voice_call_duration_seconds",
            "Duration of voice calls",
            buckets=[30, 60, 120, 180, 300, 600, 1200],
        )

        self.turn_count = Histogram(
            "voice_call_turns",
            "Number of conversation turns per call",
            buckets=[1, 2, 3, 5, 8, 13, 21],
        )

        # Cost tracking
        self.call_cost = Histogram(
            "voice_call_cost_dollars",
            "Estimated cost per call in USD",
            buckets=[0.01, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00],
        )

    def start(self) -> None:
        """Start the Prometheus metrics HTTP server."""
        start_http_server(self.port)
        print(f"Prometheus metrics available at http://localhost:{self.port}/metrics")

    def track_asr(self, engine: str):
        """Context manager for tracking ASR latency."""
        return self._latency_tracker(self.asr_latency, {"engine": engine})

    def track_llm(self, model: str):
        """Context manager for tracking LLM latency."""
        return self._latency_tracker(self.llm_latency, {"model": model})

    def track_tts(self, engine: str):
        """Context manager for tracking TTS latency."""
        return self._latency_tracker(self.tts_latency, {"engine": engine})

    def _latency_tracker(self, histogram: Histogram, labels: dict):
        """Generic latency tracking context manager."""
        class LatencyTracker:
            def __init__(self, hist, lbls):
                self.hist = hist
                self.lbls = lbls
                self.start = None

            def __enter__(self):
                self.start = time.perf_counter()
                return self

            def __exit__(self, *args):
                elapsed = time.perf_counter() - self.start
                self.hist.labels(**self.lbls).observe(elapsed)

        return LatencyTracker(histogram, labels)

    def record_call_complete(
        self,
        duration_seconds: float,
        turns: int,
        status: str,
        language: str,
        cost_dollars: float,
        worker_id: str,
    ) -> None:
        """Record metrics when a call completes."""
        self.calls_total.labels(status=status, language=language).inc()
        self.call_duration.observe(duration_seconds)
        self.turn_count.observe(turns)
        self.call_cost.observe(cost_dollars)
        self.active_calls.labels(worker_id=worker_id).dec()


# Global monitor instance
monitor = VoiceAgentMonitor()
```

### Prometheus Configuration

```yaml
# prometheus.yml  Prometheus scraping configuration
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'voice-agent'
    static_configs:
      - targets:
          - 'voice-worker:8001'
          - 'voice-gateway:8001'
    metrics_path: /metrics

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:9121']  # redis_exporter

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:9187']  # postgres_exporter

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - 'alerts/*.yml'
```

### Alerting Rules

```yaml
# alerts/voice_agent_alerts.yml
groups:
  - name: voice_agent_sla
    rules:
      - alert: HighE2ELatency
        expr: histogram_quantile(0.95, rate(voice_e2e_latency_seconds_bucket[5m])) > 2.0
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Voice agent p95 latency above 2s"
          description: "p95 E2E latency is {{ $value }}s"

      - alert: CriticalE2ELatency
        expr: histogram_quantile(0.95, rate(voice_e2e_latency_seconds_bucket[5m])) > 4.0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Voice agent p95 latency critically high"

      - alert: HighCallAbandonRate
        expr: |
          rate(voice_calls_total{status="abandoned"}[10m]) /
          rate(voice_calls_total[10m]) > 0.15
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Call abandon rate above 15%"

      - alert: WorkerCapacityHigh
        expr: voice_worker_capacity_ratio > 0.85
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Worker at {{ $value | humanizePercentage }} capacity"

      - alert: ASRErrorRate
        expr: rate(voice_asr_errors_total[5m]) > 0.05
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "ASR error rate above 5%"
```

---

## Section 5: Conversation Analytics

```python
"""
conversation_analytics.py  Track conversation metrics in PostgreSQL.
"""
import asyncio
import time
from dataclasses import dataclass
from typing import Optional
import asyncpg
import json


@dataclass
class TurnRecord:
    """Database record for one conversation turn."""
    turn_id: str
    session_id: str
    user_id: Optional[str]
    speaker: str             # "user" or "agent"
    text: str
    audio_duration_seconds: float
    asr_confidence: float
    asr_latency_ms: float
    llm_latency_ms: float
    tts_latency_ms: float
    e2e_latency_ms: float
    emotion: Optional[str]
    timestamp: float


@dataclass
class CallRecord:
    """Database record for a complete call."""
    session_id: str
    user_id: Optional[str]
    phone_number: Optional[str]
    started_at: float
    ended_at: Optional[float]
    duration_seconds: float
    turn_count: int
    intent: Optional[str]
    resolution_status: str
    escalated: bool
    total_cost_dollars: float
    language: str


class ConversationAnalytics:
    """
    Store and query conversation analytics in PostgreSQL.
    """

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._pool: Optional[asyncpg.Pool] = None

    async def _get_pool(self) -> asyncpg.Pool:
        if not self._pool:
            self._pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
            )
            await self._ensure_tables()
        return self._pool

    async def _ensure_tables(self) -> None:
        """Create analytics tables if they don't exist."""
        pool = self._pool
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS calls (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    phone_number TEXT,
                    started_at DOUBLE PRECISION,
                    ended_at DOUBLE PRECISION,
                    duration_seconds DOUBLE PRECISION,
                    turn_count INTEGER,
                    intent TEXT,
                    resolution_status TEXT,
                    escalated BOOLEAN DEFAULT FALSE,
                    total_cost_dollars DOUBLE PRECISION DEFAULT 0,
                    language TEXT DEFAULT 'en-US'
                );

                CREATE TABLE IF NOT EXISTS turns (
                    turn_id TEXT PRIMARY KEY,
                    session_id TEXT REFERENCES calls(session_id),
                    user_id TEXT,
                    speaker TEXT,
                    text TEXT,
                    audio_duration_seconds DOUBLE PRECISION,
                    asr_confidence DOUBLE PRECISION,
                    asr_latency_ms DOUBLE PRECISION,
                    llm_latency_ms DOUBLE PRECISION,
                    tts_latency_ms DOUBLE PRECISION,
                    e2e_latency_ms DOUBLE PRECISION,
                    emotion TEXT,
                    timestamp DOUBLE PRECISION
                );

                CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
                CREATE INDEX IF NOT EXISTS idx_calls_user ON calls(user_id);
                CREATE INDEX IF NOT EXISTS idx_calls_started ON calls(started_at);
            """)

    async def record_turn(self, turn: TurnRecord) -> None:
        """Record a single conversation turn."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO turns (
                    turn_id, session_id, user_id, speaker, text,
                    audio_duration_seconds, asr_confidence,
                    asr_latency_ms, llm_latency_ms, tts_latency_ms, e2e_latency_ms,
                    emotion, timestamp
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ON CONFLICT (turn_id) DO NOTHING
            """,
                turn.turn_id, turn.session_id, turn.user_id, turn.speaker, turn.text,
                turn.audio_duration_seconds, turn.asr_confidence,
                turn.asr_latency_ms, turn.llm_latency_ms, turn.tts_latency_ms, turn.e2e_latency_ms,
                turn.emotion, turn.timestamp,
            )

    async def record_call(self, call: CallRecord) -> None:
        """Record or update a call record."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO calls (
                    session_id, user_id, phone_number, started_at, ended_at,
                    duration_seconds, turn_count, intent, resolution_status,
                    escalated, total_cost_dollars, language
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (session_id) DO UPDATE SET
                    ended_at = EXCLUDED.ended_at,
                    duration_seconds = EXCLUDED.duration_seconds,
                    turn_count = EXCLUDED.turn_count,
                    intent = EXCLUDED.intent,
                    resolution_status = EXCLUDED.resolution_status,
                    escalated = EXCLUDED.escalated,
                    total_cost_dollars = EXCLUDED.total_cost_dollars
            """,
                call.session_id, call.user_id, call.phone_number,
                call.started_at, call.ended_at,
                call.duration_seconds, call.turn_count, call.intent,
                call.resolution_status, call.escalated,
                call.total_cost_dollars, call.language,
            )

    async def get_dashboard_metrics(
        self, hours_back: int = 24
    ) -> dict:
        """Get key metrics for the dashboard."""
        pool = await self._get_pool()
        cutoff = time.time() - (hours_back * 3600)

        async with pool.acquire() as conn:
            # Total calls
            total = await conn.fetchval(
                "SELECT COUNT(*) FROM calls WHERE started_at > $1", cutoff
            )

            # Completion rate
            completed = await conn.fetchval(
                "SELECT COUNT(*) FROM calls WHERE started_at > $1 AND resolution_status = 'resolved'",
                cutoff,
            )

            # Average duration
            avg_duration = await conn.fetchval(
                "SELECT AVG(duration_seconds) FROM calls WHERE started_at > $1 AND ended_at IS NOT NULL",
                cutoff,
            )

            # Escalation rate
            escalated = await conn.fetchval(
                "SELECT COUNT(*) FROM calls WHERE started_at > $1 AND escalated = TRUE",
                cutoff,
            )

            # P95 E2E latency
            p95_latency = await conn.fetchval(
                """SELECT PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY e2e_latency_ms)
                   FROM turns WHERE timestamp > $1 AND speaker = 'agent'""",
                cutoff,
            )

            # Average cost per call
            avg_cost = await conn.fetchval(
                "SELECT AVG(total_cost_dollars) FROM calls WHERE started_at > $1",
                cutoff,
            )

            # Most common intents
            intents = await conn.fetch(
                """SELECT intent, COUNT(*) as count
                   FROM calls WHERE started_at > $1 AND intent IS NOT NULL
                   GROUP BY intent ORDER BY count DESC LIMIT 10""",
                cutoff,
            )

        completion_rate = (completed / total * 100) if total > 0 else 0
        escalation_rate = (escalated / total * 100) if total > 0 else 0

        return {
            "period_hours": hours_back,
            "total_calls": total,
            "completion_rate_pct": round(completion_rate, 1),
            "escalation_rate_pct": round(escalation_rate, 1),
            "avg_duration_seconds": round(avg_duration or 0, 1),
            "p95_e2e_latency_ms": round(p95_latency or 0, 0),
            "avg_cost_per_call_dollars": round(avg_cost or 0, 4),
            "top_intents": [{"intent": r["intent"], "count": r["count"]} for r in intents],
        }
```

---

## Section 6: Cost Optimization

Understanding and managing cost is critical at scale:

| Component | Cost Model | Typical Cost |
|-----------|-----------|--------------|
| Twilio Inbound | $0.0085/min + phone number | ~$5/month + usage |
| Deepgram Nova-2 | $0.0059/min streaming | ~$0.006/min |
| OpenAI gpt-4o-mini | $0.15/1M input, $0.60/1M output | ~$0.003/call |
| OpenAI TTS | $0.015/1K characters | ~$0.005/call |
| ElevenLabs | $0.30/1K characters | ~$0.015/call |
| Server (GCP n2-standard-4) | $0.19/hour | ~$0.0001/min |
| **Total (OpenAI TTS)** | | **~$0.014/min** |
| **Total (ElevenLabs)** | | **~$0.024/min** |

```python
"""
cost_tracker.py  Real-time cost tracking per call.
"""
from dataclasses import dataclass, field
import time


@dataclass
class CostModel:
    """Cost rates per service component."""
    # Per minute
    telephony_per_min: float = 0.0085       # Twilio
    asr_per_min: float = 0.0059             # Deepgram Nova-2

    # Per 1K tokens
    llm_input_per_1k: float = 0.00015      # GPT-4o-mini input
    llm_output_per_1k: float = 0.00060     # GPT-4o-mini output

    # Per 1K characters
    tts_per_1k_chars: float = 0.015        # OpenAI TTS
    # tts_per_1k_chars: float = 0.30       # ElevenLabs

    # Per minute (infrastructure)
    infra_per_min: float = 0.0002          # ~$0.12/hour per call


@dataclass
class CallCostAccumulator:
    """Track costs for a single call in real-time."""
    session_id: str
    cost_model: CostModel
    start_time: float = field(default_factory=time.time)

    # Accumulated usage
    call_duration_minutes: float = 0.0
    asr_duration_minutes: float = 0.0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    tts_characters: int = 0

    def record_asr_usage(self, duration_seconds: float) -> None:
        self.asr_duration_minutes += duration_seconds / 60.0

    def record_llm_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.llm_input_tokens += input_tokens
        self.llm_output_tokens += output_tokens

    def record_tts_usage(self, text: str) -> None:
        self.tts_characters += len(text)

    def compute_total_cost(self) -> dict:
        """Compute total cost breakdown."""
        duration_minutes = (time.time() - self.start_time) / 60.0

        telephony_cost = duration_minutes * self.cost_model.telephony_per_min
        asr_cost = self.asr_duration_minutes * self.cost_model.asr_per_min
        llm_cost = (
            (self.llm_input_tokens / 1000) * self.cost_model.llm_input_per_1k
            + (self.llm_output_tokens / 1000) * self.cost_model.llm_output_per_1k
        )
        tts_cost = (self.tts_characters / 1000) * self.cost_model.tts_per_1k_chars
        infra_cost = duration_minutes * self.cost_model.infra_per_min

        total = telephony_cost + asr_cost + llm_cost + tts_cost + infra_cost

        return {
            "session_id": self.session_id,
            "duration_minutes": round(duration_minutes, 2),
            "telephony": round(telephony_cost, 4),
            "asr": round(asr_cost, 4),
            "llm": round(llm_cost, 4),
            "tts": round(tts_cost, 4),
            "infrastructure": round(infra_cost, 4),
            "total": round(total, 4),
        }
```

---

## Section 7: Circuit Breakers and High Availability

```python
"""
circuit_breaker.py  Circuit breakers for external voice service calls.
"""
import asyncio
import time
from enum import Enum
from dataclasses import dataclass
from typing import Callable, TypeVar, Any

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing  reject calls immediately
    HALF_OPEN = "half_open"  # Testing  allow limited calls through


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 2          # Successes in half-open to close
    timeout_seconds: float = 60.0       # How long to stay open before trying
    request_timeout: float = 5.0        # Per-request timeout


class CircuitBreaker:
    """
    Circuit breaker for external voice service calls (ASR, TTS, LLM).
    Prevents cascade failures when an external service is down.
    """

    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float = 0
        self._lock = asyncio.Lock()

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function through the circuit breaker.
        Raises CircuitOpenError if the circuit is open.
        """
        async with self._lock:
            if self.state == CircuitState.OPEN:
                elapsed = time.time() - self.last_failure_time
                if elapsed > self.config.timeout_seconds:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    print(f"[CircuitBreaker:{self.name}] Testing (HALF_OPEN)")
                else:
                    remaining = self.config.timeout_seconds - elapsed
                    raise CircuitOpenError(
                        f"Circuit {self.name} is open. Retry in {remaining:.0f}s"
                    )

        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.request_timeout,
            )
            await self._on_success()
            return result

        except Exception as e:
            await self._on_failure(str(e))
            raise

    async def _on_success(self) -> None:
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    print(f"[CircuitBreaker:{self.name}] CLOSED (recovered)")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)

    async def _on_failure(self, reason: str) -> None:
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.config.failure_threshold:
                if self.state != CircuitState.OPEN:
                    self.state = CircuitState.OPEN
                    print(f"[CircuitBreaker:{self.name}] OPENED after {self.failure_count} failures. Reason: {reason}")
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                print(f"[CircuitBreaker:{self.name}] Back to OPEN (failed in HALF_OPEN)")


class CircuitOpenError(Exception):
    pass


class ResilientVoicePipeline:
    """
    Voice pipeline with circuit breakers for each external service.
    Falls back gracefully when services fail.
    """

    def __init__(self):
        self.asr_breaker = CircuitBreaker("ASR", CircuitBreakerConfig(failure_threshold=3))
        self.llm_breaker = CircuitBreaker("LLM", CircuitBreakerConfig(failure_threshold=3))
        self.tts_breaker = CircuitBreaker("TTS", CircuitBreakerConfig(failure_threshold=3))

        # Fallback engines
        self.primary_asr = None    # Deepgram
        self.fallback_asr = None   # Whisper (local)
        self.primary_llm = None    # GPT-4o-mini
        self.fallback_llm = None   # GPT-3.5-turbo
        self.primary_tts = None    # ElevenLabs
        self.fallback_tts = None   # OpenAI TTS

    async def transcribe(self, audio, language: str) -> str:
        """Transcribe with automatic fallback."""
        try:
            return await self.asr_breaker.call(
                self.primary_asr.transcribe, audio, language
            )
        except CircuitOpenError:
            print("ASR primary circuit open, using fallback (Whisper)")
            if self.fallback_asr:
                return await self.fallback_asr.transcribe(audio, language)
            raise
        except Exception as e:
            print(f"ASR error: {e}, falling back")
            if self.fallback_asr:
                return await self.fallback_asr.transcribe(audio, language)
            raise

    async def generate_response(self, messages: list) -> str:
        """Generate LLM response with fallback."""
        try:
            return await self.llm_breaker.call(
                self._call_llm, self.primary_llm, messages
            )
        except (CircuitOpenError, Exception) as e:
            print(f"LLM primary failed ({e}), using fallback")
            if self.fallback_llm:
                return await self._call_llm(self.fallback_llm, messages)
            raise

    async def _call_llm(self, llm_client, messages: list) -> str:
        """Call LLM and extract text response."""
        import httpx
        # Implementation depends on LLM client type
        raise NotImplementedError("Implement with your LLM client")

    async def synthesize(self, text: str, language: str) -> bytes:
        """Synthesize speech with fallback."""
        try:
            return await self.tts_breaker.call(
                self.primary_tts.synthesize, text, language
            )
        except (CircuitOpenError, Exception) as e:
            print(f"TTS primary failed ({e}), using fallback")
            if self.fallback_tts:
                return await self.fallback_tts.synthesize(text, language)
            raise
```

---

## Section 8: Load Testing

```python
"""
load_tester.py  Simulate concurrent voice agent calls for load testing.
"""
import asyncio
import time
import random
from dataclasses import dataclass, field


@dataclass
class LoadTestConfig:
    target_url: str
    concurrent_calls: int = 50
    call_duration_seconds: float = 60.0
    ramp_up_seconds: float = 30.0    # Gradually add calls
    think_time_seconds: float = 2.0  # Pause between turns


@dataclass
class LoadTestResults:
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    avg_e2e_latency_ms: float = 0.0
    p95_e2e_latency_ms: float = 0.0
    p99_e2e_latency_ms: float = 0.0
    latencies: list[float] = field(default_factory=list)


class VoiceAgentLoadTester:
    """
    Simulate concurrent voice agent calls.
    Uses text-based simulation (no actual audio) for efficiency.
    """

    TEST_UTTERANCES = [
        "What are your business hours?",
        "I need help with my account",
        "Can I speak to a human?",
        "What is the status of my order?",
        "How do I cancel my subscription?",
        "I'm having trouble logging in",
        "Do you offer refunds?",
    ]

    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results = LoadTestResults()

    async def simulate_call(self, call_id: int) -> dict:
        """Simulate a single voice agent call."""
        session_id = f"load_test_{call_id}"
        start_time = time.perf_counter()

        call_latencies = []
        turns = random.randint(3, 8)

        for turn in range(turns):
            utterance = random.choice(self.TEST_UTTERANCES)
            turn_start = time.perf_counter()

            try:
                # In real load test: send audio bytes via WebSocket
                # Here we simulate with HTTP
                import httpx
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.post(
                        f"{self.config.target_url}/simulate_turn",
                        json={"session_id": session_id, "text": utterance},
                    )
                    resp.raise_for_status()

                turn_latency = (time.perf_counter() - turn_start) * 1000
                call_latencies.append(turn_latency)

            except Exception as e:
                return {"call_id": call_id, "status": "failed", "error": str(e)}

            await asyncio.sleep(self.config.think_time_seconds)

        total_duration = time.perf_counter() - start_time

        return {
            "call_id": call_id,
            "status": "completed",
            "duration_seconds": total_duration,
            "turns": turns,
            "avg_turn_latency_ms": sum(call_latencies) / len(call_latencies),
            "max_turn_latency_ms": max(call_latencies),
        }

    async def run(self) -> LoadTestResults:
        """Run the full load test with ramp-up."""
        print(f"Starting load test: {self.config.concurrent_calls} concurrent calls")
        print(f"Ramp-up: {self.config.ramp_up_seconds}s")

        # Calculate launch times (ramp-up)
        launch_delays = [
            (i / self.config.concurrent_calls) * self.config.ramp_up_seconds
            for i in range(self.config.concurrent_calls)
        ]

        all_tasks = []
        start_time = time.time()

        for call_id, delay in enumerate(launch_delays):
            async def launch_call(cid=call_id, d=delay):
                await asyncio.sleep(d)
                return await self.simulate_call(cid)

            all_tasks.append(asyncio.create_task(launch_call()))

        # Wait for all calls
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Aggregate results
        all_latencies = []
        success = 0
        failed = 0

        for result in results:
            if isinstance(result, Exception):
                failed += 1
            elif result.get("status") == "completed":
                success += 1
                all_latencies.append(result["avg_turn_latency_ms"])
            else:
                failed += 1

        if all_latencies:
            all_latencies.sort()
            self.results.avg_e2e_latency_ms = sum(all_latencies) / len(all_latencies)
            p95_idx = int(len(all_latencies) * 0.95)
            p99_idx = int(len(all_latencies) * 0.99)
            self.results.p95_e2e_latency_ms = all_latencies[p95_idx]
            self.results.p99_e2e_latency_ms = all_latencies[p99_idx]

        self.results.total_calls = self.config.concurrent_calls
        self.results.successful_calls = success
        self.results.failed_calls = failed

        print(f"\n=== Load Test Results ===")
        print(f"Duration: {total_time:.1f}s")
        print(f"Calls: {success}/{self.config.concurrent_calls} succeeded")
        print(f"Avg latency: {self.results.avg_e2e_latency_ms:.0f}ms")
        print(f"P95 latency: {self.results.p95_e2e_latency_ms:.0f}ms")
        print(f"P99 latency: {self.results.p99_e2e_latency_ms:.0f}ms")

        return self.results


# Run load test
async def main():
    config = LoadTestConfig(
        target_url="http://localhost:8000",
        concurrent_calls=50,
        ramp_up_seconds=30,
    )
    tester = VoiceAgentLoadTester(config)
    await tester.run()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Section 9: Graceful Shutdown and Zero-Downtime Deployments

The hardest part of deploying new versions of a voice agent is not breaking active calls. A rolling update that kills a pod mid-call drops that call  unacceptable for production.

```python
"""
graceful_shutdown.py  Handle SIGTERM without dropping active calls.
"""
import asyncio
import signal
import logging
from typing import Set
import time

logger = logging.getLogger("voice-agent.shutdown")


class GracefulShutdownManager:
    """
    Manages graceful shutdown of voice agent workers.

    On SIGTERM:
    1. Stop accepting new calls
    2. Wait for active calls to complete (up to max_drain_seconds)
    3. Force-close any remaining calls with a polite goodbye message
    4. Exit cleanly

    This allows Kubernetes rolling updates to work without dropping calls.
    """

    def __init__(self, max_drain_seconds: float = 300.0):
        self.max_drain_seconds = max_drain_seconds
        self._active_call_ids: Set[str] = set()
        self._is_draining = False
        self._shutdown_event = asyncio.Event()
        self._drain_event = asyncio.Event()

    def register_signal_handlers(self) -> None:
        """Register SIGTERM and SIGINT handlers."""
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(
            signal.SIGTERM,
            lambda: asyncio.create_task(self._on_sigterm()),
        )
        loop.add_signal_handler(
            signal.SIGINT,
            lambda: asyncio.create_task(self._on_sigterm()),
        )
        logger.info("Signal handlers registered")

    def register_call(self, call_id: str) -> None:
        """Register a new active call."""
        if self._is_draining:
            raise RuntimeError("Worker is draining  not accepting new calls")
        self._active_call_ids.add(call_id)
        logger.debug(f"Call registered: {call_id} (active: {len(self._active_call_ids)})")

    def unregister_call(self, call_id: str) -> None:
        """Mark a call as complete."""
        self._active_call_ids.discard(call_id)
        active_count = len(self._active_call_ids)
        logger.debug(f"Call unregistered: {call_id} (active: {active_count})")

        if self._is_draining and active_count == 0:
            logger.info("All calls drained. Ready to shutdown.")
            self._drain_event.set()

    @property
    def is_draining(self) -> bool:
        return self._is_draining

    @property
    def active_call_count(self) -> int:
        return len(self._active_call_ids)

    async def _on_sigterm(self) -> None:
        """Handle SIGTERM: drain active calls, then shutdown."""
        if self._is_draining:
            return  # Already handling

        logger.warning(
            f"SIGTERM received. Draining {len(self._active_call_ids)} active calls. "
            f"Max drain time: {self.max_drain_seconds}s"
        )
        self._is_draining = True

        if not self._active_call_ids:
            logger.info("No active calls. Shutting down immediately.")
            self._shutdown_event.set()
            return

        # Wait for all calls to drain or timeout
        try:
            await asyncio.wait_for(
                self._drain_event.wait(),
                timeout=self.max_drain_seconds,
            )
            logger.info("All calls drained gracefully.")
        except asyncio.TimeoutError:
            remaining = list(self._active_call_ids)
            logger.warning(
                f"Drain timeout. Force-closing {len(remaining)} calls: {remaining}"
            )
            # In production: send goodbye message to each remaining call
            for call_id in remaining:
                await self._force_close_call(call_id)

        self._shutdown_event.set()

    async def _force_close_call(self, call_id: str) -> None:
        """Force-close a call with a polite message."""
        logger.info(f"Force-closing call: {call_id}")
        # In production: send "Sorry, we need to restart. Call back in a moment."
        # Then close the WebSocket connection
        self._active_call_ids.discard(call_id)

    async def wait_for_shutdown(self) -> None:
        """Block until shutdown is complete."""
        await self._shutdown_event.wait()


class ShutdownAwareCallHandler:
    """
    A call handler that integrates with GracefulShutdownManager.
    """

    def __init__(self, shutdown_manager: GracefulShutdownManager):
        self.shutdown_manager = shutdown_manager

    async def handle_call(self, call_id: str, websocket) -> None:
        """Handle a call, registering and unregistering with the shutdown manager."""

        # Check if we're draining before accepting
        if self.shutdown_manager.is_draining:
            await websocket.close(code=1001, reason="Server is restarting. Please call back.")
            return

        # Register call
        try:
            self.shutdown_manager.register_call(call_id)
        except RuntimeError:
            await websocket.close(code=1001, reason="Server is restarting.")
            return

        try:
            # Run the actual call logic
            await self._run_call(call_id, websocket)
        finally:
            # Always unregister, even if call throws
            self.shutdown_manager.unregister_call(call_id)

    async def _run_call(self, call_id: str, websocket) -> None:
        """Run the actual call logic (implement your pipeline here)."""
        try:
            async for message in websocket:
                # Process audio, run STT/LLM/TTS
                pass
        except Exception as e:
            logger.error(f"[{call_id}] Call error: {e}")


# Integration with FastAPI
async def main():
    """Run a FastAPI WebSocket server with graceful shutdown."""
    from fastapi import FastAPI, WebSocket
    import uvicorn

    shutdown_mgr = GracefulShutdownManager(max_drain_seconds=300)
    handler = ShutdownAwareCallHandler(shutdown_mgr)
    app = FastAPI()

    @app.get("/ready")
    async def readiness_check():
        """Kubernetes readiness probe  returns 503 when draining."""
        if shutdown_mgr.is_draining:
            from fastapi import Response
            return Response(
                content="draining",
                status_code=503,
                headers={"Retry-After": "30"},
            )
        return {"status": "ready", "active_calls": shutdown_mgr.active_call_count}

    @app.websocket("/call/{call_id}")
    async def websocket_endpoint(websocket: WebSocket, call_id: str):
        await websocket.accept()
        await handler.handle_call(call_id, websocket)

    # Register signals
    shutdown_mgr.register_signal_handlers()

    # Run until SIGTERM
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)

    # Run server and wait for shutdown
    server_task = asyncio.create_task(server.serve())
    await shutdown_mgr.wait_for_shutdown()
    server.should_exit = True
    await server_task
```

### Kubernetes Rolling Update Configuration

```yaml
# kubernetes/rolling-update-strategy.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: voice-worker
  namespace: voice-agent
spec:
  # Rolling update: replace pods one at a time
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 0      # Never have fewer pods than desired
      maxSurge: 1            # Allow 1 extra pod during update

  template:
    spec:
      # Give pods 5 minutes to drain before force-killing
      terminationGracePeriodSeconds: 300

      containers:
        - name: voice-worker
          lifecycle:
            preStop:
              # Wait 5s for load balancer to stop routing new calls
              # before the SIGTERM is sent
              exec:
                command: ["sleep", "5"]

          # Readiness probe: remove from load balancer when draining
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
            failureThreshold: 1  # Remove from LB after 1 failure (fast)

          # Liveness probe: restart if completely broken
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
            failureThreshold: 3
```

```
Deployment flow with graceful shutdown:
1. kubectl apply -f deployment.yaml (new version)
2. Kubernetes creates 1 new pod (maxSurge=1)
3. New pod starts, passes readiness probe
4. Kubernetes sends SIGTERM to 1 old pod
5. preStop hook: sleep 5s (load balancer stops routing to it)
6. SIGTERM received: set is_draining=True
7. Readiness probe fails: pod removed from load balancer
8. Active calls continue; no new calls arrive
9. Calls complete; drain_event fires
10. Process exits cleanly
11. Kubernetes deletes old pod
12. Repeat for remaining old pods
```

---

## Section 10: Session Recovery After Worker Crash

Workers crash. Kubernetes restarts them. But what about the calls in progress?

```python
"""
session_recovery.py  Recover abandoned sessions after worker crash.
"""
import asyncio
import redis.asyncio as aioredis
import json
import time
import logging
from dataclasses import dataclass
from typing import Optional, List

logger = logging.getLogger("voice-agent.recovery")


@dataclass
class OrphanedSession:
    """A session whose worker is no longer alive."""
    session_id: str
    original_worker: str
    user_phone: Optional[str]
    call_start_time: float
    turn_count: int
    last_activity: float
    conversation_history: list


class SessionRecoveryService:
    """
    Detects and recovers sessions orphaned by worker crashes.

    When a worker crashes without graceful shutdown, its sessions
    remain in Redis with stale worker_ids. This service detects them
    and either reassigns them or closes them gracefully.
    """

    WORKER_HEARTBEAT_TTL = 30.0  # Seconds before a worker is considered dead
    SESSION_RECOVERY_INTERVAL = 15.0  # How often to check for orphans

    def __init__(self, redis_url: str, current_worker_id: str):
        self.redis_url = redis_url
        self.current_worker_id = current_worker_id
        self._redis: Optional[aioredis.Redis] = None

    async def _get_redis(self) -> aioredis.Redis:
        if not self._redis:
            self._redis = await aioredis.from_url(self.redis_url, decode_responses=True)
        return self._redis

    async def get_live_worker_ids(self) -> set[str]:
        """Get set of worker IDs that are currently alive (recent heartbeat)."""
        r = await self._get_redis()
        workers_data = await r.hgetall("voice:workers")

        now = time.time()
        live_workers = set()
        for worker_id, data_str in workers_data.items():
            try:
                data = json.loads(data_str)
                if now - data.get("last_heartbeat", 0) < self.WORKER_HEARTBEAT_TTL:
                    live_workers.add(worker_id)
            except json.JSONDecodeError:
                pass

        return live_workers

    async def find_orphaned_sessions(self) -> List[OrphanedSession]:
        """Find sessions whose worker is no longer alive."""
        r = await self._get_redis()
        live_workers = await self.get_live_worker_ids()

        orphans = []
        async for key in r.scan_iter("voice:session:*"):
            data_str = await r.get(key)
            if not data_str:
                continue

            try:
                session_data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            worker_id = session_data.get("worker_id", "")
            if worker_id not in live_workers:
                orphans.append(OrphanedSession(
                    session_id=session_data.get("session_id", ""),
                    original_worker=worker_id,
                    user_phone=session_data.get("phone_number"),
                    call_start_time=session_data.get("call_start_time", 0),
                    turn_count=session_data.get("turn_count", 0),
                    last_activity=session_data.get("last_activity", 0),
                    conversation_history=session_data.get("conversation_history", []),
                ))

        return orphans

    async def handle_orphaned_session(self, orphan: OrphanedSession) -> str:
        """
        Handle a single orphaned session.
        Returns: "reassigned", "closed_gracefully", "expired"
        """
        # Check if session is still recent enough to recover
        age_seconds = time.time() - orphan.last_activity
        if age_seconds > 120:
            # Session is too old  the caller has definitely hung up
            logger.info(
                f"Session {orphan.session_id} expired ({age_seconds:.0f}s old). Cleaning up."
            )
            await self._clean_up_session(orphan.session_id)
            return "expired"

        # Recent session  caller might still be connected
        logger.warning(
            f"Orphaned session {orphan.session_id} from dead worker {orphan.original_worker}. "
            f"Age: {age_seconds:.0f}s, turns: {orphan.turn_count}"
        )

        # Strategy 1: If the session came from Twilio, we can send a goodbye SMS
        if orphan.user_phone:
            await self._notify_user_of_disconnect(orphan.user_phone)

        # Strategy 2: Reassign to current worker
        # (only works if the underlying connection is still alive  usually it's not)
        await self._reassign_session(orphan.session_id)
        return "reassigned"

    async def _clean_up_session(self, session_id: str) -> None:
        """Remove a session from Redis and log it to the analytics DB."""
        r = await self._get_redis()
        await r.delete(f"voice:session:{session_id}")
        # In production: log to PostgreSQL with status="recovered_orphan"

    async def _reassign_session(self, session_id: str) -> None:
        """Reassign a session to the current worker."""
        r = await self._get_redis()
        key = f"voice:session:{session_id}"
        data_str = await r.get(key)
        if data_str:
            data = json.loads(data_str)
            data["worker_id"] = self.current_worker_id
            await r.setex(key, 3600, json.dumps(data))

    async def _notify_user_of_disconnect(self, phone_number: str) -> None:
        """Send a recovery SMS to a disconnected user (if you have SMS capability)."""
        logger.info(f"Would notify {phone_number} of disconnect")
        # In production: use Twilio SMS API

    async def run_recovery_loop(self) -> None:
        """Continuously check for and handle orphaned sessions."""
        logger.info("Session recovery loop started")
        while True:
            try:
                orphans = await self.find_orphaned_sessions()
                if orphans:
                    logger.warning(f"Found {len(orphans)} orphaned sessions")
                    for orphan in orphans:
                        result = await self.handle_orphaned_session(orphan)
                        logger.info(f"Session {orphan.session_id}: {result}")
            except Exception as e:
                logger.error(f"Recovery loop error: {e}")

            await asyncio.sleep(self.SESSION_RECOVERY_INTERVAL)
```

---

## Section 11: Rate Limiting and Capacity Planning

### Rate Limiting Inbound Calls

```python
"""
rate_limiter.py  Rate limiting for voice agent endpoints.
"""
import redis.asyncio as aioredis
import time
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger("voice-agent.ratelimit")


@dataclass
class RateLimitConfig:
    """Rate limit configuration for voice endpoints."""
    calls_per_minute: int = 60           # Max new calls per minute system-wide
    calls_per_user_per_hour: int = 10    # Max calls per unique user per hour
    concurrent_calls_max: int = 500      # Absolute maximum concurrent calls
    burst_factor: float = 1.5            # Allow 50% burst above per-minute limit


class VoiceRateLimiter:
    """
    Redis-backed rate limiter for voice agent calls.
    Uses sliding window algorithm for accurate rate limiting.
    """

    def __init__(self, redis_url: str, config: RateLimitConfig = None):
        self.redis_url = redis_url
        self.config = config or RateLimitConfig()
        self._redis: Optional[aioredis.Redis] = None

    async def _get_redis(self) -> aioredis.Redis:
        if not self._redis:
            self._redis = await aioredis.from_url(self.redis_url, decode_responses=True)
        return self._redis

    async def check_and_record_call(
        self,
        user_id: Optional[str] = None,
        current_concurrent: int = 0,
    ) -> Tuple[bool, str]:
        """
        Check if a new call should be allowed.
        Returns (allowed: bool, reason: str).
        """
        r = await self._get_redis()
        now = time.time()

        # 1. Check absolute concurrent limit
        if current_concurrent >= self.config.concurrent_calls_max:
            return False, f"System at capacity ({current_concurrent} concurrent calls)"

        # 2. Check system-wide calls per minute (sliding window)
        minute_key = "voice:rate:system:minute"
        minute_window_start = now - 60

        # Remove old entries from sliding window
        await r.zremrangebyscore(minute_key, 0, minute_window_start)
        minute_count = await r.zcard(minute_key)

        burst_limit = int(self.config.calls_per_minute * self.config.burst_factor)
        if minute_count >= burst_limit:
            return False, f"System rate limit exceeded ({minute_count}/{burst_limit} calls/min)"

        # 3. Check per-user limit (if user_id provided)
        if user_id:
            user_key = f"voice:rate:user:{user_id}:hour"
            hour_window_start = now - 3600
            await r.zremrangebyscore(user_key, 0, hour_window_start)
            user_count = await r.zcard(user_key)

            if user_count >= self.config.calls_per_user_per_hour:
                return False, (
                    f"User rate limit exceeded ({user_count}/{self.config.calls_per_user_per_hour} "
                    f"calls/hour)"
                )

        # 4. All checks passed  record this call
        call_id = f"{now}-{id(now)}"
        pipe = r.pipeline()
        pipe.zadd(minute_key, {call_id: now})
        pipe.expire(minute_key, 120)  # 2-minute TTL

        if user_id:
            pipe.zadd(f"voice:rate:user:{user_id}:hour", {call_id: now})
            pipe.expire(f"voice:rate:user:{user_id}:hour", 7200)

        await pipe.execute()

        return True, "allowed"

    async def get_system_stats(self) -> dict:
        """Get current rate limiting statistics."""
        r = await self._get_redis()
        now = time.time()

        # Current minute call count
        minute_key = "voice:rate:system:minute"
        await r.zremrangebyscore(minute_key, 0, now - 60)
        calls_last_minute = await r.zcard(minute_key)

        return {
            "calls_last_minute": calls_last_minute,
            "calls_per_minute_limit": self.config.calls_per_minute,
            "burst_limit": int(self.config.calls_per_minute * self.config.burst_factor),
            "utilization_pct": round(
                calls_last_minute / self.config.calls_per_minute * 100, 1
            ),
        }
```

### Capacity Planning Calculator

```python
"""
capacity_planner.py  Calculate required infrastructure for voice workloads.
"""
from dataclasses import dataclass
import math


@dataclass
class WorkloadProfile:
    """Define your expected voice agent workload."""
    peak_concurrent_calls: int      # Max simultaneous calls you need to handle
    avg_call_duration_minutes: float  # Average call length
    calls_per_hour_peak: int        # Calls arriving per hour at peak
    calls_per_hour_average: int     # Average calls per hour


@dataclass
class ServiceCosts:
    """API service costs per unit."""
    asr_per_minute: float = 0.0059      # Deepgram Nova-2
    llm_input_per_1k: float = 0.00015   # GPT-4o-mini input
    llm_output_per_1k: float = 0.00060  # GPT-4o-mini output
    tts_per_1k_chars: float = 0.015     # OpenAI TTS
    telephony_per_minute: float = 0.0085  # Twilio


@dataclass
class InfraCapacityPlan:
    """Output of capacity planning calculation."""
    # Compute
    min_worker_pods: int
    max_worker_pods: int
    cpu_cores_per_pod: float
    memory_gb_per_pod: float
    pod_spec: str

    # Monthly cost estimates
    api_cost_per_call: float
    api_cost_monthly: float
    infra_cost_monthly: float
    total_cost_monthly: float

    # Redis sizing
    redis_memory_gb: float

    # Database sizing
    postgres_storage_gb_per_month: float


def calculate_capacity(
    workload: WorkloadProfile,
    costs: ServiceCosts = None,
    calls_per_worker: int = 50,  # Tune based on benchmarking
) -> InfraCapacityPlan:
    """Calculate required infrastructure for a voice agent deployment."""
    if costs is None:
        costs = ServiceCosts()

    # Workers needed
    min_pods = math.ceil(workload.calls_per_hour_average / 60 * workload.avg_call_duration_minutes / calls_per_worker)
    max_pods = math.ceil(workload.peak_concurrent_calls / calls_per_worker)
    min_pods = max(2, min_pods)  # Always at least 2 for HA

    # Per-call API cost estimate
    avg_tokens_per_call = 2000  # Input + output tokens
    avg_chars_per_call = 800    # TTS characters
    call_asr_cost = workload.avg_call_duration_minutes * costs.asr_per_minute
    call_llm_cost = (avg_tokens_per_call / 1000) * costs.llm_input_per_1k * 3  # 3x output ratio
    call_tts_cost = (avg_chars_per_call / 1000) * costs.tts_per_1k_chars
    call_tel_cost = workload.avg_call_duration_minutes * costs.telephony_per_minute
    api_cost_per_call = call_asr_cost + call_llm_cost + call_tts_cost + call_tel_cost

    # Monthly API cost
    monthly_calls = workload.calls_per_hour_average * 24 * 30
    api_cost_monthly = monthly_calls * api_cost_per_call

    # Infra cost (GCP n2-standard-2 ≈ $0.10/hr)
    pod_cost_per_hour = 0.10
    avg_pods = (min_pods + max_pods) / 2
    infra_cost_monthly = avg_pods * pod_cost_per_hour * 24 * 30

    # Redis sizing: ~2KB per session, keep 2x for overhead
    redis_sessions = workload.peak_concurrent_calls * 2
    redis_memory_gb = max(1.0, redis_sessions * 0.002)  # 2KB per session

    # Postgres: ~10KB per call record + turns
    postgres_storage_gb_per_month = (monthly_calls * 0.00001)  # 10KB per call

    return InfraCapacityPlan(
        min_worker_pods=min_pods,
        max_worker_pods=max_pods,
        cpu_cores_per_pod=2.0,
        memory_gb_per_pod=2.0,
        pod_spec="n2-standard-2 (2 vCPU, 8GB RAM)",
        api_cost_per_call=round(api_cost_per_call, 4),
        api_cost_monthly=round(api_cost_monthly, 2),
        infra_cost_monthly=round(infra_cost_monthly, 2),
        total_cost_monthly=round(api_cost_monthly + infra_cost_monthly, 2),
        redis_memory_gb=round(redis_memory_gb, 1),
        postgres_storage_gb_per_month=round(postgres_storage_gb_per_month, 2),
    )


# Example: Customer service bot handling 500 calls/day
example_workload = WorkloadProfile(
    peak_concurrent_calls=50,
    avg_call_duration_minutes=3.5,
    calls_per_hour_peak=80,
    calls_per_hour_average=21,  # 500/day / 24h
)

plan = calculate_capacity(example_workload)

print(f"""
=== Capacity Plan: 500 calls/day ===

Infrastructure:
  Min pods: {plan.min_worker_pods}
  Max pods: {plan.max_worker_pods} (HPA target)
  Pod size: {plan.pod_spec}
  Redis: {plan.redis_memory_gb}GB

Costs:
  API cost/call:  ${plan.api_cost_per_call:.4f}
  API cost/month: ${plan.api_cost_monthly:,.2f}
  Infra/month:    ${plan.infra_cost_monthly:,.2f}
  Total/month:    ${plan.total_cost_monthly:,.2f}
""")
```

### Kubernetes HPA with Custom Metrics

To scale based on active calls per pod (not just CPU), you need the Prometheus Adapter:

```yaml
# kubernetes/hpa-custom-metrics.yaml
# Requires: prometheus-adapter installed in cluster

apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: voice-worker-hpa
  namespace: voice-agent
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: voice-worker
  minReplicas: 2
  maxReplicas: 30

  metrics:
    # Scale based on active calls per pod
    - type: Pods
      pods:
        metric:
          name: voice_active_calls
        target:
          type: AverageValue
          averageValue: "30"   # Scale when avg >30 calls/pod

    # Also scale on CPU as a safety net
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70

  behavior:
    scaleUp:
      # Scale up quickly: allow adding 4 pods at once
      policies:
        - type: Pods
          value: 4
          periodSeconds: 60
      stabilizationWindowSeconds: 30   # Start scaling up after 30s above target

    scaleDown:
      # Scale down slowly: drain calls first
      policies:
        - type: Pods
          value: 1
          periodSeconds: 300   # Remove max 1 pod every 5 minutes
      stabilizationWindowSeconds: 600  # Wait 10min before scaling down
```

---

## Vocabulary Cheat Sheet

| Term | Definition |
|------|-----------|
| **Sticky session** | Load balancer feature that routes all requests from one client to the same server |
| **HPA** | Horizontal Pod Autoscaler  Kubernetes feature that scales pods based on metrics |
| **MOS score** | Mean Opinion Score  1-5 scale for audio quality (5 = excellent) |
| **Circuit breaker** | Design pattern that stops calls to a failing service to prevent cascade failures |
| **Prometheus** | Open-source time-series monitoring and alerting system |
| **Grafana** | Visualization platform for Prometheus metrics (dashboards) |
| **Histogram** | Prometheus metric type that tracks value distributions (useful for latency) |
| **Gauge** | Prometheus metric for values that go up and down (e.g., active calls) |
| **Counter** | Prometheus metric that only increases (e.g., total calls) |
| **p95 latency** | 95th percentile latency  95% of requests are faster than this |
| **Worker pool** | A set of processes that handle concurrent calls |
| **Session affinity** | Kubernetes service configuration for sticky sessions |
| **Pre-warm** | Loading models before calls arrive to reduce first-call latency |
| **Half-open state** | Circuit breaker state where limited requests test if the service recovered |
| **Ramp-up** | Gradually increasing load during load testing to avoid thundering herd |
| **Thundering herd** | Many requests arriving simultaneously, often after an outage ends |
| **APScheduler** | Python library for scheduling periodic tasks |

---

## What's Next

In **Part 18: Security, Testing, and Compliance**, we add the enterprise-grade protections that make voice agents trustworthy:

- **Threat modeling**  what can go wrong with voice AI systems
- **Prompt injection via voice**  detecting adversarial speech
- **PII detection and redaction**  ensuring sensitive data doesn't leak into logs
- **GDPR, HIPAA, TCPA, PCI DSS**  compliance requirements for voice agents
- **Conversation simulation testing**  testing without real audio
- **Red teaming**  adversarial testing of your voice agent
- **A/B testing**  data-driven improvement of voice agent performance

By the end of Part 18, your voice agent will be ready for enterprise deployment.

---

*Part 17 of 19  Building Voice Agents: A Developer's Deep Dive*
