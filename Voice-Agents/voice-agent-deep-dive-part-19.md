# Voice Agents Deep Dive  Part 19: The Capstone  Building a Production Voice Agent Platform

---

**Series:** Building Voice Agents  A Developer's Deep Dive from Audio Fundamentals to Production
**Part:** 19 of 19 (Capstone)
**Audience:** Developers with Python experience who want to build voice-powered AI agents from the ground up
**Reading time:** ~60 minutes

---

## Introduction and Recap of Part 18

In **Part 18**, we tackled the unglamorous but absolutely essential work of **Security, Testing, and Compliance**. We built a threat model for voice agents, implemented PII redaction pipelines that strip sensitive data before it ever reaches logs or storage, and walked through regulatory frameworks: GDPR's right-to-erasure requirements, HIPAA's strict rules around PHI in healthcare voice workflows, and TCPA's consent requirements for outbound calling in the United States. We assembled automated test suites that cover unit tests for VAD and ASR mocking, integration tests for the full pipeline, and adversarial tests that probe the agent for prompt injection and data leakage.

Security and compliance are not afterthoughts. They are the foundation on which a trustworthy voice product is built. Without them, everything else we have built  however technically elegant  cannot be deployed in production.

And now, with all of that foundation in place, we arrive at **Part 19: The Capstone**.

---

## Welcome to the Grand Finale

This is it. Part 19 of 19.

Over the previous eighteen parts, we have traveled from raw audio bytes to production-grade AI systems. We have explored:

- **Part 0:** Why voice agents are hard and what makes them different from chatbots
- **Part 1:** Audio fundamentals  sample rates, PCM, codec tradeoffs
- **Part 2:** Voice Activity Detection  energy-based and ML-based VAD
- **Part 3:** Automatic Speech Recognition  streaming ASR with Whisper and Deepgram
- **Part 4:** Text-to-Speech  neural TTS, SSML, prosody control
- **Part 5:** End-to-end latency  measurement, budgeting, optimization
- **Part 6:** Telephony integration  Twilio, SIP trunks, WebRTC
- **Part 7:** Barge-in and interruption handling
- **Part 8:** LLM integration  streaming completions, function calling
- **Part 9:** Intent recognition and entity extraction
- **Part 10:** Dialog management  turn-taking, state machines, slot filling
- **Part 11:** Memory systems  short-term context, long-term retrieval
- **Part 12:** Multi-modal agents  DTMF, transfers, IVR fallback
- **Part 13:** Streaming architectures  WebSocket pipelines, backpressure
- **Part 14:** Observability  distributed tracing, metrics, dashboards
- **Part 15:** Cost optimization  model routing, caching, batching
- **Part 16:** Scalability  horizontal scaling, load balancing, stateless workers
- **Part 17:** Agent personas and customization  voice, tone, DSL config
- **Part 18:** Security, testing, and compliance

Now we bring it all together into a **complete, production-ready voice agent platform**  one that you could actually deploy, scale, and monetize.

This article is long by design. It is the reference implementation. Take your time with it.

---

## 1. System Architecture Overview

### 1.1 The Full Platform Diagram

Before writing a single line of code, we need a clear picture of what we are building. A production voice agent platform is not a single service  it is an ecosystem of cooperating components, each with a well-defined responsibility.

```mermaid
graph TB
    subgraph Ingress["Ingress Layer"]
        PSTN["PSTN / Phone Network"]
        WEB["WebRTC Browser Client"]
        SDK["Mobile SDK"]
    end

    subgraph Gateway["Voice Gateway (FastAPI)"]
        TW["Twilio Webhook Handler"]
        WS["WebSocket Audio Handler"]
        AUTH["Auth Middleware"]
        SM["Session Manager"]
    end

    subgraph Workers["Agent Worker Pool (Celery / asyncio)"]
        PM["Pipeline Manager"]
        VAD["VAD Engine"]
        ASR["ASR Engine (Whisper/Deepgram)"]
        LLM["LLM Engine (GPT-4o / Claude)"]
        TTS["TTS Engine (ElevenLabs/Azure)"]
    end

    subgraph State["State Layer"]
        REDIS["Redis (Session + Cache)"]
        CHROMA["ChromaDB (Long-term Memory)"]
        PG["PostgreSQL (Analytics + Audit)"]
    end

    subgraph Models["Model Services"]
        OPENAI["OpenAI API"]
        ANTHROPIC["Anthropic API"]
        DEEPGRAM["Deepgram ASR"]
        ELEVENLABS["ElevenLabs TTS"]
    end

    subgraph Ops["Observability Stack"]
        PROM["Prometheus"]
        GRAF["Grafana"]
        JAEGER["Jaeger Tracing"]
        ALERT["AlertManager"]
    end

    PSTN --> TW
    WEB --> WS
    SDK --> WS
    TW --> AUTH
    WS --> AUTH
    AUTH --> SM
    SM --> PM
    PM --> VAD
    PM --> ASR
    PM --> LLM
    PM --> TTS
    SM --> REDIS
    LLM --> CHROMA
    PM --> PG
    ASR --> DEEPGRAM
    LLM --> OPENAI
    LLM --> ANTHROPIC
    TTS --> ELEVENLABS
    PM --> PROM
    PROM --> GRAF
    PROM --> ALERT
```

### 1.2 Design Principles

Every architectural decision in this platform flows from three core principles:

**Principle 1: Stateless Workers**
Worker processes hold zero conversation state in memory between requests. All state  session data, conversation history, slot values  lives in Redis. This means any worker can handle any request, enabling horizontal scaling and zero-downtime deployments.

**Principle 2: Externalized State**
The boundary between compute and state is explicit and enforced. Workers read from and write to Redis and PostgreSQL. They never share memory. This allows the state layer to be scaled, replicated, and backed up independently from the compute layer.

**Principle 3: Graceful Degradation**
Every external dependency has a fallback. If Deepgram ASR is unavailable, fall back to local Whisper. If ElevenLabs TTS is unavailable, fall back to Azure TTS or a cached audio file. If the LLM is slow, return a "please hold" audio fragment and retry. The platform never hangs silently  it always takes the best available action.

### 1.3 Technology Stack Decision Table

| Layer | Primary Choice | Fallback / Alternative | Rationale |
|---|---|---|---|
| **Telephony** | Twilio | Vonage, Telnyx | Largest ecosystem, best docs |
| **WebRTC** | aiortc (Python) | Pion (Go sidecar) | Pure Python for simplicity |
| **API Framework** | FastAPI | Flask | Async-native, auto OpenAPI docs |
| **Task Queue** | asyncio + Redis Pub/Sub | Celery + RabbitMQ | Lower latency for real-time audio |
| **ASR** | Deepgram Nova-2 | OpenAI Whisper (local) | Streaming, 300ms latency |
| **LLM** | GPT-4o | Claude 3.5 Sonnet | Function calling, speed |
| **TTS** | ElevenLabs | Azure Neural TTS | Voice quality, streaming |
| **Short-term Memory** | Redis (TTL-keyed) | In-process dict | Sub-ms reads, persistence |
| **Long-term Memory** | ChromaDB | Pinecone | Local-first, no API cost |
| **Analytics DB** | PostgreSQL + TimescaleDB | SQLite (dev) | Time-series queries, ACID |
| **Metrics** | Prometheus + Grafana | Datadog | Open-source, self-hosted |
| **Tracing** | OpenTelemetry + Jaeger | AWS X-Ray | Vendor-neutral |
| **Container Orchestration** | Kubernetes | Docker Compose (dev) | Production-grade |
| **Secret Management** | HashiCorp Vault | AWS Secrets Manager | Multi-cloud compatible |

---

## 2. Data Models (Pydantic)

A production system needs rigorous data models. These are not just type hints  they are the contract between every service in the platform. We use Pydantic v2 throughout.

```python
# platform/models.py
"""
Core data models for the Voice Agent Platform.
All inter-service communication uses these validated models.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class CallDirection(str, Enum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"

class CallStatus(str, Enum):
    INITIATED = "initiated"
    RINGING = "ringing"
    ANSWERED = "answered"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NO_ANSWER = "no_answer"

class SessionStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"

class TurnRole(str, Enum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"

class AgentStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


# ---------------------------------------------------------------------------
# Core Models
# ---------------------------------------------------------------------------

class User(BaseModel):
    """A caller / end-user of the voice platform."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    phone_number: Optional[str] = None          # E.164 format
    email: Optional[str] = None
    display_name: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    consent_recorded_at: Optional[datetime] = None   # TCPA / GDPR consent
    pii_redacted: bool = False

    @field_validator("phone_number")
    @classmethod
    def validate_e164(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.startswith("+"):
            raise ValueError("Phone number must be in E.164 format (+1234567890)")
        return v


class AgentConfig(BaseModel):
    """Configuration for a single deployed voice agent."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    status: AgentStatus = AgentStatus.DRAFT
    version: str = "1.0.0"

    # Persona
    system_prompt: str
    voice_id: str = "rachel"               # TTS voice identifier
    language: str = "en-US"
    speaking_rate: float = Field(1.0, ge=0.5, le=2.0)

    # Model routing
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    llm_temperature: float = Field(0.7, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(512, ge=64, le=4096)

    # ASR config
    asr_provider: str = "deepgram"
    asr_model: str = "nova-2"
    asr_language: str = "en-US"

    # TTS config
    tts_provider: str = "elevenlabs"
    tts_model: str = "eleven_turbo_v2"

    # Behavior
    max_call_duration_seconds: int = 1800   # 30 minutes
    silence_timeout_seconds: int = 10
    barge_in_enabled: bool = True
    max_turns: int = 100
    fallback_message: str = "I'm sorry, I encountered an error. Please hold."

    # Tools / functions available to this agent
    tools: list[dict[str, Any]] = Field(default_factory=list)

    # Compliance
    requires_consent: bool = False
    gdpr_enabled: bool = True
    hipaa_enabled: bool = False
    tcpa_enabled: bool = False
    recording_enabled: bool = True
    pii_redaction_enabled: bool = True

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None

    @field_validator("system_prompt")
    @classmethod
    def prompt_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("system_prompt cannot be empty")
        return v


class Turn(BaseModel):
    """A single conversational turn (one utterance from user or agent)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    sequence_number: int
    role: TurnRole
    transcript: str
    audio_url: Optional[str] = None        # Stored audio for this turn
    intent: Optional[str] = None
    entities: dict[str, Any] = Field(default_factory=dict)

    # Latency breakdown (milliseconds)
    vad_latency_ms: Optional[float] = None
    asr_latency_ms: Optional[float] = None
    llm_latency_ms: Optional[float] = None
    tts_latency_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None

    # Cost tracking
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    tts_characters: int = 0
    estimated_cost_usd: float = 0.0

    # Quality signals
    asr_confidence: Optional[float] = None
    user_sentiment: Optional[str] = None   # positive / neutral / negative
    was_barge_in: bool = False
    was_interrupted: bool = False

    created_at: datetime = Field(default_factory=datetime.utcnow)


class ConversationMemory(BaseModel):
    """The working memory for an active conversation."""
    session_id: str
    short_term: list[dict[str, str]] = Field(default_factory=list)  # Recent turns as {role, content}
    slot_values: dict[str, Any] = Field(default_factory=dict)       # Extracted entities
    intent_history: list[str] = Field(default_factory=list)
    retrieved_context: list[str] = Field(default_factory=list)      # RAG results
    summary: Optional[str] = None                                    # LLM-generated summary
    turn_count: int = 0
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def add_turn(self, role: TurnRole, content: str, max_short_term: int = 20) -> None:
        """Append a turn and trim to max_short_term entries."""
        self.short_term.append({"role": role.value, "content": content})
        if len(self.short_term) > max_short_term:
            self.short_term = self.short_term[-max_short_term:]
        self.turn_count += 1
        self.updated_at = datetime.utcnow()

    def to_llm_messages(self) -> list[dict[str, str]]:
        """Format memory as OpenAI-compatible message list."""
        messages = []
        if self.retrieved_context:
            context_block = "\n\n".join(self.retrieved_context)
            messages.append({
                "role": "system",
                "content": f"Relevant context from memory:\n{context_block}"
            })
        messages.extend(self.short_term)
        return messages


class Session(BaseModel):
    """A single call/conversation session."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    call_id: str
    agent_id: str
    user_id: Optional[str] = None
    status: SessionStatus = SessionStatus.ACTIVE
    worker_id: Optional[str] = None        # Which worker process owns this session
    channel: str = "phone"                 # phone | webrtc | sdk

    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # Aggregated metrics (updated per turn)
    total_turns: int = 0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    task_completed: Optional[bool] = None
    csat_score: Optional[int] = None       # 1-5

    # Compliance
    recording_url: Optional[str] = None
    consent_obtained: bool = False
    pii_redacted: bool = False

    metadata: dict[str, Any] = Field(default_factory=dict)

    def close(self, task_completed: Optional[bool] = None) -> None:
        self.status = SessionStatus.ENDED
        self.ended_at = datetime.utcnow()
        if self.started_at:
            self.duration_seconds = (self.ended_at - self.started_at).total_seconds()
        if task_completed is not None:
            self.task_completed = task_completed


class Call(BaseModel):
    """A telephony call record (one-to-one with a Session for phone calls)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    external_id: Optional[str] = None      # Twilio CallSid
    direction: CallDirection
    status: CallStatus = CallStatus.INITIATED
    from_number: str
    to_number: str
    agent_id: str
    session_id: Optional[str] = None

    initiated_at: datetime = Field(default_factory=datetime.utcnow)
    answered_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    billable_duration_seconds: Optional[float] = None
    telephony_cost_usd: Optional[float] = None

    metadata: dict[str, Any] = Field(default_factory=dict)
```

### 2.1 Entity Relationship Diagram

```mermaid
erDiagram
    USER {
        string id PK
        string phone_number
        string email
        string display_name
        datetime consent_recorded_at
        bool pii_redacted
    }

    AGENT_CONFIG {
        string id PK
        string name
        string system_prompt
        string llm_model
        string asr_provider
        string tts_provider
        string version
        string status
    }

    CALL {
        string id PK
        string external_id
        string direction
        string status
        string from_number
        string to_number
        string agent_id FK
        string session_id FK
    }

    SESSION {
        string id PK
        string call_id FK
        string agent_id FK
        string user_id FK
        string status
        int total_turns
        float total_cost_usd
        float avg_latency_ms
        bool task_completed
    }

    TURN {
        string id PK
        string session_id FK
        int sequence_number
        string role
        string transcript
        float asr_latency_ms
        float llm_latency_ms
        float tts_latency_ms
        float estimated_cost_usd
    }

    CONVERSATION_MEMORY {
        string session_id PK
        json short_term
        json slot_values
        string summary
        int turn_count
    }

    USER ||--o{ SESSION : "participates in"
    AGENT_CONFIG ||--o{ SESSION : "handles"
    SESSION ||--o{ TURN : "contains"
    SESSION ||--|| CALL : "associated with"
    SESSION ||--|| CONVERSATION_MEMORY : "has"
```

---

## 3. Core Services

### 3.1 Session Manager

The `SessionManager` is the gatekeeper. Every call starts and ends here. It writes sessions to Redis with a TTL, ensuring that if a worker crashes mid-call, the session can be recovered or expired cleanly.

```python
# platform/services/session_manager.py
"""
SessionManager: Create, retrieve, update, and destroy conversation sessions.
Backed by Redis for sub-millisecond access and automatic TTL-based cleanup.
"""

import json
import logging
from datetime import timedelta
from typing import Optional

import redis.asyncio as aioredis

from platform.models import ConversationMemory, Session, SessionStatus

logger = logging.getLogger(__name__)

SESSION_TTL = timedelta(hours=2)        # Sessions expire 2h after last activity
MEMORY_TTL = timedelta(hours=24)        # Memory kept longer for post-call analysis
SESSION_PREFIX = "session:"
MEMORY_PREFIX = "memory:"


class SessionManager:
    """
    Manages conversation sessions in Redis.

    All methods are async to avoid blocking the event loop.
    Sessions are stored as JSON blobs with configurable TTL.
    """

    def __init__(self, redis_client: aioredis.Redis) -> None:
        self._redis = redis_client

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    async def create_session(self, session: Session) -> Session:
        """Persist a new session to Redis."""
        key = self._session_key(session.id)
        payload = session.model_dump_json()
        await self._redis.setex(key, SESSION_TTL, payload)
        logger.info("Session created: %s (agent=%s)", session.id, session.agent_id)
        return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by ID. Returns None if not found or expired."""
        key = self._session_key(session_id)
        raw = await self._redis.get(key)
        if raw is None:
            return None
        # Refresh TTL on access (sliding expiry)
        await self._redis.expire(key, SESSION_TTL)
        return Session.model_validate_json(raw)

    async def update_session(self, session: Session) -> None:
        """Overwrite session data and refresh TTL."""
        key = self._session_key(session.id)
        await self._redis.setex(key, SESSION_TTL, session.model_dump_json())

    async def end_session(
        self,
        session_id: str,
        task_completed: Optional[bool] = None,
    ) -> Optional[Session]:
        """Mark session as ended, persist final state, return closed session."""
        session = await self.get_session(session_id)
        if session is None:
            logger.warning("end_session called for unknown session: %s", session_id)
            return None

        session.close(task_completed=task_completed)
        await self.update_session(session)
        logger.info(
            "Session ended: %s | duration=%.1fs | turns=%d | cost=$%.4f",
            session_id,
            session.duration_seconds or 0,
            session.total_turns,
            session.total_cost_usd,
        )
        return session

    async def delete_session(self, session_id: str) -> None:
        """Hard-delete session and memory (GDPR erasure)."""
        await self._redis.delete(
            self._session_key(session_id),
            self._memory_key(session_id),
        )
        logger.info("Session and memory deleted (GDPR): %s", session_id)

    # ------------------------------------------------------------------
    # Conversation Memory
    # ------------------------------------------------------------------

    async def get_memory(self, session_id: str) -> ConversationMemory:
        """Load memory for a session, or create empty memory if none exists."""
        key = self._memory_key(session_id)
        raw = await self._redis.get(key)
        if raw is None:
            return ConversationMemory(session_id=session_id)
        await self._redis.expire(key, MEMORY_TTL)
        return ConversationMemory.model_validate_json(raw)

    async def save_memory(self, memory: ConversationMemory) -> None:
        """Persist updated memory back to Redis."""
        key = self._memory_key(memory.session_id)
        await self._redis.setex(key, MEMORY_TTL, memory.model_dump_json())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _session_key(self, session_id: str) -> str:
        return f"{SESSION_PREFIX}{session_id}"

    def _memory_key(self, session_id: str) -> str:
        return f"{MEMORY_PREFIX}{session_id}"
```

### 3.2 Memory Service

The `MemoryService` wraps both short-term (Redis) and long-term (ChromaDB) memory. When an agent needs context for a new caller, it retrieves relevant past interactions and user facts from the vector store.

```python
# platform/services/memory_service.py
"""
MemoryService: Unified interface for short-term and long-term memory.

Short-term: Redis (conversation turns, slot values)  managed by SessionManager.
Long-term: ChromaDB (user facts, past interaction summaries)  managed here.
"""

import logging
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

from platform.models import ConversationMemory, Session, TurnRole
from platform.services.session_manager import SessionManager

logger = logging.getLogger(__name__)

CHROMA_COLLECTION = "voice_agent_memory"
MAX_RETRIEVAL_RESULTS = 5


class MemoryService:
    """
    Retrieval-augmented memory for voice agents.

    On each turn:
      1. Retrieve relevant long-term context for the user query.
      2. Inject retrieved context into the conversation memory.
      3. After session ends, summarize and store summary to long-term.
    """

    def __init__(
        self,
        session_manager: SessionManager,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
    ) -> None:
        self._sm = session_manager
        self._chroma = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        self._embed_fn = embedding_functions.OpenAIEmbeddingFunction(
            model_name="text-embedding-3-small"
        )
        self._collection = self._chroma.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=self._embed_fn,
        )

    async def enrich_memory(
        self,
        session: Session,
        user_utterance: str,
    ) -> ConversationMemory:
        """
        Load short-term memory and inject retrieved long-term context.
        Call this BEFORE constructing the LLM prompt.
        """
        memory = await self._sm.get_memory(session.id)

        # Retrieve from long-term store using the user's utterance as query
        if session.user_id:
            retrieved = self._retrieve_long_term(
                user_id=session.user_id,
                query=user_utterance,
            )
            memory.retrieved_context = retrieved

        return memory

    async def update_memory(
        self,
        session: Session,
        user_utterance: str,
        agent_response: str,
        slots: Optional[dict] = None,
    ) -> None:
        """
        Append the latest turn to short-term memory and persist.
        Optionally update slot values.
        """
        memory = await self._sm.get_memory(session.id)
        memory.add_turn(TurnRole.USER, user_utterance)
        memory.add_turn(TurnRole.AGENT, agent_response)

        if slots:
            memory.slot_values.update(slots)

        await self._sm.save_memory(memory)

    async def consolidate_to_long_term(
        self,
        session: Session,
        llm_summary: str,
    ) -> None:
        """
        After a session ends, store the conversation summary to ChromaDB
        so it can be retrieved in future calls from the same user.
        """
        if not session.user_id:
            return  # Anonymous call  nothing to store

        doc_id = f"session:{session.id}"
        metadata = {
            "user_id": session.user_id,
            "agent_id": session.agent_id,
            "session_id": session.id,
            "task_completed": str(session.task_completed),
            "turn_count": session.total_turns,
        }

        self._collection.upsert(
            ids=[doc_id],
            documents=[llm_summary],
            metadatas=[metadata],
        )
        logger.info("Stored long-term memory for user %s", session.user_id)

    def _retrieve_long_term(
        self,
        user_id: str,
        query: str,
        n_results: int = MAX_RETRIEVAL_RESULTS,
    ) -> list[str]:
        """Query ChromaDB for relevant past interactions for this user."""
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"user_id": user_id},
            )
            documents: list[list[str]] = results.get("documents", [[]])
            return documents[0] if documents else []
        except Exception as exc:
            logger.warning("Long-term memory retrieval failed: %s", exc)
            return []
```

### 3.3 Agent Service

```python
# platform/services/agent_service.py
"""
AgentService: Load, validate, cache, and serve agent configurations.

Agents are stored in PostgreSQL and cached in Redis.
Cache TTL is 5 minutes  short enough to pick up config changes quickly.
"""

import json
import logging
from datetime import timedelta
from typing import Optional

import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession as DBSession
from sqlalchemy import select, text

from platform.models import AgentConfig, AgentStatus

logger = logging.getLogger(__name__)

AGENT_CACHE_TTL = timedelta(minutes=5)
AGENT_CACHE_PREFIX = "agent:"


class AgentNotFoundError(Exception):
    pass


class AgentService:
    """CRUD + caching layer for AgentConfig objects."""

    def __init__(self, db: DBSession, redis: aioredis.Redis) -> None:
        self._db = db
        self._redis = redis

    async def get_agent(self, agent_id: str) -> AgentConfig:
        """
        Retrieve agent config. Check Redis cache first, then PostgreSQL.
        Raises AgentNotFoundError if the agent does not exist or is not active.
        """
        # 1. Cache hit?
        cached = await self._redis.get(f"{AGENT_CACHE_PREFIX}{agent_id}")
        if cached:
            return AgentConfig.model_validate_json(cached)

        # 2. Database lookup
        result = await self._db.execute(
            text("SELECT config_json FROM agents WHERE id = :id AND status = 'active'"),
            {"id": agent_id},
        )
        row = result.fetchone()
        if row is None:
            raise AgentNotFoundError(f"Agent {agent_id} not found or not active")

        agent = AgentConfig.model_validate_json(row[0])

        # 3. Populate cache
        await self._redis.setex(
            f"{AGENT_CACHE_PREFIX}{agent_id}",
            AGENT_CACHE_TTL,
            agent.model_dump_json(),
        )
        return agent

    async def create_agent(self, config: AgentConfig) -> AgentConfig:
        """Persist a new agent configuration."""
        config.status = AgentStatus.DRAFT
        await self._db.execute(
            text(
                "INSERT INTO agents (id, name, status, config_json, created_at) "
                "VALUES (:id, :name, :status, :config, :created_at)"
            ),
            {
                "id": config.id,
                "name": config.name,
                "status": config.status.value,
                "config": config.model_dump_json(),
                "created_at": config.created_at,
            },
        )
        await self._db.commit()
        logger.info("Agent created: %s (%s)", config.name, config.id)
        return config

    async def activate_agent(self, agent_id: str) -> AgentConfig:
        """Promote an agent from DRAFT to ACTIVE."""
        await self._db.execute(
            text("UPDATE agents SET status = 'active' WHERE id = :id"),
            {"id": agent_id},
        )
        await self._db.commit()
        # Invalidate cache
        await self._redis.delete(f"{AGENT_CACHE_PREFIX}{agent_id}")
        return await self.get_agent(agent_id)

    async def invalidate_cache(self, agent_id: str) -> None:
        await self._redis.delete(f"{AGENT_CACHE_PREFIX}{agent_id}")
```

### 3.4 Analytics Service

```python
# platform/services/analytics_service.py
"""
AnalyticsService: Log turns to PostgreSQL (TimescaleDB) and compute metrics.

Uses async SQLAlchemy for non-blocking writes. Metrics are computed
via SQL aggregations and exposed through the /analytics/dashboard endpoint.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession as DBSession

from platform.models import Session, Turn

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Write and read voice agent analytics."""

    def __init__(self, db: DBSession) -> None:
        self._db = db

    async def log_turn(self, turn: Turn) -> None:
        """Persist a completed turn to the turns table."""
        await self._db.execute(
            text("""
                INSERT INTO turns (
                    id, session_id, sequence_number, role, transcript,
                    vad_latency_ms, asr_latency_ms, llm_latency_ms,
                    tts_latency_ms, total_latency_ms, llm_input_tokens,
                    llm_output_tokens, tts_characters, estimated_cost_usd,
                    asr_confidence, was_barge_in, created_at
                ) VALUES (
                    :id, :session_id, :seq, :role, :transcript,
                    :vad, :asr, :llm, :tts, :total, :in_tok,
                    :out_tok, :chars, :cost, :conf, :barge_in, :created_at
                )
            """),
            {
                "id": turn.id,
                "session_id": turn.session_id,
                "seq": turn.sequence_number,
                "role": turn.role.value,
                "transcript": turn.transcript,
                "vad": turn.vad_latency_ms,
                "asr": turn.asr_latency_ms,
                "llm": turn.llm_latency_ms,
                "tts": turn.tts_latency_ms,
                "total": turn.total_latency_ms,
                "in_tok": turn.llm_input_tokens,
                "out_tok": turn.llm_output_tokens,
                "chars": turn.tts_characters,
                "cost": turn.estimated_cost_usd,
                "conf": turn.asr_confidence,
                "barge_in": turn.was_barge_in,
                "created_at": turn.created_at,
            },
        )
        await self._db.commit()

    async def log_session(self, session: Session) -> None:
        """Upsert session summary after it ends."""
        await self._db.execute(
            text("""
                INSERT INTO sessions (
                    id, call_id, agent_id, user_id, status, channel,
                    started_at, ended_at, duration_seconds, total_turns,
                    total_cost_usd, avg_latency_ms, task_completed, csat_score
                ) VALUES (
                    :id, :call_id, :agent_id, :user_id, :status, :channel,
                    :started_at, :ended_at, :duration, :turns,
                    :cost, :latency, :task, :csat
                )
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    ended_at = EXCLUDED.ended_at,
                    duration_seconds = EXCLUDED.duration_seconds,
                    total_turns = EXCLUDED.total_turns,
                    total_cost_usd = EXCLUDED.total_cost_usd,
                    avg_latency_ms = EXCLUDED.avg_latency_ms,
                    task_completed = EXCLUDED.task_completed,
                    csat_score = EXCLUDED.csat_score
            """),
            {
                "id": session.id,
                "call_id": session.call_id,
                "agent_id": session.agent_id,
                "user_id": session.user_id,
                "status": session.status.value,
                "channel": session.channel,
                "started_at": session.started_at,
                "ended_at": session.ended_at,
                "duration": session.duration_seconds,
                "turns": session.total_turns,
                "cost": session.total_cost_usd,
                "latency": session.avg_latency_ms,
                "task": session.task_completed,
                "csat": session.csat_score,
            },
        )
        await self._db.commit()

    async def get_dashboard_metrics(
        self,
        agent_id: str,
        hours: int = 24,
    ) -> dict[str, Any]:
        """
        Compute key metrics over a rolling time window.
        Returns a dict suitable for JSON serialization.
        """
        since = datetime.utcnow() - timedelta(hours=hours)

        result = await self._db.execute(
            text("""
                SELECT
                    COUNT(*)                                        AS total_sessions,
                    AVG(duration_seconds)                           AS avg_duration_s,
                    AVG(total_turns)                                AS avg_turns,
                    AVG(total_cost_usd)                             AS avg_cost_usd,
                    SUM(total_cost_usd)                             AS total_cost_usd,
                    AVG(avg_latency_ms)                             AS avg_latency_ms,
                    SUM(CASE WHEN task_completed THEN 1 ELSE 0 END) AS completed_tasks,
                    AVG(csat_score)                                 AS avg_csat,
                    COUNT(DISTINCT user_id)                         AS unique_users
                FROM sessions
                WHERE agent_id = :agent_id
                  AND started_at >= :since
            """),
            {"agent_id": agent_id, "since": since},
        )
        row = result.fetchone()

        # Latency percentiles from turns table
        p_result = await self._db.execute(
            text("""
                SELECT
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY total_latency_ms) AS p50,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_latency_ms) AS p95,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY total_latency_ms) AS p99
                FROM turns t
                JOIN sessions s ON t.session_id = s.id
                WHERE s.agent_id = :agent_id
                  AND t.created_at >= :since
                  AND t.role = 'agent'
            """),
            {"agent_id": agent_id, "since": since},
        )
        p_row = p_result.fetchone()

        total_sessions = row[0] or 0
        completed_tasks = row[6] or 0

        return {
            "agent_id": agent_id,
            "window_hours": hours,
            "total_sessions": total_sessions,
            "avg_duration_seconds": round(row[1] or 0, 1),
            "avg_turns_per_session": round(row[2] or 0, 1),
            "avg_cost_per_session_usd": round(row[3] or 0, 4),
            "total_cost_usd": round(row[4] or 0, 2),
            "avg_latency_ms": round(row[5] or 0, 1),
            "task_completion_rate": (
                round(completed_tasks / total_sessions, 3) if total_sessions else 0
            ),
            "avg_csat": round(row[7] or 0, 2),
            "unique_users": row[8] or 0,
            "latency_p50_ms": round(p_row[0] or 0, 1) if p_row else None,
            "latency_p95_ms": round(p_row[1] or 0, 1) if p_row else None,
            "latency_p99_ms": round(p_row[2] or 0, 1) if p_row else None,
        }
```

---

## 4. The Core Pipeline

The `VoicePipeline` is the heart of the system. It orchestrates every step of a single conversation turn: VAD → ASR → memory enrichment → LLM → TTS → audio delivery, with streaming at every stage that supports it.

### 4.1 Pipeline Flow Diagram

```mermaid
sequenceDiagram
    participant C as Caller
    participant GW as Voice Gateway
    participant PM as Pipeline Manager
    participant VAD as VAD Engine
    participant ASR as ASR (Deepgram)
    participant MEM as Memory Service
    participant LLM as LLM (GPT-4o)
    participant TTS as TTS (ElevenLabs)

    C->>GW: Audio stream (PCM chunks)
    GW->>PM: Forward audio
    PM->>VAD: Detect speech boundaries
    VAD-->>PM: Speech segment detected
    PM->>ASR: Stream speech audio
    ASR-->>PM: Streaming transcript tokens
    ASR-->>PM: Final transcript + confidence
    PM->>MEM: Enrich memory (retrieve context)
    MEM-->>PM: Augmented ConversationMemory
    PM->>LLM: Messages + memory + tools
    LLM-->>PM: Streaming response tokens
    PM->>TTS: Stream text to TTS
    TTS-->>PM: Streaming audio chunks
    PM->>GW: Stream audio back
    GW->>C: Play agent audio
    PM->>MEM: Update memory (store turn)
    PM->>Analytics: Log turn metrics
```

### 4.2 Complete VoicePipeline Implementation

```python
# platform/pipeline/voice_pipeline.py
"""
VoicePipeline: The core turn-processing engine.

Responsibilities:
  - Receive raw PCM audio frames from the gateway
  - Run VAD to detect utterance boundaries
  - Stream audio to ASR for transcription
  - Enrich memory and construct LLM prompt
  - Stream LLM response to TTS
  - Stream synthesized audio back to caller
  - Track latency, cost, and quality per turn
  - Handle barge-in and error recovery
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncGenerator, AsyncIterator, Optional

from openai import AsyncOpenAI
from platform.models import AgentConfig, ConversationMemory, Session, Turn, TurnRole
from platform.services.analytics_service import AnalyticsService
from platform.services.memory_service import MemoryService
from platform.engines.vad_engine import VADEngine, SpeechSegment
from platform.engines.asr_engine import ASREngine, ASRResult
from platform.engines.tts_engine import TTSEngine

logger = logging.getLogger(__name__)

# Cost constants (USD per unit)  update as provider pricing changes
COST_GPT4O_INPUT_PER_1K = 0.005
COST_GPT4O_OUTPUT_PER_1K = 0.015
COST_ELEVENLABS_PER_CHAR = 0.000030
COST_DEEPGRAM_PER_SECOND = 0.0059 / 60  # ~$0.0059/min


@dataclass
class TurnMetrics:
    """Accumulated timing and cost for one pipeline turn."""
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    vad_start: float = 0.0
    asr_start: float = 0.0
    asr_end: float = 0.0
    llm_start: float = 0.0
    llm_first_token: float = 0.0
    llm_end: float = 0.0
    tts_start: float = 0.0
    tts_first_audio: float = 0.0
    tts_end: float = 0.0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    tts_characters: int = 0
    audio_duration_seconds: float = 0.0

    @property
    def asr_latency_ms(self) -> float:
        return (self.asr_end - self.asr_start) * 1000

    @property
    def llm_latency_ms(self) -> float:
        return (self.llm_end - self.llm_start) * 1000

    @property
    def llm_ttft_ms(self) -> float:
        """Time to first token."""
        return (self.llm_first_token - self.llm_start) * 1000

    @property
    def tts_latency_ms(self) -> float:
        return (self.tts_end - self.tts_start) * 1000

    @property
    def total_latency_ms(self) -> float:
        return (self.tts_first_audio - self.vad_start) * 1000

    @property
    def estimated_cost_usd(self) -> float:
        llm_cost = (
            (self.llm_input_tokens / 1000) * COST_GPT4O_INPUT_PER_1K
            + (self.llm_output_tokens / 1000) * COST_GPT4O_OUTPUT_PER_1K
        )
        tts_cost = self.tts_characters * COST_ELEVENLABS_PER_CHAR
        asr_cost = self.audio_duration_seconds * COST_DEEPGRAM_PER_SECOND
        return llm_cost + tts_cost + asr_cost


class VoicePipeline:
    """
    Orchestrates the full VAD → ASR → LLM → TTS pipeline for one session.

    Instantiated per session. Holds no cross-session state.
    """

    def __init__(
        self,
        session: Session,
        agent: AgentConfig,
        memory_service: MemoryService,
        analytics_service: AnalyticsService,
        openai_client: AsyncOpenAI,
        vad_engine: VADEngine,
        asr_engine: ASREngine,
        tts_engine: TTSEngine,
    ) -> None:
        self.session = session
        self.agent = agent
        self._memory = memory_service
        self._analytics = analytics_service
        self._openai = openai_client
        self._vad = vad_engine
        self._asr = asr_engine
        self._tts = tts_engine
        self._turn_sequence = 0
        self._active_turn: Optional[asyncio.Task] = None
        self._barge_in_event = asyncio.Event()

    async def process_audio_chunk(self, pcm_chunk: bytes) -> None:
        """
        Feed a raw PCM chunk into the VAD engine.
        When VAD detects end-of-utterance, trigger a pipeline turn.
        This method is called continuously by the gateway for every audio frame.
        """
        segment: Optional[SpeechSegment] = await self._vad.feed(pcm_chunk)
        if segment is not None:
            # Cancel any in-flight agent turn if barge-in is enabled
            if self.agent.barge_in_enabled and self._active_turn and not self._active_turn.done():
                self._barge_in_event.set()
                self._active_turn.cancel()
                logger.info("Barge-in detected for session %s", self.session.id)

            self._active_turn = asyncio.create_task(
                self._run_turn(segment)
            )

    async def _run_turn(self, segment: SpeechSegment) -> AsyncGenerator[bytes, None]:
        """
        Full pipeline for one user utterance.
        Yields audio chunks as they become available from TTS.
        """
        metrics = TurnMetrics()
        metrics.vad_start = segment.start_time
        self._barge_in_event.clear()

        try:
            # 1. ASR -------------------------------------------------------
            metrics.asr_start = time.monotonic()
            asr_result: ASRResult = await self._asr.transcribe(segment.audio_bytes)
            metrics.asr_end = time.monotonic()
            metrics.audio_duration_seconds = segment.duration_seconds

            if not asr_result.transcript.strip():
                logger.debug("Empty transcript  skipping turn")
                return

            logger.info(
                "ASR [%.0fms] %.40r (conf=%.2f)",
                metrics.asr_latency_ms,
                asr_result.transcript,
                asr_result.confidence,
            )

            # 2. Memory enrichment -----------------------------------------
            memory: ConversationMemory = await self._memory.enrich_memory(
                session=self.session,
                user_utterance=asr_result.transcript,
            )

            # 3. Build LLM messages ----------------------------------------
            messages = [{"role": "system", "content": self.agent.system_prompt}]
            messages.extend(memory.to_llm_messages())
            messages.append({"role": "user", "content": asr_result.transcript})

            # 4. LLM (streaming) -------------------------------------------
            metrics.llm_start = time.monotonic()
            llm_response_text = ""

            async def stream_llm() -> str:
                nonlocal llm_response_text
                first_token = True
                stream = await self._openai.chat.completions.create(
                    model=self.agent.llm_model,
                    messages=messages,
                    tools=self.agent.tools or None,
                    temperature=self.agent.llm_temperature,
                    max_tokens=self.agent.llm_max_tokens,
                    stream=True,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        if first_token:
                            metrics.llm_first_token = time.monotonic()
                            first_token = False
                        llm_response_text += delta.content
                        yield delta.content
                metrics.llm_end = time.monotonic()
                metrics.llm_output_tokens = len(llm_response_text.split())  # approx

            # 5. TTS streaming  pipe LLM tokens directly to TTS -----------
            metrics.tts_start = time.monotonic()
            first_audio = True

            async for audio_chunk in self._tts.synthesize_stream(
                text_stream=stream_llm(),
                voice_id=self.agent.voice_id,
            ):
                if self._barge_in_event.is_set():
                    logger.info("Barge-in: stopping TTS playback")
                    break
                if first_audio:
                    metrics.tts_first_audio = time.monotonic()
                    first_audio = False
                yield audio_chunk

            metrics.tts_end = time.monotonic()
            metrics.tts_characters = len(llm_response_text)

            logger.info(
                "Turn complete | ASR=%.0fms LLM=%.0fms (TTFT=%.0fms) TTS=%.0fms Total=%.0fms",
                metrics.asr_latency_ms,
                metrics.llm_latency_ms,
                metrics.llm_ttft_ms,
                metrics.tts_latency_ms,
                metrics.total_latency_ms,
            )

            # 6. Update memory and analytics --------------------------------
            await self._memory.update_memory(
                session=self.session,
                user_utterance=asr_result.transcript,
                agent_response=llm_response_text,
            )

            self._turn_sequence += 1
            turn = Turn(
                session_id=self.session.id,
                sequence_number=self._turn_sequence,
                role=TurnRole.AGENT,
                transcript=llm_response_text,
                asr_latency_ms=metrics.asr_latency_ms,
                llm_latency_ms=metrics.llm_latency_ms,
                tts_latency_ms=metrics.tts_latency_ms,
                total_latency_ms=metrics.total_latency_ms,
                llm_input_tokens=metrics.llm_input_tokens,
                llm_output_tokens=metrics.llm_output_tokens,
                tts_characters=metrics.tts_characters,
                estimated_cost_usd=metrics.estimated_cost_usd,
                asr_confidence=asr_result.confidence,
                was_barge_in=self._barge_in_event.is_set(),
            )
            await self._analytics.log_turn(turn)

        except asyncio.CancelledError:
            logger.info("Turn cancelled (barge-in) for session %s", self.session.id)
        except Exception as exc:
            logger.error("Pipeline error for session %s: %s", self.session.id, exc, exc_info=True)
            # Graceful degradation: yield fallback audio
            async for audio_chunk in self._tts.synthesize_text(
                self.agent.fallback_message,
                voice_id=self.agent.voice_id,
            ):
                yield audio_chunk
```

---

## 5. Agent Configuration DSL

One of the most powerful features of a platform is the ability to define agents declaratively, without writing code. We implement a JSON Schema-validated configuration system with a fluent Python builder.

### 5.1 JSON Schema

```python
# platform/config/agent_schema.py
"""
JSON Schema for AgentConfig validation.
Used by the API layer to validate incoming agent creation requests.
"""

AGENT_CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "AgentConfig",
    "type": "object",
    "required": ["name", "system_prompt"],
    "additionalProperties": False,
    "properties": {
        "name": {
            "type": "string",
            "minLength": 1,
            "maxLength": 100,
            "description": "Human-readable agent name",
        },
        "description": {"type": "string", "maxLength": 500},
        "system_prompt": {
            "type": "string",
            "minLength": 10,
            "description": "The base system prompt that defines agent behavior",
        },
        "voice_id": {
            "type": "string",
            "description": "TTS voice identifier (provider-specific)",
        },
        "language": {
            "type": "string",
            "pattern": "^[a-z]{2}-[A-Z]{2}$",
            "default": "en-US",
        },
        "llm_provider": {
            "type": "string",
            "enum": ["openai", "anthropic", "cohere"],
            "default": "openai",
        },
        "llm_model": {"type": "string", "default": "gpt-4o"},
        "llm_temperature": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 2.0,
            "default": 0.7,
        },
        "asr_provider": {
            "type": "string",
            "enum": ["deepgram", "openai", "assemblyai"],
            "default": "deepgram",
        },
        "tts_provider": {
            "type": "string",
            "enum": ["elevenlabs", "azure", "google"],
            "default": "elevenlabs",
        },
        "max_call_duration_seconds": {
            "type": "integer",
            "minimum": 60,
            "maximum": 7200,
            "default": 1800,
        },
        "barge_in_enabled": {"type": "boolean", "default": True},
        "tools": {
            "type": "array",
            "items": {"type": "object"},
            "description": "OpenAI function-calling tool definitions",
        },
        "requires_consent": {"type": "boolean", "default": False},
        "gdpr_enabled": {"type": "boolean", "default": True},
        "hipaa_enabled": {"type": "boolean", "default": False},
        "recording_enabled": {"type": "boolean", "default": True},
        "pii_redaction_enabled": {"type": "boolean", "default": True},
    },
}
```

### 5.2 Fluent Builder

```python
# platform/config/agent_builder.py
"""
AgentConfigBuilder: A fluent builder for constructing AgentConfig objects.

Usage:
    agent = (
        AgentConfigBuilder("Customer Support Agent")
        .with_prompt("You are a helpful customer support agent for Acme Corp...")
        .with_voice("rachel", provider="elevenlabs")
        .with_llm("gpt-4o", temperature=0.5)
        .with_tool(lookup_order_tool)
        .with_compliance(gdpr=True, recording=True)
        .build()
    )
"""

from __future__ import annotations
from typing import Any, Optional
from platform.models import AgentConfig


class AgentConfigBuilder:
    """Fluent builder for AgentConfig. Method-chaining API."""

    def __init__(self, name: str) -> None:
        self._data: dict[str, Any] = {
            "name": name,
            "system_prompt": "",
            "tools": [],
        }

    def with_prompt(self, prompt: str) -> "AgentConfigBuilder":
        self._data["system_prompt"] = prompt
        return self

    def with_description(self, description: str) -> "AgentConfigBuilder":
        self._data["description"] = description
        return self

    def with_voice(
        self,
        voice_id: str,
        provider: str = "elevenlabs",
        model: str = "eleven_turbo_v2",
        language: str = "en-US",
        speaking_rate: float = 1.0,
    ) -> "AgentConfigBuilder":
        self._data.update({
            "voice_id": voice_id,
            "tts_provider": provider,
            "tts_model": model,
            "language": language,
            "speaking_rate": speaking_rate,
        })
        return self

    def with_llm(
        self,
        model: str = "gpt-4o",
        provider: str = "openai",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> "AgentConfigBuilder":
        self._data.update({
            "llm_model": model,
            "llm_provider": provider,
            "llm_temperature": temperature,
            "llm_max_tokens": max_tokens,
        })
        return self

    def with_asr(
        self,
        provider: str = "deepgram",
        model: str = "nova-2",
        language: str = "en-US",
    ) -> "AgentConfigBuilder":
        self._data.update({
            "asr_provider": provider,
            "asr_model": model,
            "asr_language": language,
        })
        return self

    def with_tool(self, tool_definition: dict[str, Any]) -> "AgentConfigBuilder":
        self._data["tools"].append(tool_definition)
        return self

    def with_compliance(
        self,
        gdpr: bool = True,
        hipaa: bool = False,
        tcpa: bool = False,
        requires_consent: bool = False,
        recording: bool = True,
        pii_redaction: bool = True,
    ) -> "AgentConfigBuilder":
        self._data.update({
            "gdpr_enabled": gdpr,
            "hipaa_enabled": hipaa,
            "tcpa_enabled": tcpa,
            "requires_consent": requires_consent,
            "recording_enabled": recording,
            "pii_redaction_enabled": pii_redaction,
        })
        return self

    def with_behavior(
        self,
        barge_in: bool = True,
        max_duration: int = 1800,
        silence_timeout: int = 10,
        max_turns: int = 100,
        fallback_message: str = "I'm sorry, please hold.",
    ) -> "AgentConfigBuilder":
        self._data.update({
            "barge_in_enabled": barge_in,
            "max_call_duration_seconds": max_duration,
            "silence_timeout_seconds": silence_timeout,
            "max_turns": max_turns,
            "fallback_message": fallback_message,
        })
        return self

    def build(self) -> AgentConfig:
        """Validate and construct the AgentConfig."""
        return AgentConfig(**self._data)


# ---------------------------------------------------------------------------
# Example Agent Configurations
# ---------------------------------------------------------------------------

CUSTOMER_SUPPORT_AGENT = (
    AgentConfigBuilder("Acme Customer Support")
    .with_description("Handles order inquiries, returns, and general support for Acme Corp")
    .with_prompt(
        "You are Alex, a friendly and efficient customer support agent for Acme Corp. "
        "You help customers with order status, returns, and product questions. "
        "Always verify the customer's order number before accessing their account. "
        "If you cannot resolve an issue, offer to escalate to a human agent. "
        "Keep responses concise  this is a phone conversation, not an email. "
        "Never invent information. If you don't know, say so and offer to find out."
    )
    .with_voice("rachel", provider="elevenlabs", speaking_rate=1.05)
    .with_llm("gpt-4o", temperature=0.4, max_tokens=256)
    .with_asr("deepgram", "nova-2")
    .with_tool({
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order by order number",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_number": {"type": "string", "description": "The order number"}
                },
                "required": ["order_number"],
            },
        },
    })
    .with_compliance(gdpr=True, recording=True, pii_redaction=True)
    .with_behavior(barge_in=True, max_duration=900, silence_timeout=8)
    .build()
)

RESTAURANT_BOOKING_AGENT = (
    AgentConfigBuilder("La Bella Restaurant Reservations")
    .with_description("Takes table reservations for La Bella Italian Restaurant")
    .with_prompt(
        "You are Sofia, the reservations host at La Bella, an upscale Italian restaurant. "
        "You are warm, gracious, and efficient. "
        "To make a reservation you need: date, time, party size, and guest name. "
        "Available times are 5:30 PM, 7:00 PM, and 8:30 PM. "
        "Maximum party size is 12. Minimum 2 hours notice required. "
        "Always confirm the reservation details back to the guest before finalizing. "
        "Speak naturally and warmly  you are creating a first impression of the restaurant."
    )
    .with_voice("bella", provider="elevenlabs", speaking_rate=0.95)
    .with_llm("gpt-4o", temperature=0.6, max_tokens=300)
    .with_tool({
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Check table availability for a given date, time, and party size",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                    "time": {"type": "string", "description": "Time in HH:MM 24h format"},
                    "party_size": {"type": "integer", "description": "Number of guests"},
                },
                "required": ["date", "time", "party_size"],
            },
        },
    })
    .with_tool({
        "type": "function",
        "function": {
            "name": "create_reservation",
            "description": "Create a confirmed restaurant reservation",
            "parameters": {
                "type": "object",
                "properties": {
                    "guest_name": {"type": "string"},
                    "date": {"type": "string"},
                    "time": {"type": "string"},
                    "party_size": {"type": "integer"},
                    "phone_number": {"type": "string"},
                    "special_requests": {"type": "string"},
                },
                "required": ["guest_name", "date", "time", "party_size"],
            },
        },
    })
    .with_compliance(gdpr=True, recording=False, requires_consent=False)
    .with_behavior(barge_in=True, max_duration=600, silence_timeout=12)
    .build()
)

HEALTHCARE_SCHEDULING_AGENT = (
    AgentConfigBuilder("HealthFirst Appointment Scheduler")
    .with_description("Schedules patient appointments for HealthFirst Medical Group")
    .with_prompt(
        "You are a medical appointment scheduler for HealthFirst Medical Group. "
        "You schedule appointments for existing patients only. "
        "Collect: patient name, date of birth, reason for visit, preferred date/time, and insurance. "
        "You cannot provide medical advice. For urgent symptoms, direct to 911 or ER. "
        "Speak clearly and slowly  many callers may be elderly or anxious. "
        "Always confirm appointment details and provide a confirmation number."
    )
    .with_voice("aria", provider="elevenlabs", speaking_rate=0.90)
    .with_llm("gpt-4o", temperature=0.2, max_tokens=400)
    .with_compliance(
        gdpr=True,
        hipaa=True,
        requires_consent=True,
        recording=True,
        pii_redaction=True,
    )
    .with_behavior(barge_in=False, max_duration=1200, silence_timeout=15)
    .build()
)
```

---

## 6. FastAPI API Layer

The API layer exposes everything the platform does to the outside world: REST endpoints for management, WebSocket for real-time audio, and webhook handlers for Twilio.

```python
# platform/api/app.py
"""
Main FastAPI application for the Voice Agent Platform.

Endpoints:
  REST:
    POST   /agents                       Create a new agent
    GET    /agents/{id}                  Get agent config
    PATCH  /agents/{id}/activate         Activate a draft agent
    POST   /calls/outbound               Initiate an outbound call
    GET    /calls/{id}                   Get call status
    GET    /analytics/dashboard          Dashboard metrics

  WebSocket:
    WS     /ws/audio/{session_id}        Real-time audio stream

  Webhooks:
    POST   /webhooks/twilio/inbound      Twilio inbound call webhook
    POST   /webhooks/twilio/status       Twilio call status callback
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

import redis.asyncio as aioredis
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

from platform.models import AgentConfig, Call, CallDirection, Session, SessionStatus
from platform.services.agent_service import AgentService, AgentNotFoundError
from platform.services.analytics_service import AnalyticsService
from platform.services.memory_service import MemoryService
from platform.services.session_manager import SessionManager
from platform.pipeline.voice_pipeline import VoicePipeline
from platform.engines.vad_engine import VADEngine
from platform.engines.asr_engine import ASREngine
from platform.engines.tts_engine import TTSEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and teardown shared resources."""
    # Redis
    app.state.redis = await aioredis.from_url(
        os.environ["REDIS_URL"],
        encoding="utf-8",
        decode_responses=True,
    )

    # PostgreSQL
    engine = create_async_engine(os.environ["DATABASE_URL"], echo=False, pool_size=10)
    app.state.db_factory = async_sessionmaker(engine, expire_on_commit=False)

    logger.info("Voice Agent Platform started")
    yield

    await app.state.redis.aclose()
    await engine.dispose()
    logger.info("Voice Agent Platform shut down")


app = FastAPI(
    title="Voice Agent Platform",
    version="1.0.0",
    description="Production-grade voice agent platform",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------

async def get_db(request: Request) -> AsyncSession:
    async with request.app.state.db_factory() as session:
        yield session

async def get_redis(request: Request) -> aioredis.Redis:
    return request.app.state.redis

async def get_session_manager(
    redis: aioredis.Redis = Depends(get_redis),
) -> SessionManager:
    return SessionManager(redis)

async def get_agent_service(
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
) -> AgentService:
    return AgentService(db, redis)

async def get_analytics_service(
    db: AsyncSession = Depends(get_db),
) -> AnalyticsService:
    return AnalyticsService(db)


# ---------------------------------------------------------------------------
# REST: Agents
# ---------------------------------------------------------------------------

@app.post("/agents", status_code=201, response_model=AgentConfig)
async def create_agent(
    config: AgentConfig,
    agent_svc: AgentService = Depends(get_agent_service),
) -> AgentConfig:
    """Create a new agent configuration (starts in DRAFT status)."""
    return await agent_svc.create_agent(config)


@app.get("/agents/{agent_id}", response_model=AgentConfig)
async def get_agent(
    agent_id: str,
    agent_svc: AgentService = Depends(get_agent_service),
) -> AgentConfig:
    try:
        return await agent_svc.get_agent(agent_id)
    except AgentNotFoundError:
        raise HTTPException(status_code=404, detail="Agent not found")


@app.patch("/agents/{agent_id}/activate", response_model=AgentConfig)
async def activate_agent(
    agent_id: str,
    agent_svc: AgentService = Depends(get_agent_service),
) -> AgentConfig:
    try:
        return await agent_svc.activate_agent(agent_id)
    except AgentNotFoundError:
        raise HTTPException(status_code=404, detail="Agent not found")


# ---------------------------------------------------------------------------
# REST: Calls
# ---------------------------------------------------------------------------

class OutboundCallRequest(BaseModel):
    agent_id: str
    to_number: str
    from_number: str
    user_id: Optional[str] = None
    metadata: dict[str, Any] = {}


@app.post("/calls/outbound", status_code=202)
async def initiate_outbound_call(
    req: OutboundCallRequest,
    background_tasks: BackgroundTasks,
    agent_svc: AgentService = Depends(get_agent_service),
    session_mgr: SessionManager = Depends(get_session_manager),
) -> dict[str, str]:
    """Initiate an outbound call. Returns call_id immediately; call is placed async."""
    try:
        agent = await agent_svc.get_agent(req.agent_id)
    except AgentNotFoundError:
        raise HTTPException(status_code=404, detail="Agent not found")

    import uuid
    call_id = str(uuid.uuid4())
    background_tasks.add_task(
        _place_outbound_call,
        call_id=call_id,
        agent=agent,
        to_number=req.to_number,
        from_number=req.from_number,
    )
    return {"call_id": call_id, "status": "initiated"}


async def _place_outbound_call(
    call_id: str,
    agent: AgentConfig,
    to_number: str,
    from_number: str,
) -> None:
    """Background task: use Twilio to place the outbound call."""
    from twilio.rest import Client as TwilioClient
    client = TwilioClient(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
    webhook_base = os.environ["WEBHOOK_BASE_URL"]
    client.calls.create(
        to=to_number,
        from_=from_number,
        url=f"{webhook_base}/webhooks/twilio/inbound?agent_id={agent.id}&call_id={call_id}",
        status_callback=f"{webhook_base}/webhooks/twilio/status",
        status_callback_method="POST",
    )
    logger.info("Outbound call placed: %s → %s (agent=%s)", from_number, to_number, agent.id)


# ---------------------------------------------------------------------------
# REST: Analytics
# ---------------------------------------------------------------------------

@app.get("/analytics/dashboard")
async def get_dashboard(
    agent_id: str,
    hours: int = 24,
    analytics: AnalyticsService = Depends(get_analytics_service),
) -> dict[str, Any]:
    """Return dashboard metrics for an agent over a rolling time window."""
    return await analytics.get_dashboard_metrics(agent_id=agent_id, hours=hours)


# ---------------------------------------------------------------------------
# Webhooks: Twilio
# ---------------------------------------------------------------------------

@app.post("/webhooks/twilio/inbound", response_class=PlainTextResponse)
async def twilio_inbound(
    request: Request,
    agent_id: str,
    call_id: Optional[str] = None,
    session_mgr: SessionManager = Depends(get_session_manager),
) -> str:
    """
    Twilio calls this URL when an inbound call arrives (or outbound is answered).
    We respond with TwiML that connects the call to our WebSocket stream.
    """
    form = await request.form()
    twilio_call_sid = form.get("CallSid", "")
    from_number = form.get("From", "")
    to_number = form.get("To", "")

    import uuid
    if call_id is None:
        call_id = str(uuid.uuid4())

    # Create session
    session = Session(
        call_id=call_id,
        agent_id=agent_id,
        channel="phone",
    )
    await session_mgr.create_session(session)

    # Build TwiML to connect audio to our WebSocket
    websocket_url = os.environ["WEBSOCKET_BASE_URL"]
    response = VoiceResponse()
    connect = Connect()
    stream = Stream(url=f"{websocket_url}/ws/audio/{session.id}")
    stream.parameter(name="agentId", value=agent_id)
    connect.append(stream)
    response.append(connect)

    logger.info(
        "Inbound call connected: %s → session=%s", twilio_call_sid, session.id
    )
    return str(response)


@app.post("/webhooks/twilio/status")
async def twilio_status(
    request: Request,
    session_mgr: SessionManager = Depends(get_session_manager),
    analytics: AnalyticsService = Depends(get_analytics_service),
) -> dict[str, str]:
    """Handle Twilio call status callbacks (completed, failed, etc.)."""
    form = await request.form()
    call_sid = form.get("CallSid", "")
    call_status = form.get("CallStatus", "")
    logger.info("Twilio status callback: %s → %s", call_sid, call_status)
    return {"received": "ok"}


# ---------------------------------------------------------------------------
# WebSocket: Real-time audio
# ---------------------------------------------------------------------------

@app.websocket("/ws/audio/{session_id}")
async def audio_websocket(
    websocket: WebSocket,
    session_id: str,
    session_mgr: SessionManager = Depends(get_session_manager),
    agent_svc: AgentService = Depends(get_agent_service),
) -> None:
    """
    WebSocket endpoint for real-time bidirectional audio.

    Twilio Media Streams sends base64-encoded mulaw audio.
    WebRTC clients send raw PCM (16kHz, 16-bit, mono).
    """
    await websocket.accept()

    session = await session_mgr.get_session(session_id)
    if session is None:
        await websocket.close(code=4404, reason="Session not found")
        return

    try:
        agent = await agent_svc.get_agent(session.agent_id)
    except AgentNotFoundError:
        await websocket.close(code=4404, reason="Agent not found")
        return

    # Instantiate pipeline for this session
    from openai import AsyncOpenAI
    pipeline = VoicePipeline(
        session=session,
        agent=agent,
        memory_service=MemoryService(session_mgr),
        analytics_service=AnalyticsService(websocket.app.state.db_factory),
        openai_client=AsyncOpenAI(),
        vad_engine=VADEngine(),
        asr_engine=ASREngine(provider=agent.asr_provider),
        tts_engine=TTSEngine(provider=agent.tts_provider),
    )

    try:
        async for message in websocket.iter_json():
            event = message.get("event", "")

            if event == "media":
                import base64
                audio_bytes = base64.b64decode(message["media"]["payload"])
                # Process audio; collect any outgoing audio
                async for audio_chunk in pipeline.process_audio_chunk(audio_bytes):
                    encoded = base64.b64encode(audio_chunk).decode()
                    await websocket.send_json({
                        "event": "media",
                        "media": {"payload": encoded},
                    })

            elif event == "stop":
                logger.info("WebSocket stop event for session %s", session_id)
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for session %s", session_id)
    finally:
        await session_mgr.end_session(session_id)
        await websocket.close()
```

---

## 7. Evaluation Framework

A platform without measurement is just hope. The `VoiceAgentEvaluator` computes the metrics that tell you whether your agent is actually working.

### 7.1 Evaluation Metrics Table

| Metric | Definition | Target | Alert Threshold |
|---|---|---|---|
| **Task Completion Rate** | % sessions where task was completed | > 85% | < 75% |
| **Handle Time (AHT)** | Average session duration in seconds | < 180s | > 300s |
| **CSAT Score** | Average 1-5 customer satisfaction | > 4.2 | < 3.8 |
| **First Call Resolution** | % issues resolved without callback | > 80% | < 70% |
| **Barge-in Rate** | % turns where user interrupted agent | < 15% | > 30% |
| **ASR Error Rate** | % turns with low confidence (<0.7) | < 5% | > 15% |
| **P50 Latency** | Median turn-response latency (ms) | < 800ms | > 1200ms |
| **P95 Latency** | 95th percentile latency (ms) | < 1500ms | > 2500ms |
| **Cost per Call** | Average $ spend per completed session | < $0.15 | > $0.30 |
| **Escalation Rate** | % calls transferred to human agent | < 10% | > 25% |
| **Drop Rate** | % calls that ended unexpectedly | < 2% | > 5% |

```python
# platform/evaluation/evaluator.py
"""
VoiceAgentEvaluator: Comprehensive evaluation framework.

Computes metrics from the analytics database and supports:
  - Point-in-time reporting
  - A/B test comparison
  - Automated regression detection
  - Latency distribution analysis
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """Complete evaluation snapshot for an agent."""
    agent_id: str
    window_hours: int
    generated_at: datetime = field(default_factory=datetime.utcnow)

    # Volume
    total_sessions: int = 0
    total_turns: int = 0
    unique_users: int = 0

    # Quality
    task_completion_rate: float = 0.0
    first_call_resolution_rate: float = 0.0
    escalation_rate: float = 0.0
    drop_rate: float = 0.0
    avg_csat: float = 0.0

    # Efficiency
    avg_handle_time_seconds: float = 0.0
    avg_turns_per_session: float = 0.0
    barge_in_rate: float = 0.0
    asr_low_confidence_rate: float = 0.0

    # Latency (milliseconds)
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    avg_asr_latency: float = 0.0
    avg_llm_latency: float = 0.0
    avg_tts_latency: float = 0.0

    # Cost
    avg_cost_per_session: float = 0.0
    total_cost: float = 0.0

    # Alerts (metrics that breached thresholds)
    alerts: list[str] = field(default_factory=list)

    def check_thresholds(self) -> None:
        """Populate alerts list based on known good/bad thresholds."""
        if self.task_completion_rate < 0.75:
            self.alerts.append(
                f"LOW task completion rate: {self.task_completion_rate:.1%} (threshold: 75%)"
            )
        if self.avg_csat < 3.8:
            self.alerts.append(
                f"LOW CSAT: {self.avg_csat:.2f} (threshold: 3.8)"
            )
        if self.latency_p95 > 2500:
            self.alerts.append(
                f"HIGH P95 latency: {self.latency_p95:.0f}ms (threshold: 2500ms)"
            )
        if self.avg_cost_per_session > 0.30:
            self.alerts.append(
                f"HIGH cost per session: ${self.avg_cost_per_session:.3f} (threshold: $0.30)"
            )
        if self.barge_in_rate > 0.30:
            self.alerts.append(
                f"HIGH barge-in rate: {self.barge_in_rate:.1%}  agent may be speaking too long"
            )
        if self.drop_rate > 0.05:
            self.alerts.append(
                f"HIGH drop rate: {self.drop_rate:.1%} (threshold: 5%)"
            )

    def summary(self) -> str:
        status = "HEALTHY" if not self.alerts else f"ATTENTION ({len(self.alerts)} alerts)"
        return (
            f"=== Evaluation Report: {self.agent_id} [{status}] ===\n"
            f"Window: last {self.window_hours}h | Sessions: {self.total_sessions} "
            f"| Users: {self.unique_users}\n"
            f"Task Completion: {self.task_completion_rate:.1%} | "
            f"CSAT: {self.avg_csat:.2f}/5 | AHT: {self.avg_handle_time_seconds:.0f}s\n"
            f"Latency P50/P95/P99: {self.latency_p50:.0f}/{self.latency_p95:.0f}/{self.latency_p99:.0f}ms\n"
            f"Cost/session: ${self.avg_cost_per_session:.4f} | Total: ${self.total_cost:.2f}\n"
            + ("\nAlerts:\n" + "\n".join(f"  ! {a}" for a in self.alerts) if self.alerts else "")
        )


class VoiceAgentEvaluator:
    """
    Compute comprehensive evaluation reports from the analytics database.

    Usage:
        evaluator = VoiceAgentEvaluator(db_session)
        report = await evaluator.evaluate(agent_id="...", hours=24)
        print(report.summary())
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def evaluate(
        self,
        agent_id: str,
        hours: int = 24,
    ) -> EvaluationReport:
        """Run a full evaluation for an agent over the specified time window."""
        report = EvaluationReport(agent_id=agent_id, window_hours=hours)
        since = datetime.utcnow() - timedelta(hours=hours)

        await self._compute_session_metrics(report, agent_id, since)
        await self._compute_turn_metrics(report, agent_id, since)
        report.check_thresholds()
        return report

    async def _compute_session_metrics(
        self,
        report: EvaluationReport,
        agent_id: str,
        since: datetime,
    ) -> None:
        result = await self._db.execute(
            text("""
                SELECT
                    COUNT(*)                                                      AS total,
                    COUNT(DISTINCT user_id)                                       AS unique_users,
                    AVG(CASE WHEN task_completed THEN 1.0 ELSE 0.0 END)          AS completion_rate,
                    AVG(duration_seconds)                                          AS avg_duration,
                    AVG(total_turns)                                               AS avg_turns,
                    AVG(total_cost_usd)                                            AS avg_cost,
                    SUM(total_cost_usd)                                            AS total_cost,
                    AVG(csat_score)                                                AS avg_csat,
                    AVG(CASE WHEN status = 'ended' AND duration_seconds < 10
                             THEN 1.0 ELSE 0.0 END)                               AS drop_rate
                FROM sessions
                WHERE agent_id = :agent_id AND started_at >= :since
            """),
            {"agent_id": agent_id, "since": since},
        )
        row = result.fetchone()
        if row:
            report.total_sessions = row[0] or 0
            report.unique_users = row[1] or 0
            report.task_completion_rate = float(row[2] or 0)
            report.avg_handle_time_seconds = float(row[3] or 0)
            report.avg_turns_per_session = float(row[4] or 0)
            report.avg_cost_per_session = float(row[5] or 0)
            report.total_cost = float(row[6] or 0)
            report.avg_csat = float(row[7] or 0)
            report.drop_rate = float(row[8] or 0)

    async def _compute_turn_metrics(
        self,
        report: EvaluationReport,
        agent_id: str,
        since: datetime,
    ) -> None:
        result = await self._db.execute(
            text("""
                SELECT
                    COUNT(*)                                                                AS total_turns,
                    AVG(CASE WHEN was_barge_in THEN 1.0 ELSE 0.0 END)                     AS barge_in_rate,
                    AVG(CASE WHEN asr_confidence < 0.7 THEN 1.0 ELSE 0.0 END)             AS low_conf_rate,
                    AVG(asr_latency_ms)                                                    AS avg_asr,
                    AVG(llm_latency_ms)                                                    AS avg_llm,
                    AVG(tts_latency_ms)                                                    AS avg_tts,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY total_latency_ms)        AS p50,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_latency_ms)        AS p95,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY total_latency_ms)        AS p99
                FROM turns t
                JOIN sessions s ON t.session_id = s.id
                WHERE s.agent_id = :agent_id
                  AND t.created_at >= :since
                  AND t.role = 'agent'
            """),
            {"agent_id": agent_id, "since": since},
        )
        row = result.fetchone()
        if row:
            report.total_turns = row[0] or 0
            report.barge_in_rate = float(row[1] or 0)
            report.asr_low_confidence_rate = float(row[2] or 0)
            report.avg_asr_latency = float(row[3] or 0)
            report.avg_llm_latency = float(row[4] or 0)
            report.avg_tts_latency = float(row[5] or 0)
            report.latency_p50 = float(row[6] or 0)
            report.latency_p95 = float(row[7] or 0)
            report.latency_p99 = float(row[8] or 0)

    async def compare_ab(
        self,
        agent_id_a: str,
        agent_id_b: str,
        hours: int = 168,  # 1 week
    ) -> dict[str, EvaluationReport]:
        """Compare two agent configurations (A/B test)."""
        report_a = await self.evaluate(agent_id_a, hours)
        report_b = await self.evaluate(agent_id_b, hours)
        return {"A": report_a, "B": report_b}
```

---

## 8. Deployment

### 8.1 Deployment Architecture Diagram

```mermaid
graph TB
    subgraph Internet
        LB["Load Balancer (nginx/ALB)"]
    end

    subgraph K8s["Kubernetes Cluster"]
        subgraph GW_NS["Namespace: voice-gateway"]
            GW1["Gateway Pod 1"]
            GW2["Gateway Pod 2"]
            GW3["Gateway Pod 3"]
            GW_HPA["HPA: 2-10 replicas"]
        end

        subgraph WK_NS["Namespace: voice-workers"]
            WK1["Worker Pod 1"]
            WK2["Worker Pod 2"]
            WK3["Worker Pod 3"]
            WK_HPA["HPA: 2-20 replicas"]
        end

        subgraph DATA_NS["Namespace: data"]
            REDIS_SS["Redis StatefulSet"]
            PG_SS["PostgreSQL StatefulSet"]
            CHROMA_SS["ChromaDB StatefulSet"]
        end

        subgraph OBS_NS["Namespace: observability"]
            PROM["Prometheus"]
            GRAF["Grafana"]
            JAEGER["Jaeger"]
        end
    end

    LB --> GW1
    LB --> GW2
    LB --> GW3
    GW1 --> WK1
    GW1 --> WK2
    GW2 --> WK2
    GW2 --> WK3
    WK1 --> REDIS_SS
    WK2 --> REDIS_SS
    WK3 --> REDIS_SS
    WK1 --> PG_SS
    WK1 --> CHROMA_SS
    GW1 --> PROM
    WK1 --> PROM
    PROM --> GRAF
```

### 8.2 Docker Compose (Development)

```yaml
# docker-compose.yml
# Full local development environment for the Voice Agent Platform.
# Run: docker compose up -d
# Access:
#   API:      http://localhost:8000
#   Grafana:  http://localhost:3000 (admin/admin)
#   Jaeger:   http://localhost:16686
#   ChromaDB: http://localhost:8001

version: "3.9"

x-common-env: &common-env
  REDIS_URL: redis://redis:6379/0
  DATABASE_URL: postgresql+asyncpg://voice:voice@postgres:5432/voiceagent
  CHROMA_HOST: chromadb
  CHROMA_PORT: 8000
  OPENAI_API_KEY: ${OPENAI_API_KEY}
  ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
  DEEPGRAM_API_KEY: ${DEEPGRAM_API_KEY}
  ELEVENLABS_API_KEY: ${ELEVENLABS_API_KEY}
  TWILIO_ACCOUNT_SID: ${TWILIO_ACCOUNT_SID}
  TWILIO_AUTH_TOKEN: ${TWILIO_AUTH_TOKEN}
  WEBHOOK_BASE_URL: ${WEBHOOK_BASE_URL:-https://your-ngrok-url.ngrok.io}
  WEBSOCKET_BASE_URL: ${WEBSOCKET_BASE_URL:-wss://your-ngrok-url.ngrok.io}
  OTEL_EXPORTER_JAEGER_ENDPOINT: http://jaeger:14268/api/traces
  LOG_LEVEL: INFO

services:
  # -----------------------------------------------------------------------
  # Voice Gateway  FastAPI, handles Twilio webhooks and WebSocket audio
  # -----------------------------------------------------------------------
  gateway:
    build:
      context: .
      dockerfile: Dockerfile.gateway
    image: voice-platform/gateway:latest
    ports:
      - "8000:8000"
    environment:
      <<: *common-env
      SERVICE_NAME: gateway
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 10s
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 512M

  # -----------------------------------------------------------------------
  # Worker  Pipeline execution (VAD, ASR, LLM, TTS)
  # -----------------------------------------------------------------------
  worker:
    build:
      context: .
      dockerfile: Dockerfile.worker
    image: voice-platform/worker:latest
    environment:
      <<: *common-env
      SERVICE_NAME: worker
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
      chromadb:
        condition: service_started
    restart: unless-stopped
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: "2.0"
          memory: 2G

  # -----------------------------------------------------------------------
  # Redis  Session state and caching
  # -----------------------------------------------------------------------
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
    restart: unless-stopped

  # -----------------------------------------------------------------------
  # PostgreSQL + TimescaleDB  Analytics and audit logs
  # -----------------------------------------------------------------------
  postgres:
    image: timescale/timescaledb:latest-pg16
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: voice
      POSTGRES_PASSWORD: voice
      POSTGRES_DB: voiceagent
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U voice -d voiceagent"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # -----------------------------------------------------------------------
  # ChromaDB  Long-term vector memory
  # -----------------------------------------------------------------------
  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      ANONYMIZED_TELEMETRY: "false"
    restart: unless-stopped

  # -----------------------------------------------------------------------
  # Prometheus  Metrics collection
  # -----------------------------------------------------------------------
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.retention.time=30d"
      - "--web.enable-lifecycle"
    restart: unless-stopped

  # -----------------------------------------------------------------------
  # Grafana  Dashboards
  # -----------------------------------------------------------------------
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_USERS_ALLOW_SIGN_UP: "false"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    depends_on:
      - prometheus
    restart: unless-stopped

  # -----------------------------------------------------------------------
  # Jaeger  Distributed tracing
  # -----------------------------------------------------------------------
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"   # Jaeger UI
      - "14268:14268"   # Collector HTTP
      - "6831:6831/udp" # Compact Thrift
    environment:
      COLLECTOR_OTLP_ENABLED: "true"
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
  chroma_data:
  prometheus_data:
  grafana_data:
```

### 8.3 Kubernetes Manifests

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: voice-platform
  labels:
    app.kubernetes.io/name: voice-platform
    environment: production
---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: voice-platform-config
  namespace: voice-platform
data:
  REDIS_URL: "redis://redis-service:6379/0"
  DATABASE_URL: "postgresql+asyncpg://voice:$(POSTGRES_PASSWORD)@postgres-service:5432/voiceagent"
  CHROMA_HOST: "chromadb-service"
  CHROMA_PORT: "8000"
  LOG_LEVEL: "INFO"
  WEBHOOK_BASE_URL: "https://api.yourplatform.com"
  WEBSOCKET_BASE_URL: "wss://api.yourplatform.com"
---
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: voice-platform-secrets
  namespace: voice-platform
type: Opaque
stringData:
  OPENAI_API_KEY: "sk-..."
  ANTHROPIC_API_KEY: "sk-ant-..."
  DEEPGRAM_API_KEY: "..."
  ELEVENLABS_API_KEY: "..."
  TWILIO_ACCOUNT_SID: "AC..."
  TWILIO_AUTH_TOKEN: "..."
  POSTGRES_PASSWORD: "strongpassword"
---
# k8s/gateway-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: voice-gateway
  namespace: voice-platform
  labels:
    app: voice-gateway
    version: "1.0.0"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: voice-gateway
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  template:
    metadata:
      labels:
        app: voice-gateway
        version: "1.0.0"
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      terminationGracePeriodSeconds: 30
      containers:
        - name: gateway
          image: your-registry/voice-platform/gateway:1.0.0
          ports:
            - containerPort: 8000
              name: http
          envFrom:
            - configMapRef:
                name: voice-platform-config
            - secretRef:
                name: voice-platform-secrets
          resources:
            requests:
              cpu: "250m"
              memory: "256Mi"
            limits:
              cpu: "1000m"
              memory: "512Mi"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 15
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
            failureThreshold: 2
          lifecycle:
            preStop:
              exec:
                command: ["/bin/sh", "-c", "sleep 5"]
---
# k8s/gateway-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: voice-gateway-hpa
  namespace: voice-platform
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: voice-gateway
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
---
# k8s/worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: voice-worker
  namespace: voice-platform
  labels:
    app: voice-worker
spec:
  replicas: 5
  selector:
    matchLabels:
      app: voice-worker
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 2
      maxSurge: 2
  template:
    metadata:
      labels:
        app: voice-worker
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9000"
    spec:
      terminationGracePeriodSeconds: 60
      containers:
        - name: worker
          image: your-registry/voice-platform/worker:1.0.0
          envFrom:
            - configMapRef:
                name: voice-platform-config
            - secretRef:
                name: voice-platform-secrets
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "2000m"
              memory: "2Gi"
          livenessProbe:
            httpGet:
              path: /health
              port: 9000
            initialDelaySeconds: 15
            periodSeconds: 20
---
# k8s/worker-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: voice-worker-hpa
  namespace: voice-platform
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: voice-worker
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 65
---
# k8s/redis-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
  namespace: voice-platform
spec:
  serviceName: redis-service
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
        - name: redis
          image: redis:7-alpine
          ports:
            - containerPort: 6379
          command:
            - redis-server
            - "--appendonly"
            - "yes"
            - "--maxmemory"
            - "2gb"
            - "--maxmemory-policy"
            - "allkeys-lru"
          resources:
            requests:
              cpu: "250m"
              memory: "512Mi"
            limits:
              cpu: "500m"
              memory: "2Gi"
          volumeMounts:
            - name: redis-storage
              mountPath: /data
  volumeClaimTemplates:
    - metadata:
        name: redis-storage
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 10Gi
---
# k8s/services.yaml
apiVersion: v1
kind: Service
metadata:
  name: gateway-service
  namespace: voice-platform
spec:
  selector:
    app: voice-gateway
  ports:
    - name: http
      port: 80
      targetPort: 8000
  type: LoadBalancer
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: voice-platform
spec:
  selector:
    app: redis
  ports:
    - port: 6379
      targetPort: 6379
  type: ClusterIP
```

### 8.4 Health and Readiness Endpoints

```python
# platform/api/health.py
"""
Health, readiness, and liveness endpoints.
These are used by Kubernetes probes and load balancers.
"""

import logging
from typing import Any

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from platform.api.app import get_db, get_redis

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Basic liveness check  always returns 200 if the process is running."""
    return {"status": "ok"}


@router.get("/ready")
async def readiness_check(
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
) -> JSONResponse:
    """
    Readiness check  verifies all downstream dependencies are reachable.
    Returns 200 only when the service is ready to handle traffic.
    Returns 503 if any dependency is unhealthy.
    """
    checks: dict[str, Any] = {}
    all_healthy = True

    # Check PostgreSQL
    try:
        await db.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {e}"
        all_healthy = False

    # Check Redis
    try:
        await redis.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"
        all_healthy = False

    http_status = status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(
        status_code=http_status,
        content={"status": "ready" if all_healthy else "degraded", "checks": checks},
    )


@router.get("/metrics/summary")
async def metrics_summary(request: Request) -> dict[str, Any]:
    """Quick metrics snapshot for internal dashboards (not Prometheus  use /metrics for that)."""
    return {
        "active_websockets": getattr(request.app.state, "active_ws_count", 0),
        "uptime_seconds": getattr(request.app.state, "uptime_seconds", 0),
    }
```

### 8.5 Deployment Checklist

| Step | Action | Verification |
|---|---|---|
| **1. Secrets** | Load all API keys into Kubernetes Secret or Vault | `kubectl get secret voice-platform-secrets -n voice-platform` |
| **2. Database** | Run init.sql migration | `psql -c "SELECT COUNT(*) FROM sessions"` returns 0 |
| **3. Redis** | Verify Redis is reachable | `redis-cli -h redis-service ping` returns PONG |
| **4. ChromaDB** | Verify collection creation | GET /api/v1/collections returns `voice_agent_memory` |
| **5. Gateway** | Deploy gateway pods | All pods Running, readiness probe passes |
| **6. Workers** | Deploy worker pods | All pods Running |
| **7. HPA** | Apply HPA manifests | `kubectl get hpa -n voice-platform` shows correct min/max |
| **8. Ingress** | Configure TLS cert and domain | `curl https://api.yourplatform.com/health` returns 200 |
| **9. Twilio** | Update webhook URLs in Twilio console | Inbound test call reaches `/webhooks/twilio/inbound` |
| **10. Monitoring** | Confirm metrics in Grafana | Dashboard shows real-time latency and session counts |
| **11. Smoke test** | Run automated smoke test suite | All tests pass |
| **12. Alerting** | Configure AlertManager rules | Test alert fires and reaches PagerDuty/Slack |

---

## 9. Observability and Monitoring

Production-grade observability means you know exactly what is happening inside your platform at any moment  before your customers tell you something is wrong.

### 9.1 Prometheus Metrics

```python
# platform/observability/metrics.py
"""
Prometheus metrics for the Voice Agent Platform.
Uses prometheus-fastapi-instrumentator for automatic HTTP metrics
plus custom metrics for voice-specific measurements.
"""

from prometheus_client import Counter, Histogram, Gauge, Summary

# -----------------------------------------------------------------------
# Call and session metrics
# -----------------------------------------------------------------------

CALLS_TOTAL = Counter(
    "voice_calls_total",
    "Total number of calls",
    ["direction", "agent_id", "status"],
)

ACTIVE_SESSIONS = Gauge(
    "voice_active_sessions",
    "Number of currently active voice sessions",
    ["agent_id"],
)

SESSION_DURATION = Histogram(
    "voice_session_duration_seconds",
    "Duration of voice sessions",
    ["agent_id"],
    buckets=[30, 60, 120, 180, 300, 600, 900, 1800],
)

TASK_COMPLETION = Counter(
    "voice_task_completion_total",
    "Task completion outcomes",
    ["agent_id", "completed"],
)

# -----------------------------------------------------------------------
# Pipeline latency metrics
# -----------------------------------------------------------------------

ASR_LATENCY = Histogram(
    "voice_asr_latency_seconds",
    "ASR transcription latency",
    ["provider", "agent_id"],
    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0],
)

LLM_LATENCY = Histogram(
    "voice_llm_latency_seconds",
    "LLM response latency (full completion)",
    ["model", "agent_id"],
    buckets=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
)

LLM_TTFT = Histogram(
    "voice_llm_ttft_seconds",
    "LLM time to first token",
    ["model", "agent_id"],
    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
)

TTS_LATENCY = Histogram(
    "voice_tts_latency_seconds",
    "TTS synthesis latency (first audio chunk)",
    ["provider", "agent_id"],
    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
)

TOTAL_TURN_LATENCY = Histogram(
    "voice_turn_latency_seconds",
    "End-to-end turn latency (VAD detection to first audio out)",
    ["agent_id"],
    buckets=[0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0],
)

# -----------------------------------------------------------------------
# Cost metrics
# -----------------------------------------------------------------------

COST_PER_TURN = Histogram(
    "voice_turn_cost_usd",
    "Estimated cost per pipeline turn in USD",
    ["agent_id"],
    buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.10],
)

CUMULATIVE_COST = Counter(
    "voice_cumulative_cost_usd_total",
    "Cumulative platform cost in USD",
    ["agent_id", "component"],  # component: llm | tts | asr
)

# -----------------------------------------------------------------------
# Error metrics
# -----------------------------------------------------------------------

PIPELINE_ERRORS = Counter(
    "voice_pipeline_errors_total",
    "Pipeline errors by stage",
    ["stage", "agent_id", "error_type"],
)

BARGE_IN_EVENTS = Counter(
    "voice_barge_in_events_total",
    "Number of barge-in events",
    ["agent_id"],
)
```

### 9.2 Structured Logging

```python
# platform/observability/logging_config.py
"""
Structured JSON logging configuration.
All log records include session_id, agent_id, and turn_id for correlation.
"""

import logging
import sys
from typing import Any

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog for JSON output in production, pretty output in development."""

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
    )


def bind_session_context(
    session_id: str,
    agent_id: str,
    turn_id: str | None = None,
) -> None:
    """Bind context variables that will appear in all subsequent log records."""
    structlog.contextvars.bind_contextvars(
        session_id=session_id,
        agent_id=agent_id,
        **({"turn_id": turn_id} if turn_id else {}),
    )


def clear_session_context() -> None:
    structlog.contextvars.clear_contextvars()
```

---

## 10. Quickstart Guide

This section is your fast path from zero to a working voice agent in under 30 minutes.

### 10.1 Prerequisites

- Docker Desktop (or Podman) installed
- Python 3.11+
- A Twilio account with a phone number
- An OpenAI API key
- A Deepgram API key
- An ElevenLabs API key
- ngrok (for local Twilio webhook routing during development)

### 10.2 Step-by-Step Setup

```bash
# Step 1: Clone the repository
git clone https://github.com/yourorg/voice-agent-platform.git
cd voice-agent-platform

# Step 2: Configure environment variables
cp .env.example .env
# Edit .env with your API keys:
#   OPENAI_API_KEY=sk-...
#   DEEPGRAM_API_KEY=...
#   ELEVENLABS_API_KEY=...
#   TWILIO_ACCOUNT_SID=AC...
#   TWILIO_AUTH_TOKEN=...

# Step 3: Start ngrok to expose your local server to Twilio
ngrok http 8000
# Copy the https:// URL (e.g., https://abc123.ngrok.io)
# Add to .env:  WEBHOOK_BASE_URL=https://abc123.ngrok.io
#               WEBSOCKET_BASE_URL=wss://abc123.ngrok.io

# Step 4: Launch the full platform
docker compose up -d

# Step 5: Wait for services to be healthy
docker compose ps
# All services should show "healthy" or "running"

# Step 6: Run database migrations
docker compose exec postgres psql -U voice -d voiceagent -f /docker-entrypoint-initdb.d/init.sql

# Step 7: Verify the API is up
curl http://localhost:8000/health
# {"status": "ok"}
```

### 10.3 Create Your First Agent

```python
# quickstart/create_agent.py
"""
Create a simple demo agent using the platform API.
Run: python quickstart/create_agent.py
"""

import requests

API_BASE = "http://localhost:8000"

# Create the agent
response = requests.post(
    f"{API_BASE}/agents",
    json={
        "name": "Demo Assistant",
        "description": "A simple demo voice agent",
        "system_prompt": (
            "You are a friendly demo assistant. "
            "Your name is Sam. "
            "Keep responses to 1-2 sentences  this is a voice call. "
            "Be warm and conversational."
        ),
        "voice_id": "rachel",
        "llm_model": "gpt-4o",
        "llm_temperature": 0.7,
        "asr_provider": "deepgram",
        "tts_provider": "elevenlabs",
        "barge_in_enabled": True,
        "gdpr_enabled": True,
        "recording_enabled": True,
        "pii_redaction_enabled": True,
    },
)
response.raise_for_status()
agent = response.json()
print(f"Agent created: {agent['id']} (status: {agent['status']})")

# Activate the agent
activate_response = requests.patch(f"{API_BASE}/agents/{agent['id']}/activate")
activate_response.raise_for_status()
print(f"Agent activated: {agent['id']}")
print(f"\nAgent ID to use: {agent['id']}")
print(f"Set this as your Twilio webhook: {API_BASE}/webhooks/twilio/inbound?agent_id={agent['id']}")
```

### 10.4 Configure Twilio

```python
# quickstart/configure_twilio.py
"""
Automatically configure your Twilio phone number to point to the platform.
Run after create_agent.py.
"""

import os
from twilio.rest import Client

TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
WEBHOOK_BASE_URL = os.environ["WEBHOOK_BASE_URL"]
AGENT_ID = input("Paste your agent ID: ").strip()
PHONE_NUMBER = input("Paste your Twilio phone number (E.164): ").strip()

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Find the phone number SID
numbers = client.incoming_phone_numbers.list(phone_number=PHONE_NUMBER)
if not numbers:
    print(f"Phone number {PHONE_NUMBER} not found in your Twilio account.")
    exit(1)

number_sid = numbers[0].sid

# Update the webhook URL
client.incoming_phone_numbers(number_sid).update(
    voice_url=f"{WEBHOOK_BASE_URL}/webhooks/twilio/inbound?agent_id={AGENT_ID}",
    voice_method="POST",
    status_callback=f"{WEBHOOK_BASE_URL}/webhooks/twilio/status",
    status_callback_method="POST",
)

print(f"\nTwilio number {PHONE_NUMBER} configured!")
print(f"Call {PHONE_NUMBER} to test your agent.")
```

### 10.5 View the Dashboard

```python
# quickstart/view_dashboard.py
"""
Print a text-based dashboard summary for your agent.
Run after making a few test calls.
"""

import os
import requests

API_BASE = "http://localhost:8000"
AGENT_ID = input("Paste your agent ID: ").strip()

response = requests.get(
    f"{API_BASE}/analytics/dashboard",
    params={"agent_id": AGENT_ID, "hours": 24},
)
response.raise_for_status()
metrics = response.json()

print("\n" + "=" * 50)
print(f"DASHBOARD: {AGENT_ID}")
print("=" * 50)
print(f"Sessions (24h):       {metrics['total_sessions']}")
print(f"Unique users:         {metrics['unique_users']}")
print(f"Task completion:      {metrics['task_completion_rate']:.1%}")
print(f"Avg CSAT:             {metrics['avg_csat']:.2f}/5")
print(f"Avg handle time:      {metrics['avg_duration_seconds']:.0f}s")
print(f"Avg latency:          {metrics['avg_latency_ms']:.0f}ms")
print(f"P50/P95/P99 latency:  "
      f"{metrics['latency_p50_ms']:.0f}/"
      f"{metrics['latency_p95_ms']:.0f}/"
      f"{metrics['latency_p99_ms']:.0f}ms")
print(f"Avg cost/session:     ${metrics['avg_cost_per_session_usd']:.4f}")
print(f"Total cost:           ${metrics['total_cost_usd']:.2f}")
print("=" * 50)
```

### 10.6 Run the Test Suite

```python
# tests/test_platform_smoke.py
"""
Smoke tests for the Voice Agent Platform.
Run: pytest tests/test_platform_smoke.py -v

These tests hit the live API running in Docker Compose.
They are designed to run in CI/CD after every deployment.
"""

import pytest
import httpx
import asyncio


API_BASE = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}


@pytest.fixture
def client():
    return httpx.Client(base_url=API_BASE, timeout=10.0)


def test_health_check(client):
    """Platform responds to health checks."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readiness_check(client):
    """All dependencies are healthy."""
    r = client.get("/ready")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ready"
    assert data["checks"]["postgres"] == "ok"
    assert data["checks"]["redis"] == "ok"


def test_create_and_activate_agent(client):
    """Can create and activate an agent."""
    # Create
    payload = {
        "name": "Smoke Test Agent",
        "system_prompt": "You are a test agent. Say 'test passed' when asked.",
        "voice_id": "rachel",
        "llm_model": "gpt-4o",
    }
    r = client.post("/agents", json=payload)
    assert r.status_code == 201
    agent = r.json()
    assert agent["status"] == "draft"
    agent_id = agent["id"]

    # Get
    r2 = client.get(f"/agents/{agent_id}")
    assert r2.status_code == 200
    assert r2.json()["name"] == "Smoke Test Agent"

    # Activate
    r3 = client.patch(f"/agents/{agent_id}/activate")
    assert r3.status_code == 200
    assert r3.json()["status"] == "active"


def test_dashboard_returns_metrics(client):
    """Dashboard endpoint returns valid metrics structure."""
    # Create an agent to use
    r = client.post("/agents", json={
        "name": "Dashboard Test Agent",
        "system_prompt": "Test agent.",
        "voice_id": "rachel",
        "llm_model": "gpt-4o",
    })
    agent_id = r.json()["id"]

    r2 = client.get("/analytics/dashboard", params={"agent_id": agent_id, "hours": 1})
    assert r2.status_code == 200
    data = r2.json()
    required_keys = [
        "total_sessions", "task_completion_rate", "avg_latency_ms",
        "avg_cost_per_session_usd", "latency_p50_ms",
    ]
    for key in required_keys:
        assert key in data, f"Missing key: {key}"


def test_unknown_agent_returns_404(client):
    """Requesting an unknown agent returns 404."""
    r = client.get("/agents/nonexistent-agent-id-xyz")
    assert r.status_code == 404
```

---

## 11. Monitoring Dashboard Layout

```mermaid
graph LR
    subgraph Grafana["Grafana Dashboard  Voice Agent Platform"]
        subgraph Row1["Row 1: Volume"]
            W1["Active Sessions\n(Gauge)"]
            W2["Calls Today\n(Counter)"]
            W3["Calls/Hour\n(Time Series)"]
        end

        subgraph Row2["Row 2: Quality"]
            W4["Task Completion Rate\n(Stat %)"]
            W5["Avg CSAT Score\n(Stat)"]
            W6["Barge-in Rate\n(Stat %)"]
        end

        subgraph Row3["Row 3: Latency"]
            W7["P50/P95/P99 Latency\n(Time Series)"]
            W8["ASR Latency\n(Histogram)"]
            W9["LLM TTFT\n(Histogram)"]
        end

        subgraph Row4["Row 4: Cost"]
            W10["Cost/Session Today\n(Stat $)"]
            W11["Cumulative Cost\n(Time Series)"]
            W12["Cost by Component\n(Pie: LLM/TTS/ASR)"]
        end

        subgraph Row5["Row 5: Errors"]
            W13["Pipeline Errors/Hour\n(Bar Chart)"]
            W14["Error Rate by Stage\n(Table)"]
            W15["Active Alerts\n(Alert List)"]
        end
    end
```

---

## 12. Production Hardening

Beyond the core platform, several cross-cutting concerns must be addressed before going live.

### 12.1 Rate Limiting and Abuse Prevention

```python
# platform/middleware/rate_limiter.py
"""
Rate limiting middleware using Redis sliding window counters.
Prevents abuse and protects downstream LLM/TTS API budgets.
"""

import time
import logging
from typing import Callable

import redis.asyncio as aioredis
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Rate limits per IP per minute
RATE_LIMITS = {
    "/webhooks/twilio/inbound": 60,   # 60 inbound calls/min per IP
    "/calls/outbound": 10,             # 10 outbound calls/min per IP
    "/agents": 20,                     # 20 agent creations/min per IP
    "default": 120,                    # Default: 120 requests/min
}


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiter backed by Redis.
    Returns 429 Too Many Requests when limits are exceeded.
    """

    def __init__(self, app, redis_client: aioredis.Redis) -> None:
        super().__init__(app)
        self._redis = redis_client

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path
        limit = RATE_LIMITS.get(path, RATE_LIMITS["default"])

        key = f"rate:{client_ip}:{path}"
        now = int(time.time())
        window_start = now - 60  # 1-minute window

        pipe = self._redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, 90)
        results = await pipe.execute()
        request_count = results[2]

        if request_count > limit:
            logger.warning("Rate limit exceeded: %s %s (%d/%d)", client_ip, path, request_count, limit)
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please slow down."},
                headers={"Retry-After": "60"},
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, limit - request_count))
        return response
```

### 12.2 Graceful Degradation with Circuit Breakers

```python
# platform/resilience/circuit_breaker.py
"""
Circuit breaker pattern for external API calls.
Prevents cascade failures when upstream services are unhealthy.

States:
  CLOSED    normal operation, requests pass through
  OPEN      too many failures, requests fail immediately
  HALF_OPEN  testing if service has recovered
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Async circuit breaker for external service calls.

    Usage:
        breaker = CircuitBreaker(name="openai", failure_threshold=5, recovery_timeout=30)

        @breaker.protect
        async def call_openai():
            ...
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 2,
    ) -> None:
        self.name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self._recovery_timeout:
                logger.info("Circuit %s: OPEN → HALF_OPEN (testing recovery)", self.name)
                self._state = CircuitState.HALF_OPEN
        return self._state

    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function through the circuit breaker."""
        current_state = self.state

        if current_state == CircuitState.OPEN:
            raise RuntimeError(f"Circuit {self.name} is OPEN  call rejected")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        self._failure_count = 0
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self._success_threshold:
                logger.info("Circuit %s: HALF_OPEN → CLOSED (recovered)", self.name)
                self._state = CircuitState.CLOSED
                self._success_count = 0

    def _on_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._state == CircuitState.HALF_OPEN:
            logger.warning("Circuit %s: HALF_OPEN → OPEN (still failing)", self.name)
            self._state = CircuitState.OPEN
        elif self._failure_count >= self._failure_threshold:
            logger.error(
                "Circuit %s: CLOSED → OPEN (threshold=%d reached)",
                self.name,
                self._failure_threshold,
            )
            self._state = CircuitState.OPEN


# Pre-built circuit breakers for each external service
openai_breaker = CircuitBreaker("openai", failure_threshold=5, recovery_timeout=30)
deepgram_breaker = CircuitBreaker("deepgram", failure_threshold=3, recovery_timeout=20)
elevenlabs_breaker = CircuitBreaker("elevenlabs", failure_threshold=3, recovery_timeout=20)
```

### 12.3 Blue-Green Deployment Script

```python
# scripts/blue_green_deploy.py
"""
Blue-green deployment script for zero-downtime releases.

Strategy:
  1. Deploy new version to "green" deployment (alongside existing "blue")
  2. Wait for green to pass readiness checks
  3. Switch load balancer traffic to green
  4. Monitor green for 5 minutes
  5. If healthy, scale down blue; if not, roll back to blue immediately

Run: python scripts/blue_green_deploy.py --image voice-platform/gateway:2.0.0
"""

import argparse
import subprocess
import sys
import time
import httpx
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NAMESPACE = "voice-platform"
GATEWAY_SERVICE = "gateway-service"
HEALTH_CHECK_URL_TEMPLATE = "http://localhost:8001/health"  # port-forwarded green pod
ROLLBACK_ON_ERROR = True


def kubectl(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["kubectl", "-n", NAMESPACE] + args
    logger.info("Running: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def deploy_green(image: str) -> None:
    """Deploy green version by patching the deployment image."""
    kubectl([
        "set", "image",
        "deployment/voice-gateway-green",
        f"gateway={image}",
    ])
    logger.info("Green deployment updated to image: %s", image)


def wait_for_green_ready(timeout: int = 120) -> bool:
    """Wait until green deployment is fully ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = kubectl([
            "rollout", "status",
            "deployment/voice-gateway-green",
            "--timeout=10s",
        ], check=False)
        if result.returncode == 0:
            logger.info("Green deployment is ready")
            return True
        time.sleep(5)
    logger.error("Green deployment did not become ready within %ds", timeout)
    return False


def switch_traffic_to_green() -> None:
    """Patch service selector to route traffic to green pods."""
    kubectl([
        "patch", "service", GATEWAY_SERVICE,
        "--patch", '{"spec": {"selector": {"version": "green"}}}',
    ])
    logger.info("Traffic switched to green deployment")


def rollback_to_blue() -> None:
    """Restore service selector to blue pods."""
    kubectl([
        "patch", "service", GATEWAY_SERVICE,
        "--patch", '{"spec": {"selector": {"version": "blue"}}}',
    ])
    logger.warning("ROLLBACK: Traffic restored to blue deployment")


def monitor_green(duration: int = 300) -> bool:
    """Monitor green for error spikes over a duration (seconds)."""
    logger.info("Monitoring green deployment for %ds...", duration)
    start = time.time()
    error_count = 0
    check_interval = 15

    while time.time() - start < duration:
        try:
            r = httpx.get(HEALTH_CHECK_URL_TEMPLATE, timeout=5)
            if r.status_code != 200:
                error_count += 1
                logger.warning("Health check failed: status=%d (errors=%d)", r.status_code, error_count)
        except Exception as e:
            error_count += 1
            logger.warning("Health check exception: %s (errors=%d)", e, error_count)

        if error_count >= 3:
            logger.error("Green deployment has %d health check failures  triggering rollback", error_count)
            return False

        time.sleep(check_interval)

    logger.info("Green deployment passed monitoring period with %d errors", error_count)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Blue-green deployment")
    parser.add_argument("--image", required=True, help="New Docker image to deploy")
    args = parser.parse_args()

    try:
        logger.info("=== Starting blue-green deployment: %s ===", args.image)

        deploy_green(args.image)

        if not wait_for_green_ready():
            logger.error("Green not ready  aborting deployment (blue still serving)")
            sys.exit(1)

        switch_traffic_to_green()

        if not monitor_green(duration=300):
            rollback_to_blue()
            sys.exit(1)

        # Scale down blue
        kubectl(["scale", "deployment/voice-gateway-blue", "--replicas=0"])
        logger.info("=== Blue-green deployment complete. Green is live. ===")

    except Exception as e:
        logger.error("Deployment failed: %s", e)
        if ROLLBACK_ON_ERROR:
            rollback_to_blue()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---

## 13. Database Schema

```sql
-- db/init.sql
-- Initialize the Voice Agent Platform database schema.
-- Compatible with PostgreSQL 16 + TimescaleDB extension.

-- Enable TimescaleDB for time-series optimization on turns table
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- -----------------------------------------------------------------------
-- Agents
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agents (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'draft',
    config_json JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_agents_status ON agents(status);
CREATE INDEX idx_agents_name ON agents(name);

-- -----------------------------------------------------------------------
-- Users
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id                   TEXT PRIMARY KEY,
    phone_number         TEXT UNIQUE,
    email                TEXT,
    display_name         TEXT,
    metadata             JSONB DEFAULT '{}',
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    consent_recorded_at  TIMESTAMPTZ,
    pii_redacted         BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_users_phone ON users(phone_number);

-- -----------------------------------------------------------------------
-- Calls
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS calls (
    id                        TEXT PRIMARY KEY,
    external_id               TEXT,          -- Twilio CallSid
    direction                 TEXT NOT NULL,
    status                    TEXT NOT NULL DEFAULT 'initiated',
    from_number               TEXT NOT NULL,
    to_number                 TEXT NOT NULL,
    agent_id                  TEXT NOT NULL REFERENCES agents(id),
    session_id                TEXT,
    initiated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    answered_at               TIMESTAMPTZ,
    ended_at                  TIMESTAMPTZ,
    billable_duration_seconds FLOAT,
    telephony_cost_usd        FLOAT,
    metadata                  JSONB DEFAULT '{}'
);

CREATE INDEX idx_calls_agent_id ON calls(agent_id);
CREATE INDEX idx_calls_initiated_at ON calls(initiated_at DESC);
CREATE INDEX idx_calls_external_id ON calls(external_id);

-- -----------------------------------------------------------------------
-- Sessions
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    id                  TEXT PRIMARY KEY,
    call_id             TEXT REFERENCES calls(id),
    agent_id            TEXT NOT NULL REFERENCES agents(id),
    user_id             TEXT REFERENCES users(id),
    status              TEXT NOT NULL DEFAULT 'active',
    channel             TEXT NOT NULL DEFAULT 'phone',
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at            TIMESTAMPTZ,
    duration_seconds    FLOAT,
    total_turns         INTEGER DEFAULT 0,
    total_cost_usd      FLOAT DEFAULT 0,
    avg_latency_ms      FLOAT DEFAULT 0,
    task_completed      BOOLEAN,
    csat_score          INTEGER CHECK (csat_score BETWEEN 1 AND 5),
    recording_url       TEXT,
    consent_obtained    BOOLEAN DEFAULT FALSE,
    pii_redacted        BOOLEAN DEFAULT FALSE,
    metadata            JSONB DEFAULT '{}'
);

CREATE INDEX idx_sessions_agent_id ON sessions(agent_id);
CREATE INDEX idx_sessions_started_at ON sessions(started_at DESC);
CREATE INDEX idx_sessions_user_id ON sessions(user_id);

-- -----------------------------------------------------------------------
-- Turns (TimescaleDB hypertable for efficient time-series queries)
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS turns (
    id                  TEXT NOT NULL,
    session_id          TEXT NOT NULL REFERENCES sessions(id),
    sequence_number     INTEGER NOT NULL,
    role                TEXT NOT NULL,
    transcript          TEXT,
    vad_latency_ms      FLOAT,
    asr_latency_ms      FLOAT,
    llm_latency_ms      FLOAT,
    tts_latency_ms      FLOAT,
    total_latency_ms    FLOAT,
    llm_input_tokens    INTEGER DEFAULT 0,
    llm_output_tokens   INTEGER DEFAULT 0,
    tts_characters      INTEGER DEFAULT 0,
    estimated_cost_usd  FLOAT DEFAULT 0,
    asr_confidence      FLOAT,
    was_barge_in        BOOLEAN DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
);

-- Convert to TimescaleDB hypertable (partitioned by time)
SELECT create_hypertable('turns', 'created_at', if_not_exists => TRUE);

CREATE INDEX idx_turns_session_id ON turns(session_id, created_at DESC);
CREATE INDEX idx_turns_role ON turns(role, created_at DESC);

-- -----------------------------------------------------------------------
-- Useful views
-- -----------------------------------------------------------------------
CREATE OR REPLACE VIEW session_summary AS
SELECT
    s.id,
    s.agent_id,
    a.name AS agent_name,
    s.user_id,
    s.channel,
    s.started_at,
    s.duration_seconds,
    s.total_turns,
    s.total_cost_usd,
    s.avg_latency_ms,
    s.task_completed,
    s.csat_score,
    COUNT(t.id) FILTER (WHERE t.was_barge_in) AS barge_in_count
FROM sessions s
JOIN agents a ON s.agent_id = a.id
LEFT JOIN turns t ON s.id = t.session_id
GROUP BY s.id, a.name;
```

---

## 14. Vocabulary Cheat Sheet

A comprehensive reference for every technical term used across this series.

| Term | Definition |
|---|---|
| **ASR** | Automatic Speech Recognition  converts spoken audio to text |
| **TTS** | Text-to-Speech  converts text to spoken audio |
| **VAD** | Voice Activity Detection  detects when speech starts and ends |
| **PCM** | Pulse-Code Modulation  raw uncompressed digital audio format |
| **Sample Rate** | Audio samples per second (e.g., 8000 Hz for telephony, 16000 Hz for ASR) |
| **Bit Depth** | Bits per audio sample (16-bit is standard for voice) |
| **Codec** | Algorithm for encoding/decoding audio (μ-law, OPUS, MP3) |
| **μ-law / mulaw** | Telephony audio codec used by PSTN and Twilio (8kHz, 8-bit) |
| **OPUS** | Modern low-latency audio codec used in WebRTC |
| **WebRTC** | Web Real-Time Communication  browser-based audio/video streaming |
| **SIP** | Session Initiation Protocol  VoIP signaling standard |
| **PSTN** | Public Switched Telephone Network  traditional phone network |
| **Twilio** | Cloud telephony platform; provides phone numbers and call routing |
| **TwiML** | Twilio Markup Language  XML instructions for call flow |
| **Media Stream** | Twilio feature that delivers call audio over WebSocket |
| **Barge-in** | User interrupts agent while agent is speaking |
| **Streaming ASR** | Transcription delivered word-by-word as speech occurs |
| **TTFT** | Time to First Token  latency from LLM request to first output token |
| **End-to-end Latency** | Time from end of user speech to start of agent audio |
| **Latency Budget** | Allocation of total latency across pipeline stages |
| **LLM** | Large Language Model (GPT-4o, Claude 3.5, etc.) |
| **Function Calling** | LLM feature to invoke external tools/APIs |
| **Streaming Completion** | LLM response delivered token-by-token |
| **RAG** | Retrieval-Augmented Generation  injecting retrieved context into LLM |
| **ChromaDB** | Open-source vector database for semantic search |
| **Embeddings** | Dense vector representations of text for semantic similarity |
| **Redis** | In-memory data store; used for session state and caching |
| **TimescaleDB** | PostgreSQL extension for time-series data |
| **Session** | A single voice conversation from start to finish |
| **Turn** | One utterance from either user or agent |
| **Slot Filling** | Collecting required entities through a conversation (name, date, etc.) |
| **Intent** | The user's goal/purpose in an utterance (book_table, check_order, etc.) |
| **Entity** | A structured data value extracted from an utterance (date, name, number) |
| **Dialog State** | The current state of the conversation (which slots are filled, etc.) |
| **System Prompt** | Instructions given to the LLM that define agent behavior |
| **Pydantic** | Python data validation library using type annotations |
| **FastAPI** | Python async web framework with automatic OpenAPI docs |
| **Prometheus** | Open-source metrics collection and alerting toolkit |
| **Grafana** | Metrics visualization and dashboarding platform |
| **Jaeger** | Distributed tracing system for microservices |
| **OpenTelemetry** | Vendor-neutral observability framework (traces, metrics, logs) |
| **HPA** | Horizontal Pod Autoscaler  Kubernetes automatic scaling |
| **Circuit Breaker** | Resilience pattern that stops calling failing services |
| **Blue-Green Deployment** | Zero-downtime deployment by running two identical environments |
| **PII** | Personally Identifiable Information (names, phone numbers, SSNs) |
| **GDPR** | General Data Protection Regulation (EU privacy law) |
| **HIPAA** | Health Insurance Portability and Accountability Act (US healthcare privacy) |
| **TCPA** | Telephone Consumer Protection Act (US telemarketing law) |
| **CSAT** | Customer Satisfaction Score (1-5 rating) |
| **AHT** | Average Handle Time  mean duration of a voice session |
| **FCR** | First Call Resolution  % of issues resolved without a callback |
| **A/B Test** | Controlled experiment comparing two agent configurations |
| **DSL** | Domain-Specific Language  a specialized configuration syntax |
| **Stateless Worker** | A compute process that holds no persistent state between requests |
| **Externalized State** | Architecture pattern where state lives in dedicated stores, not workers |
| **Graceful Degradation** | System continues operating at reduced capability when a component fails |

---

## 15. The Full Journey: Parts 0–19 Recap

We started with a simple question: *What does it actually take to build a voice agent?*

Here is the answer, compressed into nineteen parts:

```mermaid
graph LR
    P0["Part 0\nWhy Voice Is Hard"] --> P1["Part 1\nAudio Fundamentals"]
    P1 --> P2["Part 2\nVAD"]
    P2 --> P3["Part 3\nStreaming ASR"]
    P3 --> P4["Part 4\nNeural TTS"]
    P4 --> P5["Part 5\nLatency Engineering"]
    P5 --> P6["Part 6\nTelephony + WebRTC"]
    P6 --> P7["Part 7\nBarge-in"]
    P7 --> P8["Part 8\nLLM Integration"]
    P8 --> P9["Part 9\nIntent + Entities"]
    P9 --> P10["Part 10\nDialog Management"]
    P10 --> P11["Part 11\nMemory Systems"]
    P11 --> P12["Part 12\nMulti-modal IVR"]
    P12 --> P13["Part 13\nStreaming Arch"]
    P13 --> P14["Part 14\nObservability"]
    P14 --> P15["Part 15\nCost Optimization"]
    P15 --> P16["Part 16\nScalability"]
    P16 --> P17["Part 17\nAgent Personas"]
    P17 --> P18["Part 18\nSecurity + Compliance"]
    P18 --> P19["Part 19\nThe Capstone"]
```

Each part built on the last. VAD made ASR possible. ASR made LLM integration meaningful. LLM integration made dialog management necessary. Dialog management demanded memory. Memory required observability. Observability enabled optimization. Optimization enabled scale. Scale demanded security. And security enabled trust  the foundation of any production system.

---

## 16. What to Learn Next

Finishing this series is a beginning, not an end. Here is where to go from here:

### 16.1 Deepen Your Knowledge

| Topic | Resource | Why |
|---|---|---|
| **Audio DSP** | Julius O. Smith's "Mathematics of the DFT" (ccrma.stanford.edu) | Understand what is happening inside VAD and ASR at the signal level |
| **Speech Science** | "Spoken Language Processing"  Huang, Acero, Hon | The academic foundation of everything ASR/TTS does |
| **LLM Engineering** | "Building LLM-Powered Applications"  Valentina Alto | Beyond function calling to advanced agent architectures |
| **Distributed Systems** | "Designing Data-Intensive Applications"  Martin Kleppmann | The mental models behind everything in our state layer |
| **SRE** | Google's "Site Reliability Engineering" (sre.google/books) | How to operate production systems at scale |
| **Kafka** | Confluent's Kafka documentation | Next-level event streaming for high-volume voice platforms |

### 16.2 Career Paths in Voice AI

The skills you have built apply across several growing career tracks:

**Voice AI Engineer**  Build and maintain voice agent platforms. Deep expertise in audio processing, real-time streaming, and LLM integration. High demand in healthcare, fintech, and e-commerce.

**Conversational AI Designer**  Design dialog flows, agent personas, and user experiences. Combines linguistic expertise with LLM prompting and evaluation skills.

**ML Platform Engineer**  Build the infrastructure that ML models run on. Focus on model serving, latency optimization, and cost management at scale.

**AI Product Manager**  Define what voice AI products should do, measure whether they succeed, and prioritize improvements. Requires deep technical literacy.

**Voice AI Researcher**  Advance the state of the art in ASR, TTS, spoken dialog systems. Academic or industrial research track.

### 16.3 The Future of Voice AI

The landscape is moving fast. Here is what to watch:

**GPT-4o Native Audio**  OpenAI's native audio model processes speech directly without a separate ASR step, dramatically reducing latency and preserving prosody. This will reshape the pipeline architecture we built.

**Real-time Multimodal Models**  Models that simultaneously understand audio, video, and text are arriving. Your voice agent may soon also see the caller's screen or face.

**Ambient AI**  Always-on voice agents embedded in physical spaces (offices, hospitals, retail floors) that help proactively rather than reactively.

**Smaller, Faster Models**  Distilled models that run locally on devices (phones, smart speakers, embedded systems) will eliminate cloud round-trip latency entirely.

**Emotional Intelligence**  Models that detect and respond appropriately to caller emotions (frustration, urgency, distress) will dramatically improve CSAT scores.

**Voice Cloning and Persona**  Agents that sound exactly like a brand's ideal voice, with consistent emotional tone, are already commercially available.

The infrastructure patterns we have built  streaming pipelines, stateless workers, externalized state, observability-first design  will remain relevant regardless of which underlying models power them. Good architecture outlasts any individual model.

---

## 17. Series Conclusion

If you started this series with Part 0 and have made it here, take a moment to appreciate what you have built.

You understand audio at the byte level. You know why 16kHz matters and what μ-law sounds like. You have wired together VAD, streaming ASR, LLM, and TTS into a coherent pipeline and measured every millisecond of latency along the way. You have connected a real phone number to a real AI agent and heard it respond to a human voice. You have designed for failure, for privacy, for cost, for scale, and for compliance.

Most importantly, you have built something that *speaks*.

Voice is the most natural interface humans have. It is how we comfort our children, negotiate deals, tell stories, and ask for help. Building AI that participates in that  that can listen and respond with genuine usefulness  is not a small thing.

The systems we have built in this series are not perfect. No production system ever is. They will break in ways you did not anticipate. Callers will say things your prompts never imagined. Deepgram will have an outage at 2 AM. Your circuit breaker will trip on a busy Friday afternoon. These are not failures  they are the curriculum of production engineering.

Every incident is a data point. Every barge-in event is feedback about your TTS pacing. Every dropped call is a signal worth investigating. The evaluation framework you built in Part 19 gives you the instruments to hear those signals and act on them.

The future of voice AI will be built by people who understand both the technical depth and the human stakes. People who care not just about latency percentiles, but about the 80-year-old caller who needed help understanding their insurance bill. The developer who ships a robust healthcare scheduling agent is not just writing code  they are improving someone's access to care.

Build things that matter. Measure what you build. Fix what breaks. Share what you learn.

The microphone is open.

Go build something incredible.

---

## Appendix A: Project Directory Structure

```
voice-agent-platform/
├── platform/
│   ├── __init__.py
│   ├── models.py                    # Pydantic data models
│   ├── api/
│   │   ├── app.py                   # FastAPI application
│   │   ├── health.py                # Health/readiness endpoints
│   │   └── dependencies.py          # Dependency injection helpers
│   ├── services/
│   │   ├── session_manager.py       # Redis-backed session management
│   │   ├── memory_service.py        # Short + long-term memory
│   │   ├── agent_service.py         # Agent config CRUD + cache
│   │   └── analytics_service.py     # Turn logging + metrics queries
│   ├── pipeline/
│   │   └── voice_pipeline.py        # VAD → ASR → LLM → TTS orchestration
│   ├── engines/
│   │   ├── vad_engine.py            # Voice activity detection
│   │   ├── asr_engine.py            # Automatic speech recognition
│   │   └── tts_engine.py            # Text-to-speech synthesis
│   ├── config/
│   │   ├── agent_schema.py          # JSON Schema for agent validation
│   │   └── agent_builder.py         # Fluent AgentConfigBuilder + examples
│   ├── evaluation/
│   │   └── evaluator.py             # VoiceAgentEvaluator + EvaluationReport
│   ├── observability/
│   │   ├── metrics.py               # Prometheus metric definitions
│   │   └── logging_config.py        # Structured JSON logging
│   ├── middleware/
│   │   └── rate_limiter.py          # Redis-based rate limiting
│   └── resilience/
│       └── circuit_breaker.py       # Async circuit breaker pattern
├── db/
│   └── init.sql                     # PostgreSQL schema + TimescaleDB setup
├── tests/
│   ├── test_platform_smoke.py        # Smoke tests (live API)
│   ├── test_models.py                # Pydantic model validation tests
│   ├── test_pipeline.py              # Pipeline unit tests (mocked engines)
│   └── test_evaluator.py             # Evaluation framework tests
├── scripts/
│   └── blue_green_deploy.py          # Zero-downtime deployment script
├── monitoring/
│   ├── prometheus.yml                # Prometheus scrape config
│   └── grafana/
│       ├── dashboards/               # Dashboard JSON definitions
│       └── datasources/              # Grafana datasource config
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── gateway-deployment.yaml
│   ├── gateway-hpa.yaml
│   ├── worker-deployment.yaml
│   ├── worker-hpa.yaml
│   ├── redis-statefulset.yaml
│   └── services.yaml
├── quickstart/
│   ├── create_agent.py
│   ├── configure_twilio.py
│   └── view_dashboard.py
├── docker-compose.yml
├── Dockerfile.gateway
├── Dockerfile.worker
├── .env.example
├── requirements.txt
└── README.md
```

---

## Appendix B: .env.example

```bash
# Voice Agent Platform  Environment Variables
# Copy to .env and fill in your values.
# NEVER commit .env to version control.

# -----------------------------------------------------------------------
# API Keys
# -----------------------------------------------------------------------
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPGRAM_API_KEY=...
ELEVENLABS_API_KEY=...
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...

# -----------------------------------------------------------------------
# Infrastructure
# -----------------------------------------------------------------------
REDIS_URL=redis://localhost:6379/0
DATABASE_URL=postgresql+asyncpg://voice:voice@localhost:5432/voiceagent
CHROMA_HOST=localhost
CHROMA_PORT=8000

# -----------------------------------------------------------------------
# Platform URLs (update with ngrok URL for local dev)
# -----------------------------------------------------------------------
WEBHOOK_BASE_URL=https://your-ngrok-url.ngrok.io
WEBSOCKET_BASE_URL=wss://your-ngrok-url.ngrok.io

# -----------------------------------------------------------------------
# Observability
# -----------------------------------------------------------------------
OTEL_EXPORTER_JAEGER_ENDPOINT=http://localhost:14268/api/traces
LOG_LEVEL=INFO

# -----------------------------------------------------------------------
# Security (generate strong random values)
# -----------------------------------------------------------------------
SECRET_KEY=change-me-to-a-long-random-string
POSTGRES_PASSWORD=change-me-to-a-strong-password
```

---

*This article is Part 19 of 19 in the "Building Voice Agents  A Developer's Deep Dive from Audio Fundamentals to Production" series.*

*Parts 0–18 are available in the same repository.*

*Thank you for reading. Build boldly.*


