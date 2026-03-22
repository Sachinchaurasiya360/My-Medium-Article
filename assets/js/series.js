// Central data for all series — used by index.html, article.html, series pages
const SERIES = [
  {
    id: "Machine-Learning",
    title: "Machine Learning",
    emoji: "🤖",
    color: "purple",
    gradient: "linear-gradient(135deg,#8b5cf6,#6366f1)",
    description: "Linear regression to production ML platforms. Every algorithm built from scratch, then with PyTorch and scikit-learn.",
    level: "Beginner → Production",
    parts: 20,
    articles: Array.from({length:20},(_,i)=>({
      num: i,
      file: `Machine-Learning/ml-deep-dive-part-${i}.md`,
      titles: [
        "The ML Landscape — What ML Actually Is (And Isn't)",
        "The Math You Actually Need",
        "Linear Regression from Scratch",
        "Classification — Logistic Regression and Evaluation",
        "Trees and Forests — Decision Trees to XGBoost",
        "The Algorithm Zoo — SVMs, KNN, Naive Bayes",
        "Unsupervised Learning — Clustering and PCA",
        "Feature Engineering",
        "Neural Networks from Scratch",
        "PyTorch Fundamentals",
        "CNNs — Teaching Machines to See",
        "Sequence Models — RNNs and LSTMs",
        "Training Deep Networks — Optimizers and Debugging",
        "Transfer Learning",
        "NLP with Transformers and BERT",
        "Advanced Computer Vision — Detection, GANs, ViT",
        "Model Evaluation and Selection",
        "ML System Design",
        "MLOps — Notebook to Production",
        "The Capstone — Production ML Platform"
      ][i]
    }))
  },
  {
    id: "AI-Memory",
    title: "AI Memory Systems",
    emoji: "🧠",
    color: "blue",
    gradient: "linear-gradient(135deg,#3b82f6,#2dd4bf)",
    description: "How machines remember, retrieve, and reason. Attention, embeddings, vector databases, and autonomous agents.",
    level: "Beginner → Production",
    parts: 20,
    articles: Array.from({length:20},(_,i)=>({
      num: i,
      file: `AI-Memory/ai-memory-deep-dive-part-${i}.md`,
      titles: [
        "What Is Memory in AI?",
        "How Machines Represent Information — Tokens, Numbers, Vectors",
        "Neural Networks as Memory Systems",
        "The Attention Mechanism — Teaching AI to Focus",
        "Transformers and Context Windows",
        "The Memory Wall and Breaking Through",
        "External Memory — When Context Isn't Enough",
        "Embeddings — Teaching Machines to Understand Meaning",
        "Building and Understanding Vector Databases",
        "Retrieval-Augmented Generation",
        "Chunking and Retrieval Optimization",
        "Short-Term vs Long-Term Memory in AI Agents",
        "Memory Compression and Summarization",
        "Updating and Editing Memory",
        "Personalization — Memory That Knows You",
        "Multi-Modal Memory — Beyond Text",
        "Autonomous Agents With Memory",
        "Scaling Memory Systems in Production",
        "Research-Level Memory Architectures",
        "Capstone — Production AI Memory Platform"
      ][i]
    }))
  },
  {
    id: "Voice-Agents",
    title: "Voice Agents",
    emoji: "🎙️",
    color: "green",
    gradient: "linear-gradient(135deg,#10b981,#3b82f6)",
    description: "Audio processing, ASR, TTS, voice cloning, real-time pipelines, WebRTC, and telephony integration.",
    level: "Beginner → Production",
    parts: 20,
    articles: Array.from({length:20},(_,i)=>({
      num: i,
      file: `Voice-Agents/voice-agent-deep-dive-part-${i}.md`,
      titles: [
        "The Voice AI Landscape",
        "Audio Fundamentals — Sound, Signals, and Digital Audio",
        "Audio Signal Processing — Spectrograms and MFCCs",
        "Speech Recognition — From Sound Waves to Text",
        "Real-Time ASR and Wake Word Detection",
        "Speech Synthesis — Teaching Machines to Talk",
        "Voice Cloning and Custom Voices",
        "Real-Time Audio Pipelines — Streaming and WebSockets",
        "WebRTC and Telephony",
        "Your First Voice Agent — ASR + LLM + TTS",
        "Dialog Management — Turn-Taking and Interruptions",
        "Voice Agent Memory",
        "Frameworks — LiveKit, Pipecat, and Vocode",
        "Phone Call Agents — Twilio and SIP",
        "Multi-Language and Emotion",
        "Advanced Voice Features",
        "Latency Optimization",
        "Production Infrastructure",
        "Security, Testing, and Compliance",
        "Capstone — Production Voice Agent Platform"
      ][i]
    }))
  },
  {
    id: "RAG",
    title: "RAG",
    emoji: "🔍",
    color: "orange",
    gradient: "linear-gradient(135deg,#f59e0b,#ef4444)",
    description: "Retrieval-Augmented Generation from scratch — chunking, embeddings, vector search, and production deployment.",
    level: "Intermediate → Production",
    parts: 10,
    articles: Array.from({length:10},(_,i)=>({
      num: i,
      file: `RAG/rag-deep-dive-part-${i}.md`,
      titles: [
        "What Is RAG? Foundations and Why It Matters",
        "Text Preprocessing and Chunking Strategies",
        "Embeddings — The Heart of RAG",
        "Vector Databases and Indexing",
        "Retrieval Strategies — Basic to Advanced",
        "Building Your First RAG Pipeline",
        "Advanced RAG Patterns — HyDE, Re-ranking, Fusion",
        "Evaluation and Debugging RAG Systems",
        "Production RAG — Scaling and Monitoring",
        "Multi-Modal RAG, Agentic RAG, and The Future"
      ][i]
    }))
  },
  {
    id: "Kafka",
    title: "Apache Kafka",
    emoji: "⚡",
    color: "red",
    gradient: "linear-gradient(135deg,#ef4444,#f59e0b)",
    description: "Distributed logs, replication internals, stream processing, performance tuning, and production operations.",
    level: "Intermediate → Advanced",
    parts: 11,
    articles: Array.from({length:11},(_,i)=>({
      num: i,
      file: `Kafka/kafka-deep-dive-part-${i}.md`,
      titles: [
        "The Foundation You Need Before Going Deep",
        "Why Kafka Exists — The Distributed Log",
        "Architecture Internals — Brokers and KRaft",
        "Replication — ISR, Leader Election, Durability",
        "Consumer Groups and Offset Management",
        "Storage Engine — Segments and Log Compaction",
        "Producers — Batching, Idempotence, Transactions",
        "Performance Engineering",
        "Stream Processing — Kafka Streams and Flink",
        "Production Operations and Monitoring",
        "Advanced Patterns — Event Sourcing, CDC, CQRS"
      ][i]
    }))
  },
  {
    id: "Redis",
    title: "Redis",
    emoji: "🔥",
    color: "pink",
    gradient: "linear-gradient(135deg,#ec4899,#f59e0b)",
    description: "Architecture internals, data structure encodings, persistence, clustering, and production engineering.",
    level: "Intermediate → Advanced",
    parts: 9,
    articles: Array.from({length:9},(_,i)=>({
      num: i,
      file: `Redis/redis-deep-dive-part-${i}.md`,
      titles: [
        "The Foundation You Need Before Going Deep",
        "Architecture and Event Loop Internals",
        "Data Structures — Internal Encoding and Complexity",
        "Memory Management and Persistence",
        "Networking Model and Performance Engineering",
        "Replication, HA, and Sentinel",
        "Redis Cluster and Distributed Systems",
        "Advanced Use Cases and Patterns",
        "Production Engineering and Scaling"
      ][i]
    }))
  },
  {
    id: "LangChain",
    title: "LangChain",
    emoji: "🔗",
    color: "teal",
    gradient: "linear-gradient(135deg,#14b8a6,#3b82f6)",
    description: "Build LLM-powered apps: chains, prompts, agents, tools, memory, LangSmith, and production best practices.",
    level: "Intermediate",
    parts: 3,
    articles: Array.from({length:3},(_,i)=>({
      num: i,
      file: `LangChain/langchain-deep-dive-part-${i}.md`,
      titles: [
        "LangChain Fundamentals — Chains, Prompts, and Models",
        "Agents, Tools, Memory, and Advanced RAG",
        "Production LangChain — LangSmith and Deployment"
      ][i]
    }))
  }
];

// Helper: find series by id
function getSeriesById(id) {
  return SERIES.find(s => s.id === id);
}

// Helper: find prev/next article across the same series
function getNeighbors(seriesId, partNum) {
  const s = getSeriesById(seriesId);
  if (!s) return { prev: null, next: null };
  const prev = s.articles.find(a => a.num === partNum - 1) || null;
  const next = s.articles.find(a => a.num === partNum + 1) || null;
  return { prev, next, series: s };
}
