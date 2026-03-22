# LangChain Deep Dive  Part 0: LangChain Fundamentals  Chains, Prompts, and Models

---

**Series:** LangChain  A Developer's Deep Dive for Building LLM Applications
**Part:** 0 of 2 (Foundations)
**Audience:** Developers who want to build production LLM applications with LangChain
**Reading time:** ~50 minutes

---

## Why This Series Exists

You've called the OpenAI API. You've written a prompt. You've gotten a response back and thought, "Cool, I can build a product on this." Then reality hit.

Your prompt needed to change based on user input. You needed to chain multiple LLM calls together. You needed to search a database before answering. You needed to parse structured output. You needed to handle streaming. You needed retries when the API flaked out. You needed to remember what the user said three messages ago. You needed to give the model access to tools  a calculator, a web browser, a database.

Suddenly your "simple LLM wrapper" is 2,000 lines of spaghetti code with string formatting everywhere, nested try-catch blocks, and a growing sense of dread every time you need to change something.

**LangChain exists to solve this problem.**

It provides a standardized set of abstractions  prompts, models, chains, agents, memory, retrievers, tools  that let you compose LLM applications the way you'd compose functions in a well-designed codebase. It's not magic. It's software engineering applied to the LLM application layer.

This 3-part series will take you from "I've called an LLM API" to "I can architect and deploy production LLM applications with LangChain." We'll write real code in every section, build real pipelines, and understand every abstraction from the inside out.

> **Who this is for:** You're a developer. You've used at least one LLM API. You know Python. You want to build real applications, not toy demos.

> **Who this is NOT for:** If you're looking for a conceptual overview of AI or a beginner's guide to prompting, this series will feel overwhelming. That's intentional.

Here's what we'll cover across the three parts:

- **Part 0 (this article):** LangChain fundamentals  chains, prompts, models, LCEL, document loading, embeddings, vector stores, and a complete RAG pipeline
- **Part 1:** Agents, tools, and memory  building autonomous LLM systems that can reason, act, and remember
- **Part 2:** Production deployment  LangServe, LangSmith observability, LangGraph for stateful agents, testing, and scaling

Let's start.

---

## 1. Why LangChain?

### The Problem

Building an LLM application is fundamentally a **composition problem**. You're gluing together:

- **Prompts**  dynamic templates that change based on user input and context
- **Models**  LLM providers with different APIs, capabilities, and pricing
- **Output parsing**  extracting structured data from unstructured text
- **Data retrieval**  searching databases, documents, and APIs for relevant context
- **Memory**  maintaining conversation state across interactions
- **Tools**  giving the model access to external capabilities (search, code execution, APIs)
- **Orchestration**  chaining all of these together in the right order with error handling

Without a framework, you end up building all of these abstractions yourself. That's fine for a one-off script. It's not fine for a production application that needs to be maintained, tested, and extended by a team.

### What LangChain Gives You

LangChain provides **standardized interfaces** for each of these components. The key insight is that most LLM application patterns are the same regardless of which model or data store you're using. A RAG pipeline with OpenAI + Pinecone has the same shape as one with Anthropic + Chroma. LangChain lets you swap components without rewriting your pipeline.

Here's a concrete example. Without LangChain, a simple "summarize this document" pipeline might look like:

```python
# Without LangChain  ad hoc composition
import openai
from pypdf import PdfReader

# Load document
reader = PdfReader("report.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

# Manually chunk (hope this works)
chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]

# Summarize each chunk
summaries = []
for chunk in chunks:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Summarize the following text concisely."},
            {"role": "user", "content": chunk}
        ],
        temperature=0.3
    )
    summaries.append(response.choices[0].message.content)

# Combine summaries
combined = "\n".join(summaries)
final_response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Combine these summaries into a single coherent summary."},
        {"role": "user", "content": combined}
    ],
    temperature=0.3
)
print(final_response.choices[0].message.content)
```

Now with LangChain:

```python
# With LangChain  composable abstractions
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.summarize import load_summarize_chain

# Load and split
loader = PyPDFLoader("report.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
split_docs = splitter.split_documents(docs)

# Summarize with map-reduce
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
chain = load_summarize_chain(llm, chain_type="map_reduce")
result = chain.invoke(split_docs)
print(result["output_text"])
```

Same result. A fraction of the code. But more importantly: **every component is swappable**. Switch to Anthropic? Change one line. Switch to a different document format? Swap the loader. Need better chunking? Swap the splitter. The pipeline structure stays the same.

### When NOT to Use LangChain

LangChain is not always the right tool. Here's when you should skip it:

| Scenario | Why Skip LangChain | What to Use Instead |
|----------|-------------------|---------------------|
| Single API call, no chaining | LangChain adds unnecessary abstraction overhead | Raw OpenAI/Anthropic SDK |
| You need maximum control over every HTTP request | LangChain abstracts away request details | Raw SDK + httpx |
| Your entire app is one simple prompt | The prompt template system is overkill | f-strings |
| You need the absolute latest provider features on release day | LangChain integrations may lag behind | Raw SDK |
| You're building a search-focused app with minimal generation | LlamaIndex has better data indexing primitives | LlamaIndex |
| You need extremely low latency | Every abstraction layer adds microseconds | Raw SDK with connection pooling |

> **Honest take:** LangChain's value increases with the complexity of your application. For a one-shot prompt-and-response, it's overkill. For a multi-step pipeline with retrieval, memory, tools, and structured output, it saves you hundreds of hours of plumbing code.

---

## 2. LangChain Architecture Overview

LangChain is not a single package  it's an ecosystem. Understanding which package does what will save you from importing the wrong thing and debugging mysterious errors.

### The Ecosystem

```mermaid
graph TB
    subgraph Core["langchain-core"]
        A[Base Abstractions]
        B[LCEL Runtime]
        C[Runnables]
        D[Message Types]
        E[Prompt Templates]
        F[Output Parsers]
    end

    subgraph Main["langchain"]
        G[Chains]
        H[Agents]
        I[Retrieval Strategies]
        J[Callbacks]
    end

    subgraph Providers["Provider Packages"]
        K[langchain-openai]
        L[langchain-anthropic]
        M[langchain-google-genai]
        N[langchain-mistralai]
        O[langchain-ollama]
    end

    subgraph Community["langchain-community"]
        P[Document Loaders]
        Q[Vector Stores]
        R[Embeddings]
        S[Tools & Toolkits]
        T[Chat History Stores]
    end

    subgraph Deployment["Deployment & Ops"]
        U[LangServe]
        V[LangSmith]
        W[LangGraph]
    end

    Core --> Main
    Core --> Providers
    Core --> Community
    Main --> Deployment
    Providers --> Main
    Community --> Main

    style Core fill:#1a1a2e,stroke:#16213e,color:#e0e0e0
    style Main fill:#16213e,stroke:#0f3460,color:#e0e0e0
    style Providers fill:#0f3460,stroke:#533483,color:#e0e0e0
    style Community fill:#533483,stroke:#e94560,color:#e0e0e0
    style Deployment fill:#e94560,stroke:#e94560,color:#ffffff
```

### Package Breakdown

**`langchain-core`**  The foundation. Contains all base abstractions: `Runnable`, `BaseMessage`, `BasePromptTemplate`, `BaseOutputParser`, `BaseRetriever`, `BaseLanguageModel`. You'll import from here when building custom components. This package has minimal dependencies by design.

**`langchain`**  The main package. Contains higher-level chains, agents, and retrieval strategies built on top of `langchain-core`. This is where you find `create_stuff_documents_chain`, `create_retrieval_chain`, `AgentExecutor`, and pre-built chain constructors.

**`langchain-community`**  Third-party integrations that don't have their own dedicated package. Document loaders (PDF, CSV, web), some vector stores, some embeddings, tools, and utilities. This is the "everything else" bucket.

**Provider packages** (`langchain-openai`, `langchain-anthropic`, `langchain-google-genai`, `langchain-mistralai`, `langchain-ollama`)  First-class integrations with specific LLM providers. These are maintained separately and versioned independently. Always prefer these over the community equivalents when available.

**`langserve`**  Deploys any LCEL chain as a REST API with a single line of code. Built on FastAPI. Gives you auto-generated docs, playground UI, and streaming support for free.

**`langsmith`**  Observability platform for debugging, testing, and monitoring LLM applications. Traces every step of your chain, logs inputs/outputs, measures latency, and enables evaluation datasets. Not open source  it's a hosted service (with a generous free tier).

**`langgraph`**  Framework for building stateful, multi-actor agent applications. If `langchain` gives you chains (linear pipelines), `langgraph` gives you graphs (cycles, branches, parallel execution, persistence). This is where LangChain is heading for complex agent workflows.

### Import Map

Knowing where to import from is half the battle. Here's the cheat sheet:

```python
# Core abstractions  always from langchain-core
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document

# Provider-specific models  from provider packages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Community integrations  from langchain-community
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, CSVLoader
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# High-level chains and agents  from langchain
from langchain.chains.summarize import load_summarize_chain
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

> **Rule of thumb:** If it's a base class or primitive, it's in `langchain-core`. If it's a specific provider, it's in `langchain-{provider}`. If it's a third-party integration, it's in `langchain-community`. If it's a high-level chain or agent, it's in `langchain`.

---

## 3. Installation and Setup

### Basic Installation

```bash
# Core + main package
pip install langchain

# This automatically installs langchain-core as a dependency.
# But you should also install at least one provider:

# OpenAI (most common)
pip install langchain-openai

# Anthropic
pip install langchain-anthropic

# Google Gemini
pip install langchain-google-genai

# Local models via Ollama
pip install langchain-ollama

# Community integrations (document loaders, vector stores, etc.)
pip install langchain-community

# Common extras you'll want
pip install faiss-cpu          # FAISS vector store (CPU version)
pip install chromadb           # Chroma vector store
pip install pypdf              # PDF document loading
pip install beautifulsoup4     # Web scraping for WebBaseLoader
pip install tiktoken           # OpenAI tokenizer (needed for token counting)
pip install pydantic           # Structured output parsing
```

### One-Shot Installation for This Series

```bash
# Everything you need for Parts 0-2
pip install langchain langchain-core langchain-community \
    langchain-openai langchain-anthropic langchain-ollama \
    faiss-cpu chromadb pypdf beautifulsoup4 tiktoken \
    python-dotenv pydantic langgraph langserve
```

### API Key Setup

LangChain reads API keys from environment variables. Here's the recommended setup:

```bash
# Create a .env file in your project root
# NEVER commit this file to version control

# .env
OPENAI_API_KEY=sk-proj-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
GOOGLE_API_KEY=your-google-key-here
LANGSMITH_API_KEY=lsv2_your-langsmith-key-here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=langchain-deep-dive
```

```python
# Load environment variables at the top of every script
import os
from dotenv import load_dotenv

load_dotenv()  # Reads .env file and sets environment variables

# Verify keys are loaded
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set!"

# LangChain models automatically read from environment variables.
# You don't need to pass the key explicitly:
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")  # Automatically uses OPENAI_API_KEY

# But you CAN pass it explicitly if needed:
llm = ChatOpenAI(model="gpt-4o", api_key="sk-proj-...")
```

### Verifying Your Installation

```python
"""Quick sanity check  run this to verify everything works."""
import langchain
import langchain_core
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

print(f"langchain version: {langchain.__version__}")
print(f"langchain-core version: {langchain_core.__version__}")

# Test a simple call
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
response = llm.invoke([HumanMessage(content="Say 'LangChain is working!' and nothing else.")])
print(response.content)
# Output: LangChain is working!
```

---

## 4. LangChain Expression Language (LCEL)

LCEL is the single most important concept in modern LangChain. If you learn nothing else from this article, learn LCEL. It's how you build everything.

### What Is LCEL?

**LangChain Expression Language (LCEL)** is a declarative way to compose LangChain components into chains using the **pipe operator** (`|`). It replaced the old `LLMChain`, `SequentialChain`, and `SimpleSequentialChain` classes, which were verbose, rigid, and hard to debug.

Think of it like Unix pipes. In Unix:

```bash
cat file.txt | grep "error" | sort | uniq -c
```

Each command takes input, transforms it, and passes output to the next. LCEL works the same way, but for LLM operations:

```python
chain = prompt | model | output_parser
```

Each component is a **Runnable**  an object that implements a standard interface for data transformation.

### The Runnable Interface

Every LCEL component implements the `Runnable` interface from `langchain-core`. This interface guarantees these methods:

| Method | What It Does | Use Case |
|--------|-------------|----------|
| `invoke(input)` | Process a single input synchronously | Most common  one request, one response |
| `batch(inputs)` | Process a list of inputs (with concurrency) | Bulk processing, parallel execution |
| `stream(input)` | Stream output tokens as they're generated | Real-time UI, chatbots |
| `ainvoke(input)` | Async version of invoke | Web servers, async pipelines |
| `abatch(inputs)` | Async version of batch | Async bulk processing |
| `astream(input)` | Async version of stream | Async streaming |
| `astream_events(input)` | Stream intermediate steps and final output | Debugging, detailed streaming |

The beauty: **every component gets all of these methods for free.** When you build a chain with LCEL, the chain automatically supports invoke, batch, stream, and all async variants without you writing any extra code.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Build a chain with LCEL
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

chain = prompt | model | parser

# All of these work automatically:
# 1. Single invocation
result = chain.invoke({"topic": "programming"})
print(result)

# 2. Batch processing (processes in parallel!)
results = chain.batch([
    {"topic": "programming"},
    {"topic": "data science"},
    {"topic": "DevOps"},
])
for r in results:
    print(r)

# 3. Streaming (tokens arrive as they're generated)
for chunk in chain.stream({"topic": "programming"}):
    print(chunk, end="", flush=True)
print()

# 4. Async invocation
import asyncio

async def main():
    result = await chain.ainvoke({"topic": "programming"})
    print(result)

asyncio.run(main())
```

### The Pipe Operator in Detail

The pipe operator `|` is syntactic sugar for `RunnableSequence`. These two are identical:

```python
from langchain_core.runnables import RunnableSequence

# These are the same:
chain_a = prompt | model | parser
chain_b = RunnableSequence(first=prompt, middle=[model], last=parser)

# You can verify:
print(type(chain_a))  # <class 'langchain_core.runnables.base.RunnableSequence'>
```

When you call `chain.invoke(input)`:
1. `input` is passed to `prompt.invoke(input)`, producing a `ChatPromptValue`
2. The `ChatPromptValue` is passed to `model.invoke(prompt_value)`, producing an `AIMessage`
3. The `AIMessage` is passed to `parser.invoke(ai_message)`, producing a `str`
4. The `str` is returned to you

```mermaid
flowchart LR
    A["Input Dict<br/>{'topic': 'Python'}"] --> B["ChatPromptTemplate<br/>.invoke()"]
    B --> C["ChatPromptValue<br/>[SystemMessage, HumanMessage]"]
    C --> D["ChatOpenAI<br/>.invoke()"]
    D --> E["AIMessage<br/>content='Here is a joke...'"]
    E --> F["StrOutputParser<br/>.invoke()"]
    F --> G["str<br/>'Here is a joke...'"]

    style A fill:#2d3436,stroke:#636e72,color:#dfe6e9
    style B fill:#0984e3,stroke:#74b9ff,color:#ffffff
    style C fill:#2d3436,stroke:#636e72,color:#dfe6e9
    style D fill:#6c5ce7,stroke:#a29bfe,color:#ffffff
    style E fill:#2d3436,stroke:#636e72,color:#dfe6e9
    style F fill:#00b894,stroke:#55efc4,color:#ffffff
    style G fill:#2d3436,stroke:#636e72,color:#dfe6e9
```

### RunnablePassthrough

`RunnablePassthrough` passes the input through unchanged. It's used when you need to forward the original input to a later step in the chain while also doing something else.

```python
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Example: pass the original question AND a formatted context to the model
retriever = vectorstore.as_retriever()

chain = (
    RunnableParallel(
        context=retriever,                        # Retrieves relevant docs
        question=RunnablePassthrough()             # Passes the question through unchanged
    )
    | prompt
    | model
    | parser
)

# When you call chain.invoke("What is LCEL?"):
# - retriever receives "What is LCEL?" and returns documents
# - RunnablePassthrough receives "What is LCEL?" and returns "What is LCEL?"
# - prompt receives {"context": [docs], "question": "What is LCEL?"}
```

`RunnablePassthrough` also has a useful `.assign()` method that adds new keys to the input while keeping existing ones:

```python
from langchain_core.runnables import RunnablePassthrough

# RunnablePassthrough.assign() adds keys to the existing dict
chain = RunnablePassthrough.assign(
    upper=lambda x: x["text"].upper(),
    length=lambda x: len(x["text"])
)

result = chain.invoke({"text": "hello world"})
print(result)
# {"text": "hello world", "upper": "HELLO WORLD", "length": 11}
# Original "text" key is preserved, new keys "upper" and "length" are added
```

### RunnableLambda

`RunnableLambda` wraps any Python function into a `Runnable`, making it composable with LCEL. This is your escape hatch  any custom logic you need can be wrapped and piped.

```python
from langchain_core.runnables import RunnableLambda

# Wrap any function as a Runnable
def format_docs(docs):
    """Join document page_content with double newlines."""
    return "\n\n".join(doc.page_content for doc in docs)

format_docs_runnable = RunnableLambda(format_docs)

# Now it works in an LCEL chain:
chain = retriever | format_docs_runnable | prompt | model | parser

# You can also use the @chain decorator as syntactic sugar:
from langchain_core.runnables import chain as chain_decorator

@chain_decorator
def format_and_log(docs):
    """Wrap a function as a Runnable with the @chain decorator."""
    formatted = "\n\n".join(doc.page_content for doc in docs)
    print(f"Formatted {len(docs)} documents into {len(formatted)} characters")
    return formatted

# Use it directly in a chain  no RunnableLambda needed
chain = retriever | format_and_log | prompt | model | parser
```

### RunnableParallel

`RunnableParallel` runs multiple Runnables concurrently and collects their outputs into a dictionary. This is essential for patterns like RAG, where you need to retrieve context AND pass the question forward simultaneously.

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Run two chains in parallel
joke_chain = ChatPromptTemplate.from_template("Tell a joke about {topic}") | model | parser
poem_chain = ChatPromptTemplate.from_template("Write a haiku about {topic}") | model | parser

parallel_chain = RunnableParallel(
    joke=joke_chain,
    poem=poem_chain,
)

result = parallel_chain.invoke({"topic": "Python programming"})
print(result["joke"])  # A joke about Python
print(result["poem"])  # A haiku about Python

# Both chains run concurrently  total time is max(joke_time, poem_time), not sum
```

You can also create a `RunnableParallel` with dict syntax:

```python
# These are equivalent:
chain_a = RunnableParallel(context=retriever, question=RunnablePassthrough())
chain_b = {"context": retriever, "question": RunnablePassthrough()}

# Dict syntax is more concise and commonly used in LCEL chains
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)
```

> **Key insight:** LCEL is not just syntactic sugar. The runtime automatically handles batching with concurrency, streaming propagation through the entire chain, and async execution. If you built this manually, you'd be writing hundreds of lines of async/threading code.

---

## 5. Chat Models

Chat models are the core of every LangChain application. They take messages in, produce messages out. LangChain provides a unified interface across all providers.

### The BaseChatModel Interface

Every chat model in LangChain implements the same interface:

```python
from langchain_core.language_models import BaseChatModel

# All chat models support:
# .invoke(messages) -> AIMessage
# .stream(messages) -> Iterator[AIMessageChunk]
# .batch(messages_list) -> List[AIMessage]
# .ainvoke(messages) -> AIMessage (async)
# .bind_tools(tools) -> ChatModel (with tool calling)
# .with_structured_output(schema) -> Runnable (forces structured output)
```

### Message Types

LLM conversations are sequences of typed messages. LangChain defines these message types:

```python
from langchain_core.messages import (
    SystemMessage,    # Instructions to the model (sets behavior/persona)
    HumanMessage,     # User input
    AIMessage,        # Model response
    ToolMessage,      # Result from a tool call
    FunctionMessage,  # (Legacy) Result from a function call
)

# Create messages directly
messages = [
    SystemMessage(content="You are a helpful Python tutor."),
    HumanMessage(content="What's a list comprehension?"),
    AIMessage(content="A list comprehension is a concise way to create lists..."),
    HumanMessage(content="Can you give me an example?"),
]
```

### ChatOpenAI

The most commonly used model. Wraps OpenAI's chat completion API.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Basic initialization
llm = ChatOpenAI(
    model="gpt-4o",           # Model name
    temperature=0.7,           # Creativity (0.0 = deterministic, 2.0 = max random)
    max_tokens=1024,           # Maximum response length
    # api_key="sk-...",        # Optional  reads from OPENAI_API_KEY env var
    # base_url="https://...",  # Optional  for Azure, proxies, or compatible APIs
)

# Invoke with messages
response = llm.invoke([
    SystemMessage(content="You are a senior Python developer. Be concise."),
    HumanMessage(content="Explain decorators in 3 sentences."),
])

print(response.content)
# Decorators are functions that modify the behavior of other functions...

print(type(response))
# <class 'langchain_core.messages.ai.AIMessage'>

# The AIMessage has useful metadata
print(response.response_metadata)
# {'token_usage': {'completion_tokens': 89, 'prompt_tokens': 28, 'total_tokens': 117},
#  'model_name': 'gpt-4o', 'finish_reason': 'stop'}
```

### ChatAnthropic

Wraps Anthropic's Claude API. Claude excels at long-form analysis, careful reasoning, and following complex instructions.

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",  # Model name
    temperature=0.7,
    max_tokens=1024,
    # api_key="sk-ant-...",            # Optional  reads from ANTHROPIC_API_KEY
)

response = llm.invoke([
    SystemMessage(content="You are a senior Python developer. Be concise."),
    HumanMessage(content="Explain decorators in 3 sentences."),
])

print(response.content)
```

### ChatOllama (Local Models)

Run models locally with Ollama. No API key needed, no data leaves your machine.

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# Requires Ollama running locally: https://ollama.ai
# Pull a model first: ollama pull llama3.1
llm = ChatOllama(
    model="llama3.1",          # Any model available in Ollama
    temperature=0.7,
    # base_url="http://localhost:11434",  # Default Ollama URL
)

response = llm.invoke([
    HumanMessage(content="Explain Python decorators in 3 sentences.")
])

print(response.content)
```

### Streaming

Streaming is critical for user-facing applications. Users expect to see tokens appear in real-time, not wait 10 seconds for a wall of text.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0.7, streaming=True)

# Streaming with a for loop
for chunk in llm.stream([HumanMessage(content="Write a limerick about Python.")]):
    print(chunk.content, end="", flush=True)
print()  # Newline at the end

# Each chunk is an AIMessageChunk with a small piece of the response
# chunk.content might be "", "There", " once", " was", " a", " snake", ...
```

### Async Streaming

For web servers (FastAPI, etc.), you'll want async streaming:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import asyncio

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

async def stream_response():
    async for chunk in llm.astream([
        HumanMessage(content="Write a limerick about Python.")
    ]):
        print(chunk.content, end="", flush=True)
    print()

asyncio.run(stream_response())
```

### Model Configuration Comparison

| Parameter | OpenAI | Anthropic | Ollama |
|-----------|--------|-----------|--------|
| `model` | gpt-4o, gpt-4o-mini | claude-sonnet-4-20250514, claude-haiku-35-20241022 | llama3.1, mistral, codellama |
| `temperature` | 0.0 - 2.0 | 0.0 - 1.0 | 0.0 - 2.0 |
| `max_tokens` | 1 - 16384 (model dependent) | 1 - 8192 | Model dependent |
| `api_key` | OPENAI_API_KEY | ANTHROPIC_API_KEY | None needed |
| `base_url` | Customizable | Customizable | http://localhost:11434 |
| Cost | Per-token pricing | Per-token pricing | Free (local compute) |

---

## 6. Prompt Templates

Raw string formatting (`f"Tell me about {topic}"`) works for simple cases, but breaks down fast. You need:
- Different message types (system, human, AI)
- Dynamic few-shot examples
- Placeholders for conversation history
- Reusable templates shared across chains
- Validation that all required variables are provided

LangChain's prompt templates solve all of these.

### ChatPromptTemplate

The workhorse. Creates a list of typed messages with variable placeholders.

```python
from langchain_core.prompts import ChatPromptTemplate

# Simple template with one variable
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")

# This creates a prompt with a single HumanMessage
result = prompt.invoke({"topic": "programming"})
print(result.to_messages())
# [HumanMessage(content='Tell me a joke about programming')]

# Multi-message template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Always respond in {language}."),
    ("human", "{question}"),
])

result = prompt.invoke({
    "role": "Python tutor",
    "language": "simple English",
    "question": "What is a generator?",
})

print(result.to_messages())
# [
#   SystemMessage(content='You are a Python tutor. Always respond in simple English.'),
#   HumanMessage(content='What is a generator?')
# ]
```

### Message Tuples vs Message Objects

You can define messages as tuples `("role", "content")` or as message objects:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# Tuple syntax (more concise)
prompt_a = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
])

# Message object syntax (more explicit)
prompt_b = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant."),
    ("human", "{question}"),  # Can mix and match
])

# Both produce the same result
```

### MessagesPlaceholder

`MessagesPlaceholder` reserves a slot for a list of messages  essential for conversation history, few-shot examples, or agent scratchpads.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# Pass conversation history as a list of messages
result = prompt.invoke({
    "chat_history": [
        HumanMessage(content="What is Python?"),
        AIMessage(content="Python is a high-level programming language..."),
        HumanMessage(content="What are its main features?"),
        AIMessage(content="Key features include dynamic typing, readability..."),
    ],
    "question": "Can you give me a code example?",
})

for msg in result.to_messages():
    print(f"{msg.type}: {msg.content[:80]}...")
# system: You are a helpful assistant....
# human: What is Python?...
# ai: Python is a high-level programming language......
# human: What are its main features?...
# ai: Key features include dynamic typing, readability......
# human: Can you give me a code example?...
```

### Few-Shot Prompting Templates

Few-shot prompting provides examples to guide the model's output format and behavior.

```python
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# Define examples
examples = [
    {"input": "What is 2+2?", "output": '{"question": "2+2", "answer": 4, "confidence": 1.0}'},
    {"input": "What is the capital of France?", "output": '{"question": "capital of France", "answer": "Paris", "confidence": 0.99}'},
    {"input": "Who wrote Hamlet?", "output": '{"question": "author of Hamlet", "answer": "William Shakespeare", "confidence": 0.98}'},
]

# Create the few-shot prompt
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# Embed in a larger prompt
full_prompt = ChatPromptTemplate.from_messages([
    ("system", "You answer questions and always respond in valid JSON with the fields: question, answer, confidence."),
    few_shot_prompt,
    ("human", "{input}"),
])

# Test it
result = full_prompt.invoke({"input": "What is the speed of light?"})
for msg in result.to_messages():
    print(f"{msg.type}: {msg.content}")
```

### Partial Prompts

Sometimes you know some variables at template creation time, not at invocation time. Partial prompts let you fill in variables incrementally.

```python
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

# Partial with a static value
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Today's date is {date}."),
    ("human", "{question}"),
])

# Fill in the date now, question later
partial_prompt = prompt.partial(date=datetime.now().strftime("%Y-%m-%d"))

# Later, invoke with just the question
result = partial_prompt.invoke({"question": "What day of the week is it?"})
print(result.to_messages())

# Partial with a function (evaluated at invoke time)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Current time: {current_time}. You are a helpful assistant."),
    ("human", "{question}"),
])

partial_prompt = prompt.partial(
    current_time=lambda: datetime.now().strftime("%H:%M:%S")
)

# current_time is computed fresh each time invoke is called
```

### Complete Prompt Template Example

Here's a production-style prompt template combining everything:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# A complete RAG-style prompt with all the bells and whistles
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert technical assistant for {company_name}.

INSTRUCTIONS:
- Answer the user's question based ONLY on the provided context
- If the context doesn't contain the answer, say "I don't have enough information to answer that"
- Always cite which document(s) you used in your answer
- Be concise but thorough
- Use code examples when relevant

CONTEXT:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{question}"),
])

# This prompt accepts:
# - company_name: str
# - context: str (formatted document text)
# - chat_history: list[BaseMessage] (optional)
# - question: str
```

---

## 7. Output Parsers

LLMs return strings. You want structured data. Output parsers bridge the gap. They do two things:
1. **Provide format instructions** that you inject into the prompt (telling the model HOW to format its response)
2. **Parse the model's response** into the desired data structure

### StrOutputParser

The simplest parser. Extracts the string content from an `AIMessage`. You'll use this in almost every chain.

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage

parser = StrOutputParser()

# Extracts .content from an AIMessage
message = AIMessage(content="Hello, world!")
result = parser.invoke(message)
print(result)       # "Hello, world!"
print(type(result)) # <class 'str'>

# In an LCEL chain:
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

chain = (
    ChatPromptTemplate.from_template("Tell me a fact about {topic}")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

result = chain.invoke({"topic": "black holes"})
print(result)  # A string, not an AIMessage
```

### JsonOutputParser

Parses the model's response as JSON. Provides format instructions to the model.

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_template(
    """Extract the key information from this text and return it as JSON.

{format_instructions}

Text: {text}"""
)

chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | parser

result = chain.invoke({
    "text": "John Smith is a 35-year-old software engineer living in San Francisco. He works at Google and earns $200,000 per year.",
    "format_instructions": parser.get_format_instructions(),
})

print(result)
# {'name': 'John Smith', 'age': 35, 'occupation': 'software engineer',
#  'city': 'San Francisco', 'employer': 'Google', 'salary': 200000}
print(type(result))  # <class 'dict'>
```

### PydanticOutputParser

The gold standard for structured output. Defines the schema using Pydantic models and validates the parsed output.

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional

# Define your output schema with Pydantic
class MovieReview(BaseModel):
    """Structured movie review."""
    title: str = Field(description="The movie title")
    year: int = Field(description="Release year")
    director: str = Field(description="Director's name")
    rating: float = Field(description="Rating out of 10", ge=0, le=10)
    genres: List[str] = Field(description="List of genres")
    summary: str = Field(description="Brief plot summary in 2-3 sentences")
    recommended: bool = Field(description="Whether you'd recommend the movie")
    similar_movies: Optional[List[str]] = Field(
        default=None, description="List of similar movies"
    )

# Create the parser
parser = PydanticOutputParser(pydantic_object=MovieReview)

prompt = ChatPromptTemplate.from_template(
    """You are a movie critic. Analyze the following movie and provide a structured review.

{format_instructions}

Movie: {movie}"""
)

chain = (
    prompt.partial(format_instructions=parser.get_format_instructions())
    | ChatOpenAI(model="gpt-4o", temperature=0.3)
    | parser
)

result = chain.invoke({"movie": "The Matrix (1999)"})

# result is a validated MovieReview Pydantic object
print(f"Title: {result.title}")
print(f"Year: {result.year}")
print(f"Director: {result.director}")
print(f"Rating: {result.rating}/10")
print(f"Genres: {', '.join(result.genres)}")
print(f"Summary: {result.summary}")
print(f"Recommended: {result.recommended}")
print(f"Similar: {result.similar_movies}")

# You get full Pydantic validation:
print(result.model_dump_json(indent=2))
```

### with_structured_output (The Modern Approach)

Modern LLMs (GPT-4o, Claude 3.5+) support native structured output. Instead of prompt engineering the format, you can tell the model to output a specific schema directly. LangChain wraps this with `.with_structured_output()`:

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List

class MovieReview(BaseModel):
    """Structured movie review."""
    title: str = Field(description="The movie title")
    year: int = Field(description="Release year")
    rating: float = Field(description="Rating out of 10")
    genres: List[str] = Field(description="List of genres")
    summary: str = Field(description="Brief plot summary")

# No parser needed  the model outputs the schema directly
llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm = llm.with_structured_output(MovieReview)

result = structured_llm.invoke("Analyze the movie Inception (2010)")
print(type(result))  # <class 'MovieReview'>
print(result.title)  # Inception
print(result.rating) # 9.2

# This works in LCEL chains too:
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Analyze this movie: {movie}")
chain = prompt | structured_llm

result = chain.invoke({"movie": "Interstellar (2014)"})
```

> **Best practice:** Use `.with_structured_output()` when your model supports it (GPT-4o, Claude 3.5+). It's more reliable than prompt-based parsing because the model's token generation is constrained to the schema. Fall back to `PydanticOutputParser` for models that don't support structured output natively.

### CommaSeparatedListOutputParser

A simple parser for when you just need a list of items.

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

parser = CommaSeparatedListOutputParser()

prompt = ChatPromptTemplate.from_template(
    """List 5 {category}.

{format_instructions}"""
)

chain = (
    prompt.partial(format_instructions=parser.get_format_instructions())
    | ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    | parser
)

result = chain.invoke({"category": "programming languages for data science"})
print(result)
# ['Python', 'R', 'Julia', 'Scala', 'SQL']
print(type(result))  # <class 'list'>
```

### Output Parser Comparison

| Parser | Output Type | Schema Required? | Validation? | Best For |
|--------|------------|-----------------|-------------|----------|
| `StrOutputParser` | `str` | No | No | Simple text responses |
| `JsonOutputParser` | `dict` | Optional (Pydantic) | Basic JSON | Flexible JSON output |
| `PydanticOutputParser` | Pydantic model | Yes | Full Pydantic | Structured data with validation |
| `.with_structured_output()` | Pydantic model | Yes | Full Pydantic | Production structured output (modern models) |
| `CommaSeparatedListOutputParser` | `list[str]` | No | No | Simple lists |

---

## 8. Chains with LCEL

Now we combine everything. Chains are the core pattern in LangChain  they're how you build multi-step LLM pipelines. With LCEL, chains are just sequences of Runnables connected by pipes.

### Pattern 1: Simple Chain

The most basic pattern: prompt -> model -> parser.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Simple chain: take a topic, generate an explanation
chain = (
    ChatPromptTemplate.from_template(
        "Explain {concept} to a {audience} in exactly 3 sentences."
    )
    | ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    | StrOutputParser()
)

result = chain.invoke({
    "concept": "recursion",
    "audience": "5-year-old"
})
print(result)
# Recursion is like standing between two mirrors and seeing yourself
# repeated forever. In programming, it means a function that calls
# itself to solve smaller pieces of a bigger problem. It's like
# opening a box that has a smaller box inside, which has an even
# smaller box inside!
```

### Pattern 2: Sequential Chains

The output of one chain feeds into the next. This is how you build multi-step reasoning pipelines.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
parser = StrOutputParser()

# Step 1: Generate a story outline
outline_chain = (
    ChatPromptTemplate.from_template(
        "Create a brief story outline (3-5 bullet points) for a {genre} story about {topic}."
    )
    | llm
    | parser
)

# Step 2: Write the story based on the outline
story_chain = (
    ChatPromptTemplate.from_template(
        """Based on this outline, write a short story (200-300 words).

Outline:
{outline}

Write the story:"""
    )
    | llm
    | parser
)

# Step 3: Generate a title based on the story
title_chain = (
    ChatPromptTemplate.from_template(
        """Based on this story, generate a creative title (just the title, nothing else).

Story:
{story}"""
    )
    | llm
    | parser
)

# Compose them sequentially
full_chain = (
    # Step 1: generate outline
    {"outline": outline_chain, "genre": lambda x: x["genre"], "topic": lambda x: x["topic"]}
    # Step 2: generate story from outline
    | {"story": story_chain, "outline": lambda x: x["outline"]}
    # Step 3: generate title from story
    | {"title": title_chain, "story": lambda x: x["story"], "outline": lambda x: x.get("outline", "")}
)

result = full_chain.invoke({"genre": "sci-fi", "topic": "time travel"})
print(f"Title: {result['title']}")
print(f"\n{result['story']}")
```

A cleaner way to compose sequential chains using `RunnablePassthrough.assign()`:

```python
from langchain_core.runnables import RunnablePassthrough

# Cleaner sequential composition with .assign()
full_chain = (
    RunnablePassthrough.assign(
        outline=outline_chain  # Adds 'outline' key to the dict
    )
    | RunnablePassthrough.assign(
        story=story_chain      # Adds 'story' key (can use 'outline' from previous step)
    )
    | RunnablePassthrough.assign(
        title=title_chain      # Adds 'title' key (can use 'story' from previous step)
    )
)

result = full_chain.invoke({"genre": "sci-fi", "topic": "time travel"})
# result has: genre, topic, outline, story, title
```

### Pattern 3: Parallel Chains

Run multiple chains concurrently. Use when steps are independent of each other.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()

# Three independent analyses of the same topic
pros_chain = (
    ChatPromptTemplate.from_template("List 3 advantages of {technology}. Be concise.")
    | llm | parser
)

cons_chain = (
    ChatPromptTemplate.from_template("List 3 disadvantages of {technology}. Be concise.")
    | llm | parser
)

alternatives_chain = (
    ChatPromptTemplate.from_template("List 3 alternatives to {technology} with one-line descriptions.")
    | llm | parser
)

# Run all three in parallel
analysis_chain = RunnableParallel(
    pros=pros_chain,
    cons=cons_chain,
    alternatives=alternatives_chain,
)

# Optionally, add a synthesis step
synthesis_prompt = ChatPromptTemplate.from_template(
    """Based on this analysis of {technology}:

PROS:
{pros}

CONS:
{cons}

ALTERNATIVES:
{alternatives}

Write a balanced 2-paragraph recommendation."""
)

full_chain = (
    RunnablePassthrough.assign(**{
        k: v for k, v in {
            "pros": pros_chain,
            "cons": cons_chain,
            "alternatives": alternatives_chain,
        }.items()
    })
    | RunnablePassthrough.assign(
        recommendation=synthesis_prompt | llm | parser
    )
)

result = full_chain.invoke({"technology": "Kubernetes"})
print("Pros:", result["pros"])
print("\nCons:", result["cons"])
print("\nAlternatives:", result["alternatives"])
print("\nRecommendation:", result["recommendation"])
```

### Pattern 4: Branching / Routing Chains

Route input to different chains based on conditions. Essential for building classifiers, triage systems, and dynamic pipelines.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

# Step 1: Classify the question
classifier_chain = (
    ChatPromptTemplate.from_template(
        """Classify this question into exactly one category: 'technical', 'creative', or 'factual'.
Respond with ONLY the category name, nothing else.

Question: {question}"""
    )
    | llm
    | parser
    | (lambda x: x.strip().lower())
)

# Step 2: Different chains for different categories
technical_chain = (
    ChatPromptTemplate.from_template(
        """You are a senior software engineer. Answer this technical question thoroughly with code examples.

Question: {question}"""
    )
    | llm | parser
)

creative_chain = (
    ChatPromptTemplate.from_template(
        """You are a creative writer. Answer this question with flair, metaphors, and imagination.

Question: {question}"""
    )
    | llm | parser
)

factual_chain = (
    ChatPromptTemplate.from_template(
        """You are an encyclopedia. Answer this factual question precisely and cite your reasoning.

Question: {question}"""
    )
    | llm | parser
)

# Step 3: Route based on classification
def route(info):
    """Route to the appropriate chain based on classification."""
    category = info["category"]
    if category == "technical":
        return technical_chain
    elif category == "creative":
        return creative_chain
    else:
        return factual_chain

# Full routing chain
routing_chain = (
    RunnablePassthrough.assign(category=classifier_chain)
    | RunnableLambda(lambda x: route(x).invoke(x))
)

# Test with different question types
questions = [
    "How do I implement a binary search tree in Python?",
    "Write me a poem about debugging at 3am",
    "What year was the transistor invented?",
]

for q in questions:
    print(f"\nQ: {q}")
    result = routing_chain.invoke({"question": q})
    print(f"A: {result[:200]}...")
```

Alternatively, use `RunnableBranch` for cleaner syntax:

```python
from langchain_core.runnables import RunnableBranch

# RunnableBranch takes (condition, runnable) pairs and a default
branch = RunnableBranch(
    (lambda x: x["category"] == "technical", technical_chain),
    (lambda x: x["category"] == "creative", creative_chain),
    factual_chain,  # Default fallback
)

routing_chain = (
    RunnablePassthrough.assign(category=classifier_chain)
    | branch
)
```

### Chain Composition Patterns Summary

```mermaid
graph TD
    subgraph Sequential["Sequential Chain"]
        S1[Step 1] --> S2[Step 2] --> S3[Step 3]
    end

    subgraph Parallel["Parallel Chain"]
        P0[Input] --> P1[Branch A]
        P0 --> P2[Branch B]
        P0 --> P3[Branch C]
        P1 --> P4[Collect Results]
        P2 --> P4
        P3 --> P4
    end

    subgraph Routing["Routing Chain"]
        R0[Input] --> R1{Classifier}
        R1 -->|Type A| R2[Chain A]
        R1 -->|Type B| R3[Chain B]
        R1 -->|Type C| R4[Chain C]
    end

    style Sequential fill:#1a1a2e,stroke:#e94560,color:#e0e0e0
    style Parallel fill:#16213e,stroke:#0f3460,color:#e0e0e0
    style Routing fill:#0f3460,stroke:#533483,color:#e0e0e0
```

---

## 9. Document Loaders

LLMs need data. Document loaders bring data into LangChain from dozens of sources  PDFs, web pages, CSVs, databases, APIs, and more. Every loader returns a list of `Document` objects.

### The Document Object

```python
from langchain_core.documents import Document

# A Document is just text + metadata
doc = Document(
    page_content="LangChain is a framework for building LLM applications.",
    metadata={
        "source": "langchain_docs.pdf",
        "page": 1,
        "author": "Harrison Chase",
    }
)

print(doc.page_content)  # The actual text
print(doc.metadata)      # Metadata (source, page, etc.)  useful for citations
```

### TextLoader

Load plain text files.

```python
from langchain_community.document_loaders import TextLoader

# Load a single text file
loader = TextLoader("data/notes.txt", encoding="utf-8")
docs = loader.load()

print(f"Loaded {len(docs)} document(s)")
print(f"Content preview: {docs[0].page_content[:200]}")
print(f"Metadata: {docs[0].metadata}")
# Metadata: {'source': 'data/notes.txt'}
```

### PyPDFLoader

Load PDFs  one document per page.

```python
from langchain_community.document_loaders import PyPDFLoader

# Load a PDF (one Document per page)
loader = PyPDFLoader("data/research_paper.pdf")
docs = loader.load()

print(f"Loaded {len(docs)} pages")
print(f"Page 1 preview: {docs[0].page_content[:200]}")
print(f"Metadata: {docs[0].metadata}")
# Metadata: {'source': 'data/research_paper.pdf', 'page': 0}

# Lazy loading for large PDFs (doesn't load everything into memory)
for doc in loader.lazy_load():
    process(doc)  # Process one page at a time
```

### WebBaseLoader

Load web pages. Uses BeautifulSoup under the hood.

```python
from langchain_community.document_loaders import WebBaseLoader

# Load a single web page
loader = WebBaseLoader("https://docs.python.org/3/tutorial/classes.html")
docs = loader.load()

print(f"Loaded {len(docs)} document(s)")
print(f"Content length: {len(docs[0].page_content)} characters")
print(f"Preview: {docs[0].page_content[:300]}")

# Load multiple pages at once
loader = WebBaseLoader([
    "https://docs.python.org/3/tutorial/classes.html",
    "https://docs.python.org/3/tutorial/modules.html",
    "https://docs.python.org/3/tutorial/errors.html",
])
docs = loader.load()
print(f"Loaded {len(docs)} documents from 3 URLs")
```

### CSVLoader

Load CSV files  one document per row.

```python
from langchain_community.document_loaders import CSVLoader

# Load a CSV (one Document per row)
loader = CSVLoader(
    file_path="data/products.csv",
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
    },
    # source_column="product_name",  # Optional: use a column as the 'source' metadata
)
docs = loader.load()

print(f"Loaded {len(docs)} rows as documents")
print(f"Row 1: {docs[0].page_content}")
# Row 1: product_name: Widget A
#         price: 29.99
#         category: Electronics
print(f"Metadata: {docs[0].metadata}")
# Metadata: {'source': 'data/products.csv', 'row': 0}
```

### DirectoryLoader

Load all files from a directory, with glob pattern matching.

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Load all .txt files from a directory
loader = DirectoryLoader(
    "data/documents/",
    glob="**/*.txt",         # Recursive glob pattern
    loader_cls=TextLoader,   # Which loader to use for each file
    show_progress=True,      # Show a progress bar
    use_multithreading=True, # Parallelize file loading
)
docs = loader.load()
print(f"Loaded {len(docs)} documents from directory")

# Load all PDFs from a directory
from langchain_community.document_loaders import PyPDFLoader

pdf_loader = DirectoryLoader(
    "data/pdfs/",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True,
)
pdf_docs = pdf_loader.load()
print(f"Loaded {len(pdf_docs)} pages from PDF directory")
```

### Document Loader Comparison

| Loader | Source | Docs Per | Metadata |
|--------|--------|----------|----------|
| `TextLoader` | .txt files | 1 per file | source |
| `PyPDFLoader` | .pdf files | 1 per page | source, page |
| `WebBaseLoader` | URLs | 1 per URL | source, title |
| `CSVLoader` | .csv files | 1 per row | source, row |
| `DirectoryLoader` | Directories | Varies | Depends on inner loader |
| `JSONLoader` | .json files | Configurable | source |
| `UnstructuredMarkdownLoader` | .md files | Configurable | source |
| `NotionDirectoryLoader` | Notion export | 1 per page | source |
| `GitLoader` | Git repos | 1 per file | source, file_path, file_type |

> **Pro tip:** Use `lazy_load()` instead of `load()` when dealing with large datasets. `load()` pulls everything into memory at once. `lazy_load()` yields documents one at a time, which is essential when you're loading thousands of PDFs or web pages.

---

## 10. Text Splitters

Documents are usually too long to fit in a single LLM context window, or too long to embed effectively. Text splitters break documents into smaller, semantically meaningful chunks.

### Why Chunking Matters

Bad chunking destroys information. If you split in the middle of a sentence, or split a code block from its explanation, the chunk becomes meaningless. Good chunking preserves semantic boundaries  paragraphs, sections, code blocks.

```mermaid
flowchart LR
    A["Raw Document<br/>(10,000 tokens)"] --> B["Text Splitter"]
    B --> C["Chunk 1<br/>(500 tokens)"]
    B --> D["Chunk 2<br/>(500 tokens)"]
    B --> E["Chunk 3<br/>(500 tokens)"]
    B --> F["..."]
    B --> G["Chunk N<br/>(500 tokens)"]

    C --> H["Overlap region<br/>(shared context)"]
    D --> H

    style A fill:#2d3436,stroke:#636e72,color:#dfe6e9
    style B fill:#e17055,stroke:#d63031,color:#ffffff
    style C fill:#0984e3,stroke:#74b9ff,color:#ffffff
    style D fill:#0984e3,stroke:#74b9ff,color:#ffffff
    style E fill:#0984e3,stroke:#74b9ff,color:#ffffff
    style G fill:#0984e3,stroke:#74b9ff,color:#ffffff
    style H fill:#fdcb6e,stroke:#f39c12,color:#2d3436
```

### RecursiveCharacterTextSplitter

The **default choice** for most use cases. It tries to split on the most semantically meaningful boundary:
1. First tries `\n\n` (paragraph breaks)
2. Then `\n` (line breaks)
3. Then `" "` (spaces)
4. Then individual characters

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Maximum characters per chunk
    chunk_overlap=200,      # Characters of overlap between consecutive chunks
    length_function=len,    # How to measure chunk size (len = character count)
    separators=["\n\n", "\n", " ", ""],  # Default separators (most to least preferred)
)

# Split a raw string
text = """
# Introduction to Python

Python is a high-level, interpreted programming language created by Guido van Rossum.
It emphasizes code readability and allows developers to express concepts in fewer lines
of code than languages like C++ or Java.

## Key Features

- Dynamic typing
- Garbage collection
- Multiple programming paradigms
- Extensive standard library

Python supports modules and packages, which encourages program modularity and code reuse.

## History

Python was conceived in the late 1980s by Guido van Rossum at Centrum Wiskunde &
Informatica (CWI) in the Netherlands. Its implementation began in December 1989.
"""

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i} ({len(chunk)} chars) ---")
    print(chunk[:100] + "...")
    print()

# Split LangChain Document objects (preserves metadata)
from langchain_core.documents import Document

docs = [Document(page_content=text, metadata={"source": "python_intro.md"})]
split_docs = splitter.split_documents(docs)

for doc in split_docs:
    print(f"Chunk ({len(doc.page_content)} chars): {doc.page_content[:80]}...")
    print(f"Metadata: {doc.metadata}")
    print()
```

### TokenTextSplitter

Splits based on token count instead of character count. Essential when you need precise control over how many tokens each chunk uses (for embedding models with token limits, for example).

```python
from langchain.text_splitter import TokenTextSplitter

# Split by token count (uses tiktoken for OpenAI tokenization)
splitter = TokenTextSplitter(
    chunk_size=256,          # Maximum tokens per chunk
    chunk_overlap=50,        # Token overlap between chunks
    encoding_name="cl100k_base",  # Tokenizer (GPT-4, GPT-3.5-turbo)
)

text = "Your long document text here..." * 100
chunks = splitter.split_text(text)
print(f"Split into {len(chunks)} chunks")
for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i}: {len(chunk)} chars")
```

### MarkdownHeaderTextSplitter

Splits Markdown documents by headers, preserving the document structure in metadata. Perfect for documentation, README files, and knowledge bases.

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

markdown_text = """# Machine Learning Guide

## Supervised Learning

### Classification
Classification is the task of predicting a discrete class label.
Common algorithms include logistic regression, decision trees, and SVMs.

### Regression
Regression predicts continuous values.
Linear regression and polynomial regression are common examples.

## Unsupervised Learning

### Clustering
Clustering groups similar data points together.
K-means and DBSCAN are popular clustering algorithms.

### Dimensionality Reduction
PCA and t-SNE are used to reduce the number of features.
"""

# Define which headers to split on
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(markdown_text)

for chunk in chunks:
    print(f"Content: {chunk.page_content[:80]}...")
    print(f"Metadata: {chunk.metadata}")
    print()
# Content: Classification is the task of predicting a discrete class label...
# Metadata: {'Header 1': 'Machine Learning Guide', 'Header 2': 'Supervised Learning', 'Header 3': 'Classification'}
```

### Chunking Strategy Comparison

| Splitter | Splits On | Best For | Token-Aware? |
|----------|-----------|----------|--------------|
| `RecursiveCharacterTextSplitter` | Characters with smart separators | General text, default choice | No (character-based) |
| `TokenTextSplitter` | Token count | Precise token budgets | Yes |
| `MarkdownHeaderTextSplitter` | Markdown headers | Documentation, structured text | No |
| `HTMLHeaderTextSplitter` | HTML headers | Web content | No |
| `PythonCodeTextSplitter` | Python syntax | Source code | No |
| `LatexTextSplitter` | LaTeX commands | Academic papers | No |

> **Chunking rules of thumb:**
> - **chunk_size=500-1000** characters for embedding + retrieval (smaller chunks = more precise retrieval)
> - **chunk_size=2000-4000** characters for summarization (larger chunks = more context per summary)
> - **chunk_overlap=10-20%** of chunk_size (enough to maintain context across boundaries)
> - Always use `split_documents()` when working with loaded documents (preserves metadata)

---

## 11. Embeddings

Embeddings convert text into dense numerical vectors that capture semantic meaning. Two sentences with similar meanings will have similar embedding vectors, even if they use completely different words. This is the foundation of semantic search and RAG.

### The Embeddings Interface

All embedding models in LangChain implement the same interface:

```python
from langchain_core.embeddings import Embeddings

# All embedding models support:
# .embed_documents(texts: List[str]) -> List[List[float]]  # Embed multiple texts
# .embed_query(text: str) -> List[float]                    # Embed a single query
```

**Why two methods?** Some providers use different models or parameters for documents vs. queries. For example, some models prepend "query: " or "passage: " to the text depending on whether it's a query or a document. Using the right method ensures optimal retrieval performance.

### OpenAIEmbeddings

The most popular choice. Uses OpenAI's embedding models.

```python
from langchain_openai import OpenAIEmbeddings

# Initialize (reads OPENAI_API_KEY from environment)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # Cheaper, good for most use cases
    # model="text-embedding-3-large",  # Higher quality, more expensive
    # dimensions=256,  # Optional: reduce dimensions for text-embedding-3-* models
)

# Embed documents (for indexing)
texts = [
    "Python is a programming language",
    "JavaScript runs in browsers",
    "LangChain is an LLM framework",
]
doc_vectors = embeddings.embed_documents(texts)

print(f"Number of vectors: {len(doc_vectors)}")
print(f"Vector dimension: {len(doc_vectors[0])}")
print(f"First 5 values: {doc_vectors[0][:5]}")
# Number of vectors: 3
# Vector dimension: 1536
# First 5 values: [0.0023, -0.0145, 0.0067, ...]

# Embed a query (for searching)
query_vector = embeddings.embed_query("What programming languages are popular?")
print(f"Query vector dimension: {len(query_vector)}")

# Compute similarity manually (cosine similarity)
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

for i, text in enumerate(texts):
    sim = cosine_similarity(query_vector, doc_vectors[i])
    print(f"Similarity to '{text}': {sim:.4f}")
# Similarity to 'Python is a programming language': 0.8234
# Similarity to 'JavaScript runs in browsers': 0.7891
# Similarity to 'LangChain is an LLM framework': 0.6543
```

### HuggingFaceEmbeddings

Run embedding models locally. No API costs, no data leaving your machine.

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize with a model from HuggingFace
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Fast, good quality
    # model_name="BAAI/bge-large-en-v1.5",  # Higher quality, slower
    model_kwargs={"device": "cpu"},  # or "cuda" for GPU
    encode_kwargs={"normalize_embeddings": True},  # L2 normalize for cosine similarity
)

# Same interface as OpenAI
texts = ["Python is great", "I love coding", "The weather is nice"]
vectors = embeddings.embed_documents(texts)
print(f"Dimension: {len(vectors[0])}")  # 384 for all-MiniLM-L6-v2
```

### CacheBackedEmbeddings

Embedding API calls cost money and take time. Caching avoids re-embedding the same text.

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.storage import LocalFileStore

# Create a file-based cache
underlying_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
store = LocalFileStore("./embedding_cache/")

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace=underlying_embeddings.model,  # Namespace prevents collisions
)

# First call: hits the API and caches
texts = ["Hello world", "LangChain is awesome"]
vectors = cached_embeddings.embed_documents(texts)

# Second call: instant (reads from cache)
vectors_again = cached_embeddings.embed_documents(texts)

# Only new texts hit the API
vectors_mixed = cached_embeddings.embed_documents([
    "Hello world",          # Cached  no API call
    "This is a new text",   # Not cached  API call
])
```

### Embedding Model Comparison

| Model | Provider | Dimensions | Speed | Cost | Quality |
|-------|----------|-----------|-------|------|---------|
| `text-embedding-3-small` | OpenAI | 1536 | Fast | $0.02/1M tokens | Good |
| `text-embedding-3-large` | OpenAI | 3072 | Fast | $0.13/1M tokens | Best (OpenAI) |
| `all-MiniLM-L6-v2` | HuggingFace (local) | 384 | Very fast | Free | Good |
| `bge-large-en-v1.5` | HuggingFace (local) | 1024 | Moderate | Free | Very good |
| `voyage-3` | Voyage AI | 1024 | Fast | $0.06/1M tokens | Excellent |
| `embed-english-v3.0` | Cohere | 1024 | Fast | Pay per use | Very good |

---

## 12. Vector Stores

Vector stores are databases optimized for storing and searching embedding vectors. They're the backbone of RAG  you embed your documents, store them in a vector store, then search for relevant documents at query time.

### The Vector Store Interface

```python
from langchain_core.vectorstores import VectorStore

# All vector stores support:
# .from_documents(docs, embedding) -> VectorStore       # Create from documents
# .from_texts(texts, embedding, metadatas) -> VectorStore # Create from raw texts
# .add_documents(docs) -> List[str]                      # Add more documents
# .similarity_search(query, k=4) -> List[Document]       # Search by text
# .similarity_search_with_score(query, k=4) -> List[Tuple[Document, float]]
# .as_retriever(**kwargs) -> VectorStoreRetriever        # Convert to a Retriever
# .delete(ids) -> None                                    # Delete by ID
```

### FAISS (Local, In-Memory)

Facebook AI Similarity Search. Fast, local, no server required. Great for development and small-to-medium datasets.

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create from documents
docs = [
    Document(page_content="Python was created by Guido van Rossum", metadata={"source": "python.md"}),
    Document(page_content="JavaScript was created by Brendan Eich", metadata={"source": "js.md"}),
    Document(page_content="Rust was created by Graydon Hoare at Mozilla", metadata={"source": "rust.md"}),
    Document(page_content="Go was created at Google by Robert Griesemer, Rob Pike, and Ken Thompson", metadata={"source": "go.md"}),
    Document(page_content="LangChain is a framework for building LLM applications", metadata={"source": "langchain.md"}),
]

vectorstore = FAISS.from_documents(docs, embeddings)

# Similarity search
results = vectorstore.similarity_search("Who created Python?", k=2)
for doc in results:
    print(f"[{doc.metadata['source']}] {doc.page_content}")
# [python.md] Python was created by Guido van Rossum
# [go.md] Go was created at Google by Robert Griesemer...

# Search with similarity scores
results_with_scores = vectorstore.similarity_search_with_score("LLM framework", k=2)
for doc, score in results_with_scores:
    print(f"Score: {score:.4f} | {doc.page_content}")
# Score: 0.2341 | LangChain is a framework for building LLM applications
# Score: 0.8912 | Python was created by Guido van Rossum

# Save to disk (persist across sessions)
vectorstore.save_local("./faiss_index")

# Load from disk
loaded_vectorstore = FAISS.load_local(
    "./faiss_index",
    embeddings,
    allow_dangerous_deserialization=True  # Required for pickle-based loading
)

# Add more documents later
new_docs = [Document(page_content="TypeScript is a superset of JavaScript", metadata={"source": "ts.md"})]
vectorstore.add_documents(new_docs)
```

### Chroma (Local, Persistent)

An open-source embedding database with built-in persistence. More features than FAISS (metadata filtering, collections).

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

docs = [
    Document(page_content="React is a JavaScript library for building UIs", metadata={"category": "frontend", "year": 2013}),
    Document(page_content="Django is a Python web framework", metadata={"category": "backend", "year": 2005}),
    Document(page_content="FastAPI is a modern Python web framework", metadata={"category": "backend", "year": 2018}),
    Document(page_content="Vue.js is a progressive JavaScript framework", metadata={"category": "frontend", "year": 2014}),
    Document(page_content="Flask is a lightweight Python web framework", metadata={"category": "backend", "year": 2010}),
]

# Create with persistence
vectorstore = Chroma.from_documents(
    docs,
    embeddings,
    collection_name="web_frameworks",
    persist_directory="./chroma_db",  # Data persists here
)

# Basic similarity search
results = vectorstore.similarity_search("Python web development", k=3)
for doc in results:
    print(f"[{doc.metadata['category']}] {doc.page_content}")

# Metadata filtering  powerful feature of Chroma
results = vectorstore.similarity_search(
    "web framework",
    k=3,
    filter={"category": "backend"},  # Only search backend frameworks
)
for doc in results:
    print(f"[{doc.metadata['year']}] {doc.page_content}")
# All results will be backend frameworks

# Load existing collection (persistence)
loaded_vectorstore = Chroma(
    collection_name="web_frameworks",
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
```

### Using as_retriever()

The `.as_retriever()` method converts a vector store into a LangChain `Retriever`, which plugs directly into LCEL chains.

```python
# Convert to a retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",       # "similarity", "mmr", or "similarity_score_threshold"
    search_kwargs={
        "k": 4,                     # Number of documents to retrieve
        # "score_threshold": 0.5,   # For similarity_score_threshold
        # "fetch_k": 20,            # For MMR: fetch this many, then diversify to k
        # "lambda_mult": 0.5,       # For MMR: 0 = max diversity, 1 = max relevance
    },
)

# Use in an LCEL chain
docs = retriever.invoke("Python web frameworks")
for doc in docs:
    print(doc.page_content)

# MMR (Maximum Marginal Relevance)  balances relevance and diversity
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5},
)
```

### Vector Store Comparison

| Store | Type | Persistence | Metadata Filter | Best For |
|-------|------|-------------|-----------------|----------|
| **FAISS** | Local, in-memory | Save/load to disk | No | Development, small datasets, fast prototyping |
| **Chroma** | Local, embedded | Built-in persistence | Yes | Small-to-medium production, metadata-heavy |
| **Pinecone** | Cloud-hosted | Fully managed | Yes | Production at scale, managed infrastructure |
| **Qdrant** | Self-hosted or cloud | Built-in persistence | Yes (advanced) | Production, complex filtering needs |
| **Weaviate** | Self-hosted or cloud | Built-in persistence | Yes | Production, hybrid search (vector + keyword) |
| **pgvector** | PostgreSQL extension | PostgreSQL | Yes (SQL) | Teams already using PostgreSQL |

---

## 13. Building a RAG Chain with LCEL

This is where everything comes together. We'll build a complete Retrieval-Augmented Generation pipeline that:
1. Loads documents
2. Splits them into chunks
3. Embeds and stores them in a vector store
4. Retrieves relevant chunks for a query
5. Generates an answer with source citations

### The RAG Pipeline

```mermaid
flowchart LR
    subgraph Indexing["Indexing Pipeline (run once)"]
        A["Documents"] --> B["Text Splitter"]
        B --> C["Chunks"]
        C --> D["Embedding Model"]
        D --> E["Vector Store"]
    end

    subgraph Query["Query Pipeline (run per question)"]
        F["User Question"] --> G["Embedding Model"]
        G --> H["Vector Store Search"]
        H --> I["Relevant Chunks"]
        I --> J["Prompt Template"]
        F --> J
        J --> K["LLM"]
        K --> L["Answer with Citations"]
    end

    E -.->|"stored vectors"| H

    style Indexing fill:#1a1a2e,stroke:#e94560,color:#e0e0e0
    style Query fill:#16213e,stroke:#0f3460,color:#e0e0e0
```

### Complete RAG Pipeline

```python
"""
Complete RAG pipeline with LangChain and LCEL.
Load documents, split, embed, store, retrieve, and generate answers.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Step 1: Load Documents ---
from langchain_community.document_loaders import WebBaseLoader

# Load some web pages as our knowledge base
urls = [
    "https://python.langchain.com/docs/concepts/",
    "https://python.langchain.com/docs/concepts/lcel/",
    "https://python.langchain.com/docs/concepts/chat_models/",
]
loader = WebBaseLoader(urls)
raw_docs = loader.load()
print(f"Loaded {len(raw_docs)} documents")

# --- Step 2: Split Documents ---
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
split_docs = splitter.split_documents(raw_docs)
print(f"Split into {len(split_docs)} chunks")

# --- Step 3: Create Embeddings and Vector Store ---
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(split_docs, embeddings)
print(f"Stored {len(split_docs)} chunks in FAISS")

# --- Step 4: Create Retriever ---
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)

# --- Step 5: Build RAG Chain with LCEL ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def format_docs(docs):
    """Format retrieved documents into a single string with source references."""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Source {i+1}: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

prompt = ChatPromptTemplate.from_template("""You are a helpful technical assistant.
Answer the question based ONLY on the following context.
If the context doesn't contain the answer, say "I don't have enough information."
Always cite which source(s) you used by referencing [Source N].

Context:
{context}

Question: {question}

Answer:""")

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# The RAG chain in LCEL
rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --- Step 6: Query the RAG Pipeline ---
question = "What is LCEL and how does the pipe operator work?"
answer = rag_chain.invoke(question)
print(f"\nQuestion: {question}")
print(f"\nAnswer: {answer}")
```

### RAG Chain with Source Tracking

Often you need to return both the answer AND the source documents. Here's how to track sources through the chain:

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel

def format_docs_with_sources(docs):
    """Format docs and also return source metadata."""
    formatted_text = []
    sources = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        sources.append({"source": source, "chunk": doc.page_content[:100]})
        formatted_text.append(f"[{i+1}] {doc.page_content}")
    return {
        "formatted_context": "\n\n".join(formatted_text),
        "sources": sources,
    }

# Chain that returns answer + sources
rag_chain_with_sources = (
    # Step 1: Retrieve and format
    RunnablePassthrough.assign(
        retrieval_result=lambda x: format_docs_with_sources(
            retriever.invoke(x["question"])
        )
    )
    # Step 2: Extract formatted context
    | RunnablePassthrough.assign(
        context=lambda x: x["retrieval_result"]["formatted_context"],
        sources=lambda x: x["retrieval_result"]["sources"],
    )
    # Step 3: Generate answer
    | RunnablePassthrough.assign(
        answer=prompt | llm | StrOutputParser()
    )
    # Step 4: Return only what we need
    | RunnableLambda(lambda x: {
        "question": x["question"],
        "answer": x["answer"],
        "sources": x["sources"],
    })
)

result = rag_chain_with_sources.invoke({"question": "What is LCEL?"})
print(f"Answer: {result['answer']}")
print(f"\nSources used:")
for src in result["sources"]:
    print(f"  - {src['source']}: {src['chunk']}...")
```

### RAG with Streaming

For user-facing applications, stream the RAG response:

```python
# Stream the answer while still using RAG
for chunk in rag_chain.stream("What are the core abstractions in LangChain?"):
    print(chunk, end="", flush=True)
print()
```

> **RAG best practices:**
> - **chunk_size** of 500-1000 characters works well for most Q&A use cases
> - **chunk_overlap** of 100-200 characters prevents losing context at boundaries
> - **k=3-5** retrieved documents is usually sufficient; more can dilute relevance
> - Always include **source citations**  they improve user trust AND help you debug retrieval quality
> - Use **MMR retrieval** (`search_type="mmr"`) to get diverse results when documents are repetitive
> - **Test retrieval separately** from generation  if retrieval returns bad documents, no amount of prompt engineering will fix the answer

---

## 14. Error Handling and Fallbacks

Production LLM applications fail. APIs go down. Rate limits hit. Models hallucinate. LangChain provides built-in mechanisms for resilience.

### .with_fallbacks()

Define backup chains that execute when the primary chain fails.

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Primary model: GPT-4o
primary_llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Fallback model: Claude (different provider = different failure domain)
fallback_llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

# Second fallback: cheaper model (less likely to hit rate limits)
second_fallback_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Chain with fallbacks: tries primary, then fallback, then second fallback
resilient_llm = primary_llm.with_fallbacks([fallback_llm, second_fallback_llm])

prompt = ChatPromptTemplate.from_template("Explain {concept} in one paragraph.")
chain = prompt | resilient_llm | StrOutputParser()

# If GPT-4o fails (rate limit, downtime), automatically tries Claude, then GPT-4o-mini
result = chain.invoke({"concept": "quantum computing"})
print(result)
```

### .with_retry()

Automatically retry on transient failures (rate limits, network errors).

```python
from langchain_openai import ChatOpenAI

# Retry configuration
llm = ChatOpenAI(model="gpt-4o", temperature=0).with_retry(
    stop_after_attempt=3,        # Maximum number of attempts
    wait_exponential_jitter=True, # Exponential backoff with jitter
)

# This will retry up to 3 times with exponential backoff if the API returns
# a transient error (429 rate limit, 500 server error, network timeout)
```

### Combining Fallbacks and Retries

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Primary: GPT-4o with retries
primary = ChatOpenAI(model="gpt-4o").with_retry(
    stop_after_attempt=2,
    wait_exponential_jitter=True,
)

# Fallback: Claude with retries
fallback = ChatAnthropic(model="claude-sonnet-4-20250514").with_retry(
    stop_after_attempt=2,
    wait_exponential_jitter=True,
)

# Combined: retry primary twice, then retry fallback twice
resilient_llm = primary.with_fallbacks([fallback])
```

### Custom Error Handling in LCEL

For more control, wrap error handling in a `RunnableLambda`:

```python
from langchain_core.runnables import RunnableLambda
import logging

logger = logging.getLogger(__name__)

def safe_invoke(chain):
    """Wrap a chain with custom error handling."""
    def _invoke(input_data):
        try:
            return chain.invoke(input_data)
        except Exception as e:
            logger.error(f"Chain failed: {e}")
            return {
                "error": True,
                "message": f"I encountered an error: {str(e)}. Please try again.",
                "original_input": input_data,
            }
    return RunnableLambda(_invoke)

# Usage
safe_chain = safe_invoke(rag_chain)
result = safe_chain.invoke("What is LCEL?")
```

### Error Handling Decision Tree

| Error Type | Strategy | LangChain Tool |
|-----------|----------|----------------|
| Rate limit (429) | Retry with backoff | `.with_retry()` |
| Provider outage | Switch provider | `.with_fallbacks()` |
| Parsing error | Retry with clearer prompt, or fallback parser | `.with_fallbacks()` on parser |
| Context too long | Truncate, summarize, or use a model with larger context | Pre-processing in `RunnableLambda` |
| Invalid API key | Fail fast with clear error | Environment validation at startup |
| Network timeout | Retry, then fallback | `.with_retry()` + `.with_fallbacks()` |

---

## 15. Comparison: LangChain vs LlamaIndex vs Raw Code

This is the question every developer asks: "Do I even need a framework?" Here's an honest comparison.

### Feature Comparison

| Feature | LangChain | LlamaIndex | Raw SDK Code |
|---------|-----------|------------|--------------|
| **Primary focus** | Composable LLM pipelines | Data indexing and retrieval | Maximum control |
| **Composition model** | LCEL (pipe operator) | Query engines, indices | Manual function calls |
| **Chain building** | Excellent (LCEL) | Good (query pipelines) | Manual (custom code) |
| **RAG** | Very good | Excellent (purpose-built) | Manual (most work) |
| **Agent support** | Excellent (LangGraph) | Good (agents) | Manual (most work) |
| **Tool calling** | Excellent | Good | Moderate (SDK support) |
| **Memory** | Good (LangGraph) | Good | Manual |
| **Observability** | LangSmith (excellent) | LlamaTrace | Custom logging |
| **Deployment** | LangServe | LlamaDeploy | FastAPI / custom |
| **Document loaders** | 160+ integrations | 160+ integrations | Manual per source |
| **Vector stores** | 70+ integrations | 40+ integrations | Direct SDK calls |
| **Learning curve** | Moderate-steep | Moderate | Low (but high total effort) |
| **Abstraction level** | High | High | None |
| **Debugging** | Moderate (LangSmith helps) | Moderate | Easy (your code) |
| **Community** | Very large | Large | N/A |
| **When to choose** | Multi-step LLM pipelines, agents, diverse integrations | Data-heavy RAG applications, knowledge bases | Simple apps, maximum control, minimal dependencies |

### Decision Framework

Choose **LangChain** when:
- You're building multi-step pipelines (chain multiple LLM calls)
- You need agents with tool calling
- You want to swap LLM providers easily
- You're building complex RAG with routing, branching, or multi-query
- You need LangSmith for production observability
- You need LangGraph for stateful agents

Choose **LlamaIndex** when:
- Your primary use case is RAG over your data
- You need advanced retrieval strategies (auto-merge, sentence window, knowledge graphs)
- You're building a "chat with your data" application
- You need sophisticated document parsing and chunking

Choose **Raw SDK code** when:
- You're making simple, single-step LLM calls
- You need maximum control over every HTTP request
- You want zero abstraction overhead
- Your application is small and won't grow
- You need the absolute latest provider features immediately

> **Hot take:** In practice, many production applications use **LangChain for orchestration** and **LlamaIndex for indexing** together. They're not mutually exclusive. LlamaIndex has a LangChain integration that lets you use LlamaIndex retrievers inside LangChain chains.

---

## 16. Key Vocabulary

| Term | Definition |
|------|-----------|
| **LCEL** | LangChain Expression Language  the declarative composition system using the pipe (`\|`) operator |
| **Runnable** | Any object that implements the LangChain Runnable interface (invoke, batch, stream, etc.) |
| **Chain** | A sequence of Runnables composed together; the fundamental building block of LangChain applications |
| **Prompt Template** | A parameterized template that produces formatted messages for an LLM |
| **Chat Model** | An LLM that takes messages (SystemMessage, HumanMessage, etc.) and produces an AIMessage |
| **Output Parser** | Transforms raw LLM output (string) into structured data (dict, Pydantic model, list, etc.) |
| **Document** | A LangChain object containing `page_content` (text) and `metadata` (dict) |
| **Document Loader** | Loads data from a source (file, URL, database) into LangChain Document objects |
| **Text Splitter** | Breaks documents into smaller chunks for embedding and retrieval |
| **Embedding** | A dense numerical vector that represents the semantic meaning of a piece of text |
| **Vector Store** | A database optimized for storing and searching embedding vectors |
| **Retriever** | An object that takes a query and returns relevant documents; the search component of RAG |
| **RAG** | Retrieval-Augmented Generation  augmenting LLM generation with retrieved external knowledge |
| **RunnablePassthrough** | An LCEL component that passes input through unchanged (used for forwarding data) |
| **RunnableLambda** | An LCEL wrapper that turns any Python function into a composable Runnable |
| **RunnableParallel** | An LCEL component that runs multiple Runnables concurrently and collects their outputs |
| **RunnableBranch** | An LCEL component that routes input to different Runnables based on conditions |
| **Fallback** | A backup Runnable that executes when the primary one fails |
| **LangSmith** | LangChain's observability platform for tracing, debugging, and evaluating LLM applications |
| **LangServe** | LangChain's deployment tool that exposes LCEL chains as REST APIs |
| **LangGraph** | LangChain's framework for building stateful, multi-actor agent applications with cycles and persistence |
| **MMR** | Maximum Marginal Relevance  a retrieval strategy that balances relevance with diversity |
| **Chunk Overlap** | The number of characters/tokens shared between consecutive text chunks to preserve context |
| **Few-Shot Prompting** | Providing example input-output pairs in the prompt to guide the model's behavior and format |

---

## 17. What's Next

In **Part 1: Agents, Tools, and Memory**, we'll go from static chains to dynamic, autonomous systems. Here's what we'll build:

- **Tools and Tool Calling**  giving LLMs access to calculators, web search, databases, code execution, and custom APIs
- **The Agent Loop**  the ReAct pattern: Reason → Act → Observe → Repeat
- **AgentExecutor**  LangChain's agent runtime with built-in error handling and iteration limits
- **LangGraph Agents**  the modern approach: building agents as state machines with explicit control flow
- **Memory Systems**  ConversationBufferMemory, ConversationSummaryMemory, and LangGraph's built-in persistence
- **Multi-Agent Architectures**  building systems where multiple specialized agents collaborate
- **A complete project:** An AI research assistant that can search the web, read documents, write code, and maintain conversation history

If Part 0 taught you how to build pipelines, Part 1 will teach you how to build systems that think.

---

*Part 0 complete. Continue to [Part 1: Agents, Tools, and Memory](./langchain-deep-dive-part-1.md).*
