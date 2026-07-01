<div align="center">

# ✨ Auralis — RAG-Powered Document Chatbot with MCP Tools ✨

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-FF6B6B?style=for-the-badge&logoColor=white" />
  <img src="https://img.shields.io/badge/multi--qa--MiniLM--L6-Embeddings-FF8C00?style=for-the-badge&logoColor=white" />
  <img src="https://img.shields.io/badge/Cerebras-gpt--oss--120b-8A2BE2?style=for-the-badge&logoColor=white" />
  <img src="https://img.shields.io/badge/MCP-Tools-0EA5E9?style=for-the-badge&logoColor=white" />
  <img src="https://img.shields.io/badge/SSE-Streaming-22C55E?style=for-the-badge&logoColor=white" />
</p>

<p>
  <strong>Ask questions across your PDFs — with dual retrieval routing, MCP tool integration, citations, streaming answers, and faithfulness verification.</strong>
</p>

<p>
  <img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square" />
</p>

</div>

---

## 🌈 Features

| | Feature | What it does |
|---|---|---|
| 🔍 | **Hybrid Retrieval** | Dense (MiniLM vectors) + Sparse (BM25 keywords), merged with Reciprocal Rank Fusion |
| 🧠 | **Multi-Query Expansion** | LLM generates multiple search variants per question to catch more relevant chunks |
| 📊 | **Cross-Encoder Reranking** | `ms-marco-MiniLM-L-6-v2` re-scores the top candidates by reading query+passage together |
| 🛠️ | **MCP Tool Integration** | Standalone MCP server exposes calculator, weather, and document retrieval as tools |
| 🔀 | **Dual Retrieval Routing** | LLM decides: deep document analysis → RAG pipeline, quick lookup → MCP retrieve tool |
| 💬 | **Multi-Turn Memory** | Conversation history with automatic query contextualization (follow-up rewriting) |
| 📎 | **Source Citations** | Every answer cites numbered passages `[1]`, `[2]` with source filenames |
| ✅ | **Faithfulness Check** | Second LLM pass verifies the answer is grounded in retrieved documents |
| ⚡ | **SSE Streaming** | Tokens stream live to the browser — sources appear first, then the answer |
| 💾 | **Browser Persistence** | Chat history saved in `localStorage` (up to 50 chats, zero backend required) |

---

## 🏗️ Architecture

```
                         ┌──────────────────────────────────────────────────┐
                         │                   server.py                      │
  User Message ────────► │                                                  │
                         │   RAG Check ──YES──► Condense ──► Retrieve       │
                         │       │               (deep doc questions)       │
                         │      NO               grounding + faithfulness   │
                         │       │                                          │
                         │       ▼                                          │
                         │   llm_with_tools (MCP tool binding)              │
                         │       │                                          │
                         │       ├── calculator  ──► mcp_server.py          │
                         │       ├── get_weather ──► mcp_server.py          │
                         │       ├── retrieve_documents ──► mcp_server.py   │
                         │       └── no tool ──► direct LLM reply           │
                         └──────────────────────────────────────────────────┘
                                          │
                              SSE stream ──► Browser
```

---

## 🛠️ MCP Tools

`mcp_server.py` runs as a **standalone subprocess** and exposes three tools via the MCP protocol. `server.py` connects to it at startup using `MultiServerMCPClient` and binds its tools to the LLM.

| Tool | Description |
|---|---|
| `calculator` | Evaluates any math or arithmetic expression |
| `get_weather` | Returns dummy weather data for a given city |
| `retrieve_documents` | Quick factual lookups from uploaded documents |

The LLM reads tool descriptions and decides which to call (or none) based on the user's query.

---

## 🔀 Routing Logic

Every query goes through two decision steps:

1. **RAG Check** — Is this a deep document question (summarize, compare, multi-part)? → Full RAG pipeline with grounding and faithfulness check
2. **Tool Binding** — Otherwise, `llm_with_tools` picks the right MCP tool, or answers directly

Terminal logs show which path fired:
```
[ROUTER] → RAG             | query: 'summarize the paper'
[ROUTER] → MCP tool: calculator  | args: {'expression': '5*9'}
[ROUTER] → MCP tool: retrieve_documents  | args: {'query': 'who wrote the paper'}
[ROUTER] → DIRECT          | query: 'hello how are you'
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/auralis.git
cd auralis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> The first run downloads `multi-qa-MiniLM-L6-cos-v1` (~90 MB) and `ms-marco-MiniLM-L-6-v2` from HuggingFace automatically.

### 3. Set up your API key

```bash
cp .env.example .env
```

Open `.env` and replace `your_key_here` with your Cerebras key. Get a free key at [cloud.cerebras.ai](https://cloud.cerebras.ai).

### 4. Add your PDFs

```bash
mkdir pdfs
# Copy your .pdf files into pdfs/
```

### 5. Build the index

```bash
python src/ingest.py
```

This embeds all PDFs and stores them in ChromaDB. Re-run whenever you add or change PDFs — existing data is cleared automatically (no duplicates).

### 6. Start the server

```bash
python src/server.py
```

Server starts at `http://localhost:8000`. The MCP server subprocess starts automatically.

### 7. Open the UI

Open `frontend/index.html` directly in your browser. No web server needed.

---

## ⚙️ Configuration

| Variable | File | Default | Description |
|---|---|---|---|
| `CEREBRAS_API_KEY` | `.env` | *(required)* | API key for the LLM |
| `CHUNK_SIZE` | `src/ingest.py` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `src/ingest.py` | `120` | Overlap between adjacent chunks |
| `FETCH_K` | `src/retrieval.py` | `15` | Candidates fetched per retriever |
| `TOP_K` | `src/retrieval.py` | `5` | Passages kept after reranking |

---

## 📁 Project Structure

```
auralis/
├── pdfs/                    ← drop your PDFs here
├── chroma_db/               ← auto-generated vector store (git-ignored)
├── chunks.pkl               ← auto-generated BM25 source (git-ignored)
├── frontend/
│   └── index.html           ← open this in your browser
├── src/
│   ├── embeddings.py        ← multi-qa-MiniLM-L6-cos-v1 LangChain wrapper
│   ├── ingest.py            ← PDF → chunks → Chroma + pickle
│   ├── retrieval.py         ← hybrid retriever pipeline
│   ├── grounding.py         ← prompt, context formatting, faithfulness check
│   ├── mcp_server.py        ← standalone MCP server (calculator, weather, retrieval)
│   └── server.py            ← FastAPI SSE endpoint with dual routing
├── .env                     ← your secrets (git-ignored)
├── .env.example             ← template to copy from
└── requirements.txt
```

---

## 💡 How It Works

1. **Ingest** — PDFs are split into 800-char overlapping chunks, embedded with `multi-qa-MiniLM-L6-cos-v1`, and stored in ChromaDB. The raw text is also pickled for BM25.

2. **MCP Startup** — When the server starts, it spawns `mcp_server.py` as a subprocess and connects to it via the MCP stdio protocol, loading its tools and binding them to the LLM.

3. **Route** — Every query goes through a RAG check first. Deep analysis questions go to the full RAG pipeline. Everything else goes to the tool-binding path where the LLM decides which MCP tool to call (or none).

4. **RAG Path** — Follow-up questions are condensed into standalone queries. Each query spawns multiple LLM-generated variants (multi-query), hits both Chroma (dense) and BM25 (sparse), merges with Reciprocal Rank Fusion, and reranks to the top 5 with a cross-encoder.

5. **MCP Tool Path** — The LLM reads tool descriptions and calls the appropriate MCP tool. The tool runs in the `mcp_server.py` subprocess and returns the result. The LLM then forms a natural answer from the result.

6. **Ground** — On the RAG path, the 5 passages are numbered and injected into the prompt. The LLM answers only from these passages and cites `[1]`, `[2]`, etc.

7. **Verify** — A second LLM call checks if the RAG answer is supported by the context. The UI shows a **Verified** or **Unverified** badge.

8. **Stream** — Tokens stream back via Server-Sent Events. Sources appear first, then the answer streams in, then the verification badge.

---

## 🔒 Security

- `.env`, `chroma_db/`, `chunks.pkl`, and `pdfs/` are all in `.gitignore` — your API key and documents stay local.
- The frontend uses `textContent` during streaming (no XSS) and only switches to `innerHTML` for markdown rendering after the stream completes.

---

## 📜 License

MIT — feel free to use, modify, and share.
