<div align="center">

# ✨ Auralis — RAG-Powered Document Chatbot ✨

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-FF6B6B?style=for-the-badge&logoColor=white" />
  <img src="https://img.shields.io/badge/Qwen3-Embeddings-FF8C00?style=for-the-badge&logoColor=white" />
  <img src="https://img.shields.io/badge/Cerebras-gpt--oss--120b-8A2BE2?style=for-the-badge&logoColor=white" />
  <img src="https://img.shields.io/badge/SSE-Streaming-22C55E?style=for-the-badge&logoColor=white" />
</p>

<p>
  <strong>Ask questions across your PDFs — with citations, streaming answers, faithfulness verification, and multi-turn memory.</strong>
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
| 🔍 | **Hybrid Retrieval** | Dense (Qwen3 vectors) + Sparse (BM25 keywords), merged with Reciprocal Rank Fusion |
| 🧠 | **Multi-Query Expansion** | LLM generates multiple search variants per question to catch more relevant chunks |
| 📊 | **Cross-Encoder Reranking** | `ms-marco-MiniLM-L-6-v2` re-scores the top candidates by reading query+passage together |
| 💬 | **Multi-Turn Memory** | Conversation history with automatic query contextualization (follow-up rewriting) |
| 📎 | **Source Citations** | Every answer cites numbered passages `[1]`, `[2]` with source filenames |
| ✅ | **Faithfulness Check** | Second LLM pass verifies the answer is grounded in retrieved documents |
| ⚡ | **SSE Streaming** | Tokens stream live to the browser — sources appear first, then the answer |
| 💾 | **Browser Persistence** | Chat history saved in `localStorage` (up to 50 chats, zero backend required) |
| 🤖 | **Casual Chat** | Greetings and small talk answered naturally — no "document not found" for "hi" |

---

## 🏗️ Architecture

```
                         ┌─────────────────────────────────────────┐
                         │              server.py                  │
  User Message ────────► │                                         │
                         │   Is casual? ──YES──► LLM direct reply  │
                         │       │                                  │
                         │      NO                                  │
                         │       │                                  │
                         │  Condense follow-up into standalone      │
                         │  query (history-aware rewrite)           │
                         │       │                                  │
                         │       ▼                                  │
                         │ ┌──────────────────────────────────┐    │
                         │ │         retrieval.py             │    │
                         │ │                                  │    │
                         │ │  MultiQuery ──► [q1, q2, q3]    │    │
                         │ │                    │             │    │
                         │ │         ┌──────────┴─────────┐  │    │
                         │ │   Chroma Dense          BM25  │  │    │
                         │ │  (Qwen3 vectors)   (keywords) │  │    │
                         │ │         └──────────┬─────────┘  │    │
                         │ │        EnsembleRetriever (RRF)   │    │
                         │ │                    │             │    │
                         │ │      CrossEncoderReranker        │    │
                         │ │         (top 5 passages)         │    │
                         │ └──────────────┬───────────────────┘    │
                         │               │                         │
                         │               ▼                         │
                         │ ┌──────────────────────────────────┐    │
                         │ │         grounding.py             │    │
                         │ │  Format [1]..[5] → Prompt → LLM │    │
                         │ │  → is_grounded() verification    │    │
                         │ └──────────────────────────────────┘    │
                         │               │                         │
                         │     SSE stream ──► Browser              │
                         └─────────────────────────────────────────┘
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

> The first run also downloads `Qwen3-Embedding-0.6B` (~300 MB) and `ms-marco-MiniLM-L-6-v2` from HuggingFace automatically.

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

Server starts at `http://localhost:8000`.

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
│   ├── embeddings.py        ← Qwen3-Embedding-0.6B LangChain wrapper
│   ├── ingest.py            ← PDF → chunks → Chroma + pickle
│   ├── retrieval.py         ← hybrid retriever pipeline
│   ├── grounding.py         ← prompt, context formatting, faithfulness check
│   └── server.py            ← FastAPI SSE endpoint
├── .env                     ← your secrets (git-ignored)
├── .env.example             ← template to copy from
└── requirements.txt
```

---

## 💡 How It Works

1. **Ingest** — PDFs are split into 800-char overlapping chunks, embedded with `Qwen3-Embedding-0.6B`, and stored in ChromaDB. The raw text is also pickled for BM25.

2. **Route** — Casual messages like "hi" or "thanks" bypass retrieval entirely and are answered by the LLM directly.

3. **Condense** — Follow-up questions ("tell me more about that") are rewritten by the LLM into self-contained queries before retrieval.

4. **Retrieve** — Each query spawns multiple LLM-generated variants (multi-query). Each variant hits both Chroma (dense) and BM25 (sparse). Results are merged with Reciprocal Rank Fusion, then a cross-encoder reranks to the top 5.

5. **Ground** — The 5 passages are numbered and injected into the prompt. The LLM answers only from these passages and cites `[1]`, `[2]`, etc.

6. **Verify** — A second LLM call checks if the answer is actually supported by the context. The UI shows a **Verified** or **Unverified** badge. Skipped for casual replies and refusals.

7. **Stream** — Tokens stream back via Server-Sent Events. Sources appear first, then the answer streams in, then the verification badge.

---

## 🔒 Security

- `.env`, `chroma_db/`, `chunks.pkl`, and `pdfs/` are all in `.gitignore` — your API key and documents stay local.
- The frontend uses `textContent` during streaming (no XSS) and only switches to `innerHTML` for markdown rendering after the stream completes.

---

## 📜 License

MIT — feel free to use, modify, and share.
