"""
server.py - Route -> Retrieve -> Condense -> Ground -> Stream -> Verify.

Two paths:
  RAG          -> deep document questions (condense -> retrieve -> ground -> verify)
  Tool binding -> LLM picks from MCP tools loaded from mcp_server.py at startup
"""

import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_mcp_adapters.client import MultiServerMCPClient

from retrieval import build_retriever
from grounding import GROUNDED_PROMPT, REFUSAL, format_context, is_grounded

load_dotenv()

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
if not CEREBRAS_API_KEY:
    raise RuntimeError("CEREBRAS_API_KEY is missing in .env")

llm = ChatOpenAI(
    model="gpt-oss-120b",
    api_key=CEREBRAS_API_KEY,
    base_url="https://api.cerebras.ai/v1",
    temperature=0,
    max_tokens=1024,
)

DEFAULT_RETRIEVER = build_retriever(llm=llm, use_multiquery=True)

_MCP_SERVER_PATH = str(Path(__file__).parent / "mcp_server.py")

# Set during lifespan startup when MCP server connects
_llm_with_tools = None
_tool_map: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Connect to MCP server subprocess, load its tools, bind to LLM."""
    global _llm_with_tools, _tool_map

    mcp_client = MultiServerMCPClient(
        {
            "rag-chatbot-tools": {
                "command": "python",
                "args": [_MCP_SERVER_PATH],
                "transport": "stdio",
            }
        }
    )
    tools = await mcp_client.get_tools()
    _llm_with_tools = llm.bind_tools(tools)
    _tool_map = {t.name: t for t in tools}
    yield


# ── Prompts ───────────────────────────────────────────────────────────────────

_RAG_CHECK_SYSTEM = (
    "Does the user's message require deep document analysis such as: "
    "summarizing a document, comparing multiple sections, multi-part questions, "
    "or questions needing strict citation-based answers from uploaded documents? "
    "Reply with YES or NO only."
)

_DIRECT_SYSTEM = (
    "You are Auralis, a helpful AI assistant for document Q&A. "
    "Respond naturally and helpfully."
)

_CONDENSE_SYSTEM = (
    "Given the conversation history and a follow-up question, rewrite the "
    "follow-up as a self-contained question that can be understood without "
    "the history. If it already makes sense on its own, return it unchanged. "
    "Return only the rewritten question, nothing else."
)

_CITATION_RE = re.compile(r'\[\d+\]')

# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class HistoryMessage(BaseModel):
    role: str       # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    source: Optional[str] = None
    history: list[HistoryMessage] = []


def _to_lc_messages(history: list[HistoryMessage]):
    """Strip [n] markers from AI turns so they don't collide with current turn's passage numbers."""
    out = []
    for m in history[-10:]:
        if m.role == "user":
            out.append(HumanMessage(content=m.content))
        else:
            clean = _CITATION_RE.sub("", m.content).strip()
            out.append(AIMessage(content=clean))
    return out


@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Message is required")

    retriever = (
        build_retriever(llm=llm, metadata_filter={"source": req.source}, use_multiquery=True)
        if req.source else DEFAULT_RETRIEVER
    )

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            history = _to_lc_messages(req.history)

            # Step 1: Check if this needs the full RAG pipeline (deep analysis)
            rag_check = await llm.ainvoke([
                SystemMessage(content=_RAG_CHECK_SYSTEM),
                HumanMessage(content=req.message),
            ])

            if "YES" in rag_check.content.strip().upper():
                print(f"\n[ROUTER] → RAG  | query: {req.message!r}")
                # ── RAG: deep document questions ──────────────────────────────
                query = req.message
                if history:
                    try:
                        result = await llm.ainvoke([
                            SystemMessage(content=_CONDENSE_SYSTEM),
                            *history,
                            HumanMessage(content=req.message),
                        ])
                        query = result.content.strip() or req.message
                    except Exception:
                        query = req.message

                docs = await retriever.ainvoke(query)
                context = format_context(docs) if docs else ""
                sources = sorted({d.metadata.get("source", "?") for d in docs}) if docs else []

                chain = GROUNDED_PROMPT | llm | StrOutputParser()
                yield f'data: {json.dumps({"type":"sources","data":sources})}\n\n'

                full_answer = ""
                async for token in chain.astream(
                    {"context": context, "question": req.message, "history": history}
                ):
                    full_answer += token
                    yield f'data: {json.dumps({"type":"token","data":token})}\n\n'

                if docs and REFUSAL not in full_answer:
                    grounded = await is_grounded(llm, context, full_answer)
                    yield f'data: {json.dumps({"type":"grounded","data":grounded})}\n\n'

                return

            # Step 2: Tool binding — LLM picks from MCP tools or answers directly
            messages = [SystemMessage(content=_DIRECT_SYSTEM), *history, HumanMessage(content=req.message)]
            resp = await _llm_with_tools.ainvoke(messages)

            if not resp.tool_calls:
                print(f"\n[ROUTER] → DIRECT  | query: {req.message!r}")
                # No tool needed — direct conversational answer
                yield f'data: {json.dumps({"type":"sources","data":[]})}\n\n'
                async for chunk in llm.astream(messages):
                    if chunk.content:
                        yield f'data: {json.dumps({"type":"token","data":chunk.content})}\n\n'
                return

            # Execute the tool the LLM chose — runs inside mcp_server.py subprocess
            tool_call = resp.tool_calls[0]
            print(f"\n[ROUTER] → MCP tool: {tool_call['name']}  | args: {tool_call['args']}")
            chosen_tool = _tool_map[tool_call["name"]]
            tool_result = await chosen_tool.ainvoke(tool_call["args"])

            yield f'data: {json.dumps({"type":"sources","data":[]})}\n\n'

            # Pass tool result back to LLM and stream the final answer
            async for chunk in llm.astream([
                *messages,
                resp,
                ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]),
            ]):
                if chunk.content:
                    yield f'data: {json.dumps({"type":"token","data":chunk.content})}\n\n'

        except Exception as exc:
            yield f'data: {json.dumps({"type":"error","data":str(exc)})}\n\n'

        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
