"""
server.py - Route -> Retrieve -> Condense -> Ground -> Stream -> Verify.

Router: one cheap LLM call decides if the message needs document retrieval.
  YES  -> full RAG pipeline (condense -> retrieve -> ground -> verify)
  NO   -> LLM answers directly from conversation history (greetings, jokes, general chat)
"""

import json
import os
import re
from typing import Optional, AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

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

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# Router: classify whether the message needs document retrieval
_ROUTER_SYSTEM = (
    "Decide whether answering the user's message requires searching their uploaded documents. "
    "Reply with exactly one word — YES or NO.\n\n"
    "Say NO for: greetings, casual conversation, jokes, general knowledge questions, "
    "questions about yourself, anything that doesn't need the user's specific documents.\n"
    "Say YES for: questions about content in the user's documents, specific facts, "
    "summaries, comparisons, or anything that requires looking up uploaded material."
)

# Used when the router says NO — answer directly without any RAG
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

            # Router: one cheap call to decide if retrieval is needed
            router_resp = await llm.ainvoke([
                SystemMessage(content=_ROUTER_SYSTEM),
                HumanMessage(content=req.message),
            ])
            needs_docs = "YES" in router_resp.content.strip().upper()

            if not needs_docs:
                # Direct conversational reply — no retrieval, no grounding check
                yield f'data: {json.dumps({"type":"sources","data":[]})}\n\n'
                async for chunk in llm.astream([
                    SystemMessage(content=_DIRECT_SYSTEM),
                    *history,
                    HumanMessage(content=req.message),
                ]):
                    if chunk.content:
                        yield f'data: {json.dumps({"type":"token","data":chunk.content})}\n\n'
                return

            # Rewrite follow-up questions into standalone queries before retrieval
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

            # Only verify faithfulness when we retrieved docs and got a real answer
            if docs and REFUSAL not in full_answer:
                grounded = await is_grounded(llm, context, full_answer)
                yield f'data: {json.dumps({"type":"grounded","data":grounded})}\n\n'

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
