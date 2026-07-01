"""
mcp_server.py - MCP server exposing three tools:
  1. calculator       - evaluate math expressions
  2. get_weather      - dummy weather data
  3. retrieve_documents - same retriever as RAG pipeline (retrieval.py)
"""

import math
import os

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from langchain_openai import ChatOpenAI

from retrieval import build_retriever

load_dotenv()

mcp = FastMCP("rag-chatbot-tools")

_llm = ChatOpenAI(
    model="gpt-oss-120b",
    api_key=os.getenv("CEREBRAS_API_KEY"),
    base_url="https://api.cerebras.ai/v1",
    temperature=0,
    max_tokens=1024,
)

_retriever = build_retriever(llm=_llm, use_multiquery=True)


@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use this for any arithmetic or math calculation."""
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


@mcp.tool()
def get_weather(city: str) -> str:
    """Get current weather for a city. Use this when the user asks about weather."""
    dummy = {
        "lahore":     "Sunny, 35°C, humidity 60%",
        "karachi":    "Partly cloudy, 32°C, humidity 75%",
        "islamabad":  "Clear, 28°C, humidity 50%",
        "london":     "Rainy, 12°C, humidity 90%",
        "new york":   "Cloudy, 18°C, humidity 65%",
        "dubai":      "Hot and sunny, 42°C, humidity 45%",
    }
    info = dummy.get(city.lower(), "Sunny, 25°C, humidity 55%")
    return f"Weather in {city}: {info}"


@mcp.tool()
async def retrieve_documents(query: str) -> str:
    """
    Search and retrieve relevant passages from the uploaded documents.
    Use this when the user asks a question that requires looking up information from documents.
    """
    docs = await _retriever.ainvoke(query)
    if not docs:
        return "No relevant documents found."
    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i + 1}] (source: {source})\n{doc.page_content}")
    return "\n\n".join(parts)


if __name__ == "__main__":
    mcp.run()
