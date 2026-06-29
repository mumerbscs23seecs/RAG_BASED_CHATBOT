"""
grounding.py - keep the answer tied to the retrieved text.

Three levers:
  1. Numbered context passages, each tagged with its source.
  2. A strict prompt: answer only from context, cite [n], refuse with a fixed
     string when the answer is not present.
  3. (optional) a faithfulness check: a second, cheap LLM pass that verifies the
     answer is supported by the context. Costs one extra call, so it is opt-in.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

REFUSAL = "I could not find this in the documents."

GROUNDED_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful AI assistant. "
     "For simple greetings, casual conversation, or general questions that don't require documents, respond naturally and briefly. "
     "For any question that requires knowledge from the user's documents, answer ONLY from the numbered context passages below, "
     "cite the passages you rely on using their [n] markers, and use markdown formatting (bold, lists, headers) to structure your answer clearly. "
     f"If a document question cannot be answered from the context, reply exactly: \"{REFUSAL}\" "
     "Never use outside knowledge for document questions."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def format_context(docs) -> str:
    """Number each passage and show its source, so the model can cite it."""
    return "\n\n".join(
        f"[{i + 1}] (source: {d.metadata.get('source', '?')})\n{d.page_content}"
        for i, d in enumerate(docs)
    )


# Optional faithfulness check. Returns True if the answer is supported.
FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Decide if the ANSWER is fully supported by the CONTEXT. "
     "Reply with one word: SUPPORTED or UNSUPPORTED."),
    ("human", "Context:\n{context}\n\nAnswer:\n{answer}"),
])


async def is_grounded(llm, context: str, answer: str) -> bool:
    from langchain_core.output_parsers import StrOutputParser
    chain = FAITHFULNESS_PROMPT | llm | StrOutputParser()
    verdict = await chain.ainvoke({"context": context, "answer": answer})
    return "UNSUPPORTED" not in verdict.upper()
