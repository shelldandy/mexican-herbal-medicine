"""RAG query engine for the Herbolaria application."""

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI

SYSTEM_PROMPT = """Eres un asistente experto en medicina tradicional mexicana.
Responde en español usando la información proporcionada del acervo de la
Biblioteca Digital de la Medicina Tradicional Mexicana de la UNAM.

Cuando respondas:
- Cita las fuentes específicas cuando sea relevante (nombre de la planta, entrada del diccionario, etc.)
- Si la información proviene de conocimientos de pueblos indígenas específicos, menciónalos
- Si no encuentras información relevante en el contexto proporcionado, indícalo claramente
- Mantén un tono respetuoso hacia el conocimiento tradicional

El contexto incluye información de:
- Diccionario Enciclopédico de la Medicina Tradicional Mexicana
- Atlas de las Plantas de la Medicina Tradicional Mexicana
- La Medicina Tradicional de los Pueblos Indígenas de México
- Flora Medicinal Indígena de México"""

MODELS = {
    "openai": {
        "gpt-4o": "GPT-4o (más capaz)",
        "gpt-4o-mini": "GPT-4o Mini (más rápido)",
    },
    "anthropic": {
        "claude-sonnet-4-20250514": "Claude Sonnet 4 (equilibrado)",
        "claude-3-5-haiku-latest": "Claude 3.5 Haiku (más rápido)",
    },
}


def create_llm(provider: str, model: str, api_key: str):
    """Create an LLM instance based on provider and model."""
    if provider == "openai":
        return OpenAI(model=model, api_key=api_key, temperature=0.1)
    elif provider == "anthropic":
        return Anthropic(model=model, api_key=api_key, temperature=0.1)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def create_chat_engine(
    index: VectorStoreIndex,
    provider: str,
    model: str,
    api_key: str,
    openai_api_key: str | None = None,
):
    """Create a chat engine with the specified LLM."""
    llm = create_llm(provider, model, api_key)
    Settings.llm = llm

    if provider == "openai":
        Settings.embed_model = OpenAIEmbedding(api_key=api_key)
    elif openai_api_key:
        Settings.embed_model = OpenAIEmbedding(api_key=openai_api_key)

    retriever = index.as_retriever(similarity_top_k=5)

    memory = ChatMemoryBuffer.from_defaults(token_limit=4096)

    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        memory=memory,
        system_prompt=SYSTEM_PROMPT,
        verbose=False,
    )

    return chat_engine


def format_sources(response) -> list[dict]:
    """Extract and format source information from a response."""
    sources = []
    seen = set()

    if hasattr(response, "source_nodes"):
        for node in response.source_nodes:
            metadata = node.metadata
            file_path = metadata.get("file_path", "")
            title = metadata.get("title", "")
            section = metadata.get("section", "")

            key = f"{file_path}:{title}"
            if key in seen:
                continue
            seen.add(key)

            sources.append(
                {
                    "title": title,
                    "section": section,
                    "file_path": file_path,
                    "score": getattr(node, "score", None),
                    "text_preview": node.text[:200] + "..."
                    if len(node.text) > 200
                    else node.text,
                }
            )

    return sources
