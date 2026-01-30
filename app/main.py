"""Streamlit entry point for the Herbolaria RAG application."""

import sys
from pathlib import Path

# Add parent directory to path for imports when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from app.components import render_chat, render_sidebar
from app.indexer import load_index
from app.query_engine import create_chat_engine

st.set_page_config(
    page_title="Herbolaria - Medicina Tradicional Mexicana",
    page_icon="🌿",
    layout="wide",
)


def main():
    """Main application entry point."""
    config = render_sidebar()

    has_api_key = bool(config["api_key"])
    needs_openai_for_embeddings = (
        config["provider"] == "anthropic"
        and not config["openai_key_for_embeddings"]
    )

    if has_api_key and needs_openai_for_embeddings:
        st.warning(
            "Para usar Anthropic, también necesitas una API key de OpenAI "
            "para generar los embeddings."
        )
        has_api_key = False

    chat_engine = None
    if has_api_key:
        openai_key = (
            config["api_key"]
            if config["provider"] == "openai"
            else config["openai_key_for_embeddings"]
        )

        if "index" not in st.session_state:
            with st.spinner("Cargando índice..."):
                st.session_state.index = load_index(api_key=openai_key)

        if st.session_state.index is not None:
            engine_key = f"{config['provider']}_{config['model']}"
            if (
                "chat_engine" not in st.session_state
                or st.session_state.get("engine_key") != engine_key
            ):
                st.session_state.chat_engine = create_chat_engine(
                    index=st.session_state.index,
                    provider=config["provider"],
                    model=config["model"],
                    api_key=config["api_key"],
                    openai_api_key=openai_key,
                )
                st.session_state.engine_key = engine_key
                st.session_state.messages = []

            chat_engine = st.session_state.chat_engine

    render_chat(chat_engine, has_api_key)


if __name__ == "__main__":
    main()
