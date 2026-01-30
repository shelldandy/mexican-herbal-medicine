"""Chat interface component."""

import streamlit as st

from ..query_engine import format_sources

EXAMPLE_QUERIES = [
    "¿Qué plantas se usan para tratar la diabetes?",
    "¿Cómo usan los mayas las plantas medicinales?",
    "¿Qué es el temascal y para qué se usa?",
    "¿Cuáles son los usos medicinales del nopal?",
    "¿Qué plantas ayudan con problemas digestivos?",
]


def render_chat(chat_engine, has_api_key: bool):
    """Render the chat interface."""
    st.header("Medicina Tradicional Mexicana")
    st.caption("Consulta el acervo de la UNAM sobre medicina tradicional")

    if not has_api_key:
        st.info(
            "Ingresa tu API key en la barra lateral para comenzar a hacer preguntas."
        )

        st.subheader("Preguntas de ejemplo")
        for query in EXAMPLE_QUERIES:
            st.markdown(f"- {query}")
        return

    if chat_engine is None:
        st.warning(
            "El índice no está disponible. "
            "Ejecuta `python -m app.indexer --build` primero."
        )
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("Ver fuentes"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**{source['title']}** ({source['section']})\n\n"
                            f"_{source['text_preview']}_"
                        )

    if len(st.session_state.messages) == 0:
        st.markdown("**Algunas preguntas que puedes hacer:**")
        cols = st.columns(2)
        for i, query in enumerate(EXAMPLE_QUERIES[:4]):
            with cols[i % 2]:
                if st.button(query, key=f"example_{i}"):
                    st.session_state.pending_query = query
                    st.rerun()

    if "pending_query" in st.session_state:
        prompt = st.session_state.pending_query
        del st.session_state.pending_query
    else:
        prompt = st.chat_input("Escribe tu pregunta sobre medicina tradicional...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Buscando información..."):
                response = chat_engine.chat(prompt)
                sources = format_sources(response)

            st.markdown(str(response))

            if sources:
                with st.expander("Ver fuentes"):
                    for source in sources:
                        st.markdown(
                            f"**{source['title']}** ({source['section']})\n\n"
                            f"_{source['text_preview']}_"
                        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": str(response),
                "sources": sources,
            }
        )
