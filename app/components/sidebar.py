"""Sidebar component for settings and API key input."""

import streamlit as st

from ..query_engine import MODELS


def render_sidebar() -> dict:
    """Render the sidebar with settings and return configuration."""
    with st.sidebar:
        st.header("Configuración")

        provider = st.selectbox(
            "Proveedor LLM",
            options=["openai", "anthropic"],
            format_func=lambda x: "OpenAI" if x == "openai" else "Anthropic",
            help="Selecciona el proveedor de modelo de lenguaje",
        )

        model_options = MODELS[provider]
        model = st.selectbox(
            "Modelo",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            help="Selecciona el modelo a usar",
        )

        st.divider()

        api_key = st.text_input(
            f"API Key de {'OpenAI' if provider == 'openai' else 'Anthropic'}",
            type="password",
            help="Tu clave API (no se almacena)",
        )

        openai_key_for_embeddings = None
        if provider == "anthropic":
            st.caption(
                "Los embeddings requieren OpenAI. "
                "Ingresa tu clave de OpenAI para los embeddings."
            )
            openai_key_for_embeddings = st.text_input(
                "API Key de OpenAI (para embeddings)",
                type="password",
                help="Necesaria para generar embeddings con Anthropic",
            )

        st.divider()

        st.subheader("Acerca de")
        st.markdown(
            """
            Esta aplicación permite consultar el acervo de la
            [Biblioteca Digital de la Medicina Tradicional Mexicana](http://www.medicinatradicionalmexicana.unam.mx/)
            de la UNAM usando RAG (Retrieval-Augmented Generation).

            **Secciones incluidas:**
            - Diccionario Enciclopédico
            - Atlas de Plantas Medicinales
            - Medicina de Pueblos Indígenas
            - Flora Medicinal Indígena
            """
        )

        st.divider()

        if st.button("Limpiar conversación"):
            st.session_state.messages = []
            st.session_state.chat_engine = None
            st.rerun()

    return {
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "openai_key_for_embeddings": openai_key_for_embeddings,
    }
