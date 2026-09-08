import streamlit as st
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.llms import ChatMessage, MessageRole

from src.engine import get_chat_engine, get_vector_store
from src.model_loader import get_embedding_model, initialise_llm

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Transformer Circuits RAG",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Transformer Circuits Research Assistant")
st.caption(
    "Powered by GPT-OSS 120B · cross-encoder reranker · "
    "Transformer Circuits corpus"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    show_sources = st.toggle("Show retrieved sources", value=True)

    st.divider()
    st.markdown("**Documents loaded:**")
    st.markdown(
        "- A Mathematical Framework for Transformer Circuits\n"
        "- In-Context Learning and Induction Heads\n"
        "- Mechanistic Interpretability Essay\n"
        "- Interpretability Dreams\n"
        "- Towards Monosemanticity\n"
        "- Scaling Monosemanticity"
    )
    st.divider()

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.chat_engine.reset()
        st.rerun()


# ── Load engine (cached so it only builds once per session) ───────────────────
@st.cache_resource(show_spinner="Loading models and vector store...")
def load_engine() -> CondensePlusContextChatEngine:
    llm = initialise_llm()
    embed_model = get_embedding_model()
    return get_chat_engine(llm=llm, embed_model=embed_model)


if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = load_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_sources and msg.get("sources"):
            with st.expander(f"📄 Retrieved sources ({len(msg['sources'])})"):
                for i, source in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk {i}**")
                    st.markdown(source)
                    if i < len(msg["sources"]):
                        st.divider()

# ── Chat input ────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask about transformer circuits, induction heads, monosemanticity..."):

    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Query the chat engine
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            response = st.session_state.chat_engine.chat(user_input)

        answer = str(response)
        st.markdown(answer)

        # Extract retrieved source chunks
        sources: list[str] = []
        if hasattr(response, "source_nodes") and response.source_nodes:
            for node in response.source_nodes:
                text = node.get_content().strip()
                # Strip HTML tags for cleaner display
                import re
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    sources.append(text[:800] + ("..." if len(text) > 800 else ""))

        if show_sources and sources:
            with st.expander(f"📄 Retrieved sources ({len(sources)})"):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"**Chunk {i}**")
                    st.markdown(source)
                    if i < len(sources):
                        st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
