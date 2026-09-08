from llama_index.core import (
    StorageContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_PATH,
    LLM_SYSTEM_PROMPT,
    RERANKER_MODEL_NAME,
    RERANKER_TOP_N,
    SIMILARITY_TOP_K,
    VECTOR_STORE_PATH,
    CHAT_MEMORY_TOKEN_LIMIT,
)
from src.model_loader import (
    get_embedding_model,
    initialise_llm,
)


def _create_new_vector_store(
        embed_model: HuggingFaceEmbedding,
) -> VectorStoreIndex:
    """Creates, saves, and returns a new vector store from documents."""

    print(
        "Creating new vector store from all files in the 'data' directory..."
    )

    documents: list[Document] = SimpleDirectoryReader(
        input_dir=DATA_PATH
    ).load_data()

    if not documents:
        raise ValueError(
            f"No documents found in {DATA_PATH}. Cannot create vector store."
        )

    text_splitter: SentenceSplitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    index: VectorStoreIndex = VectorStoreIndex.from_documents(
        documents,
        transformations=[text_splitter],
        embed_model=embed_model,
    )

    index.storage_context.persist(persist_dir=VECTOR_STORE_PATH.as_posix())
    print("Vector store created and saved.")
    return index


def get_vector_store(embed_model: HuggingFaceEmbedding) -> VectorStoreIndex:
    """
    Loads the vector store from disk if it exists;
    otherwise, creates a new one.
    """

    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

    if any(VECTOR_STORE_PATH.iterdir()):
        print("Loading existing vector store from disk...")
        storage_context: StorageContext = StorageContext.from_defaults(
            persist_dir=VECTOR_STORE_PATH.as_posix()
        )
        return load_index_from_storage(
            storage_context,
            embed_model=embed_model,
        )
    else:
        return _create_new_vector_store(embed_model)


def get_chat_engine(
        llm: Groq,
        embed_model: HuggingFaceEmbedding,
) -> CondensePlusContextChatEngine:
    """
    Initialises and returns the main conversational RAG chat engine.

    The pipeline is: retriever -> cross-encoder reranker -> LLM,
    using the winning configuration discovered by the evaluation
    pipeline (chunk=512/50, retriever_k=10, reranker_n=5, no HyDE).
    """

    vector_index: VectorStoreIndex = get_vector_store(embed_model)

    retriever = vector_index.as_retriever(similarity_top_k=SIMILARITY_TOP_K)

    reranker: SentenceTransformerRerank = SentenceTransformerRerank(
        top_n=RERANKER_TOP_N,
        model=RERANKER_MODEL_NAME,
    )

    memory: ChatMemoryBuffer = ChatMemoryBuffer.from_defaults(
        token_limit=CHAT_MEMORY_TOKEN_LIMIT
    )

    chat_engine: CondensePlusContextChatEngine = (
        CondensePlusContextChatEngine.from_defaults(
            retriever=retriever,
            llm=llm,
            memory=memory,
            system_prompt=LLM_SYSTEM_PROMPT,
            node_postprocessors=[reranker],
        )
    )
    return chat_engine


def main_chat_loop() -> None:
    """Main application loop to run the RAG chatbot."""

    print("--- Initialising models... ---")
    llm: Groq = initialise_llm()
    embed_model: HuggingFaceEmbedding = get_embedding_model()

    chat_engine: CondensePlusContextChatEngine = get_chat_engine(
        llm=llm,
        embed_model=embed_model,
    )
    print("--- RAG Chatbot Initialised. ---")
    chat_engine.chat_repl()
