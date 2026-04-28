"""Document loading, chunking, embedding (stub)."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from devcontext.config.settings import settings


def load_documents(docs_dir: Path = settings.docs_dir) -> list:
    """Load all markdown and text files from docs directory."""
    loader = DirectoryLoader(
        str(docs_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s) from {docs_dir}")
    return documents


def chunk_documents(documents: list) -> list:
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunk(s)")
    return chunks


def get_embeddings() -> OllamaEmbeddings:
    """Return the embedding model."""
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )


def build_vectorstore(chunks: list) -> Chroma:
    """Embed chunks and store in ChromaDB."""
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_db_dir)
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB at {settings.chroma_db_dir}")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load existing ChromaDB vectorstore from disk."""
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_db_dir)
    )
    return vectorstore


def ingest(force: bool = False) -> Chroma:
    """
    Full ingestion pipeline.
    Reuses the persisted collection if it already has documents (unless force=True).
    """
    if not force:
        existing = load_vectorstore()
        try:
            count = existing._collection.count()
        except Exception:
            count = 0
        if count > 0:
            print(f"ChromaDB collection has {count} chunk(s) — skipping ingestion.")
            return existing

    documents = load_documents()
    if not documents:
        raise ValueError(f"No markdown files found in {settings.docs_dir}")
    chunks = chunk_documents(documents)
    return build_vectorstore(chunks)