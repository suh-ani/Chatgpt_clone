from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader
)
from langchain.text_splitter import CharacterTextSplitter

import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Embedding model using HuggingFace
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': device}
)

# Global vector database
vector_db = None


def update_vector_store(file_path: str) -> str:
    """
    Loads a document and updates the FAISS vector store.
    Accepts .pdf, .docx, .pptx files.
    """
    global vector_db

    try:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = UnstructuredWordDocumentLoader(file_path)
        elif ext == ".pptx":
            loader = UnstructuredPowerPointLoader(file_path)
        else:
            return f"❌ Unsupported file type: {ext}"

        documents = loader.load()

        # Split into smaller chunks
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        # Create or update FAISS vector store
        vector_db = FAISS.from_documents(chunks, embedding_model)
        return f"✅ {ext.upper()} file indexed successfully."

    except Exception as e:
        return f"❌ Failed to process document: {str(e)}"


def retrieve_docs(query: str) -> str:
    """
    Searches the FAISS vector store using the query.
    """
    if vector_db is None:
        return "⚠️ Please upload a document first."
    
    docs = vector_db.similarity_search(query, k=3)
    return "\n\n".join(doc.page_content for doc in docs)
