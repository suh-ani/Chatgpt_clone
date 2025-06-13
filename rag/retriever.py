from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import torch

device = "cpu"  # Force CPU to avoid memory spikes

vector_db = None  # Don't load immediately

def load_vector_store():
    global vector_db

    if vector_db is not None:
        return vector_db

    loader = DirectoryLoader("docs", loader_cls=PyPDFLoader)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Load embedding model only when needed
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )

    # FAISS vector store in memory
    vector_db = FAISS.from_documents(docs, embeddings)
    return vector_db

def retrieve_docs(query: str):
    db = load_vector_store()
    docs = db.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])
