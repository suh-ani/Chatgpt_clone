from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import torch

device="cuda" if torch.cuda.is_available() else "cpu"

def load_vector_store():
    loader = DirectoryLoader("docs", loader_cls=PyPDFLoader)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"
,model_kwargs={'device': device})
    db = FAISS.from_documents(docs, embeddings)
    return db

vector_db = load_vector_store()

def retrieve_docs(query: str):
    docs = vector_db.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])
