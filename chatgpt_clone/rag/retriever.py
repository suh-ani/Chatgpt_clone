import os
import torch
import fitz  # PyMuPDF
import camelot
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader
)
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": device}
)

# Global vector DB
vector_db = None


def is_scientific_pdf(file_path: str) -> bool:
    try:
        reader = PdfReader(file_path)
        text = " ".join(page.extract_text() or "" for page in reader.pages[:2])
        keywords = ["abstract", "introduction", "methods", "results", "conclusion"]
        return sum(k in text.lower() for k in keywords) >= 2
    except Exception as e:
        print(f"❌ PDF check failed: {e}")
        return False


def extract_tables(file_path: str) -> list[str]:
    try:
        tables = camelot.read_pdf(file_path, pages="all", flavor="stream")
        return [f"Table {i+1}:\n{t.df.to_string()}" for i, t in enumerate(tables)]
    except Exception as e:
        print(f"❌ Table extraction failed: {e}")
        return []


def extract_figures(file_path: str) -> list[str]:
    captions = []
    try:
        doc = fitz.open(file_path)
        for i, page in enumerate(doc):
            images = page.get_images(full=True)
            for j, _ in enumerate(images):
                captions.append(f"Figure {j+1} on page {i+1}: [Image content not extracted]")
        return captions
    except Exception as e:
        print(f"❌ Figure extraction failed: {e}")
        return []


def update_vector_store(file_path: str) -> str:
    global vector_db

    try:
        ext = os.path.splitext(file_path)[1].lower()
        all_docs = []

        if ext == ".pdf":
            if is_scientific_pdf(file_path):
                print("🔬 Detected scientific PDF, extracting text, tables, and figures...")

                reader = PdfReader(file_path)
                pages_text_docs = []
                for i, page in enumerate(reader.pages):
                    print(f"📄 Reading page {i+1}")
                    text = page.extract_text() or ""
                    pages_text_docs.append(Document(page_content=text))
                print(f"📄 Total pages read: {len(pages_text_docs)}")

                splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                print("✂️ Splitting text into chunks...")
                text_chunks = splitter.split_documents(pages_text_docs)
                print(f"✂️ Created {len(text_chunks)} text chunks")

                print("📊 Extracting tables...")
                table_strings = extract_tables(file_path)
                table_docs = [Document(page_content=tbl, metadata={"type": "table"}) for tbl in table_strings]
                print(f"📊 Extracted {len(table_docs)} tables")

                print("🖼️ Extracting figures...")
                figure_strings = extract_figures(file_path)
                figure_docs = [Document(page_content=fig, metadata={"type": "image"}) for fig in figure_strings]
                print(f"🖼️ Extracted {len(figure_docs)} figures")

                all_docs = text_chunks + table_docs + figure_docs
                print(f"📦 Total loaded chunks (text + tables + figures): {len(all_docs)}")

            else:
                print("📄 Standard PDF detected, extracting text...")
                loader = PyPDFLoader(file_path)
                raw_docs = loader.load()
                splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                print("✂️ Splitting loaded document into chunks...")
                all_docs = splitter.split_documents(raw_docs)
                print(f"✂️ Created {len(all_docs)} chunks")

        elif ext == ".docx":
            loader = UnstructuredWordDocumentLoader(file_path)
            raw_docs = loader.load()
            splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            print("✂️ Splitting Word document into chunks...")
            all_docs = splitter.split_documents(raw_docs)
            print(f"✂️ Created {len(all_docs)} chunks")

        elif ext == ".pptx":
            loader = UnstructuredPowerPointLoader(file_path)
            raw_docs = loader.load()
            splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            print("✂️ Splitting PowerPoint into chunks...")
            all_docs = splitter.split_documents(raw_docs)
            print(f"✂️ Created {len(all_docs)} chunks")

        else:
            return f"❌ Unsupported file type: {ext}"

        if not all_docs:
            return "⚠️ No content extracted from file."

        print("🧠 Creating embeddings and updating FAISS index (this may take a while)...")
        vector_db = FAISS.from_documents(all_docs, embedding_model)
        print("✅ FAISS vector store created/updated.")
        return f"✅ {ext.upper()} file indexed successfully with {len(all_docs)} chunks."

    except Exception as e:
        return f"❌ Failed to process document: {str(e)}"


def estimate_token_count(text: str) -> int:
    return max(1, len(text) // 4)  # Approximate 1 token per 4 characters


def retrieve_docs(query: str, max_tokens_context: int = 2300) -> str:
    """Retrieve most relevant chunks based on the query, constrained by max token budget."""
    if vector_db is None:
        return "⚠️ Please upload a document first."

    docs = vector_db.similarity_search(query, k=15)  # Retrieve extra for filtering

    grouped = {"text": [], "table": [], "image": []}
    token_total = 0

    for doc in docs:
        content = doc.page_content
        tokens = estimate_token_count(content)

        if token_total + tokens > max_tokens_context:
            continue  # Skip if exceeds token budget

        dtype = doc.metadata.get("type", "text")
        grouped[dtype].append(content)
        token_total += tokens

    sections = []
    for dtype in ["text", "table", "image"]:
        if grouped[dtype]:
            sections.append(f"🔹 {dtype.upper()}:\n" + "\n\n".join(grouped[dtype]))

    return "\n\n".join(sections) or "❌ No relevant content found."