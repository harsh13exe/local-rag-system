import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# ---- CONFIG ----
DOCS_DIR = "./documents"          # put your PDFs/txt/md files here
PERSIST_DIR = "./chroma_db"       # where the vector DB will be saved
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBED_MODEL = "nomic-embed-text"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

def load_documents():
    docs = []
    for filename in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        elif filename.endswith((".txt", ".md")):
            loader = TextLoader(path)
            docs.extend(loader.load())
        else:
            print(f"Skipping unsupported file: {filename}")
    return docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(docs)

def build_vectorstore(chunks):
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    return vectorstore

if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents()
    print(f"Loaded {len(docs)} document(s)/page(s).")

    print("Chunking...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("Embedding and storing in ChromaDB (this may take a bit on CPU)...")
    build_vectorstore(chunks)
    print(f"Done. Vector store saved to '{PERSIST_DIR}'.")