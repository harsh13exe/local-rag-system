import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma

# ---- CONFIG ----
PERSIST_DIR = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "harsh-resume-model"
TOP_K = 6

# In Docker, this points to the 'ollama' service name (see docker-compose.yml).
# Locally without Docker, it falls back to your normal localhost Ollama.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
print("=" * 50)
print("OLLAMA_BASE_URL:", OLLAMA_BASE_URL)
print("LLM_MODEL:", LLM_MODEL)
print("=" * 50)

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions using ONLY the context provided below.
The context may contain multiple jobs/entries with different date ranges. To find the CURRENT or MOST RECENT one, look for the entry whose date range ends with "Present" — that is the current role, even if other entries have similar-looking dates.
Read the entire context carefully before answering. If the answer is genuinely not present anywhere in the context, say "I don't have enough information to answer that."
Cite the source number(s) you used, like [1], [2].

Context:
{context}

Question: {question}

Answer:"""

app = FastAPI(title="RAG API")

# Load once at startup, reused across requests
embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
llm = ChatOllama(model=LLM_MODEL, temperature=0, base_url=OLLAMA_BASE_URL)


class QuestionRequest(BaseModel):
    question: str
    top_k: int | None = None


class SourceInfo(BaseModel):
    source: str
    page: str
    score: float


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]


def build_context(results):
    context_blocks = []
    for i, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        context_blocks.append(f"[{i}] (source: {source}, page: {page})\n{doc.page_content}")
    return "\n\n".join(context_blocks)


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    k = request.top_k or TOP_K

    try:
        results = vectorstore.similarity_search_with_score(request.question, k=k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    if not results:
        return AnswerResponse(answer="No relevant documents found.", sources=[])

    context = build_context(results)
    prompt = PROMPT_TEMPLATE.format(context=context, question=request.question)

    try:
        response = llm.invoke(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

    sources = [
        SourceInfo(
            source=doc.metadata.get("source", "unknown"),
            page=str(doc.metadata.get("page", "?")),
            score=round(float(score), 4)
        )
        for doc, score in results
    ]

    return AnswerResponse(answer=response.content, sources=sources)


@app.get("/health")
def health():
    return {"status": "ok"}