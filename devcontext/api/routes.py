"""FastAPI routes (stub)."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from devcontext.agents.supervisor import run
from devcontext.rag.evaluator import run_evaluation, print_eval_report

app = FastAPI(
    title="DevContext API",
    description="API for the DevContext multi-agent developer assistant",
    version="0.1.0"
)

# --- Request / Response models ---

class QueryRequest(BaseModel):
    query: str
    filepath: Optional[str] = None

class QueryResponse(BaseModel):
    agent_used: str
    response: Optional[str] = None 
    error: Optional[str] = None


# --- Routes ---

@app.get("/health")
def health():
    return {"status": "ok", "service": "devcontext"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Main endpoint. Routes query to the correct agent and returns response.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    result = run(request.query, filepath=request.filepath)

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    return QueryResponse(
        agent_used=result["agent_used"],
        response=result["response"] or "",
        error=result.get("error")
    )


@app.post("/code", response_model=QueryResponse)
def ask_code(request: QueryRequest):
    """Force route to code_agent regardless of query content."""
    if not request.filepath:
        raise HTTPException(status_code=400, detail="filepath is required for /code endpoint")
    result = run(request.query, filepath=request.filepath)
    return QueryResponse(**result)


@app.post("/review", response_model=QueryResponse)
def review_code(request: QueryRequest):
    """Force route to review_agent."""
    if not request.filepath:
        raise HTTPException(status_code=400, detail="filepath is required for /review endpoint")
    result = run(f"review {request.query}", filepath=request.filepath)
    return QueryResponse(**result)


@app.post("/docs", response_model=QueryResponse)
def search_docs_endpoint(request: QueryRequest):
    """Force route to docs_agent."""
    result = run(request.query, filepath=None)
    return QueryResponse(**result)

@app.post("/eval")
def evaluate_rag():
    """
    Run RAGAS evaluation on the RAG pipeline.
    Returns faithfulness and answer_relevancy (no context_precision without a `reference` column).
    Warning: can take several minutes (LLM + embeddings calls).
    """
    scores = run_evaluation()
    print_eval_report(scores)
    return scores