from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents import run_product_research, run_niche_analysis, run_product_comparison, run_content_brief
from memory import (
    save_product_research, save_niche_analysis, save_content_brief,
    get_similar_products, get_top_products,
    get_memory_stats, build_memory_context
)
import re
import os

app = FastAPI(
    title="SmartHaven AI Backend",
    description="Multi-agent product research system with RAG memory",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProductRequest(BaseModel):
    input: str
    mode: str

class HealthResponse(BaseModel):
    status: str
    message: str

def extract_scores(text: str) -> dict:
    match = re.search(r'SCORES:(\{[^}]+\})', text)
    if match:
        try:
            import json
            return json.loads(match.group(1))
        except:
            pass
    return {"demand": 0, "margin": 0, "competition": "MED", "trend": 0, "verdict": "CAUTION"}

@app.get("/", response_model=HealthResponse)
def root():
    return {"status": "online", "message": "SmartHaven AI Backend v2.0 with RAG memory"}

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "healthy", "message": "All agents and memory ready"}

@app.post("/analyze")
async def analyze(request: ProductRequest):
    try:
        memory_context = build_memory_context(request.input)

        if request.mode == "research":
            result = run_product_research(request.input, memory_context)
            scores = extract_scores(result)
            save_product_research(
                product_name=request.input,
                analysis=result,
                verdict=scores.get("verdict", "CAUTION"),
                scores=scores
            )
        elif request.mode == "analyze":
            result = run_niche_analysis(request.input, memory_context)
            scores = extract_scores(result)
            save_niche_analysis(
                niche_name=request.input,
                analysis=result,
                verdict=scores.get("verdict", "CAUTION")
            )
        elif request.mode == "compare":
            result = run_product_comparison(request.input, memory_context)
            scores = extract_scores(result)
        elif request.mode == "content":
            result = run_content_brief(request.input, memory_context)
            scores = extract_scores(result)
            save_content_brief(product_name=request.input, content=result)
        else:
            raise HTTPException(status_code=400, detail="Invalid mode")

        return {
            "success": True,
            "result": result,
            "scores": scores,
            "memory_used": bool(memory_context)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/stats")
def memory_stats():
    return get_memory_stats()

@app.get("/memory/top-products")
def top_products():
    return {"products": get_top_products(limit=10)}

@app.get("/memory/similar")
def similar_products(query: str):
    return {"similar": get_similar_products(query, limit=5)}
