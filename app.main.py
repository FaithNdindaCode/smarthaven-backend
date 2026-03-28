from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents import run_product_research, run_niche_analysis, run_product_comparison, run_content_brief
import os

app = FastAPI(
    title="SmartHaven AI Backend",
    description="Multi-agent product research system for SmartHaven Digital",
    version="1.0.0"
)

# Allow requests from your Netlify frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request models ---
class ProductRequest(BaseModel):
    input: str
    mode: str  # research | analyze | compare | content

class HealthResponse(BaseModel):
    status: str
    message: str

# --- Routes ---
@app.get("/", response_model=HealthResponse)
def root():
    return {"status": "online", "message": "SmartHaven AI Backend is running"}

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "healthy", "message": "All agents ready"}

@app.post("/analyze")
async def analyze(request: ProductRequest):
    try:
        if request.mode == "research":
            result = run_product_research(request.input)
        elif request.mode == "analyze":
            result = run_niche_analysis(request.input)
        elif request.mode == "compare":
            result = run_product_comparison(request.input)
        elif request.mode == "content":
            result = run_content_brief(request.input)
        else:
            raise HTTPException(status_code=400, detail="Invalid mode")

        return {"success": True, "result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
