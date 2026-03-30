from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from datetime import datetime
import os
import json

# ============================================================
# EMBEDDING MODEL (free, runs locally)
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# ============================================================
# VECTOR STORES (separate collections for different memory types)
# ============================================================

CHROMA_PATH = "./chroma_db"

def get_product_store():
    """Store for researched products"""
    return Chroma(
        collection_name="product_research",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

def get_niche_store():
    """Store for analyzed niches"""
    return Chroma(
        collection_name="niche_analysis",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

def get_content_store():
    """Store for generated content packages"""
    return Chroma(
        collection_name="content_briefs",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

# ============================================================
# SAVE FUNCTIONS
# ============================================================

def save_product_research(product_name: str, analysis: str, verdict: str, scores: dict):
    """Save a product research result to memory"""
    store = get_product_store()
    
    doc = Document(
        page_content=analysis,
        metadata={
            "product": product_name,
            "verdict": verdict,
            "demand": scores.get("demand", 0),
            "margin": scores.get("margin", 0),
            "competition": scores.get("competition", "MED"),
            "trend": scores.get("trend", 0),
            "date": datetime.now().isoformat(),
            "type": "product_research"
        }
    )
    
    store.add_documents([doc])
    print(f"✅ Saved product research: {product_name}")

def save_niche_analysis(niche_name: str, analysis: str, verdict: str):
    """Save a niche analysis to memory"""
    store = get_niche_store()
    
    doc = Document(
        page_content=analysis,
        metadata={
            "niche": niche_name,
            "verdict": verdict,
            "date": datetime.now().isoformat(),
            "type": "niche_analysis"
        }
    )
    
    store.add_documents([doc])
    print(f"✅ Saved niche analysis: {niche_name}")

def save_content_brief(product_name: str, content: str):
    """Save a content brief to memory"""
    store = get_content_store()
    
    doc = Document(
        page_content=content,
        metadata={
            "product": product_name,
            "date": datetime.now().isoformat(),
            "type": "content_brief"
        }
    )
    
    store.add_documents([doc])
    print(f"✅ Saved content brief: {product_name}")

# ============================================================
# RETRIEVE FUNCTIONS
# ============================================================

def get_similar_products(query: str, limit: int = 3) -> list:
    """Find previously researched similar products"""
    try:
        store = get_product_store()
        results = store.similarity_search(query, k=limit)
        return [
            {
                "product": r.metadata.get("product"),
                "verdict": r.metadata.get("verdict"),
                "demand": r.metadata.get("demand"),
                "margin": r.metadata.get("margin"),
                "date": r.metadata.get("date"),
                "summary": r.page_content[:300]
            }
            for r in results
        ]
    except Exception:
        return []

def get_similar_niches(query: str, limit: int = 3) -> list:
    """Find previously analyzed similar niches"""
    try:
        store = get_niche_store()
        results = store.similarity_search(query, k=limit)
        return [
            {
                "niche": r.metadata.get("niche"),
                "verdict": r.metadata.get("verdict"),
                "date": r.metadata.get("date"),
                "summary": r.page_content[:300]
            }
            for r in results
        ]
    except Exception:
        return []

def get_all_products(verdict_filter: str = None) -> list:
    """Get all researched products, optionally filtered by verdict"""
    try:
        store = get_product_store()
        results = store.get()
        
        products = []
        for i, metadata in enumerate(results["metadatas"]):
            if verdict_filter and metadata.get("verdict") != verdict_filter:
                continue
            products.append({
                "product": metadata.get("product"),
                "verdict": metadata.get("verdict"),
                "demand": metadata.get("demand"),
                "margin": metadata.get("margin"),
                "competition": metadata.get("competition"),
                "trend": metadata.get("trend"),
                "date": metadata.get("date")
            })
        
        return sorted(products, key=lambda x: x.get("demand", 0), reverse=True)
    except Exception:
        return []

def get_top_products(limit: int = 5) -> list:
    """Get top GO products by demand score"""
    all_products = get_all_products(verdict_filter="GO")
    return all_products[:limit]

def build_memory_context(query: str) -> str:
    """Build a memory context string to inject into agent prompts"""
    similar = get_similar_products(query, limit=3)
    
    if not similar:
        return ""
    
    context = "\n\n📚 MEMORY CONTEXT — Previously researched similar products:\n"
    for p in similar:
        context += f"""
- {p['product']} | Verdict: {p['verdict']} | Demand: {p['demand']} | Margin: {p['margin']}%
  Summary: {p['summary'][:150]}...
"""
    context += "\nUse this context to inform your analysis but still evaluate the new product independently.\n"
    return context

# ============================================================
# MEMORY STATS
# ============================================================

def get_memory_stats() -> dict:
    """Get overview of what's stored in memory"""
    try:
        product_store = get_product_store()
        niche_store = get_niche_store()
        content_store = get_content_store()
        
        products = product_store.get()
        niches = niche_store.get()
        contents = content_store.get()
        
        go_count = sum(1 for m in products["metadatas"] if m.get("verdict") == "GO")
        caution_count = sum(1 for m in products["metadatas"] if m.get("verdict") == "CAUTION")
        skip_count = sum(1 for m in products["metadatas"] if m.get("verdict") == "SKIP")
        
        return {
            "total_products_researched": len(products["ids"]),
            "total_niches_analyzed": len(niches["ids"]),
            "total_content_briefs": len(contents["ids"]),
            "verdicts": {
                "GO": go_count,
                "CAUTION": caution_count,
                "SKIP": skip_count
            }
        }
    except Exception:
        return {
            "total_products_researched": 0,
            "total_niches_analyzed": 0,
            "total_content_briefs": 0,
            "verdicts": {"GO": 0, "CAUTION": 0, "SKIP": 0}
        }
