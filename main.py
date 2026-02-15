import os
import json
import time
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Semantic Search API")

# Load reviews
REVIEWS = []
try:
    with open("reviews.json", "r") as f:
        REVIEWS = json.load(f)
    logger.info(f"Loaded {len(REVIEWS)} reviews.")
except FileNotFoundError:
    logger.error("reviews.json not found!")

# --- Embedding & Search Logic ---

class SearchSystem:
    def __init__(self):
        self.use_openai = bool(os.environ.get("OPENAI_API_KEY"))
        self.vectorizer = None
        self.tfidf_matrix = None
        self.openai_client = None
        
        # Simple cache for document embeddings to avoid recomputing on startup if using API
        # In a real app, use a vector DB or persistent cache
        self.doc_embeddings = {} 

        if self.use_openai:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI()
                logger.info("Using OpenAI for embeddings.")
            except ImportError:
                logger.warning("openai module not found, falling back to TF-IDF")
                self.use_openai = False
        
        if not self.use_openai:
            logger.info("Using TF-IDF for embeddings (Fallback).")
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.vectorizer = TfidfVectorizer(stop_words='english')
                if REVIEWS:
                    self.tfidf_matrix = self.vectorizer.fit_transform([r['content'] for r in REVIEWS])
            except ImportError:
                logger.error("scikit-learn not found! Search will fail.")

    def get_embedding(self, text: str):
        if self.use_openai and self.openai_client:
            try:
                response = self.openai_client.embeddings.create(
                    input=text,
                    model="text-embedding-3-small"
                )
                return response.data[0].embedding
            except Exception as e:
                logger.error(f"OpenAI Embedding Error: {e}")
                return []
        return None

    def search(self, query: str, k: int = 10):
        results = []
        if self.use_openai:
            query_vec = self.get_embedding(query)
            if not query_vec:
                return []
            
            # Compute cosine similarity manually for this small dataset
            # (In production, use FAISS/Chroma)
            query_vec = np.array(query_vec)
            norm_q = np.linalg.norm(query_vec)
            
            for doc in REVIEWS:
                # Naively recompute doc embedding if not cached (very slow for real usage)
                # Ideally, we cache these. For assignment 59 docs, we can try caching or just compute.
                # BUT computing 59 embeddings per request is too slow (>200ms).
                # So we MUST cache or pre-compute.
                if doc['id'] not in self.doc_embeddings:
                     self.doc_embeddings[doc['id']] = self.get_embedding(doc['content'])
                
                doc_vec = np.array(self.doc_embeddings[doc['id']])
                norm_d = np.linalg.norm(doc_vec)
                
                if norm_q == 0 or norm_d == 0:
                    score = 0.0
                else:
                    score = np.dot(query_vec, doc_vec) / (norm_q * norm_d)
                    
                results.append({
                    "id": doc['id'],
                    "score": float(score),
                    "content": doc['content'],
                    "metadata": {"source": doc.get('source')}
                })
        
        elif self.vectorizer:
            # TF-IDF Search
            from sklearn.metrics.pairwise import cosine_similarity
            query_vec = self.vectorizer.transform([query])
            cosine_sim = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get top k indices
            top_k_indices = cosine_sim.argsort()[::-1][:k]
            for idx in top_k_indices:
                results.append({
                    "id": REVIEWS[idx]['id'],
                    "score": float(cosine_sim[idx]),
                    "content": REVIEWS[idx]['content'],
                    "metadata": {"source": REVIEWS[idx].get('source')}
                })
        
        # Sort and return top k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]

    def rerank(self, query: str, candidates: List[Dict], k: int = 6):
        if self.use_openai and self.openai_client:
            reranked = []
            for doc in candidates:
                try:
                    # Simple re-ranking logic using LLM
                    prompt = f"Query: {query}\nDocument: {doc['content']}\n\nRate relevance 0-10. Return number only."
                    response = self.openai_client.chat.completions.create(
                        model="gpt-3.5-turbo", # or gpt-4o-mini
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=5
                    )
                    score_str = response.choices[0].message.content.strip()
                    try:
                        new_score = float(score_str) / 10.0
                    except ValueError:
                        new_score = doc['score'] # Keep original if parse fails
                    
                    doc['score'] = new_score
                    reranked.append(doc)
                except Exception:
                    reranked.append(doc)
            
            reranked.sort(key=lambda x: x['score'], reverse=True)
            return reranked[:k]
        
        return candidates[:k]

search_system = SearchSystem()

class SearchRequest(BaseModel):
    query: str
    k: int = 10
    rerank: bool = True
    rerankK: int = 6

@app.post("/search")
async def search_endpoint(req: SearchRequest):
    start_time = time.time()
    
    results = search_system.search(req.query, k=req.k)
    
    if req.rerank:
        results = search_system.rerank(req.query, results, k=req.rerankK)
    
    latency = (time.time() - start_time) * 1000
    
    return {
        "results": results,
        "reranked": req.rerank,
        "metrics": {
            "latency": round(latency, 2),
            "totalDocs": len(REVIEWS)
        }
    }

@app.get("/")
def home():
    return {"message": "Semantic Search API Running"}
