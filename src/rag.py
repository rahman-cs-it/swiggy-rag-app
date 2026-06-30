"""
Core RAG pipeline with hybrid Multi-API Support (Ollama & Cloud).
  1. Embed the user's question with a local SentenceTransformer model.
  2. Retrieve the top-k most similar chunks from the FAISS index.
  3. Route the structured context prompt to either Ollama or a Cloud provider
     using LiteLLM based on your .env configuration.
"""

import os
import pickle
import faiss
from litellm import completion
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "chunks.pkl")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Pull model from env. Defaulting to local ollama/llama3.1 if not specified.
LLM_MODEL = os.environ.get("LLM_MODEL", "ollama/llama3.1")
TOP_K = 5

SYSTEM_PROMPT = """You are a precise financial document assistant.
Answer the user's question using ONLY the information in the provided
context, which is extracted from Swiggy's Annual Report.

Rules:
- If the answer is not contained in the context, respond exactly:
  "I could not find this information in the Swiggy Annual Report."
- Do not use any outside knowledge or make assumptions.
- Be concise and cite page numbers from the context when possible.
"""


class SwiggyRAG:
    def __init__(self):
        if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
            raise FileNotFoundError(
                "Vector store not found. Run `python src/ingest.py` first."
            )
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            self.chunks = pickle.load(f)
            
        print(f"-> SwiggyRAG initialized utilizing LLM: {LLM_MODEL}")

    def retrieve(self, query: str, top_k: int = TOP_K):
        query_vec = self.embed_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "page_number": chunk["page_number"],
                "score": float(score),
            })
        return results

    def build_prompt(self, query: str, contexts: list) -> str:
        context_block = "\n\n".join(
            f"[Page {c['page_number']}]\n{c['text']}" for c in contexts
        )
        return (
            f"Context from the Swiggy Annual Report:\n\n{context_block}\n\n"
            f"Question: {query}\n\nAnswer:"
        )

    def answer(self, query: str, top_k: int = TOP_K):
        contexts = self.retrieve(query, top_k=top_k)
        prompt = self.build_prompt(query, contexts)

        # litellm structural abstraction handles all providers behind the scenes
        response = completion(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        
        answer_text = response.choices[0].message.content
        return {
            "answer": answer_text,
            "contexts": contexts,
        }