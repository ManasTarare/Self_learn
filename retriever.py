"""In-memory hybrid retriever (same idea as service/scripts/hybrid_retriever.py, no Qdrant/embed server)."""
import hashlib, json, pathlib, numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

class HybridRetriever:
    def __init__(self, snippets_path, alpha=0.3, top_k=10, model="BAAI/bge-small-en-v1.5"):
        self.docs = [json.loads(l) for l in open(snippets_path, encoding="utf-8")]
        texts = [d["index_text"] for d in self.docs]
        self.alpha, self.top_k = alpha, top_k
        self.bm25 = BM25Okapi([t.lower().split() for t in texts])
        self.emb = SentenceTransformer(model)
        # cache embeddings on disk so reruns start instantly (invalidated if texts or model change)
        sig = hashlib.md5((model + "\n".join(texts)).encode()).hexdigest()
        cache = pathlib.Path(snippets_path).with_suffix(".emb.npz")
        if cache.exists() and str(np.load(cache)["sig"]) == sig:
            self.vecs = np.load(cache)["vecs"]
        else:
            self.vecs = self.emb.encode(texts, normalize_embeddings=True, show_progress_bar=True)
            np.savez(cache, vecs=self.vecs, sig=sig)

    @staticmethod
    def _norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-9)   # original mixed raw BM25 with cosine: fix

    def search(self, query, exclude_id=None, exclude_ids=None):
        exclude_ids = set(exclude_ids or [])
        b = self._norm(self.bm25.get_scores(query.lower().split()))
        d = self._norm(self.vecs @ self.emb.encode(query, normalize_embeddings=True))
        score = self.alpha * b + (1 - self.alpha) * d
        out = []
        for i in np.argsort(-score):
            if self.docs[i]["id"] == exclude_id or self.docs[i]["id"] in exclude_ids: continue
            d = self.docs[i]
            topics = [u["name"] for u in d["units"] if u["type"] == "LECTURE"][:8]
            out.append({"score": float(score[i]), "metadata": {
                "id": d["id"], "title": d["title"], "description": d["description"],
                "duration_min": d["duration_min"], "topics": topics, "has_lesson_text": bool(d.get("text"))}})
            if len(out) == self.top_k: break
        return out

