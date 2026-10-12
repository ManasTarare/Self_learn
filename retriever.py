"""In-memory hybrid retriever. BM25 by default; sentence embeddings are opt-in (USE_EMBEDDINGS=1)."""
import hashlib, json, os, pathlib
import numpy as np
from rank_bm25 import BM25Okapi


class HybridRetriever:
    def __init__(self, snippets_path, alpha=0.3, top_k=10, model="BAAI/bge-small-en-v1.5"):
        with open(snippets_path, encoding="utf-8") as f:
            self.docs = [json.loads(l) for l in f]
        texts = [d["index_text"] for d in self.docs]
        self.alpha, self.top_k, self.model_name = alpha, top_k, model
        self.bm25 = BM25Okapi([t.lower().split() for t in texts])

        self.emb = None
        self.vecs = None
        self.use_embeddings = os.getenv("USE_EMBEDDINGS", "0") == "1"
        if self.use_embeddings:
            self._init_embeddings(snippets_path, texts)

    def _get_model(self):
        # lazy import so torch is never loaded unless embeddings are enabled
        if self.emb is None:
            from sentence_transformers import SentenceTransformer
            self.emb = SentenceTransformer(self.model_name, device="cpu")
        return self.emb

    def _init_embeddings(self, snippets_path, texts):
        sig = hashlib.md5((self.model_name + "\n".join(texts)).encode()).hexdigest()
        cache = pathlib.Path(snippets_path).with_suffix(".emb.npz")
        if cache.exists():
            data = np.load(cache)
            if str(data["sig"]) == sig:
                self.vecs = data["vecs"]
                return
        self.vecs = self._get_model().encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=16)
        np.savez(cache, vecs=self.vecs, sig=sig)

    @staticmethod
    def _norm(x):
        x = np.asarray(x, dtype=float)
        return (x - x.min()) / (x.max() - x.min() + 1e-9)

    def search(self, query, exclude_id=None, exclude_ids=None):
        exclude_ids = set(exclude_ids or [])
        if exclude_id is not None:
            exclude_ids.add(exclude_id)

        bm25_scores = self._norm(self.bm25.get_scores(query.lower().split()))
        if self.vecs is not None:
            q = self._get_model().encode(query, normalize_embeddings=True)
            dense_scores = self._norm(self.vecs @ q)
            score = self.alpha * bm25_scores + (1 - self.alpha) * dense_scores
        else:
            score = bm25_scores

        out = []
        for i in np.argsort(-score):
            doc = self.docs[i]
            if doc["id"] in exclude_ids:
                continue
            topics = [u["name"] for u in doc["units"] if u["type"] == "LECTURE"][:8]
            out.append({"score": float(score[i]), "metadata": {
                "id": doc["id"], "title": doc["title"], "description": doc["description"],
                "duration_min": doc["duration_min"], "topics": topics,
                "has_lesson_text": bool(doc.get("text"))}})
            if len(out) == self.top_k:
                break
        return out
