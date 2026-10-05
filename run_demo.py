"""End-to-end demo: retrieve -> plan (LLM picks next module) -> adapt (LLM rewrites intro/outro).
Usage: python run_demo.py [data/snippets_full.jsonl]   (falls back to data/snippets.jsonl)
"""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows console: allow non-English output
from dotenv import load_dotenv
load_dotenv()   # reads .env in this folder
from retriever import HybridRetriever
from pipeline import plan, adapt

path = sys.argv[1] if len(sys.argv) > 1 else "data/snippets.jsonl"
if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
    provider, model = "gemini", os.getenv("PX_MODEL", "gemini-2.5-flash")
elif os.getenv("ANTHROPIC_API_KEY"):
    provider, model = "anthropic", os.getenv("PX_MODEL", "claude-sonnet-5-5")
else:
    provider, model = "openai", os.getenv("PX_MODEL", "gpt-4o")
kw = dict(provider=provider, model=model)

retriever = HybridRetriever(path)
by_id = {d["id"]: d for d in retriever.docs}

student_profile = {
    "level": "beginner",
    "long_term_goal": "deploy machine learning models on Azure",
    "short_term_goal": "understand how to track and compare experiments",
    "learning_style": "prefers short hands-on exercises",
}
history = "Student just finished an introduction to Azure Machine Learning and asked about comparing models."

cands, choice = plan(retriever, student_profile, history, **kw)
print("\nTOP CANDIDATES:")
for c in cands[:5]:
    print(f"  {c['score']:.3f}  {c['metadata']['title']}")

sel = choice["selected_candidate"]
print("\nLLM PICKED:", sel.get("id"), "\nREASON:", choice["reason"])

snippet = by_id[sel["id"]]
result = adapt(history, snippet, choice["reason"], **kw)
print("\nSUGGESTION:\n", result["suggestion"])
print("\nSTART:", result["start_speech"], "\nEND:  ", result["end_speech"])
