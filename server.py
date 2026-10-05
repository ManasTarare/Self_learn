"""Pxplore - standalone Flask web app (no Streamlit needed).

Run:  python server.py     then open http://localhost:5000
Needs the same sibling modules as before: learner_store.py, pipeline.py, retriever.py, data/snippets*.jsonl
"""
import json
import os
import pathlib
import secrets
import threading
from datetime import datetime
from functools import wraps

from dotenv import load_dotenv
from flask import (Flask, Response, abort, flash, redirect, render_template, request,
                   session, url_for)
from markupsafe import Markup, escape

load_dotenv()

from learner_store import (  # noqa: E402
    account_exists, add_learning_note, complete_course, create_account, erase_account, get_course,
    get_learning_record, get_profile, init_db, memory_summary, record_assessment,
    save_course, set_learning_style, update_password, update_profile, verify_login,
)
from pipeline import generate_course, grade_assignment, infer_learning_style, plan  # noqa: E402
from retriever import HybridRetriever  # noqa: E402

BASE = pathlib.Path(__file__).parent
DATA_DIR = BASE / "data"
LEVELS = ["beginner", "intermediate", "advanced"]
PROVIDERS = ["gemini", "anthropic", "openai"]
DEFAULT_MODELS = {"gemini": "gemini-3.6-flash", "anthropic": "claude-sonnet-5-5", "openai": "gpt-4o"}

app = Flask(__name__, template_folder=str(BASE / "templates"), static_folder=str(BASE / "static"))
_key_file = BASE / ".secret_key"
if not os.getenv("SECRET_KEY") and not _key_file.exists():
    _key_file.write_text(secrets.token_hex(32))
app.secret_key = os.getenv("SECRET_KEY") or _key_file.read_text().strip()
app.config.update(SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_HTTPONLY=True)
init_db()

# ---------- helpers ----------
_retrievers, _lock = {}, threading.Lock()
LAST = {}  # (user_id, course_id) -> latest quiz/assignment result shown on the course page


def get_retriever(path):
    with _lock:
        if path not in _retrievers:
            _retrievers[path] = HybridRetriever(path)
        return _retrievers[path]


def default_provider():
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    return "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "openai"


def llm_kw():
    provider = session.get("provider") or default_provider()
    model = session.get("model") or os.getenv("PX_MODEL") or DEFAULT_MODELS[provider]
    return {"provider": provider, "model": model}


def catalog_files():
    files = [p.name for p in sorted(DATA_DIR.glob("snippets*.jsonl"))]
    return sorted(files, key=lambda n: (n != "snippets_full.jsonl", n))


def uid():
    return session["user"]["id"]


def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **k):
        if "user" not in session:
            return redirect(url_for("login"))
        return fn(*a, **k)
    return wrapper


@app.context_processor
def inject():
    name = None
    if "user" in session:
        try:
            name = get_profile(uid())["display_name"]
        except Exception:
            session.clear()
    return {"display_name": name, "year": datetime.now().year}


@app.template_filter("md")
def md_filter(text):
    text = str(text or "")
    try:
        import markdown
        html = markdown.markdown(text, extensions=["fenced_code", "tables"])
        try:
            import bleach
            tags = list(bleach.sanitizer.ALLOWED_TAGS) + ["p", "pre", "h1", "h2", "h3", "h4", "table", "thead",
                                                         "tbody", "tr", "th", "td", "br", "hr"]
            html = bleach.clean(html, tags=tags, strip=True)
        except ImportError:  # never show unsanitized HTML: fall back to plain text
            return Markup("<pre class='plain'>%s</pre>" % escape(text))
        return Markup(html)
    except ImportError:
        return Markup("<pre class='plain'>%s</pre>" % escape(text))


def build_graph_data(profile_data, learning_record):
    concepts = learning_record.get("concepts", [])
    courses = learning_record.get("courses", [])
    links = learning_record.get("links", [])
    course_concepts = learning_record.get("course_concepts", [])
    nodes, edges = [], []
    node_ids, edge_ids, concept_ids = set(), set(), {}
    concept_info = {item["name"]: item for item in concepts}

    def add_node(node_id, label, kind, detail="", anchor=None, **extra):
        if node_id in node_ids:
            return
        node_ids.add(node_id)
        nodes.append({"id": node_id, "label": str(label), "kind": kind,
                      "detail": str(detail), "anchor": anchor, **extra})

    def add_edge(source, target, relation=""):
        edge_id = (source, target, relation)
        if source in node_ids and target in node_ids and edge_id not in edge_ids:
            edge_ids.add(edge_id)
            edges.append({"source": source, "target": target, "relation": relation})

    def add_concept(name, anchor=None):
        name = str(name or "").strip()
        if not name:
            return None
        if name not in concept_ids:
            concept_id = f"concept-{len(concept_ids)}"
            concept_ids[name] = concept_id
            data = concept_info.get(name, {})
            mastery = round(float(data.get("mastery", 0.5)) * 100)
            evidence = int(data.get("evidence_count", 0))
            detail = (f"Estimated mastery: {mastery}% · {evidence} assessment "
                      f"evidence" if evidence else "No assessment evidence yet")
            add_node(concept_id, name, "concept", detail, anchor,
                     **({"mastery": max(0.0, min(1.0, mastery / 100))} if evidence else {}))
        return concept_ids[name]

    profile_id = "learner-profile"
    add_node(profile_id, profile_data.get("display_name", "Learner"), "profile",
             "Your learner profile")

    for key, label in (("long_term_goal", "Long-term goal"),
                       ("short_term_goal", "Short-term goal")):
        goal = str(profile_data.get(key) or "").strip()
        if goal:
            goal_id = f"goal-{key}"
            add_node(goal_id, label, "goal", goal, profile_id)
            add_edge(profile_id, goal_id, "goal")

    course_concepts_by_id = {}
    for item in course_concepts:
        course_concepts_by_id.setdefault(item["course_id"], []).append(item["concept"])

    for course in courses:
        course_id = f"course-{course['id']}"
        title = course.get("title", "Course")
        course_detail = " · ".join(filter(None, [
            str(course.get("level", "")).strip(),
            str(course.get("status", "")).replace("_", " ").strip(),
            f"Assessment average: {course['result_pct']}%" if course.get("result_pct") is not None else "",
        ]))
        add_node(course_id, title, "course", course_detail, profile_id)
        add_edge(profile_id, course_id, "studies")

        saved = get_course(course["id"])
        course_data = saved.get("course", {}) if saved else {}
        lessons = course_data.get("lessons", [])
        connected_concepts = set()
        for lesson_index, lesson in enumerate(lessons):
            lesson_id = f"lesson-{course['id']}-{lesson_index}"
            lesson_title = lesson.get("title") or f"Lesson {lesson_index + 1}"
            duration = lesson.get("duration_min")
            lesson_detail = f"{title}" + (f" · {duration} min" if duration else "")
            add_node(lesson_id, lesson_title, "lesson", lesson_detail, course_id)
            add_edge(course_id, lesson_id, "lesson")
            for name in lesson.get("concepts", []) or []:
                concept_id = add_concept(name, lesson_id)
                if concept_id:
                    add_edge(lesson_id, concept_id, "covers")
                    connected_concepts.add(str(name).strip())

        for name in course_concepts_by_id.get(course["id"], []):
            concept_id = add_concept(name, course_id)
            if concept_id and name not in connected_concepts:
                add_edge(course_id, concept_id, "teaches")

    for concept in concepts:
        add_concept(concept.get("name"))
    for relation in links:
        source = add_concept(relation.get("source"))
        target = add_concept(relation.get("target"))
        if source and target:
            add_edge(source, target, relation.get("relation", "related"))

    graph_data = json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False)
    graph_data = graph_data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return graph_data



# ---------- landing + auth ----------
@app.route("/")
def home():
    return render_template("index.html", app_url=url_for("learn"))


@app.route("/health")
def health():
    return {"status": "ok"}


@app.route("/setup", methods=["GET", "POST"])
def setup():
    if account_exists():
        return redirect(url_for("login"))
    if request.method == "POST":
        f = request.form
        if f["password"] != f["confirm"]:
            flash("The passwords do not match.", "error")
        else:
            try:
                create_account(f["username"], f["display_name"], f["password"])
                update_profile(f["display_name"], f.get("long_goal", ""), f.get("short_goal", ""),
                               f.get("level", "beginner"))
                u = verify_login(f["username"], f["password"])
                session["user"] = {"id": u["id"], "username": u["username"]}
                return redirect(url_for("learn"))
            except Exception as exc:
                flash(str(exc), "error")
    return render_template("auth.html", mode="setup", levels=LEVELS)


@app.route("/login", methods=["GET", "POST"])
def login():
    if not account_exists():
        return redirect(url_for("setup"))
    if request.method == "POST":
        u = verify_login(request.form["username"], request.form["password"])
        if u:
            session["user"] = {"id": u["id"], "username": u["username"]}
            return redirect(url_for("learn"))
        flash("Username or password is incorrect.", "error")
    return render_template("auth.html", mode="login")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# ---------- learn ----------
@app.route("/learn")
@login_required
def learn():
    record = get_learning_record(uid())
    courses = record["courses"]
    return render_template(
        "learn.html", active="learn", profile=get_profile(uid()), record=record,
        in_progress=[c for c in courses if c["status"] == "in_progress"],
        completed=[c for c in courses if c["status"] == "completed"],
        files=catalog_files(), providers=PROVIDERS, kw=llm_kw(),
        form=session.pop("learn_form", {}))


@app.route("/learn/generate", methods=["POST"])
@login_required
def generate():
    f = request.form
    kw_provider = f.get("provider") if f.get("provider") in PROVIDERS else default_provider()
    session["provider"], session["model"] = kw_provider, (f.get("model") or "").strip() or None
    kw = llm_kw()
    studied = f.get("studied", "").strip()
    profile_now, memory_now = get_profile(uid()), memory_summary(uid())
    if not profile_now["long_term_goal"] and not profile_now["short_term_goal"]:
        flash("Add at least one learning goal in the Me profile first.", "warn")
        return redirect(url_for("me"))
    files = catalog_files()
    if not files:
        flash("No data/snippets*.jsonl found. Run convert_mslearn.py first.", "error")
        return redirect(url_for("learn"))
    data_file = f.get("data_file") if f.get("data_file") in files else files[0]
    retriever = get_retriever(str(DATA_DIR / data_file))
    retriever.top_k = max(5, min(20, int(f.get("top_k") or 10)))
    retriever.alpha = max(0.0, min(1.0, float(f.get("alpha") or 0.3)))
    profile_for_llm = {
        "level": profile_now["level"], "long_term_goal": profile_now["long_term_goal"],
        "short_term_goal": profile_now["short_term_goal"], "learning_style": profile_now["preferred_style"],
        "learning_style_confidence": profile_now["style_confidence"],
        "completed_courses": memory_now["completed_courses"],
        "exclude_module_ids": memory_now["exclude_module_ids"],
        "priority_topics": [i["concept"] for i in memory_now["priority_gaps"]],
        "concept_mastery": memory_now["concept_mastery"],
        "recent_study_notes": memory_now["recent_study_notes"],
        "recent_assessment_results": memory_now["recent_assessment_results"],
        "assessment_average_pct": memory_now["assessment_average_pct"], "just_studied": studied,
    }
    history = studied
    if memory_now["recent_courses"]:
        recent = "; ".join(f'{c["title"]} ({c["status"]})' for c in memory_now["recent_courses"])
        history = f"Saved recent course history: {recent}. {history}".strip()
    try:
        _, choice = plan(retriever, profile_for_llm, history, **kw)
        selected_id = choice["selected_candidate"]["id"]
        by_id = {d["id"]: d for d in retriever.docs}
        if selected_id not in by_id:
            raise ValueError("The model selected a module outside the catalog.")
        snippet = by_id[selected_id]
        generated = generate_course(snippet, memory_now, choice["reason"], **kw)
        generated["recommendation_reason"] = choice["reason"]
        course_id = save_course(uid(), snippet, generated)
        add_learning_note(uid(), studied)
        flash(f"Recommended module: {snippet['title']}. Your course is ready.", "success")
        return redirect(url_for("course", cid=course_id))
    except Exception as exc:
        flash(f"Could not generate your course: {exc}", "error")
        session["learn_form"] = {"studied": studied}
        return redirect(url_for("learn"))


def _course_or_404(cid):
    row = get_course(cid)
    if not row:
        abort(404)
    return row


@app.route("/course/<int:cid>")
@login_required
def course(cid):
    row = _course_or_404(cid)
    return render_template("course.html", active="learn", row=row, c=row["course"],
                           last=LAST.get((uid(), cid), {}))


def _update_style(course_data, result):
    try:
        style = infer_learning_style(memory_summary(uid()),
                                     course_data.get("learning_approach", "mixed instruction"),
                                     result, **llm_kw())
        if isinstance(style, dict) and style.get("preferred_style"):
            set_learning_style(style["preferred_style"], style.get("confidence", "low"),
                               style.get("evidence", []), uid())
    except Exception as exc:
        flash(f"Assessment saved. Learning-approach inference will retry after another result ({exc}).", "info")


@app.route("/course/<int:cid>/quiz", methods=["POST"])
@login_required
def submit_quiz(cid):
    row = _course_or_404(cid)
    questions = row["course"].get("quiz", {}).get("questions", [])
    picks = [request.form.get(f"q{i}") for i in range(len(questions))]
    if any(p is None for p in picks):
        flash("Answer every question before submitting.", "warn")
        return redirect(url_for("course", cid=cid) + "#quiz")
    details, correct, per_concept = [], 0, {}
    for q, pick in zip(questions, picks):
        options = q.get("options", [])
        idx = int(pick)
        is_correct = idx == int(q.get("answer_index", 0))
        correct += int(is_correct)
        concept = str(q.get("concept", "Course concepts"))
        bucket = per_concept.setdefault(concept, [0, 0])
        bucket[0] += int(is_correct)
        bucket[1] += 1
        details.append({"question": q.get("question", ""), "answer": options[idx] if idx < len(options) else "",
                        "is_correct": is_correct, "concept": concept, "explanation": q.get("explanation", "")})
    result = {"details": details, "correct": correct, "total": len(questions)}
    record_assessment(cid, "quiz", correct, len(questions), result,
                      {k: r / t for k, (r, t) in per_concept.items()}, uid())
    LAST.setdefault((uid(), cid), {})["quiz"] = {"score": correct, "max_score": len(questions), "details": details}
    _update_style(row["course"], {"kind": "quiz", "score": correct, "max_score": len(questions), "details": details})
    return redirect(url_for("course", cid=cid) + "#quiz")


@app.route("/course/<int:cid>/assignment", methods=["POST"])
@login_required
def submit_assignment(cid):
    row = _course_or_404(cid)
    submission = request.form.get("submission", "").strip()
    if not submission:
        flash("Write a response before submitting.", "warn")
        return redirect(url_for("course", cid=cid) + "#assignment")
    try:
        result = grade_assignment(row["course"], submission, memory_summary(uid()), **llm_kw())
        scores = {i["concept"]: float(i.get("score", 0)) / max(1, float(i.get("max_score", 100)))
                  for i in result.get("concept_scores", [])}
        if not scores:
            scores = {n: result["score"] / 100 for n in row["course"].get("concepts", [])}
        result["submission"] = submission
        record_assessment(cid, "assignment", result["score"], 100, result, scores, uid())
        LAST.setdefault((uid(), cid), {})["assignment"] = result
        _update_style(row["course"], {"kind": "assignment", **result})
    except Exception as exc:
        flash(f"Could not grade assignment: {exc}", "error")
    return redirect(url_for("course", cid=cid) + "#assignment")


@app.route("/course/<int:cid>/complete", methods=["POST"])
@login_required
def mark_complete(cid):
    _course_or_404(cid)
    complete_course(cid, uid())
    flash("Course marked complete and added to your learning history.", "success")
    return redirect(url_for("course", cid=cid))


# ---------- me ----------
@app.route("/me")
@login_required
def me():
    record = get_learning_record(uid())
    profile = get_profile(uid())
    evidence = []
    if profile.get("style_evidence"):
        try:
            evidence = json.loads(profile["style_evidence"])
        except (TypeError, json.JSONDecodeError):
            evidence = [profile["style_evidence"]]
    assessments = []
    for a in record["assessments"]:
        try:
            detail = json.loads(a["result_json"])
        except (TypeError, json.JSONDecodeError):
            detail = {}
        assessments.append({**a, "detail": detail})
    return render_template("me.html", active="me", profile=profile, record=record, levels=LEVELS,
                           evidence=evidence, assessments=assessments)


@app.route("/me/profile", methods=["POST"])
@login_required
def save_profile():
    f = request.form
    if not f.get("display_name", "").strip():
        flash("Enter a display name.", "error")
    else:
        update_profile(f["display_name"], f.get("long_goal", ""), f.get("short_goal", ""),
                       f.get("level", "beginner"), uid())
        flash("Profile saved.", "success")
    return redirect(url_for("me"))


@app.route("/me/password", methods=["POST"])
@login_required
def change_password():
    f = request.form
    if f["new"] != f["confirm"]:
        flash("The new passwords do not match.", "error")
    else:
        try:
            update_password(session["user"]["username"], f["old"], f["new"], uid())
            flash("Password updated.", "success")
        except ValueError as exc:
            flash(str(exc), "error")
    return redirect(url_for("me"))


@app.route("/me/erase", methods=["POST"])
@login_required
def erase():
    f = request.form
    if f.get("confirm_text") != "ERASE":
        flash("Type ERASE exactly to confirm.", "error")
    elif not verify_login(session["user"]["username"], f.get("password", "")):
        flash("Password is incorrect.", "error")
    else:
        erase_account()
        session.clear()
        flash("All learner data was erased.", "success")
        return redirect(url_for("home"))
    return redirect(url_for("me"))


@app.route("/graph")
@login_required
def graph():
    raw = (BASE / "templates" / "graph.html").read_text(encoding="utf-8")
    data = build_graph_data(get_profile(uid()), get_learning_record(uid()))
    return Response(raw.replace("__GRAPH_DATA__", data), mimetype="text/html")


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "5000")),
            debug=os.getenv("FLASK_DEBUG", "0") == "1")
