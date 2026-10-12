"""Pxplore - FastAPI web app.

Run:   python server.py            (or: uvicorn server:app --host 0.0.0.0 --port 5000)
Open:  http://localhost:5000
API docs (Swagger UI): http://localhost:5000/docs      ReDoc: http://localhost:5000/redoc

Needs the same sibling modules as before: learner_store.py, pipeline.py, retriever.py, data/snippets*.jsonl
The Jinja templates in templates/ are reused as-is (url_for, get_flashed_messages, session and the
`md` filter are provided below so the Flask-style templates keep working).
"""

import json
import os
import pathlib
import secrets
import threading
from datetime import datetime
from urllib.parse import urlencode

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import pass_context
from markupsafe import Markup, escape
from starlette.concurrency import run_in_threadpool
from starlette.middleware.sessions import SessionMiddleware

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

app = FastAPI(
    title="Pxplore API",
    description="Recommends a Microsoft Learn module, then generates a personalised course with an LLM. "
                "Most endpoints return HTML pages and accept form posts; /health returns JSON. "
                "Endpoints tagged 'Learner' need a logged-in session cookie (use /login first).",
    version="1.0.0",
)

_key_file = BASE / ".secret_key"
if not os.getenv("SECRET_KEY") and not _key_file.exists():
    _key_file.write_text(secrets.token_hex(32))
SECRET_KEY = os.getenv("SECRET_KEY") or _key_file.read_text().strip()

# Signed cookie session (same idea as Flask's session). max_age=None -> cookie lasts for the browser session.
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, same_site="lax", max_age=None, https_only=False)
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE / "templates"))

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


def llm_kw(session):
    provider = session.get("provider") or default_provider()
    model = session.get("model") or os.getenv("PX_MODEL") or DEFAULT_MODELS[provider]
    return {"provider": provider, "model": model}


def catalog_files():
    files = [p.name for p in sorted(DATA_DIR.glob("snippets*.jsonl"))]
    return sorted(files, key=lambda n: (n != "snippets_full.jsonl", n))


def uid(request: Request):
    return request.session["user"]["id"]


def _num(value, cast, default):
    """Parse an optional numeric form value, falling back to the default if blank or invalid."""
    try:
        return cast(value) if str(value).strip() else default
    except ValueError:
        return default


# ---------- Flask-compatible helpers so the existing Jinja templates keep working ----------

def redirect(url: str) -> RedirectResponse:
    # 303 so the browser follows the redirect with GET (FastAPI's default 307 would re-send the POST).
    return RedirectResponse(url, status_code=303)


def url_for(endpoint: str, **values) -> str:
    """Flask-style url_for(endpoint, **values) -> path. Extra keyword args become the query string."""
    values.pop("_external", None)
    anchor = values.pop("_anchor", None)
    if endpoint == "static" and "filename" in values:
        values["path"] = values.pop("filename")
    route = next((r for r in app.routes if getattr(r, "name", None) == endpoint), None)
    if route is None:
        raise KeyError(f"No route named {endpoint!r}")
    path_params = set(getattr(route, "param_convertors", {}))
    path_values = {k: v for k, v in values.items() if k in path_params}
    query = {k: v for k, v in values.items() if k not in path_params and v is not None}
    url = str(app.url_path_for(endpoint, **path_values))
    if query:
        url += "?" + urlencode(query)
    return url + (f"#{anchor}" if anchor else "")


def flash(request: Request, message, category: str = "message"):
    # Assign a new list (not append in place) so Starlette sees the session changed and re-sends the cookie.
    request.session["_flashes"] = [*request.session.get("_flashes", []), [category, str(message)[:500]]]


@pass_context
def get_flashed_messages(context, with_categories=False, category_filter=()):
    request = context["request"]
    flashes = getattr(request.state, "_flashes", None)
    if flashes is None:
        flashes = request.session.pop("_flashes", [])
        request.state._flashes = flashes
    if category_filter:
        flashes = [f for f in flashes if f[0] in category_filter]
    return flashes if with_categories else [m for _, m in flashes]


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


templates.env.globals["url_for"] = url_for
templates.env.globals["get_flashed_messages"] = get_flashed_messages
templates.env.filters["md"] = md_filter


def render(request: Request, name: str, **ctx):
    """Render a template with the values every page used to get from Flask's context processor."""
    display_name = None
    if "user" in request.session:
        try:
            display_name = get_profile(uid(request))["display_name"]
        except Exception:
            request.session.clear()
    ctx.update(display_name=display_name, year=datetime.now().year, session=request.session)
    return templates.TemplateResponse(request, name, ctx)


# ---------- login protection ----------

class LoginRequired(Exception):
    pass


def require_login(request: Request):
    if "user" not in request.session:
        raise LoginRequired()


@app.exception_handler(LoginRequired)
def _redirect_to_login(request: Request, exc: LoginRequired):
    return redirect(url_for("login"))


LOGIN = [Depends(require_login)]


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

@app.get("/", response_class=HTMLResponse, name="home", tags=["Public"], summary="Landing page")
def home(request: Request):
    return render(request, "index.html", app_url=url_for("learn"))


@app.get("/health", name="health", tags=["Public"], summary="Health check")
def health():
    return {"status": "ok"}


@app.get("/setup", response_class=HTMLResponse, name="setup", tags=["Auth"],
         summary="Account setup page")
def setup_page(request: Request):
    if account_exists():
        return redirect(url_for("login"))
    return render(request, "auth.html", mode="setup", levels=LEVELS)


@app.post("/setup", response_class=HTMLResponse, name="setup", tags=["Auth"],
          summary="Create the local learner account and profile")
def setup_submit(request: Request, username: str = Form(...), display_name: str = Form(...),
                 password: str = Form(...), confirm: str = Form(...), long_goal: str = Form(""),
                 short_goal: str = Form(""), level: str = Form("beginner")):
    if account_exists():
        return redirect(url_for("login"))
    if password != confirm:
        flash(request, "The passwords do not match.", "error")
    else:
        try:
            create_account(username, display_name, password)
            update_profile(display_name, long_goal, short_goal, level)
            u = verify_login(username, password)
            request.session["user"] = {"id": u["id"], "username": u["username"]}
            return redirect(url_for("learn"))
        except Exception as exc:
            flash(request, str(exc), "error")
    return render(request, "auth.html", mode="setup", levels=LEVELS)


@app.get("/login", response_class=HTMLResponse, name="login", tags=["Auth"], summary="Login page")
def login_page(request: Request):
    if not account_exists():
        return redirect(url_for("setup"))
    return render(request, "auth.html", mode="login")


@app.post("/login", response_class=HTMLResponse, name="login", tags=["Auth"],
          summary="Log in (sets the session cookie)")
def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    if not account_exists():
        return redirect(url_for("setup"))
    u = verify_login(username, password)
    if u:
        request.session["user"] = {"id": u["id"], "username": u["username"]}
        return redirect(url_for("learn"))
    flash(request, "Username or password is incorrect.", "error")
    return render(request, "auth.html", mode="login")


@app.get("/logout", name="logout", tags=["Auth"], summary="Log out and clear the session")
def logout(request: Request):
    request.session.clear()
    return redirect(url_for("home"))


# ---------- learn ----------

@app.get("/learn", response_class=HTMLResponse, name="learn", tags=["Learner"],
         summary="Main learning page", dependencies=LOGIN)
def learn(request: Request):
    user_id = uid(request)
    record = get_learning_record(user_id)
    courses = record["courses"]
    return render(
        request, "learn.html", active="learn", profile=get_profile(user_id), record=record,
        in_progress=[c for c in courses if c["status"] == "in_progress"],
        completed=[c for c in courses if c["status"] == "completed"],
        files=catalog_files(), providers=PROVIDERS, kw=llm_kw(request.session),
        form=request.session.pop("learn_form", {}))


@app.post("/learn/generate", name="generate", tags=["Learner"], dependencies=LOGIN,
          summary="Recommend a Microsoft Learn module and generate a course with the LLM")
def generate(request: Request, provider: str = Form(""), model: str = Form(""),
             studied: str = Form(""), data_file: str = Form(""), top_k: str = Form(""),
             alpha: str = Form("")):
    session = request.session
    user_id = uid(request)
    session["provider"] = provider if provider in PROVIDERS else default_provider()
    session["model"] = model.strip() or None
    kw = llm_kw(session)
    studied = studied.strip()
    profile_now, memory_now = get_profile(user_id), memory_summary(user_id)
    if not profile_now["long_term_goal"] and not profile_now["short_term_goal"]:
        flash(request, "Add at least one learning goal in the Me profile first.", "warn")
        return redirect(url_for("me"))
    files = catalog_files()
    if not files:
        flash(request, "No data/snippets*.jsonl found. Run convert_mslearn.py first.", "error")
        return redirect(url_for("learn"))
    chosen_file = data_file if data_file in files else files[0]
    retriever = get_retriever(str(DATA_DIR / chosen_file))
    retriever.top_k = max(5, min(20, _num(top_k, int, 10)))
    retriever.alpha = max(0.0, min(1.0, _num(alpha, float, 0.3)))
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
        course_id = save_course(user_id, snippet, generated)
        add_learning_note(user_id, studied)
        flash(request, f"Recommended module: {snippet['title']}. Your course is ready.", "success")
        return redirect(url_for("course", cid=course_id))
    except Exception as exc:
        flash(request, f"Could not generate your course: {exc}", "error")
        session["learn_form"] = {"studied": studied[:500]}
        return redirect(url_for("learn"))


def _course_or_404(cid):
    row = get_course(cid)
    if not row:
        raise HTTPException(status_code=404, detail="Course not found")
    return row


@app.get("/course/{cid}", response_class=HTMLResponse, name="course", tags=["Learner"],
         summary="Show a generated course", dependencies=LOGIN)
def course(request: Request, cid: int):
    row = _course_or_404(cid)
    return render(request, "course.html", active="learn", row=row, c=row["course"],
                  last=LAST.get((uid(request), cid), {}))


def _update_style(request: Request, course_data, result):
    try:
        style = infer_learning_style(memory_summary(uid(request)),
                                     course_data.get("learning_approach", "mixed instruction"),
                                     result, **llm_kw(request.session))
        if isinstance(style, dict) and style.get("preferred_style"):
            set_learning_style(style["preferred_style"], style.get("confidence", "low"),
                               style.get("evidence", []), uid(request))
    except Exception as exc:
        flash(request, f"Assessment saved. Learning-approach inference will retry after another result ({exc}).", "info")


def _grade_quiz(request: Request, cid: int, form: dict):
    row = _course_or_404(cid)
    user_id = uid(request)
    questions = row["course"].get("quiz", {}).get("questions", [])
    picks = [form.get(f"q{i}") for i in range(len(questions))]
    if any(p is None or not str(p).isdigit() for p in picks):
        flash(request, "Answer every question before submitting.", "warn")
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
                      {k: r / t for k, (r, t) in per_concept.items()}, user_id)
    LAST.setdefault((user_id, cid), {})["quiz"] = {"score": correct, "max_score": len(questions), "details": details}
    _update_style(request, row["course"], {"kind": "quiz", "score": correct, "max_score": len(questions), "details": details})
    return redirect(url_for("course", cid=cid) + "#quiz")


_QUIZ_FORM_DOC = {"requestBody": {"required": True, "content": {"application/x-www-form-urlencoded": {
    "schema": {"type": "object", "additionalProperties": {"type": "string"},
               "description": "One field per question: q0, q1, q2 ... Each value is the chosen option index."},
    "example": {"q0": "1", "q1": "0", "q2": "3"}}}}}


@app.post("/course/{cid}/quiz", name="submit_quiz", tags=["Learner"], dependencies=LOGIN,
          summary="Submit quiz answers (q0, q1, ... = chosen option index)",
          openapi_extra=_QUIZ_FORM_DOC)
async def submit_quiz(request: Request, cid: int):
    form = dict(await request.form())
    return await run_in_threadpool(_grade_quiz, request, cid, form)


@app.post("/course/{cid}/assignment", name="submit_assignment", tags=["Learner"], dependencies=LOGIN,
          summary="Submit an assignment for LLM grading")
def submit_assignment(request: Request, cid: int, submission: str = Form("")):
    row = _course_or_404(cid)
    user_id = uid(request)
    submission = submission.strip()
    if not submission:
        flash(request, "Write a response before submitting.", "warn")
        return redirect(url_for("course", cid=cid) + "#assignment")
    try:
        result = grade_assignment(row["course"], submission, memory_summary(user_id), **llm_kw(request.session))
        scores = {i["concept"]: float(i.get("score", 0)) / max(1, float(i.get("max_score", 100)))
                  for i in result.get("concept_scores", [])}
        if not scores:
            scores = {n: result["score"] / 100 for n in row["course"].get("concepts", [])}
        result["submission"] = submission
        record_assessment(cid, "assignment", result["score"], 100, result, scores, user_id)
        LAST.setdefault((user_id, cid), {})["assignment"] = result
        _update_style(request, row["course"], {"kind": "assignment", **result})
    except Exception as exc:
        flash(request, f"Could not grade assignment: {exc}", "error")
    return redirect(url_for("course", cid=cid) + "#assignment")


@app.post("/course/{cid}/complete", name="mark_complete", tags=["Learner"], dependencies=LOGIN,
          summary="Mark a course as complete")
def mark_complete(request: Request, cid: int):
    _course_or_404(cid)
    complete_course(cid, uid(request))
    flash(request, "Course marked complete and added to your learning history.", "success")
    return redirect(url_for("course", cid=cid))


# ---------- me ----------

@app.get("/me", response_class=HTMLResponse, name="me", tags=["Learner"],
         summary="Profile, learning record and assessments", dependencies=LOGIN)
def me(request: Request):
    user_id = uid(request)
    record = get_learning_record(user_id)
    profile = get_profile(user_id)
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
    return render(request, "me.html", active="me", profile=profile, record=record, levels=LEVELS,
                  evidence=evidence, assessments=assessments)


@app.post("/me/profile", name="save_profile", tags=["Learner"], dependencies=LOGIN,
          summary="Update display name, goals and level")
def save_profile(request: Request, display_name: str = Form(""), long_goal: str = Form(""),
                 short_goal: str = Form(""), level: str = Form("beginner")):
    if not display_name.strip():
        flash(request, "Enter a display name.", "error")
    else:
        update_profile(display_name, long_goal, short_goal, level, uid(request))
        flash(request, "Profile saved.", "success")
    return redirect(url_for("me"))


@app.post("/me/password", name="change_password", tags=["Learner"], dependencies=LOGIN,
          summary="Change the password")
def change_password(request: Request, old: str = Form(...), new: str = Form(...), confirm: str = Form(...)):
    if new != confirm:
        flash(request, "The new passwords do not match.", "error")
    else:
        try:
            update_password(request.session["user"]["username"], old, new, uid(request))
            flash(request, "Password updated.", "success")
        except ValueError as exc:
            flash(request, str(exc), "error")
    return redirect(url_for("me"))


@app.post("/me/erase", name="erase", tags=["Learner"], dependencies=LOGIN,
          summary="Erase the profile and all learning data (type ERASE and enter the password)")
def erase(request: Request, confirm_text: str = Form(""), password: str = Form("")):
    if confirm_text != "ERASE":
        flash(request, "Type ERASE exactly to confirm.", "error")
    elif not verify_login(request.session["user"]["username"], password):
        flash(request, "Password is incorrect.", "error")
    else:
        erase_account()
        request.session.clear()
        flash(request, "All learner data was erased.", "success")
        return redirect(url_for("home"))
    return redirect(url_for("me"))


@app.get("/graph", response_class=HTMLResponse, name="graph", tags=["Learner"],
         summary="Knowledge graph page", dependencies=LOGIN)
def graph(request: Request):
    raw = (BASE / "templates" / "graph.html").read_text(encoding="utf-8")
    data = build_graph_data(get_profile(uid(request)), get_learning_record(uid(request)))
    return HTMLResponse(raw.replace("__GRAPH_DATA__", data))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "5000")),
                reload=os.getenv("PX_RELOAD", "0") == "1")
