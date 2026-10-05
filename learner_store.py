"""Local, single-learner SQLite store for the Pxplore test profile."""
import hashlib
import hmac
import json
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "student_memory.db"
PBKDF2_ROUNDS = 310_000


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA secure_delete = ON")
    return conn


def init_db():
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS user_account (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            username TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL,
            salt BLOB NOT NULL,
            password_hash BLOB NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS learner_profile (
            user_id INTEGER PRIMARY KEY REFERENCES user_account(id) ON DELETE CASCADE,
            long_term_goal TEXT NOT NULL DEFAULT '',
            short_term_goal TEXT NOT NULL DEFAULT '',
            level TEXT NOT NULL DEFAULT 'beginner',
            preferred_style TEXT NOT NULL DEFAULT 'Not enough evidence yet',
            style_confidence TEXT NOT NULL DEFAULT 'low',
            style_evidence TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES user_account(id) ON DELETE CASCADE,
            source_module_id TEXT NOT NULL,
            source_title TEXT NOT NULL,
            source_url TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL,
            level TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'in_progress',
            course_json TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS learning_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES user_account(id) ON DELETE CASCADE,
            note TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
            kind TEXT NOT NULL,
            score REAL NOT NULL,
            max_score REAL NOT NULL,
            result_json TEXT NOT NULL,
            submitted_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS concepts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS course_concepts (
            course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
            concept_id INTEGER NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
            PRIMARY KEY(course_id, concept_id)
        );
        CREATE TABLE IF NOT EXISTS concept_links (
            from_concept_id INTEGER NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
            to_concept_id INTEGER NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
            relation TEXT NOT NULL,
            PRIMARY KEY(from_concept_id, to_concept_id, relation)
        );
        CREATE TABLE IF NOT EXISTS learner_concepts (
            user_id INTEGER NOT NULL REFERENCES user_account(id) ON DELETE CASCADE,
            concept_id INTEGER NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
            mastery REAL NOT NULL DEFAULT 0.5,
            evidence_count INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(user_id, concept_id)
        );
        """)


def account_exists():
    with connect() as db:
        return db.execute("SELECT 1 FROM user_account WHERE id=1").fetchone() is not None


def _hash_password(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)


def create_account(username, display_name, password):
    username = username.strip()
    display_name = display_name.strip()
    if not username or not display_name:
        raise ValueError("Enter both a username and display name.")
    if len(password) < 8:
        raise ValueError("Use a password with at least 8 characters.")
    salt = secrets.token_bytes(16)
    digest = _hash_password(password, salt)
    with connect() as db:
        if db.execute("SELECT 1 FROM user_account").fetchone():
            raise ValueError("A local profile already exists.")
        db.execute(
            "INSERT INTO user_account(id,username,display_name,salt,password_hash,created_at) VALUES(1,?,?,?,?,?)",
            (username, display_name, salt, digest, _now()))
        db.execute("INSERT INTO learner_profile(user_id,updated_at) VALUES(1,?)", (_now(),))


def verify_login(username, password):
    with connect() as db:
        row = db.execute(
            "SELECT id,username,display_name,salt,password_hash FROM user_account WHERE username=?",
            (username.strip(),)).fetchone()
    if not row:
        return None
    expected = _hash_password(password, row["salt"])
    if not hmac.compare_digest(expected, row["password_hash"]):
        return None
    return {"id": row["id"], "username": row["username"], "display_name": row["display_name"]}


def get_profile(user_id=1):
    with connect() as db:
        row = db.execute("""
            SELECT a.username,a.display_name,p.long_term_goal,p.short_term_goal,p.level,
                   p.preferred_style,p.style_confidence,p.style_evidence,p.updated_at
            FROM user_account a JOIN learner_profile p ON p.user_id=a.id WHERE a.id=?
        """, (user_id,)).fetchone()
    return dict(row) if row else None


def update_profile(display_name, long_term_goal, short_term_goal, level, user_id=1):
    if not display_name.strip():
        raise ValueError("Enter a display name.")
    with connect() as db:
        db.execute("UPDATE user_account SET display_name=? WHERE id=?",
                   (display_name.strip(), user_id))
        db.execute("""
            UPDATE learner_profile
            SET long_term_goal=?,short_term_goal=?,level=?,updated_at=? WHERE user_id=?
        """, (long_term_goal.strip(), short_term_goal.strip(), level, _now(), user_id))


def update_password(username, current_password, new_password, user_id=1):
    if len(new_password) < 8:
        raise ValueError("Use a password with at least 8 characters.")
    with connect() as db:
        row = db.execute(
            "SELECT salt,password_hash FROM user_account WHERE id=? AND username=?",
            (user_id, username)).fetchone()
        if not row or not hmac.compare_digest(
                _hash_password(current_password, row["salt"]), row["password_hash"]):
            raise ValueError("Current password is incorrect.")
        salt = secrets.token_bytes(16)
        db.execute("UPDATE user_account SET salt=?,password_hash=? WHERE id=?",
                   (salt, _hash_password(new_password, salt), user_id))


def add_learning_note(user_id, note):
    note = (note or "").strip()
    if note:
        with connect() as db:
            db.execute("INSERT INTO learning_notes(user_id,note,created_at) VALUES(?,?,?)",
                       (user_id, note[:2000], _now()))


def _concept_id(db, name):
    name = " ".join(str(name or "").split()).strip()
    if not name:
        return None
    db.execute("INSERT OR IGNORE INTO concepts(name) VALUES(?)", (name,))
    return db.execute("SELECT id FROM concepts WHERE name=?", (name,)).fetchone()["id"]


def save_course(user_id, source_module, course):
    now = _now()
    with connect() as db:
        cur = db.execute("""
            INSERT INTO courses(user_id,source_module_id,source_title,source_url,title,level,
                                status,course_json,started_at)
            VALUES(?,?,?,?,?,?,'in_progress',?,?)
        """, (user_id, source_module["id"], source_module["title"], source_module.get("url", ""),
              course["title"], course.get("level", "Foundational"),
              json.dumps(course, ensure_ascii=False), now))
        course_id = cur.lastrowid
        concept_ids = []
        for name in course.get("concepts", []):
            cid = _concept_id(db, name)
            if cid:
                concept_ids.append(cid)
                db.execute("INSERT OR IGNORE INTO course_concepts VALUES(?,?)", (course_id, cid))
        for prerequisite in course.get("prerequisites", []):
            from_id = _concept_id(db, prerequisite)
            to_id = concept_ids[0] if concept_ids else None
            if from_id and to_id and from_id != to_id:
                db.execute("INSERT OR IGNORE INTO concept_links VALUES(?,?,?)",
                           (from_id, to_id, "prerequisite"))
    return course_id


def get_course(course_id):
    with connect() as db:
        row = db.execute("SELECT * FROM courses WHERE id=?", (course_id,)).fetchone()
    if not row:
        return None
    item = dict(row)
    item["course"] = json.loads(item.pop("course_json"))
    return item


def complete_course(course_id, user_id=1):
    with connect() as db:
        db.execute("UPDATE courses SET status='completed',completed_at=? WHERE id=? AND user_id=?",
                   (_now(), course_id, user_id))


def record_assessment(course_id, kind, score, max_score, result, concept_scores=None, user_id=1):
    concept_scores = concept_scores or {}
    with connect() as db:
        db.execute("""
            INSERT INTO assessments(course_id,kind,score,max_score,result_json,submitted_at)
            SELECT ?,?,?,?,?,? WHERE EXISTS(SELECT 1 FROM courses WHERE id=? AND user_id=?)
        """, (course_id, kind, score, max_score, json.dumps(result, ensure_ascii=False),
              _now(), course_id, user_id))
        for name, ratio in concept_scores.items():
            cid = _concept_id(db, name)
            if cid is None:
                continue
            row = db.execute(
                "SELECT mastery,evidence_count FROM learner_concepts WHERE user_id=? AND concept_id=?",
                (user_id, cid)).fetchone()
            prior = row["mastery"] if row else 0.5
            count = row["evidence_count"] if row else 0
            updated = max(0.0, min(1.0, prior * 0.65 + float(ratio) * 0.35))
            db.execute("""
                INSERT INTO learner_concepts(user_id,concept_id,mastery,evidence_count,updated_at)
                VALUES(?,?,?,?,?)
                ON CONFLICT(user_id,concept_id) DO UPDATE SET
                    mastery=excluded.mastery,evidence_count=excluded.evidence_count,
                    updated_at=excluded.updated_at
            """, (user_id, cid, updated, count + 1, _now()))


def get_learning_record(user_id=1):
    with connect() as db:
        courses = [dict(r) for r in db.execute("""
            SELECT c.id,c.source_module_id,c.title,c.source_title,c.level,c.status,c.started_at,c.completed_at,
                   COALESCE(ROUND(100.0*SUM(a.score)/NULLIF(SUM(a.max_score),0),1),NULL) AS result_pct
            FROM courses c LEFT JOIN assessments a ON a.course_id=c.id
            WHERE c.user_id=? GROUP BY c.id ORDER BY c.started_at DESC
        """, (user_id,)).fetchall()]
        concepts = [dict(r) for r in db.execute("""
            SELECT k.name,COALESCE(lc.mastery,0.5) AS mastery,
                   COALESCE(lc.evidence_count,0) AS evidence_count,lc.updated_at
            FROM concepts k LEFT JOIN learner_concepts lc
              ON lc.concept_id=k.id AND lc.user_id=?
            WHERE lc.user_id=? OR EXISTS(
                SELECT 1 FROM course_concepts cc JOIN courses c ON c.id=cc.course_id
                WHERE cc.concept_id=k.id AND c.user_id=?
            )
            ORDER BY mastery ASC,k.name
        """, (user_id,user_id,user_id)).fetchall()]
        links = [dict(r) for r in db.execute("""
            SELECT f.name AS source,t.name AS target,l.relation
            FROM concept_links l JOIN concepts f ON f.id=l.from_concept_id
            JOIN concepts t ON t.id=l.to_concept_id
        """).fetchall()]
        course_concepts = [dict(r) for r in db.execute("""
            SELECT cc.course_id,c.title,k.name AS concept
            FROM course_concepts cc JOIN courses c ON c.id=cc.course_id
            JOIN concepts k ON k.id=cc.concept_id WHERE c.user_id=?
        """, (user_id,)).fetchall()]
        assessments = [dict(r) for r in db.execute("""
            SELECT a.kind,a.score,a.max_score,a.submitted_at,a.result_json,c.title
            FROM assessments a JOIN courses c ON c.id=a.course_id
            WHERE c.user_id=? ORDER BY a.submitted_at DESC
        """, (user_id,)).fetchall()]
        notes = [dict(r) for r in db.execute("""
            SELECT note,created_at FROM learning_notes
            WHERE user_id=? ORDER BY created_at DESC LIMIT 20
        """, (user_id,)).fetchall()]
    ratios = [100 * r["score"] / r["max_score"] for r in assessments if r["max_score"]]
    return {
        "courses": courses, "concepts": concepts, "links": links, "course_concepts": course_concepts,
        "assessments": assessments, "notes": notes,
        "average_score": round(sum(ratios) / len(ratios), 1) if ratios else None,
        "completed_count": sum(1 for c in courses if c["status"] == "completed"),
    }


def set_learning_style(style, confidence, evidence, user_id=1):
    with connect() as db:
        db.execute("""
            UPDATE learner_profile SET preferred_style=?,style_confidence=?,style_evidence=?,
                                       updated_at=? WHERE user_id=?
        """, (style, confidence, json.dumps(evidence, ensure_ascii=False), _now(), user_id))


def erase_account():
    db = connect()
    try:
        db.execute("DELETE FROM user_account WHERE id=1")
        db.execute("DELETE FROM concepts")
        db.commit()
        db.execute("VACUUM")
    finally:
        db.close()


def memory_summary(user_id=1):
    profile = get_profile(user_id)
    record = get_learning_record(user_id)
    if not profile:
        return ""
    weak = [c for c in record["concepts"] if c["mastery"] < 0.5]
    recent = record["courses"][:5]
    return {
        "learner_level": profile["level"],
        "long_term_goal": profile["long_term_goal"],
        "short_term_goal": profile["short_term_goal"],
        "preferred_learning_approach": profile["preferred_style"],
        "style_confidence": profile["style_confidence"],
        "style_evidence": profile["style_evidence"],
        "completed_courses": [
            {"title": c["title"], "level": c["level"], "result_pct": c["result_pct"]}
            for c in record["courses"] if c["status"] == "completed"
        ][:10],
        "recent_courses": [
            {"title": c["title"], "level": c["level"], "status": c["status"],
             "result_pct": c["result_pct"]} for c in recent
        ],
        "concept_mastery": [
            {"concept": c["name"], "estimated_mastery": round(c["mastery"], 2),
             "evidence_count": c["evidence_count"]} for c in record["concepts"]
        ],
        "priority_gaps": [
            {"concept": c["name"], "estimated_mastery": round(c["mastery"], 2)}
            for c in weak[:8]
        ],
        "assessment_average_pct": record["average_score"],
        "exclude_module_ids": [c["source_module_id"] for c in record["courses"]],
        "recent_study_notes": [n["note"] for n in record.get("notes", [])[:10]],
        "recent_assessment_results": [
            {"course": a["title"], "kind": a["kind"], "score": a["score"],
             "max_score": a["max_score"], "date": a["submitted_at"]}
            for a in record["assessments"][:10]
        ],
    }










