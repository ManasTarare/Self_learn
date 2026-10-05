"""mslearn_course_structures.json -> snippets.jsonl (our snippet schema, no lesson text yet).
One snippet = one MS Learn module (course_id is the module id). Units become `children`.
Run: python convert_mslearn.py mslearn_course_structures.json data/snippets.jsonl
"""
import json, pathlib, re, sys

def slug_parts(url):
    m = re.search(r"/training/(modules|paths)/([^/?]+)", url)
    return m.group(2) if m else None

def convert(src, dst):
    courses = json.load(open(src, encoding="utf-8"))
    pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(dst, "w", encoding="utf-8") as f:
        for c in courses:
            for m in c["modules"]:
                units = [dict(u, order=i) for i, u in enumerate(
                    m["lectures"] + m["assessments"] + m["supplements"])]
                # text used for retrieval until real content exists
                base = " . ".join([m["module_name"], m["description"]]
                                  + [u["name"] for u in m["lectures"]])
                snippet = {
                    "id": c["course_id"],
                    "slug": slug_parts(c["url"]),
                    "url": c["url"].split("?")[0],
                    "title": m["module_name"],
                    "description": m["description"],
                    "duration_min": m["total_duration_minutes"],
                    "units": units,                      # titles only
                    "text": None,                        # filled by fetch_content.py
                    "index_base": base,                  # description-only retrieval text (never changes)
                    "index_text": base,                  # fetch_content.py rebuilds this from index_base
                }
                f.write(json.dumps(snippet, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} snippets -> {dst}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python convert_mslearn.py <mslearn_course_structures.json> <out.jsonl>")
    convert(sys.argv[1], sys.argv[2])
