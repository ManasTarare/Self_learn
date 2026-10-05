"""Fill each snippet with real lesson text from the public MicrosoftDocs/learn repo (CC BY 4.0).

  git clone --depth 1 https://github.com/MicrosoftDocs/learn
  python fetch_content.py data/snippets.jsonl learn/learn-pr data/snippets_full.jsonl

- Matches modules by slug (last part of the module URL). Only ~half of the modules in
  mslearn_course_structures.json exist in that repo; the rest keep description-only.
- Adds  text  (cleaned markdown, paragraphs separated by blank lines)  and  index_text  =
  index_base + " || " + first ~1500 characters of lesson text (better retrieval).
- Safe to re-run, even on its own output: the description-based part is kept in  index_base,
  so index_text is always rebuilt from it instead of being appended to repeatedly.
- Attribution: content is (c) Microsoft, licensed CC BY 4.0. Keep the url field for credit.
"""
import json, pathlib, re, sys

SEP = " || "

def clean_md(md: str) -> str:
    md = re.sub(r"\A---.*?---\s*", "", md, flags=re.S)                    # front matter
    md = re.sub(r"^\s*\[!INCLUDE.*?\]\(.*?\)\s*$", "", md, flags=re.M)     # includes
    md = re.sub(r":::image\b.*?:::", "", md, flags=re.S)                    # :::image ... ::: (any indent / line layout)
    md = re.sub(r"^[ \t]*:::.*$", "", md, flags=re.M)                       # other ::: markers (zone pivots etc.)
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)                           # images
    md = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", md)                       # links -> text
    md = re.sub(r"^>\s*\[!\w+\]\s*$", "", md, flags=re.M)                  # > [!NOTE]
    md = re.sub(r"<[^>]+>", "", md)                                        # html tags
    md = re.sub(r"[ \t]+\n", "\n", md)
    return re.sub(r"\n{3,}", "\n\n", md).strip()

def unit_order(p: pathlib.Path):
    m = re.match(r"(\d+)", p.name)
    return (int(m.group(1)) if m else 10**6, p.name)

def read_module(module_dir: pathlib.Path) -> str:
    parts = []
    for md in sorted((module_dir / "includes").glob("*.md"), key=unit_order):
        t = clean_md(md.read_text(encoding="utf-8", errors="ignore"))
        if t:
            parts.append(t)
    return "\n\n".join(parts)

def main(src, repo_learn_pr, dst):
    root = pathlib.Path(repo_learn_pr)
    if not root.exists():
        sys.exit(f"folder not found: {root}  (expected <clone>/learn-pr)")
    index = {}
    for p in root.rglob("index.yml"):
        index.setdefault(p.parent.name, p.parent)                          # slug -> module dir
    print(f"repo has {len(index)} modules")
    hit = 0; total = 0
    pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with open(src, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
        for line in fi:
            if not line.strip():
                continue
            s = json.loads(line); total += 1
            # description-only text; older files without index_base: strip anything after SEP
            base = s.get("index_base") or s["index_text"].split(SEP)[0]
            s["index_base"] = base
            slugs = [x for x in (s.get("slug"), s["id"].split(".")[-1]) if x]
            d = next((index[x] for x in slugs if x in index), None)
            s["text"] = read_module(d) if d else None
            if s["text"]:
                hit += 1
                s["index_text"] = base + SEP + s["text"][:1500].replace("\n", " ")
            else:
                s["index_text"] = base
            fo.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"lesson text found for {hit}/{total} modules ({100*hit//max(total,1)}%); rest keep description only")
    print(f"wrote {dst}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: python fetch_content.py <snippets.jsonl> <learn/learn-pr> <out.jsonl>")
    main(*sys.argv[1:4])
