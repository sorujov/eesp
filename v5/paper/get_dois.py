"""Look up DOIs on Crossref for the references (title match required); writes doi_cache.json."""
import json, re, os, urllib.parse, urllib.request, difflib
import importlib.util
spec = importlib.util.spec_from_file_location("br", "build_refs.py")
src = open("build_refs.py").read().split("src = ''.join")[0]
ns = {}; exec(src, ns); lib = ns["lib"]
cache = json.load(open("doi_cache.json")) if os.path.exists("doi_cache.json") else {}
def clean(t): return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\\[a-zA-Z]+|[{}$]", "", t.lower()))
import time
for k, item in lib.items():
    if k in cache: continue
    m = re.search(r"\(\d{4}[a-z]?\)\.\s*(.*?)\.\s", item, re.S)
    if not m: cache[k] = None; continue
    title = m.group(1)
    time.sleep(1.5); q = urllib.parse.quote(clean(title))
    try:
        r = json.load(urllib.request.urlopen(f"https://api.crossref.org/works?query.bibliographic={q}&rows=3&mailto=sorujov@ada.edu.az", timeout=30))
    except Exception as e:
        print("ERR", k, e); continue
    best = None
    for it in r["message"]["items"]:
        t = (it.get("title") or [""])[0]
        s = difflib.SequenceMatcher(None, clean(t), clean(title)).ratio()
        if s > 0.93 and (best is None or s > best[0]): best = (s, it["DOI"], t)
    cache[k] = best[1] if best else None
    print(k, cache[k], round(best[0],3) if best else "")
json.dump(cache, open("doi_cache.json", "w"), indent=1)
