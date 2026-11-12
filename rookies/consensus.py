"""Builds the consensus rookie board from several sites' rankings.

Input is a JSON file kept OUTSIDE this repo (the sites' own lists are not published here):
  {"class": 2027, "sources": [
     {"name": "DraftSharks", "date": "2026-11-04", "kind": "rank",  "rows": [["Jeremiah Smith","WR","Ohio State"], ...]},
     {"name": "KeepTradeCut", "date": "2026-11-12", "kind": "value", "rows": [["Jeremiah Smith", 9956], ...]}]}
"rank" lists are in order, best first. "value" lists are ranked by value, keeping only players a rank list has.
Only the averaged result is written: rookies_<class>.csv and rookies_updated.txt.
Usage: python rookies/consensus.py /path/to/input.json
"""
import csv, json, re, statistics as st, sys
from pathlib import Path

def key(n):
    n = n.lower().replace("duce alexander", "deuce alexander")
    return re.sub(r"[^a-z]", "", re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n))

def build(spec):
    info, ranks, depth = {}, {}, {}
    for s in [s for s in spec["sources"] if s["kind"] == "rank"]:
        ranks[s["name"]] = {}
        for i, (n, p, sch) in enumerate(s["rows"], 1):
            k = key(n)
            ranks[s["name"]].setdefault(k, i); info.setdefault(k, (n, p.upper(), sch))
        depth[s["name"]] = len(s["rows"])
    for s in [s for s in spec["sources"] if s["kind"] == "value"]:
        kv = sorted([(key(n), float(v)) for n, v in s["rows"] if key(n) in info], key=lambda x: -x[1])
        ranks[s["name"]] = {k: i for i, (k, _) in enumerate(kv, 1)}; depth[s["name"]] = len(kv)
    out = []
    for k, (n, p, sch) in info.items():
        have = {src: r[k] for src, r in ranks.items() if k in r}
        # A list that doesn't rank him counts him just past its last spot.
        full = [have.get(src, depth[src] + 8) for src in ranks]
        out.append((st.mean(full), n, p, sch, len(have), min(have.values()), max(have.values())))
    out.sort()
    return out

def main(path):
    spec = json.loads(Path(path).read_text())
    if len([s for s in spec["sources"] if s["kind"] == "rank"]) < 2:
        sys.exit("need at least two ranked lists")
    out = build(spec)
    root = Path(__file__).resolve().parent.parent
    with open(root / f"rookies_{spec['class']}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["rank", "name", "pos", "school", "avg", "sources", "best", "worst"])
        for i, (a, n, p, sch, c, b, wr) in enumerate(out, 1):
            w.writerow([i, n, p, sch, round(a, 1), c, b, wr])
    (root / "rookies_updated.txt").write_text(
        f"{spec['class']} class, {len(out)} players. Sources: " + "; ".join(f"{s['name']} ({s['date']})" for s in spec["sources"]) + "\n")
    print(len(out), "players")

if __name__ == "__main__":
    main(sys.argv[1])
