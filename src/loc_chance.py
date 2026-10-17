"""loc_chance.py -- chance-adjusted localization: does the slice gain in top-k exceed what a shorter input gives by chance?

For each vulnerable unit with labelled lines and each input (R1, R2, R4), the chance top-k hit rate is the probability
that k lines drawn uniformly from the lines shown to the model include a labelled line:
1 - C(n-g, k) / C(n, k), with n shown lines and g labelled lines among them. Compares each model's observed hit
rate with this chance rate. Usage: python loc_chance.py RESULTS_DIR  -> RESULTS_DIR/loc_chance.json"""
import json
import re
import sys
from math import comb
from pathlib import Path

import addons
import data_prep


def shown_lines(text):
    return {int(m.group(1)) for m in re.finditer(r"(?m)^\s*(\d+):", text)}


def main():
    d = Path(sys.argv[1])
    recs = {json.loads(l)["unit_id"]: json.loads(l) for l in open(d / "records_rq1.jsonl")}
    units, _ = data_prep.load_units(max_rows=None, allow_synthetic=False, strict=True)
    units = [u for u in units if recs.get(u.unit_id, {}).get("label") == 1 and recs[u.unit_id]["vul_lines"]]
    models = list(recs[units[0].unit_id]["by_rep"]["R1"][0])
    reps_k = [(rep, k) for rep in ("R1", "R2", "R4") for k in (1, 5)]
    acc = {(rep, k): [0.0, 0] for rep, k in reps_k}                          # chance with exactly k picks
    mch = {(m, rep, k): 0.0 for m in models for rep, k in reps_k}            # chance with as many picks as the model cited
    obs = {(m, rep, k): 0 for m in models for rep, k in reps_k}
    vis = {rep: 0 for rep in ("R1", "R2", "R4")}; nlines = {rep: 0 for rep in ("R1", "R2", "R4")}
    per_unit = []

    def chance(n, g, c):
        c = min(c, n)
        return 0.0 if n == 0 or g == 0 or c == 0 else 1 - comb(n - g, c) / comb(n, c)

    for u in units:
        reps = addons._reps_basic(u); gt = set(recs[u.unit_id]["vul_lines"])
        for rep in ("R1", "R2", "R4"):
            sl = shown_lines(reps[rep]); n = len(sl); g = len(sl & gt)
            vis[rep] += int(g > 0); nlines[rep] += n
            per_unit.append({"unit": u.unit_id, "rep": rep, "n": n, "g": g})
            for k in (1, 5):
                acc[(rep, k)][0] += chance(n, g, k); acc[(rep, k)][1] += 1
                for m in models:
                    cited = [int(x) for x in (recs[u.unit_id]["by_rep"][rep][0].get(m, {}).get("cited_lines") or [])][:k]
                    obs[(m, rep, k)] += int(bool(set(cited) & gt))
                    mch[(m, rep, k)] += chance(n, g, len(set(cited) & sl) if cited else 0)
    N = len(units)
    out = {"n_units": N,
           "labelled_line_visible_share": {r: vis[r] / N for r in vis},
           "mean_shown_lines": {r: nlines[r] / N for r in nlines},
           "chance_hit_rate": {f"{r}_top{k}": acc[(r, k)][0] / acc[(r, k)][1] for (r, k) in acc},
           "observed_hit_rate": {f"{m}|{r}_top{k}": obs[(m, r, k)] / N for (m, r, k) in obs},
           "matched_chance_hit_rate": {f"{m}|{r}_top{k}": mch[(m, r, k)] / N for (m, r, k) in mch},
           "per_unit_shown": per_unit}
    (d / "loc_chance.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
