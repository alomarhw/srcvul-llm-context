"""loc_tests.py -- paired localization tests per model: slice inputs (R2, R4) vs the whole function (R1).

For each model and vulnerable unit with labelled lines, a hit is a labelled line among the model's first k cited
lines (k = 1, 5), as in analysis._localization but per model. Exact McNemar on the paired hits, Holm over all
tests. Usage: python loc_tests.py RESULTS_DIR  -> RESULTS_DIR/loc_tests.json"""
import json
import sys
from math import comb
from pathlib import Path


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0.0] * len(ps); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i])); out[i] = run
    return out


def main():
    d = Path(sys.argv[1]); recs = [json.loads(l) for l in open(d / "records_rq1.jsonl")]
    recs = [r for r in recs if r["label"] == 1 and r["vul_lines"]]
    models = list(recs[0]["by_rep"]["R1"][0])
    tests = []
    for m in models:
        for rep in ("R2", "R4"):
            for k in (1, 5):
                b = c = hb = hr = 0
                for r in recs:
                    gt = set(r["vul_lines"])
                    hit = lambda rr: bool(set(int(x) for x in (r["by_rep"][rr][0].get(m, {}).get("cited_lines") or [])[:k]) & gt)  # noqa: E731
                    h1, h2 = hit("R1"), hit(rep)
                    hb += h1; hr += h2; b += (h2 and not h1); c += (h1 and not h2)
                tests.append({"model": m, "comparison": f"{rep} vs R1", "k": k, "n": len(recs),
                              "rate_R1": hb / len(recs), f"rate_{rep}": hr / len(recs), "rate_slice": hr / len(recs),
                              "slice_only": b, "whole_only": c, "p": exact_mcnemar(b, c)})
    for t, p in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = p
    (d / "loc_tests.json").write_text(json.dumps(tests, indent=1))
    for t in tests:
        print(f"{t['model'].split('/')[-1][:18]:18s} {t['comparison']} top-{t['k']}: {t['rate_R1']:.3f} -> {t['rate_slice']:.3f} "
              f"(+{t['slice_only']}/-{t['whole_only']}) p={t['p']:.2g} holm={t['p_holm']:.2g}")


if __name__ == "__main__":
    main()
