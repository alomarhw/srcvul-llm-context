"""loc_lift_test.py -- is the localization skill above chance different with slices?

Per model, unit and input, lift = observed top-k hit (0/1) minus the matched chance of a hit (random picks, as many
as the model cited, from the lines shown). Paired bootstrap (10,000 draws over units, seed 42) of the mean lift
difference R4 - R1 (and R2 - R1), Holm over all tests. Needs results/loc_chance.json (per_unit_shown) from
loc_chance.py. Usage: python loc_lift_test.py RESULTS_DIR  -> RESULTS_DIR/loc_lift_test.json"""
import json
import random
import sys
from math import comb
from pathlib import Path


def chance(n, g, c):
    c = min(c, n)
    return 0.0 if n == 0 or g == 0 or c == 0 else 1 - comb(n - g, c) / comb(n, c)


def main():
    d = Path(sys.argv[1])
    recs = {json.loads(l)["unit_id"]: json.loads(l) for l in open(d / "records_rq1.jsonl")}
    shown = {(x["unit"], x["rep"]): (x["n"], x["g"]) for x in json.loads((d / "loc_chance.json").read_text())["per_unit_shown"]}
    units = sorted({u for u, _ in shown})
    models = list(recs[units[0]]["by_rep"]["R1"][0])
    lift = {}
    for m in models:
        for rep in ("R1", "R2", "R4"):
            for k in (1, 5):
                vals = []
                for u in units:
                    gt = set(recs[u]["vul_lines"]); n, g = shown[(u, rep)]
                    cited = [int(x) for x in (recs[u]["by_rep"][rep][0].get(m, {}).get("cited_lines") or [])][:k]
                    vals.append(int(bool(set(cited) & gt)) - chance(n, g, len(set(cited)) if cited else 0))
                lift[(m, rep, k)] = vals
    tests, rng = [], random.Random(42)
    N = len(units)
    for m in models:
        for rep in ("R2", "R4"):
            for k in (1, 5):
                diff = [a - b for a, b in zip(lift[(m, rep, k)], lift[(m, "R1", k)])]
                obs = sum(diff) / N
                bs = sorted(sum(diff[rng.randrange(N)] for _ in range(N)) / N for _ in range(10_000))
                p = min(1.0, 2 * min(sum(b <= 0 for b in bs), sum(b >= 0 for b in bs)) / len(bs))
                tests.append({"model": m, "comparison": f"{rep} - R1", "k": k, "lift_R1": sum(lift[(m, 'R1', k)]) / N,
                              "lift_slice": sum(lift[(m, rep, k)]) / N, "diff": obs, "ci95": [bs[249], bs[9749]], "p": p})
    order = sorted(range(len(tests)), key=lambda i: tests[i]["p"]); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(tests) - rank) * tests[i]["p"])); tests[i]["p_holm"] = run
    (d / "loc_lift_test.json").write_text(json.dumps(tests, indent=1))
    for t in tests:
        print(f"{t['model'].split('/')[-1][:16]:16s} {t['comparison']} top-{t['k']}: lift {t['lift_R1']:+.3f} -> {t['lift_slice']:+.3f} "
              f"diff {t['diff']:+.3f} [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] holm p={t['p_holm']:.3g}")


if __name__ == "__main__":
    main()
