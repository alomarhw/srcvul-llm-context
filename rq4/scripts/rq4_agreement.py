"""rq4_agreement.py -- inter-coder agreement for the RQ4 coding (two independent coders), and the disagreement list.

Usage: python rq4_agreement.py DOCS_DIR
Reads  df89e_rq4_sample_{hakam,student}.csv and df89e_rq4_reasoning_sample_{hakam,student}.csv from DOCS_DIR.
Writes DOCS_DIR/df89e_rq4_agreement.json and DOCS_DIR/df89e_rq4_disagreements.csv (blind: no model names).
Cohen's kappa on primary_category (per layer; all cases and cases both coders rated high-confidence) and on
slice_effect (layer-1 slice-effect rows), with 95% bootstrap CIs over cases (10,000 draws, seed 42).
"""
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path


def kappa(a, b):
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def boot(a, b, draws=10_000, seed=42):
    rng, n, ks = random.Random(seed), len(a), []
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        k = kappa([a[i] for i in idx], [b[i] for i in idx])
        if k is not None:
            ks.append(k)
    ks.sort()
    return [ks[int(0.025 * len(ks))], ks[int(0.975 * len(ks)) - 1]]


def summary(a, b):
    k = kappa(a, b)
    return {"n": len(a), "agreement": sum(x == y for x, y in zip(a, b)) / len(a) if a else None,
            "kappa": k, "kappa_ci95": boot(a, b) if len(a) > 1 else None}


def main():
    d = Path(sys.argv[1]); out = {}; dis = []
    U = lambda s: s.strip().upper()  # noqa: E731
    for layer, stem in (("layer1", "df89e_rq4_sample"), ("layer2", "df89e_rq4_reasoning_sample")):
        h = {r["case_id"]: r for r in csv.DictReader(open(d / f"{stem}_hakam.csv", encoding="utf-8-sig"))}
        s = {r["case_id"]: r for r in csv.DictReader(open(d / f"{stem}_student.csv", encoding="utf-8-sig"))}
        ids = [i for i in h if i in s]
        a, b = [U(h[i]["primary_category"]) for i in ids], [U(s[i]["primary_category"]) for i in ids]
        hi = [j for j, i in enumerate(ids) if h[i]["confidence"].strip().lower() == "high" and s[i]["confidence"].strip().lower() == "high"]
        res = {"primary_all": summary(a, b), "primary_both_high_conf": summary([a[j] for j in hi], [b[j] for j in hi]),
               "confusion_top": Counter(f"{x}->{y}" for x, y in zip(a, b) if x != y).most_common(10)}
        if layer == "layer1":
            se = [i for i in ids if h[i]["stage"] == "slice effect"]
            res["slice_effect"] = summary([h[i]["slice_effect"].strip().lower() for i in se], [s[i]["slice_effect"].strip().lower() for i in se])
            res["primary_by_stage"] = {st: summary([U(h[i]["primary_category"]) for i in ids if h[i]["stage"] == st],
                                                    [U(s[i]["primary_category"]) for i in ids if h[i]["stage"] == st])
                                       for st in sorted({h[i]["stage"] for i in ids})}
        out[layer] = res
        for i in ids:
            pa, pb = U(h[i]["primary_category"]), U(s[i]["primary_category"])
            sea = h[i].get("slice_effect", "").strip().lower(); seb = s[i].get("slice_effect", "").strip().lower()
            if pa != pb or (layer == "layer1" and h[i]["stage"] == "slice effect" and sea != seb):
                dis.append({"layer": layer, "case_id": i, "stage": h[i]["stage"],
                            "coder1_primary": pa, "coder1_secondary": U(h[i]["secondary_category"]), "coder1_slice_effect": sea,
                            "coder1_confidence": h[i]["confidence"], "coder1_note": h[i]["note"],
                            "coder2_primary": pb, "coder2_secondary": U(s[i]["secondary_category"]), "coder2_slice_effect": seb,
                            "coder2_confidence": s[i]["confidence"], "coder2_note": s[i]["note"],
                            "adjudicated_primary": "", "adjudicated_slice_effect": "", "adjudication_note": ""})
    (d / "df89e_rq4_agreement.json").write_text(json.dumps(out, indent=1))
    with open(d / "df89e_rq4_disagreements.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dis[0])); w.writeheader(); w.writerows(dis)
    print(json.dumps({l: {k: v for k, v in r.items() if k != "primary_by_stage"} for l, r in out.items()}, indent=1))
    print("by stage:", {st: round(v["kappa"], 2) for st, v in out["layer1"]["primary_by_stage"].items()})
    print("disagreements:", len(dis))


if __name__ == "__main__":
    main()
