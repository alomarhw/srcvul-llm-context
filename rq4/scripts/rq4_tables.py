"""rq4_tables.py -- final RQ4 labels (agreed codes + adjudicated disagreements) joined with the keys; RQ4 tables.

Usage: python rq4_tables.py DOCS_DIR
Writes DOCS_DIR/df89e_rq4_final_labels.csv and DOCS_DIR/df89e_rq4_results.json.
The samples are stratified by design (20 per stratum in layer 1), so shares are within-stratum, not population rates.
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SHORT = {"Qwen/Qwen2.5-Coder-14B-Instruct": "Qwen2.5-Coder-14B", "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8": "Qwen3-Coder-30B",
         "microsoft/phi-4": "Phi-4"}


def rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def main():
    d = Path(sys.argv[1]); U = lambda s: s.strip().upper()  # noqa: E731
    adj = {(r["layer"], r["case_id"]): r for r in rd(d / "df89e_rq4_disagreements.csv")}
    final = []
    for layer, stem, keyf in (("layer1", "df89e_rq4_sample", "df89e_rq4_key_DO_NOT_SHARE_WITH_CODERS.csv"),
                              ("layer2", "df89e_rq4_reasoning_sample", "df89e_rq4_reasoning_key_DO_NOT_SHARE_WITH_CODERS.csv")):
        h = {r["case_id"]: r for r in rd(d / f"{stem}_hakam.csv")}
        s = {r["case_id"]: r for r in rd(d / f"{stem}_student.csv")}
        key = {r["case_id"]: r for r in rd(d / keyf)}
        for cid, hr in h.items():
            sr, a = s[cid], adj.get((layer, cid))
            prim = U(a["adjudicated_primary"]) if a and a["adjudicated_primary"].strip() else U(hr["primary_category"])
            assert a or U(hr["primary_category"]) == U(sr["primary_category"]), cid
            se = ""
            if layer == "layer1":
                se = (a["adjudicated_slice_effect"].strip().lower() if a and a["adjudicated_slice_effect"].strip()
                      else hr["slice_effect"].strip().lower())
            secs = {x for x in (U(hr["secondary_category"]), U(sr["secondary_category"])) if x}
            k = key[cid]
            final.append({"layer": layer, "case_id": cid, "stage": hr["stage"], "stratum": k["stratum"],
                          "model": SHORT.get(k["model"], k["model"]), "representation": k["representation"],
                          "unit_id": k["unit_id"], "primary": prim, "slice_effect": se,
                          "secondary_either_coder": " ".join(sorted(secs)), "adjudicated": bool(a)})
    with open(d / "df89e_rq4_final_labels.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(final[0])); w.writeheader(); w.writerows(final)
    L1 = [r for r in final if r["layer"] == "layer1"]; L2 = [r for r in final if r["layer"] == "layer2"]
    tab = lambda rows, by: {g: dict(Counter(r["primary"] for r in rows if r[by] == g).most_common())  # noqa: E731
                            for g in sorted({r[by] for r in rows})}
    res = {
        "n": {"layer1": len(L1), "layer2": len(L2), "adjudicated": sum(r["adjudicated"] for r in final)},
        "layer1_by_stratum": tab(L1, "stratum"),
        "layer1_by_model": tab(L1, "model"),
        "layer1_by_representation": tab([r for r in L1 if r["stratum"] != "slice_effect"], "representation"),
        "layer1_slice_effect": {e: dict(Counter(r["primary"] for r in L1 if r["slice_effect"] == e).most_common())
                                for e in ("helped", "hurt", "neutral")},
        "layer1_cx_primary_or_secondary": sum(1 for r in L1 if r["primary"] == "CX" or "CX" in r["secondary_either_coder"].split()),
        "layer1_label_problems": [r["case_id"] for r in L1 if r["primary"] == "LB"],
        "layer2_overall": dict(Counter(r["primary"] for r in L2).most_common()),
        "layer2_by_model": tab(L2, "model"),
        "layer2_by_stratum": tab(L2, "stratum"),
        "layer2_by_representation": tab(L2, "representation"),
    }
    (d / "df89e_rq4_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
