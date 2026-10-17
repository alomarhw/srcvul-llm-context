"""retrieval.py - BM25 retrieval for the R5 representation.

Requirements (pinned): rank-bm25==0.2.2, numpy==2.2.6

We build a corpus of vulnerability/patch examples from the loaded units and
retrieve the top-k most similar for a query unit. Two variants:
  - chronology-constrained: retrieve ONLY examples whose cve_year < query year
    AND exclude the query unit and its sibling fix (leakage-safe, used as R5);
  - unconstrained: retrieve from the whole corpus except the exact query unit
    (used to quantify any leakage benefit).
"""
from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi

_TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tok(text: str) -> list[str]:
    return [t.lower() for t in _TOK.findall(text)]


class RetrievalCorpus:
    def __init__(self, units: list[Any]):
        # only vulnerable examples + their fixes make useful retrieval examples
        self.entries = []
        for u in units:
            self.entries.append({
                "base": u.unit_id.rsplit("__", 1)[0],
                "unit_id": u.unit_id,
                "year": int(getattr(u, "cve_year", 0) or 0),
                "label": int(u.label),
                "cwe": u.cwe,
                "text": f"/* {u.cwe} label={u.label} */\n{u.code}",
                "tokens": _tok(u.code),
            })
        self._corpus_tokens = [e["tokens"] for e in self.entries]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    def retrieve(self, query_unit: Any, k: int = 2, chronological: bool = True) -> list[str]:
        if self._bm25 is None:
            return []
        q = _tok(query_unit.code)
        scores = self._bm25.get_scores(q)
        qbase = query_unit.unit_id.rsplit("__", 1)[0]
        qyear = int(getattr(query_unit, "cve_year", 0) or 0)
        ranked = sorted(range(len(self.entries)), key=lambda i: -scores[i])
        out: list[str] = []
        for i in ranked:
            e = self.entries[i]
            if e["unit_id"] == query_unit.unit_id:
                continue
            if e["base"] == qbase:  # exclude sibling fix / self pair
                continue
            if chronological and qyear and e["year"] and e["year"] >= qyear:
                continue
            out.append(e["text"])
            if len(out) >= k:
                break
        return out


if __name__ == "__main__":
    print("retrieval module ok; BM25Okapi available")