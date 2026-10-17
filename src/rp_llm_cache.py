"""Frozen LLM response cache for experiments that call language models (exact, offline replay).

This file is self-contained on purpose: the Engineer ships it into the experiment sandbox as
``rp_llm_cache.py`` (see :func:`llm_cache_artifact`), so generated experiment code can do::

    from rp_llm_cache import LLMCache
    llm = LLMCache()                      # cache file from RP_LLM_CACHE, else data/_llm_cache.sqlite
    text = llm.complete(system, user, model="claude-haiku-4-5", max_tokens=16, temperature=0.0,
                        sample=repeat_index)

Every response is stored in sqlite under ``sha256([model, system, user, max_tokens, temperature,
sample])``. The first write wins and identical concurrent requests wait on a per-key lock, so a
re-run served from the cache reproduces the original run exactly. Without an API key the cache still
serves every stored response (offline replay); only a miss fails, loudly, never falling back to a
proxy. ``temperature=None`` omits the parameter for models that reject it (Claude Opus 4.8 / 5.5,
Sonnet 5.5). Anthropic models (``claude-*``) use the Anthropic SDK; others use the OpenAI SDK, with open-weight
models (any ``org/name`` Hugging Face id, or ``local:...``) sent to the self-hosted server at ``RP_LOCAL_LLM_BASE_URL``.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path


# Open-weight models served on the researcher's own GPU (vLLM, OpenAI-compatible API): any Hugging Face id
# ("org/name", e.g. Qwen/..., microsoft/phi-4) or a "local:" name. Provider API ids never contain a slash.
def _is_local_model(model: str) -> bool:
    return model.startswith("local:") or "/" in model


def _default_path() -> Path:
    env = os.environ.get("RP_LLM_CACHE")
    return Path(env) if env else Path("data") / "_llm_cache.sqlite"


class LLMCache:
    """Thread-safe cached completion client; one sqlite file per experiment/project."""

    def __init__(self, path: str | os.PathLike | None = None):
        self.path = Path(path) if path else _default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self.db.commit()
        self.calls = 0
        self.hits = 0
        self.in_tokens = 0
        self.out_tokens = 0
        self.models: set[str] = set()
        self.served: dict[str, str] = {}   # requested id → exact version the provider served

    # ---- public ------------------------------------------------------------------------------
    @staticmethod
    def key(model: str, system: str, user: str, max_tokens: int, temperature, sample: int) -> str:
        return hashlib.sha256(json.dumps([model, system, user, max_tokens, temperature, sample])
                              .encode()).hexdigest()

    def complete(self, system: str, user: str, *, model: str, max_tokens: int = 1024,
                 temperature: float | None = 0.0, sample: int = 0) -> str:
        """Return the (cached) completion text. ``sample`` distinguishes independent repeats."""
        k = self.key(model, system, user, max_tokens, temperature, sample)
        with self._lock:
            key_lock = self._key_locks.setdefault(k, threading.Lock())
        with key_lock:
            with self._lock:
                row = self.db.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
                if row is not None:
                    self.hits += 1
                    return row[0]
            text, tin, tout = self._call(model, system, user, max_tokens, temperature)
            with self._lock:
                self.calls += 1
                self.in_tokens += tin
                self.out_tokens += tout
                self.models.add(model)
                self.db.execute("INSERT OR IGNORE INTO c VALUES (?, ?)", (k, text))  # first write wins
                self.db.commit()
                return self.db.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()[0]

    def stats(self) -> dict:
        return {"models": sorted(self.models), "api_calls": self.calls, "cache_hits": self.hits,
                "input_tokens": self.in_tokens, "output_tokens": self.out_tokens,
                "served_versions": dict(self.served), "cache": str(self.path)}

    def _served(self, requested: str, served) -> None:
        if served:
            with self._lock:
                self.served.setdefault(requested, str(served))

    # ---- providers ---------------------------------------------------------------------------
    def _call(self, model, system, user, max_tokens, temperature) -> tuple[str, int, int]:
        if model.startswith("claude"):
            return self._call_anthropic(model, system, user, max_tokens, temperature)
        if model.startswith("deepseek"):
            return self._call_openai(model, system, user, max_tokens, temperature,
                                     base_url="https://api.deepseek.com", key_env="DEEPSEEK_API_KEY")
        if _is_local_model(model):
            # Open-weight model served by vLLM (or any OpenAI-compatible server) on the researcher's GPU,
            # e.g. over an SSH tunnel: RP_LOCAL_LLM_BASE_URL=http://localhost:8000/v1.
            base = os.environ.get("RP_LOCAL_LLM_BASE_URL")
            if not base:
                raise RuntimeError(f"Cache miss and RP_LOCAL_LLM_BASE_URL is not set: {model} is served "
                                   f"locally; point it at the model server to query it.")
            os.environ.setdefault("RP_LOCAL_LLM_KEY", "local")
            return self._call_openai(model, system, user, max_tokens, temperature,
                                     base_url=base, key_env="RP_LOCAL_LLM_KEY")
        return self._call_openai(model, system, user, max_tokens, temperature)

    def _call_anthropic(self, model, system, user, max_tokens, temperature):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(f"Cache miss and ANTHROPIC_API_KEY is not set: this response is not in "
                               f"the frozen cache ({self.path}); set the key to query {model}.")
        import anthropic
        client = anthropic.Anthropic(max_retries=8)
        kwargs = {} if temperature is None else {"temperature": temperature}
        for attempt in range(6):
            try:
                resp = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                              messages=[{"role": "user", "content": user}], **kwargs)
                break
            except Exception:  # noqa: BLE001 — transient overloads; the SDK already retries 429/5xx
                if attempt == 5:
                    raise
                time.sleep(5 * (attempt + 1))
        text = "".join(getattr(b, "text", "") for b in resp.content)
        self._served(model, getattr(resp, "model", None))
        return text, resp.usage.input_tokens, resp.usage.output_tokens

    def _call_openai(self, model, system, user, max_tokens, temperature, *, base_url=None,
                     key_env="OPENAI_API_KEY"):
        if not os.environ.get(key_env):
            raise RuntimeError(f"Cache miss and {key_env} is not set: this response is not in "
                               f"the frozen cache ({self.path}); set the key to query {model}.")
        import openai
        from openai import OpenAI
        local = key_env == "RP_LOCAL_LLM_KEY"
        client = OpenAI(max_retries=8, api_key=os.environ[key_env], base_url=base_url,
                        **({"timeout": 300.0} if local else {}))
        kwargs = {} if temperature is None else {"temperature": temperature}
        # A self-hosted server is usually reached over a tunnel/VPN that can drop for a few minutes:
        # keep retrying connection failures (about 15 min in total) instead of failing the whole batch.
        attempts = 30 if local else 1
        for attempt in range(attempts):
            try:
                resp = client.chat.completions.create(
                    model=model, max_completion_tokens=max_tokens,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    **kwargs)
                break
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError):
                if attempt == attempts - 1:
                    raise
                time.sleep(min(60, 5 * (attempt + 1)))
        usage = resp.usage
        self._served(model, getattr(resp, "model", None))
        return (resp.choices[0].message.content or "",
                getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0))


SANDBOX_MODULE = "rp_llm_cache.py"


def llm_cache_artifact() -> dict:
    """This module as a code artifact for the experiment sandbox (importable as ``rp_llm_cache``)."""
    return {"filename": SANDBOX_MODULE, "content": Path(__file__).read_text(encoding="utf-8"),
            "language": "python", "purpose": "Frozen LLM response cache (exact offline replay)."}


def uses_llm_cache(code_artifacts: list[dict]) -> bool:
    """True when any generated Python file imports the sandbox cache module."""
    return any(str(a.get("filename", "")).endswith(".py") and a.get("filename") != SANDBOX_MODULE
               and "rp_llm_cache" in (a.get("content") or "") for a in (code_artifacts or []))
