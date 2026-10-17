"""ResearchPilot deterministic public-data runtime.

Generated helper. Fetches resolved data contracts into data/ and writes data/fetch_manifest.json.
It deliberately never invents synthetic data. The experiment's main.py should read files from data/
and decide, under the runtime policy, whether missing real data is a hard failure.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse


def safe_name(value: str) -> str:
    value = str(value or "dataset").strip().lower()
    value = value.replace("/", "_")
    value = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return value.strip("._-") or "dataset"


# --- Schema verification (mirror of utils.schema_verification, self-contained for the workspace) ---
_RP_MODALITY_SIGNATURES = {"commit_level": ["commit_id", "commit_message", "files_changed", "diff", "patch", "sha", "parents"], "cve_metadata": ["cve_id", "cwe_id", "cve_page", "publish_date", "vulnerability_classification", "access_complexity", "confidentiality_impact", "integrity_impact", "availability_impact", "score", "summary", "ref_link"], "function_level_code": ["func_before", "func_after", "func", "function", "code", "source_code", "snippet"], "tabular": [], "text_classification": ["text", "sentence", "content", "document", "review"]}
_RP_MODALITY_LABELS = {"commit_level": "commit-level metadata dataset", "cve_metadata": "CVE metadata file", "function_level_code": "function-level vulnerability dataset (function source code + a vulnerability label)", "tabular": "tabular dataset", "text_classification": "text-classification dataset", "unknown": "dataset of unknown modality"}
_RP_MODALITY_PRIORITY = ["function_level_code", "text_classification", "cve_metadata", "commit_level", "tabular"]


def _rp_norm_cols(columns):
    out = []
    for col in columns or []:
        name = col.get("name") if isinstance(col, dict) else col
        name = str(name or "").strip().lower()
        if name:
            out.append(name)
    return out


def _rp_classify_modality(columns):
    norm = set(_rp_norm_cols(columns))
    if not norm:
        return "unknown"
    scored = [(len(norm & set(markers)), m) for m, markers in _RP_MODALITY_SIGNATURES.items()]
    best_hits = max((h for h, _ in scored), default=0)
    if best_hits == 0:
        return "tabular" if len(norm) >= 2 else "unknown"
    tied = [m for h, m in scored if h == best_hits]
    return min(tied, key=lambda m: _RP_MODALITY_PRIORITY.index(m) if m in _RP_MODALITY_PRIORITY else len(_RP_MODALITY_PRIORITY))


def _rp_match(aliases, actual):
    aliases = [str(a).strip().lower() for a in (aliases or []) if str(a).strip()]
    for alias in aliases:
        if alias in actual:
            return True
    for col in actual:
        tokens = set(col.replace("-", "_").replace(" ", "_").split("_"))
        if tokens & set(aliases):
            return True
    return False


def _rp_verify_columns(columns, requirements):
    """Return (ok, failure, diagnostic, detected_modality). Pure."""
    present = _rp_norm_cols(columns)
    detected = _rp_classify_modality(columns)
    req = requirements or {}
    required = [c for c in (req.get("requiredColumns") or []) if isinstance(c, dict) and c.get("required", True)]
    reject = [m for m in (req.get("rejectModalities") or []) if m]
    required_modality = req.get("modality") or "unknown"
    if not required and required_modality == "unknown" and not reject:
        return True, None, "", detected
    if not present:
        return False, "empty_dataset", "The acquired dataset has no readable columns.", "unknown"
    actual = set(present)
    missing = [c.get("canonical") for c in required if not _rp_match(c.get("aliases") or [c.get("canonical")], actual)]
    modality_rejected = bool(reject) and detected in reject
    if not missing and not modality_rejected:
        return True, None, "", detected
    preview = ", ".join(present[:12]) + ("…" if len(present) > 12 else "")
    det_label = _RP_MODALITY_LABELS.get(detected, detected)
    req_label = _RP_MODALITY_LABELS.get(required_modality, required_modality)
    if modality_rejected:
        msg = "The provided file is a %s, not a %s." % (det_label, req_label)
        if required_modality == "function_level_code":
            msg += " This experiment requires function source code and vulnerability labels."
        return False, "wrong_modality", msg + " Got columns: %s." % preview, detected
    return (False, "missing_required_columns",
            "The acquired dataset is missing required column(s): %s (needs a %s). Got columns: %s." % (
                ", ".join(m for m in missing if m), req_label, preview), detected)


def _rp_columns_of(result):
    """Best-effort column extraction from a fetch result's sample/output file(s)."""
    cols = result.get("columns")
    if cols:
        return list(cols)
    paths = []
    if result.get("path"):
        paths.append(result["path"])
    paths.extend(result.get("files") or [])
    for p in paths:
        try:
            path = Path(p)
            if path.is_dir():
                continue
            suffix = path.suffix.lower()
            if suffix in (".jsonl", ".json"):
                with path.open("r", encoding="utf-8") as fh:
                    first = fh.readline().strip()
                if first:
                    obj = json.loads(first)
                    if isinstance(obj, dict):
                        return list(obj.keys())
            elif suffix in (".csv", ".tsv"):
                delim = "\t" if suffix == ".tsv" else ","
                with path.open("r", encoding="utf-8", newline="") as fh:
                    header = next(csv.reader(fh, delimiter=delim), [])
                if header:
                    return [str(h) for h in header]
        except Exception:
            continue
    return []


def _sha256_of(path) -> str:
    import hashlib
    try:
        p = Path(path)
        if not p.is_file():
            return ""
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _choose_split(dataset):
    preferred = ("test", "validation", "valid", "train")
    if hasattr(dataset, "keys"):
        keys = list(dataset.keys())
        for key in preferred:
            if key in keys:
                return key, dataset[key]
        if keys:
            key = keys[0]
            return key, dataset[key]
    return "default", dataset


def _write_jsonl(rows, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
    return len(rows)


def _hf_parquet_fallback(dataset_id: str, out_dir: Path, sample_rows: int) -> dict:
    """Load a dataset via its auto-converted Parquet export.

    Modern `datasets` (>=4) refuses legacy *loading-script* datasets (e.g. code_search_net)
    and no longer accepts `trust_remote_code`. Hugging Face auto-converts almost every public
    dataset to Parquet on the `refs/convert/parquet` branch, which we can read directly without
    running any dataset script. This also transparently follows datasets that were renamed.
    """
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_url

    files = HfApi().list_repo_files(dataset_id, repo_type="dataset", revision="refs/convert/parquet")
    parquet = [f for f in files if f.endswith(".parquet")]
    if not parquet:
        raise RuntimeError("no Parquet export available on refs/convert/parquet")

    def _split_of(path: str) -> str:
        parts = path.split("/")
        return parts[1] if len(parts) >= 3 else "default"

    def _config_of(path: str) -> str:
        parts = path.split("/")
        return parts[0] if len(parts) >= 3 else "default"

    # Prefer a held-out split, and stay within a single config so rows share a schema.
    config_name = _config_of(parquet[0])
    in_config = [f for f in parquet if _config_of(f) == config_name]
    chosen = []
    for pref in ("test", "validation", "valid", "train"):
        chosen = [f for f in in_config if _split_of(f) == pref]
        if chosen:
            break
    if not chosen:
        chosen = in_config
    split_name = _split_of(chosen[0])

    rows = []
    for rel in chosen:
        url = hf_hub_url(dataset_id, rel, repo_type="dataset", revision="refs/convert/parquet")
        df = pd.read_parquet(url)
        rows.extend(df.head(sample_rows - len(rows)).to_dict("records"))
        if len(rows) >= sample_rows:
            break
    sample_path = out_dir / "sample.jsonl"
    row_count = _write_jsonl(rows, sample_path)
    if row_count <= 0:
        raise RuntimeError("Parquet export returned zero rows")
    return {
        "status": "ok",
        "sourceType": "huggingface",
        "datasetId": dataset_id,
        "split": split_name,
        "path": str(sample_path),
        "rows": row_count,
        "loader": "parquet-export",
    }


def fetch_huggingface(contract: dict, root: Path) -> dict:
    dataset_id = contract.get("datasetId") or ""
    if not dataset_id:
        raise ValueError("Missing Hugging Face datasetId")
    out_dir = root / safe_name(contract.get("datasetName") or dataset_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_rows = int(os.environ.get("RP_DATA_SAMPLE_ROWS", "2000"))
    last_error = None
    try:
        import inspect
        from datasets import load_dataset
        # `trust_remote_code` was removed in datasets>=4; only pass it where supported.
        extra = {}
        if "trust_remote_code" in inspect.signature(load_dataset).parameters:
            extra["trust_remote_code"] = True
    except Exception as exc:
        load_dataset, extra, last_error = None, {}, exc
    for streaming in (True, False) if load_dataset else ():
        try:
            ds = load_dataset(dataset_id, split=None, streaming=streaming, **extra)
            split_name, split = _choose_split(ds)
            rows = []
            if streaming:
                iterator = iter(split)
                for row in iterator:
                    rows.append(row)
                    if len(rows) >= sample_rows:
                        break
            else:
                count = min(sample_rows, len(split)) if hasattr(split, "__len__") else sample_rows
                for row in split.select(range(count)) if hasattr(split, "select") else split:
                    rows.append(row)
                    if len(rows) >= sample_rows:
                        break
            sample_path = out_dir / "sample.jsonl"
            row_count = _write_jsonl(rows, sample_path)
            if row_count <= 0:
                raise RuntimeError("Hugging Face loader returned zero rows")
            return {
                "status": "ok",
                "sourceType": "huggingface",
                "datasetId": dataset_id,
                "split": split_name,
                "path": str(sample_path),
                "rows": row_count,
            }
        except Exception as exc:
            last_error = exc
    # Default config failed — many benchmarks expose the data only under a named config (e.g.
    # CodeSearchNet's per-language configs). Try each declared config before giving up on the loader.
    if load_dataset:
        try:
            from datasets import get_dataset_config_names
            configs = [c for c in (get_dataset_config_names(dataset_id, **extra) or []) if c][:12]
        except Exception:
            configs = []
        for config in configs:
            try:
                ds = load_dataset(dataset_id, config, split=None, streaming=True, **extra)
                split_name, split = _choose_split(ds)
                rows = []
                for row in iter(split):
                    rows.append(row)
                    if len(rows) >= sample_rows:
                        break
                sample_path = out_dir / "sample.jsonl"
                row_count = _write_jsonl(rows, sample_path)
                if row_count <= 0:
                    continue
                return {
                    "status": "ok",
                    "sourceType": "huggingface",
                    "datasetId": dataset_id,
                    "config": config,
                    "split": split_name,
                    "path": str(sample_path),
                    "rows": row_count,
                }
            except Exception as exc:
                last_error = exc
    # Native loader failed (e.g. legacy loading-script dataset under datasets>=4).
    # Fall back to the auto-converted Parquet export, which needs no dataset script.
    try:
        return _hf_parquet_fallback(dataset_id, out_dir, sample_rows)
    except Exception as exc:
        raise RuntimeError(
            f"Hugging Face fetch failed for {dataset_id}: "
            f"loader={last_error}; parquet-export={exc}"
        )


def fetch_openml(contract: dict, root: Path) -> dict:
    import pandas as pd
    from sklearn.datasets import fetch_openml

    dataset_id = contract.get("datasetId") or ""
    if not str(dataset_id).isdigit():
        raise ValueError("Missing numeric OpenML datasetId")
    out_dir = root / safe_name(contract.get("datasetName") or f"openml_{dataset_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_rows = int(os.environ.get("RP_DATA_SAMPLE_ROWS", "5000"))
    bunch = fetch_openml(data_id=int(dataset_id), as_frame=True, parser="auto")
    if getattr(bunch, "frame", None) is not None:
        df = bunch.frame.copy()
    else:
        df = pd.DataFrame(getattr(bunch, "data", None))
        target = getattr(bunch, "target", None)
        if target is not None:
            df["target"] = target
    if df.empty:
        raise RuntimeError("OpenML loader returned an empty frame")
    df = df.head(sample_rows)
    out_path = out_dir / "data.csv"
    df.to_csv(out_path, index=False)
    return {
        "status": "ok",
        "sourceType": "openml",
        "datasetId": str(dataset_id),
        "path": str(out_path),
        "rows": int(len(df)),
        "columns": list(map(str, df.columns[:50])),
    }


def _extract_archive(path: Path, out_dir: Path) -> list[str]:
    extracted = []
    name = path.name.lower()
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(out_dir)
        extracted = [str(p) for p in out_dir.rglob("*") if p.is_file()]
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as tf:
            tf.extractall(out_dir)
        extracted = [str(p) for p in out_dir.rglob("*") if p.is_file()]
    elif name.endswith(".gz") and not name.endswith((".tar.gz", ".tgz")):
        target = out_dir / path.name[:-3]
        with gzip.open(path, "rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        extracted = [str(target)]
    return extracted


def _download_stream(url: str, raw_path, *, retries: int = 4,
                     connect_timeout: int = 30, read_timeout: int = 120) -> None:
    """Stream ``url`` to ``raw_path``, resuming on drop and retrying transient failures.

    Large public-benchmark files (1GB+) routinely have their connection dropped or throttled
    inside the sandbox before completing, and a single requests.get that fails loses the whole
    download. This resumes from the bytes already on disk via an HTTP Range request and retries
    with backoff, so a flaky large download eventually lands instead of failing the fetch attempt.
    """
    import time

    import requests

    raw_path = Path(raw_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(retries):
        have = raw_path.stat().st_size if raw_path.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, headers=headers,
                              timeout=(connect_timeout, read_timeout)) as resp:
                if resp.status_code == 416:  # Range past EOF — file is already complete on disk.
                    return
                # 206 => server honored the Range and is sending the tail (append). Anything else
                # (200) => full body, so restart the file from scratch to avoid corrupting it.
                if have and resp.status_code == 206:
                    mode = "ab"
                else:
                    mode, have = "wb", 0
                resp.raise_for_status()
                with raw_path.open(mode) as fh:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
            return
        except requests.RequestException as exc:
            # Keep whatever bytes landed so the next attempt resumes instead of restarting.
            last_error = exc
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"download failed after {retries} attempts: {url} ({last_error})")


def fetch_direct(contract: dict, root: Path) -> dict:
    url = contract.get("accessUrl") or ""
    if not url:
        raise ValueError("Missing direct data URL")
    out_dir = root / safe_name(contract.get("datasetName") or Path(urlparse(url).path).stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(url).path).name or "downloaded_data"
    raw_path = out_dir / filename
    _download_stream(url, raw_path)
    files = [str(raw_path)]
    files.extend(_extract_archive(raw_path, out_dir))
    table_files = [
        f for f in files
        if Path(f).suffix.lower() in {".csv", ".tsv", ".json", ".jsonl", ".parquet", ".xlsx", ".xls", ".txt", ".data"}
    ]
    return {
        "status": "ok",
        "sourceType": contract.get("sourceType") or "direct_file",
        "path": str(out_dir),
        "files": table_files or files,
    }


def fetch_github_repo(contract: dict, root: Path) -> dict:
    url = contract.get("accessUrl") or ""
    dataset_id = contract.get("datasetId") or safe_name(url)
    if not url:
        raise ValueError("Missing GitHub repository URL")
    out_dir = root / "repos" / safe_name(dataset_id)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if not out_dir.exists():
        subprocess.run(["git", "clone", "--depth", "1", url, str(out_dir)], check=True, timeout=180)
    files = [str(p) for p in out_dir.rglob("*") if p.is_file()]
    return {
        "status": "ok",
        "sourceType": "github_repo",
        "datasetId": dataset_id,
        "path": str(out_dir),
        "files": files[:200],
    }


def fetch_kaggle(contract: dict, root: Path) -> dict:
    dataset_id = contract.get("datasetId") or ""
    if not dataset_id:
        raise ValueError("Missing Kaggle dataset/competition id")
    # Credentials are required; degrade to a clear per-source failure (recorded in the manifest)
    # rather than aborting the whole fetch when they're absent.
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")) and not (
        Path(os.environ.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))) / "kaggle.json"
    ).is_file():
        raise RuntimeError("Kaggle credentials unavailable — set KAGGLE_USERNAME/KAGGLE_KEY")
    is_comp = dataset_id.startswith("competition:")
    slug = dataset_id.split(":", 1)[1] if is_comp else dataset_id
    out_dir = root / safe_name(slug)
    out_dir.mkdir(parents=True, exist_ok=True)
    if is_comp:
        cmd = ["kaggle", "competitions", "download", "-c", slug, "-p", str(out_dir)]
    else:
        cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(out_dir), "--unzip"]
    try:
        subprocess.run(cmd, check=True, timeout=600)
    except FileNotFoundError:
        raise RuntimeError("Kaggle CLI not installed — `pip install kaggle`")
    # Competition downloads arrive as a zip; unpack any archives so main.py sees real files.
    for archive in list(out_dir.glob("*.zip")):
        _extract_archive(archive, out_dir)
    files = [str(p) for p in out_dir.rglob("*") if p.is_file()]
    table_files = [f for f in files if Path(f).suffix.lower() in
                   {".csv", ".tsv", ".json", ".jsonl", ".parquet", ".xlsx", ".xls", ".txt", ".data"}]
    if not files:
        raise RuntimeError("Kaggle download produced no files")
    return {
        "status": "ok",
        "sourceType": "kaggle",
        "datasetId": dataset_id,
        "path": str(out_dir),
        "files": (table_files or files)[:200],
    }


def fetch_gdrive(contract: dict, root: Path) -> dict:
    file_id = contract.get("datasetId") or ""
    if not file_id:
        raise ValueError("Missing Google Drive file id")
    out_dir = root / safe_name(contract.get("datasetName") or f"gdrive_{file_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except Exception:
        raise RuntimeError("gdown not installed — `pip install gdown`")
    target = out_dir / "download.bin"
    gdown.download(id=file_id, output=str(target), quiet=True)
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError("Google Drive download produced no file (check sharing permissions)")
    files = [str(target)] + _extract_archive(target, out_dir)
    table_files = [f for f in files if Path(f).suffix.lower() in
                   {".csv", ".tsv", ".json", ".jsonl", ".parquet", ".xlsx", ".xls", ".txt", ".data"}]
    return {"status": "ok", "sourceType": "gdrive", "datasetId": file_id,
            "path": str(out_dir), "files": table_files or files}


def fetch_zenodo(contract: dict, root: Path) -> dict:
    import requests

    record_id = contract.get("datasetId") or ""
    if not str(record_id).isdigit():
        raise ValueError("Missing numeric Zenodo record id")
    out_dir = root / safe_name(contract.get("datasetName") or f"zenodo_{record_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=60)
    meta.raise_for_status()
    entries = (meta.json() or {}).get("files") or []
    if not entries:
        raise RuntimeError(f"Zenodo record {record_id} lists no files")
    saved = []
    for entry in entries:
        link = (entry.get("links") or {}).get("self") or (entry.get("links") or {}).get("download")
        fname = entry.get("key") or entry.get("filename") or "file"
        if not link:
            continue
        raw_path = out_dir / fname
        _download_stream(link, raw_path)
        saved.append(str(raw_path))
        saved.extend(_extract_archive(raw_path, out_dir))
    table_files = [f for f in saved if Path(f).suffix.lower() in
                   {".csv", ".tsv", ".json", ".jsonl", ".parquet", ".xlsx", ".xls", ".txt", ".data"}]
    if not saved:
        raise RuntimeError(f"Zenodo record {record_id} produced no downloadable files")
    return {"status": "ok", "sourceType": "zenodo_record", "datasetId": str(record_id),
            "path": str(out_dir), "files": table_files or saved}


def fetch_contract(contract: dict, root: Path) -> dict:
    source_type = contract.get("sourceType")
    if source_type == "huggingface":
        return fetch_huggingface(contract, root)
    if source_type == "openml":
        return fetch_openml(contract, root)
    if source_type in {"direct_file", "direct_archive"}:
        return fetch_direct(contract, root)
    if source_type == "github_repo":
        return fetch_github_repo(contract, root)
    if source_type == "kaggle":
        return fetch_kaggle(contract, root)
    if source_type == "gdrive":
        return fetch_gdrive(contract, root)
    if source_type == "zenodo_record":
        return fetch_zenodo(contract, root)
    raise ValueError(f"Unsupported data contract sourceType: {source_type}")


def fetch_all(contracts: list[dict], data_root: str | Path = "data") -> dict:
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"ok": [], "failed": []}
    for contract in contracts:
        label = contract.get("datasetName") or contract.get("datasetId") or contract.get("accessUrl")
        print(f"[fetch_data.py] Fetching {label} ({contract.get('sourceType')})")
        try:
            result = fetch_contract(contract, root)
            result["contract"] = contract
            # Lineage: stamp a fingerprint + record count so provenance is reproducible.
            primary = result.get("path")
            if result.get("files"):
                primary = result["files"][0]
            result.setdefault("checksum", _sha256_of(primary))
            if result.get("rows") is not None:
                result.setdefault("recordCount", result.get("rows"))
            prov = contract.get("provenance") or {}
            for key in ("version", "license", "citation"):
                if prov.get(key):
                    result.setdefault(key, prov[key])
            # Schema gate: the bytes downloaded, but is it the RIGHT dataset for this experiment?
            requirements = contract.get("dataRequirements")
            cols = _rp_columns_of(result)
            result["columns"] = result.get("columns") or cols
            ok, failure, diagnostic, detected = _rp_verify_columns(cols, requirements)
            if not ok and failure in ("missing_required_columns", "wrong_modality"):
                manifest["failed"].append({
                    "contract": contract, "error": diagnostic, "failure": failure,
                    "detectedModality": detected, "columns": cols,
                })
                print(f"[fetch_data.py][REJECT] {label}: {diagnostic}")
            else:
                manifest["ok"].append(result)
                print(f"[fetch_data.py] OK: {label} -> {result.get('path')}")
        except Exception as exc:
            manifest["failed"].append({"contract": contract, "error": str(exc), "failure": "download_failed"})
            print(f"[fetch_data.py][WARN] Failed: {label}: {exc}")
    manifest_path = root / "fetch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[fetch_data.py] Wrote {manifest_path} ({len(manifest['ok'])} ok, {len(manifest['failed'])} failed)")
    return manifest


def _rp_simplify(value):
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _rp_identity_match(entry, key_n):
    """True if normalized key_n matches this ok manifest entry's contract identity."""
    contract = entry.get("contract") or {}
    for cand in (contract.get("datasetName"), contract.get("datasetId"),
                 contract.get("rqId"), contract.get("experimentId"), entry.get("datasetId")):
        if cand and _rp_simplify(cand) == key_n:
            return True
    name_n = _rp_simplify(contract.get("datasetName"))
    return bool(name_n) and (key_n in name_n or name_n in key_n)


def load_manifest(data_root="data"):
    """Return the parsed data/fetch_manifest.json (``{"ok": [...], "failed": [...]}``).

    Returns an empty manifest if fetch_data.py has not run yet, so callers can branch on the
    contents without a try/except. Each ``ok`` entry carries ``path`` and the originating
    ``contract`` (datasetName / rqId / experimentId / sourceType / columns).
    """
    manifest_path = Path(data_root) / "fetch_manifest.json"
    if not manifest_path.exists():
        return {"ok": [], "failed": []}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data.setdefault("ok", [])
    data.setdefault("failed", [])
    return data


def resolve_data_path(key, data_root="data", *, required=True, prefer_file=True):
    """Return the path fetch_data.py ACTUALLY wrote for a dataset — never hardcode filenames.

    ``key`` is matched (case-insensitive, punctuation-insensitive) against each ok entry's
    datasetName / datasetId / rqId / experimentId in data/fetch_manifest.json. The on-disk
    layout differs by source (sample.jsonl for Hugging Face, data.csv for OpenML, a directory
    for archives/repos), so resolving through the manifest is the only reliable way to load.

    Raises FileNotFoundError listing what IS available when nothing matches, so a path mismatch
    fails loudly instead of a confusing "data/foo.csv not found". Pass required=False for None.
    """
    root = Path(data_root)
    manifest_path = root / "fetch_manifest.json"
    if not manifest_path.exists():
        if not required:
            return None
        raise FileNotFoundError(f"{manifest_path} not found — run fetch_data.py before main.py.")
    manifest = load_manifest(data_root)
    ok = manifest.get("ok") or []
    key_n = _rp_simplify(key)
    for entry in ok:
        if _rp_identity_match(entry, key_n):
            if entry.get("files") and prefer_file:
                return entry["files"][0]
            return entry.get("path")
    if not required:
        return None
    available = "; ".join(
        f"{(e.get('contract') or {}).get('datasetName') or e.get('datasetId')!r} -> {e.get('path')}"
        for e in ok
    ) or "(none fetched successfully)"
    rejected = "; ".join(
        f"{(e.get('contract') or {}).get('datasetName')!r} ({e.get('failure')})"
        for e in (manifest.get("failed") or [])
    )
    msg = f"No fetched dataset matches {key!r}. Available (ok): {available}."
    if rejected:
        msg += f" Rejected/failed: {rejected}."
    raise FileNotFoundError(msg)
