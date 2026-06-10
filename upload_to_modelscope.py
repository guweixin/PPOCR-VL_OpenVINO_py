"""Upload the OpenVINO IR model folders to ModelScope (public repos).

Usage:
    python upload_to_modelscope.py
Auth: uses the cached login under ~/.modelscope/credentials (FionaGu1019).
"""
from pathlib import Path

from modelscope.hub.api import HubApi

USER = "FionaGu1019"
ROOT = Path(r"D:\My_internal_work\20260106_OCR\PPOCR-VL\PP-OCR-OV-models")
MODELS = ["PP-DocLayoutV3-ov", "PaddleOCR-VL-1.5-ov"]
IGNORE = ["_backup*", "*/_backup*", "**/_backup*", ".ov_cache*", "output/*"]

api = HubApi()

for name in MODELS:
    repo_id = f"{USER}/{name}"
    local = ROOT / name
    print(f"\n=== {repo_id}  <-  {local} ===", flush=True)
    if not local.is_dir():
        print(f"  SKIP: local dir not found: {local}", flush=True)
        continue
    try:
        url = api.create_repo(
            repo_id=repo_id,
            visibility="public",
            repo_type="model",
            license="Apache License 2.0",
            exist_ok=True,
        )
        print(f"  repo ready: {url}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  create_repo note: {exc}", flush=True)

    info = api.upload_folder(
        repo_id=repo_id,
        folder_path=str(local),
        repo_type="model",
        commit_message=f"Upload {name} (OpenVINO IR)",
        ignore_patterns=IGNORE,
    )
    print(f"  uploaded: {info}", flush=True)

print("\nALL DONE", flush=True)
