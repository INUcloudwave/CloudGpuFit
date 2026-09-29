#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path, manifest_name: str) -> list[str]:
    manifest_path = root / manifest_name
    if not manifest_path.is_file():
        return [f"missing manifest: {manifest_path}"]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for artifact in payload.get("artifacts", {}).values():
        path = root / artifact["path"]
        if not path.is_file():
            errors.append(f"missing artifact: {path}")
            continue
        if path.stat().st_size != int(artifact["size_bytes"]):
            errors.append(f"size mismatch: {path}")
        if sha256(path) != artifact["sha256"]:
            errors.append(f"sha256 mismatch: {path}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="g6.xlarge")
    args = parser.parse_args()
    target_root = Path(__file__).resolve().parent / "targets" / args.target
    checks = (
        (target_root / "compiled_external_model", "compiled_external_manifest.json"),
        (target_root / "compiled_fusion_model", "compiled_fusion_manifest.json"),
    )
    errors = [
        error
        for root, manifest in checks
        for error in verify_manifest(root, manifest)
    ]
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Compiled runtime artifacts: OK ({args.target})")


if __name__ == "__main__":
    main()
