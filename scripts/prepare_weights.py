"""Download the official ViT-B/16 ImageNet-21K checkpoint without loading it into RAM."""

import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
URL = "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz"
DEST = ROOT / "code/pretrained_weights/ViT-B_16_in21k.npz"
STATUS = ROOT / "manifests/pretrained_weights.json"


def save_status(payload):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(STATUS)


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    if DEST.exists():
        digest = sha256(DEST)
        if previous.get("phase") != "complete" or previous.get("sha256") != digest:
            raise RuntimeError("Existing checkpoint has no matching completed manifest. Not overwriting it.")
        expected_md5 = (previous.get("etag") or "").strip('"')
        if re.fullmatch(r"[0-9a-f]{32}", expected_md5):
            source_md5 = hashlib.md5()
            with DEST.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    source_md5.update(chunk)
            if source_md5.hexdigest() != expected_md5:
                raise RuntimeError("Existing checkpoint does not match the source object's MD5 ETag")
            previous["source_etag_md5_verified"] = True
            save_status(previous)
        print("Verified existing pretrained checkpoint", flush=True)
        return
    partial = DEST.with_suffix(".npz.part")
    details = {"phase": "downloading", "source_url": URL, "path": str(DEST)}
    save_status(details)
    count = 0
    started = time.monotonic()
    last_log = started
    digest = hashlib.sha256()
    source_md5 = hashlib.md5()
    request = urllib.request.Request(URL, headers={"User-Agent": "ISLab-PEFT-reproduction/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as target:
        expected = int(response.headers.get("Content-Length", "0"))
        details["etag"] = response.headers.get("ETag")
        details["source_md5_header"] = response.headers.get("x-goog-hash")
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            target.write(chunk)
            digest.update(chunk)
            source_md5.update(chunk)
            count += len(chunk)
            if time.monotonic() - last_log > 10:
                print(f"Downloaded {count / 1024**2:.1f} MiB", flush=True)
                last_log = time.monotonic()
        target.flush()
        os.fsync(target.fileno())
    if expected and count != expected:
        raise RuntimeError(f"Expected {expected} bytes but received {count}")
    expected_md5 = (details.get("etag") or "").strip('"')
    if re.fullmatch(r"[0-9a-f]{32}", expected_md5):
        if source_md5.hexdigest() != expected_md5:
            raise RuntimeError("Downloaded bytes do not match the source object's MD5 ETag")
        details["source_etag_md5_verified"] = True
    with partial.open("rb") as handle:
        if handle.read(4) != b"PK\x03\x04":
            raise RuntimeError("Checkpoint is not an NPZ ZIP archive")
    partial.replace(DEST)
    details.update(phase="complete", bytes=count, sha256=digest.hexdigest(),
                   elapsed_seconds=round(time.monotonic() - started, 2),
                   checksum_scope="Locally computed SHA-256, not an independently published author checksum")
    save_status(details)
    print(json.dumps(details, indent=2), flush=True)


if __name__ == "__main__":
    main()
