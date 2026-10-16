#!/usr/bin/env python3
"""Acquire pinned H3 adapters during image build; never used by a GPU Pod."""
from __future__ import annotations

import argparse
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from verify_assets import AssetError, check_file, check_header, load_lock, safe_path, verify


class HTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        parsed = urlsplit(newurl)
        if parsed.scheme != "https" or parsed.username or parsed.password:
            raise AssetError("Refusing non-HTTPS or credentialed download redirect")
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def make_opener(proxy: str | None):
    handlers = [HTTPSRedirectHandler(), urllib.request.HTTPSHandler(context=ssl.create_default_context())]
    if proxy:
        parsed = urlsplit(proxy)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.query or parsed.fragment:
            raise AssetError("Proxy must be an HTTP or HTTPS proxy URL")
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    # No explicit ProxyHandler means urllib inherits this process's proxy env.
    # Runner shell variables are not automatically passed into Docker RUN.
    return urllib.request.build_opener(*handlers)


def download(asset: dict, cache: Path, opener, attempts: int = 4, sleep=time.sleep) -> Path:
    complete = safe_path(cache, asset["sha256"] + ".safetensors")
    partial = safe_path(cache, asset["sha256"] + ".partial")
    cache.mkdir(parents=True, exist_ok=True)
    if complete.exists():
        try:
            check_file(complete, asset)
            check_header(complete, asset)
            print("Verified cached adapter: " + asset["id"], flush=True)
            return complete
        except AssetError:
            complete.unlink()
            print("Discarded corrupt cache entry: " + asset["id"], flush=True)
    retryable = (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError,
                 http.client.IncompleteRead, http.client.RemoteDisconnected, ssl.SSLError)
    for attempt in range(attempts):
        if partial.exists() and partial.stat().st_size >= asset["size_bytes"]:
            try:
                check_file(partial, asset)
                check_header(partial, asset)
                os.replace(partial, complete)
                return complete
            except AssetError:
                partial.unlink()
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "dingo-h3-distilled-assets/1", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = "bytes={}-".format(offset)
        request = urllib.request.Request(asset["url"], headers=headers)
        print("Acquiring {} (attempt {}/{}, retained bytes {})".format(asset["id"], attempt + 1, attempts, offset), flush=True)
        try:
            with opener.open(request, timeout=60) as response:
                status = response.status
                encoding = response.headers.get("Content-Encoding", "identity").lower()
                if encoding not in {"", "identity"}:
                    raise AssetError("Unexpected transfer content encoding")
                mode = "wb"
                if status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or tuple(map(int, match.groups())) != (offset, asset["size_bytes"] - 1, asset["size_bytes"]):
                        # Never append a range that belongs to a different response.
                        partial.unlink(missing_ok=True)
                        raise AssetError("Invalid resume Content-Range")
                    mode = "ab" if offset else "wb"
                elif status == 200:
                    offset = 0  # Server ignored Range: overwrite, never append.
                else:
                    raise AssetError("Unexpected download HTTP status")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != asset["size_bytes"] - offset:
                    raise AssetError("Unexpected download Content-Length")
                received = offset
                with partial.open(mode) as stream:
                    while True:
                        block = response.read(4 * 1024 * 1024)
                        if not block:
                            break
                        received += len(block)
                        if received > asset["size_bytes"]:
                            raise AssetError("Download exceeds locked file size")
                        stream.write(block)
                if received != asset["size_bytes"]:
                    raise http.client.IncompleteRead(b"", asset["size_bytes"] - received)
            check_file(partial, asset)
            check_header(partial, asset)
            os.replace(partial, complete)
            return complete
        except urllib.error.HTTPError as exc:
            if exc.code == 416:
                partial.unlink(missing_ok=True)
            elif exc.code not in {408, 429, 500, 502, 503, 504}:
                raise AssetError("Download refused with HTTP {} for {}".format(exc.code, asset["id"])) from None
            reason = "HTTP " + str(exc.code)
        except retryable as exc:
            reason = type(exc).__name__
        except AssetError as exc:
            partial.unlink(missing_ok=True)
            # A bounded fresh retry also handles a corrupt cached CDN response.
            reason = str(exc)
        except (ValueError, OSError) as exc:
            raise AssetError("Local download I/O or response format failure: " + type(exc).__name__) from None
        print("Retryable acquisition failure for {}: {}".format(asset["id"], reason), flush=True)
        if attempt + 1 < attempts:
            sleep(min(2 ** attempt, 8))
    raise AssetError("Download exhausted bounded retries for {}; checked cache and partial files are retained; no fallback source used".format(asset["id"]))


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = safe_path(destination.parent, destination.name + ".installing")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def acquire(lock_path: Path, cache: Path, output: Path, proxy: str | None = None) -> dict:
    lock = load_lock(lock_path)
    # Validate ALL destinations and vendored documents before network or copies.
    for item in lock["assets"] + lock.get("documents", []):
        safe_path(output, item["relative_path"])
    for document in lock.get("documents", []):
        check_file(safe_path(lock_path.parent, document["local_source"]), document)
    opener = make_opener(proxy)
    for asset in lock["assets"]:
        source = download(asset, cache, opener)
        atomic_copy(source, safe_path(output, asset["relative_path"]))
    for document in lock.get("documents", []):
        atomic_copy(safe_path(lock_path.parent, document["local_source"]), safe_path(output, document["relative_path"]))
    report = verify(output, lock)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proxy", help="Optional preparation-host proxy; otherwise inherit Runner environment")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        result = verify(args.output, load_lock(args.lock)) if args.verify_only else acquire(args.lock, args.cache_dir, args.output, args.proxy)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (AssetError, OSError, ValueError, KeyError, TypeError) as exc:
        print("Asset preparation failed: " + (str(exc) if isinstance(exc, AssetError) else type(exc).__name__), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
