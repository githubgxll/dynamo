#!/usr/bin/env python3
"""Prepare official VBench assets, or reproduce them from a complete reviewed lock.

No model is deserialized here. This script deliberately has no token/login logic.
Runtime consumers may import verify_manifest without performing network access.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


class AssetError(RuntimeError):
    pass


SHA256 = re.compile(r"^[0-9a-f]{64}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")
SOURCE_EXTENSIONS = {".py", ".yaml", ".yml", ".json", ".txt", ".md", ".cpp", ".cu", ".h", ".sh"}
LICENSE_NAMES = {"license", "license.txt", "license.md", "copying", "notice", "notice.txt"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise AssetError("invalid relative path in archive or manifest")
    result = PurePosixPath(value)
    if result.is_absolute() or any(part in {"..", ".", ""} for part in value.rstrip("/").split("/")):
        raise AssetError("unsafe relative path in archive or manifest")
    return result


def rooted(root, relative):
    root = Path(root).resolve()
    path = root.joinpath(*relative_path(relative).parts)
    # Existing parent links must not escape the selected root.
    if not path.parent.resolve().is_relative_to(root):
        raise AssetError("path escapes selected root")
    return path


def valid_https(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise AssetError("asset catalog must contain credential-free canonical HTTPS URLs")
    return url


class HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Signed CDN redirects may contain a query; they are never logged or saved.
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise AssetError("refused non-HTTPS download redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_error_detail(exc):
    """Expose error classes/numeric codes, never URLs or library error text."""
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}"
    if isinstance(exc, AssetError):
        return str(exc)
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    detail = type(exc).__name__
    if reason is not exc:
        detail += f" (reason_type={type(reason).__name__})"
    for attribute in ("errno", "verify_code"):
        value = getattr(reason, attribute, None)
        if type(value) is int:
            detail += f" {attribute}={value}"
    return detail


class TruncatedDownload(AssetError):
    """The server ended a response before its advertised Content-Length."""


def retryable_download_error(exc):
    """Retry transport interruptions, never authorization or integrity failures."""
    import errno
    import socket
    import ssl

    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 429, 500, 502, 503, 504}
    reason = exc
    for _ in range(3):
        if not isinstance(reason, urllib.error.URLError):
            break
        reason = reason.reason
    if isinstance(reason, (ssl.SSLCertVerificationError, ssl.CertificateError)):
        return False
    if isinstance(reason, ssl.SSLError):
        return (isinstance(reason, ssl.SSLEOFError) or reason.errno == ssl.SSL_ERROR_EOF
                or getattr(reason, "reason", None) == "UNEXPECTED_EOF_WHILE_READING")
    if isinstance(reason, socket.gaierror):
        return reason.errno == socket.EAI_AGAIN
    if isinstance(reason, (TruncatedDownload, http.client.IncompleteRead,
                           ConnectionResetError, ConnectionAbortedError, TimeoutError)):
        return True
    return isinstance(reason, OSError) and (
        reason.errno in {errno.ECONNRESET, errno.ECONNABORTED, errno.ETIMEDOUT}
        or getattr(reason, "winerror", None) in {10053, 10054, 10060})


def download(url, destination, *, expected_sha256=None, max_bytes=2 * 1024**3, timeout=90, deadline=1800, event=None):
    """Download from one canonical URL, with at most three bounded attempts.

    Optional event(dict) receives only sanitized host/status/hash/size evidence.
    Retries restart from byte zero, share the original elapsed-time deadline,
    and never replace a destination until the complete download is verified.
    """
    valid_https(url)
    if timeout <= 0 or deadline <= 0 or max_bytes <= 0:
        raise AssetError("download timeout, deadline, and size limit must be positive")
    if expected_sha256 is not None and (not isinstance(expected_sha256, str) or not SHA256.fullmatch(expected_sha256)):
        raise AssetError("invalid expected download SHA256")
    host = urllib.parse.urlsplit(url).hostname
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    started = time.monotonic()
    opener = urllib.request.build_opener(HTTPSRedirect())

    def emit(kind, attempt, **fields):
        record = {"event": kind, "host": host, "attempt": attempt, "max_attempts": 3,
                  "elapsed_seconds": round(time.monotonic() - started, 3), **fields}
        detail = fields.get("reason", kind)
        print(f"Asset download attempt {attempt}/3 {host}: {kind}; {detail}", flush=True)
        if event is not None:
            event(record)

    def remaining():
        value = deadline - (time.monotonic() - started)
        if value <= 0:
            raise AssetError("download exceeded total elapsed-time limit")
        return value

    for attempt in range(1, 4):
        temporary.unlink(missing_ok=True)
        emit("start", attempt)
        digest, size = hashlib.sha256(), 0
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "VBench-offline-assets/1"})
            with opener.open(request, timeout=min(timeout, remaining())) as response, temporary.open("wb") as output:
                remaining()
                status = response.getcode()
                if status != 200:
                    raise urllib.error.HTTPError(url, status, "unexpected download status", {}, None)
                raw_length = response.headers.get("Content-Length")
                try:
                    length = int(raw_length) if raw_length is not None else None
                except (TypeError, ValueError):
                    raise AssetError("invalid download Content-Length") from None
                if length is not None and (length < 0 or length > max_bytes):
                    raise AssetError("download exceeds configured size limit")
                while True:
                    budget = min(timeout, remaining())
                    # HTTPResponse.read1 performs at most one underlying read;
                    # update its socket timeout to the remaining shared budget.
                    stream = getattr(getattr(response, "fp", None), "raw", None)
                    sock = getattr(stream, "_sock", None)
                    if sock is not None:
                        sock.settimeout(budget)
                    reader = getattr(response, "read1", response.read)
                    chunk = reader(1024 * 1024)
                    remaining()
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes or (length is not None and size > length):
                        raise AssetError("download exceeds configured or advertised size limit")
                    digest.update(chunk)
                    output.write(chunk)
                if length is not None and size < length:
                    raise TruncatedDownload("download ended before advertised Content-Length")
            checksum = digest.hexdigest()
            if not size:
                raise AssetError("empty download")
            if expected_sha256 and checksum != expected_sha256:
                raise AssetError("download SHA256 mismatch")
            remaining()
            os.replace(temporary, destination)
        except (OSError, ValueError, http.client.HTTPException, urllib.error.URLError, AssetError) as exc:
            temporary.unlink(missing_ok=True)
            detail = download_error_detail(exc)
            delay = (2, 5)[attempt - 1] if attempt < 3 else 0
            retry = attempt < 3 and retryable_download_error(exc)
            if retry and deadline - (time.monotonic() - started) <= delay:
                retry = False
                detail += "; total elapsed-time limit leaves no retry budget"
            if retry:
                emit("retry", attempt, reason=detail, bytes_received=size, delay_seconds=delay)
                time.sleep(delay)
                continue
            emit("failure", attempt, reason=detail, bytes_received=size)
            raise AssetError(f"download failed for {host} after {attempt} attempt(s): {detail}; check Runner/build-stage egress and proxy; no fallback source was used") from None
        result = {"sha256": checksum, "bytes": size}
        emit("success", attempt, **result)
        return result


def checked_members(archive, *, max_files=10000, max_total=1024**3, max_ratio=500):
    entries = archive.infolist()
    if len(entries) > max_files or sum(info.file_size for info in entries) > max_total:
        raise AssetError("archive exceeds file count or expanded size limit")
    seen = set()
    for info in entries:
        path = relative_path(info.filename)
        key = path.as_posix().casefold()
        if key in seen:
            raise AssetError("archive has duplicate or case-colliding entries")
        seen.add(key)
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
            raise AssetError("archive contains link or special file")
        if info.flag_bits & 1 or info.file_size > max_ratio * max(1, info.compress_size):
            raise AssetError("encrypted or excessive-compression archive member")
    return entries


def extract_archive(archive_path, destination, *, prefix=None, member=None, source_kind=None):
    """Extract selected runtime code or exactly one RAFT checkpoint; never extractall."""
    selected = []
    with zipfile.ZipFile(archive_path) as archive:
        for info in checked_members(archive):
            if info.is_dir():
                continue
            name = info.filename
            if member is not None:
                if name != member:
                    continue
                target = Path(destination)
            else:
                if not prefix or not name.startswith(prefix + "/"):
                    continue
                name = name[len(prefix) + 1:]
                relative = relative_path(name)
                license_file = relative.name.lower() in LICENSE_NAMES
                if source_kind == "vbench":
                    inside = relative.parts[0] in {"vbench", "vbench2_beta_long"}
                    if not ((inside and (relative.suffix.lower() in SOURCE_EXTENSIONS or license_file)) or (len(relative.parts) == 1 and license_file)):
                        continue
                elif source_kind == "dino":
                    if len(relative.parts) != 1 or not (relative.suffix == ".py" or license_file):
                        continue
                else:
                    raise AssetError("unknown source selection policy")
                target = rooted(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            if target.stat().st_size != info.file_size:
                raise AssetError("archive member size mismatch")
            selected.append(target)
    if not selected or (member is not None and len(selected) != 1):
        raise AssetError("required source files or checkpoint member missing from archive")
    return selected


def manifest_entries(assets_root, source_root):
    records = []
    roots = {"assets": Path(assets_root).resolve(), "source": Path(source_root).resolve()}
    for root_name, root in roots.items():
        for file in sorted(root.rglob("*")):
            if file.is_symlink():
                resolved = file.resolve(strict=True)
                if not resolved.is_relative_to(root) or not resolved.is_file():
                    raise AssetError("asset link leaves its root or is not a file")
                record = {"kind": "symlink", "target": resolved.relative_to(root).as_posix()}
            elif file.is_file():
                record = {"kind": "file"}
            elif file.is_dir():
                continue
            else:
                raise AssetError("unsupported filesystem object")
            record.update(root=root_name, path=file.relative_to(root).as_posix(), sha256=file_hash(file), bytes=file.stat().st_size)
            records.append(record)
    return records


def validate_file_records(records):
    if not isinstance(records, list) or not records:
        raise AssetError("manifest has no locked files")
    seen = set()
    for item in records:
        if item.get("root") not in {"assets", "source"} or item.get("kind") not in {"file", "symlink"}:
            raise AssetError("unknown manifest root or file kind")
        relative_path(item["path"])
        key = (item["root"], item["path"].casefold())
        if key in seen:
            raise AssetError("duplicate manifest file")
        seen.add(key)
        if not isinstance(item.get("sha256"), str) or not SHA256.fullmatch(item["sha256"]) or type(item.get("bytes")) is not int or item["bytes"] < 0:
            raise AssetError("file lacks complete SHA256/size lock")
        if item["kind"] == "symlink":
            relative_path(item["target"])


def verify_manifest(lock, assets_root, source_root):
    """Verify every locked file and reject additional files; performs no network IO."""
    manifest = read_json(lock) if isinstance(lock, (str, Path)) else lock
    if manifest.get("schema_version") != 1 or manifest.get("status") != "COMPLETE_ASSET_LOCK":
        raise AssetError("not a complete supported asset lock")
    validate_file_records(manifest.get("files"))
    actual = manifest_entries(assets_root, source_root)
    expected = sorted(manifest["files"], key=lambda x: (x["root"], x["path"]))
    if sorted(actual, key=lambda x: (x["root"], x["path"])) != expected:
        actual_by_key = {(x["root"], x["path"]): x for x in actual}
        expected_by_key = {(x["root"], x["path"]): x for x in expected}
        changed = ["/".join(k) for k in sorted(set(actual_by_key) | set(expected_by_key)) if actual_by_key.get(k) != expected_by_key.get(k)]
        raise AssetError("asset/source manifest mismatch: " + ", ".join(changed[:8]))
    return {"status": "PASS_ASSET_MANIFEST", "files": len(actual), "bytes": sum(x["bytes"] for x in actual if x["kind"] == "file"), "revisions": manifest.get("revisions", {})}


def resolve_hf_revision(repo_id, directory, *, timeout=90, deadline=1800, event=None):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id):
        raise AssetError("invalid Hugging Face repository")
    file = Path(directory) / "hf-revision.json"
    # Resolve the mutable name on each execution; only the resulting immutable
    # checkpoint URL is eligible for the persistent download cache.
    download(f"https://huggingface.co/api/models/{repo_id}/revision/main", file,
             max_bytes=4 * 1024**2, timeout=timeout, deadline=deadline, event=event)
    revision = read_json(file).get("sha", "")
    if not REVISION.fullmatch(revision):
        raise AssetError("Hugging Face API returned no immutable commit")
    return revision


def validate_checkpoint(path, *, expected_sha256=None):
    """Reject an HTML error page/LFS pointer without unpickling untrusted weights."""
    with Path(path).open("rb") as file:
        signature = file.read(4)
    if not (signature.startswith(b"PK\x03\x04") or (len(signature) >= 2 and signature[0] == 0x80 and 2 <= signature[1] <= 5)):
        raise AssetError("checkpoint lacks expected PyTorch ZIP/pickle signature; not accepting an HTML page or LFS pointer")
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str) or not SHA256.fullmatch(expected_sha256):
            raise AssetError("invalid checkpoint SHA256 in catalog")
        if file_hash(path) != expected_sha256:
            raise AssetError("checkpoint SHA256 mismatch")


def run(args):
    from asset_cache import CacheError, acquire_cached

    catalog = read_json(args.catalog)
    catalog_digest = file_hash(args.catalog)
    if catalog.get("schema_version") != 1:
        raise AssetError("unsupported catalog schema")
    evidence, output, source = Path(args.evidence), Path(args.output), Path(args.vbench_source)
    if output.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(output.resolve()):
        raise AssetError("asset and source roots must be separate")
    cache_dir = getattr(args, "cache_dir", None)
    if cache_dir:
        cache_root = Path(cache_dir).resolve()
        for root in (output, source, evidence):
            if cache_root.is_relative_to(root.resolve()) or root.resolve().is_relative_to(cache_root):
                raise AssetError("download cache must be separate from assets, source and evidence")
    locked = None
    if args.phase == "publish":
        if not args.lock or not Path(args.lock).is_file():
            raise AssetError("publish requires the complete assets.lock.json from prepare")
        locked = read_json(args.lock)
        if locked.get("status") != "COMPLETE_ASSET_LOCK" or locked.get("catalog_sha256") != catalog_digest:
            raise AssetError("lock is incomplete or does not match the exact catalog")
        validate_file_records(locked.get("files"))
    for root in (output, source):
        if root.exists() and any(root.iterdir()):
            raise AssetError("asset/source output directory must be empty to prevent stale or untracked files")
        root.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    current_id = "configuration"

    def record_event(record):
        entry = {"time_utc": datetime.now(timezone.utc).isoformat(), "asset_id": current_id, **record}
        line = json.dumps(entry, ensure_ascii=True)
        # The failed RUN's filesystem is not exported. Keep the same sanitized
        # evidence in CI stdout as well as the successful preparation artifact.
        print("Asset transfer: " + line, flush=True)
        with (evidence / "asset-acquisition.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    proxies = urllib.request.getproxies()
    record_event({"event": "configuration", "cache_enabled": bool(cache_dir),
                  "https_proxy_configured": bool(proxies.get("https")),
                  "http_proxy_configured": bool(proxies.get("http")),
                  "no_proxy_configured": bool(proxies.get("no"))})
    expected_downloads = {x["id"]: x for x in locked.get("downloads", [])} if locked else {}
    expected_ids = {x["id"] for x in catalog["sources"] + catalog["weights"]}
    if locked and (set(expected_downloads) != expected_ids or len(locked["downloads"]) != len(expected_ids)):
        raise AssetError("lock does not cover every source archive and weight download")
    downloads, revisions = [], {}
    with tempfile.TemporaryDirectory(prefix="vbench-assets-") as workspace:
        temporary = Path(workspace)
        for item in catalog["sources"] + catalog["weights"]:
            item_id, url, revision = item["id"], item.get("url"), item.get("revision")
            current_id = item_id
            if item.get("hf_repository"):
                revision = expected_downloads.get(item_id, {}).get("revision") if locked else resolve_hf_revision(
                    item["hf_repository"], temporary, timeout=args.timeout, deadline=args.deadline, event=record_event)
                if not isinstance(revision, str) or not REVISION.fullmatch(revision):
                    raise AssetError("AMT needs a locked immutable Hugging Face commit")
                url = f"https://huggingface.co/{item['hf_repository']}/resolve/{revision}/{item['hf_filename']}"
            expected = expected_downloads.get(item_id, {})
            if locked and (expected.get("url") != url or expected.get("revision") != revision or not isinstance(expected.get("sha256"), str) or not SHA256.fullmatch(expected["sha256"]) or type(expected.get("bytes")) is not int or expected["bytes"] <= 0):
                raise AssetError(f"invalid download lock for {item_id}")
            known_sha = item.get("sha256")
            if known_sha and locked and known_sha != expected["sha256"]:
                raise AssetError(f"catalog/upstream checksum disagrees with lock for {item_id}")
            print(f"Acquiring {item_id} from {urllib.parse.urlsplit(url).hostname}", flush=True)
            archive = temporary / item_id
            staged = {}

            def validate_acquired(acquired):
                # A rejected cache entry must never leave partly extracted files
                # in final output. Each validation attempt gets its own staging.
                if item.get("source_kind"):
                    target = Path(tempfile.mkdtemp(prefix="source-", dir=temporary))
                    extract_archive(acquired, target, prefix=item["archive_prefix"], source_kind=item["source_kind"])
                elif item.get("archive_member"):
                    target = Path(tempfile.mkdtemp(prefix="checkpoint-", dir=temporary)) / "weight.pth"
                    extract_archive(acquired, target, member=item["archive_member"])
                    validate_checkpoint(target, expected_sha256=item.get("checkpoint_sha256"))
                else:
                    target = acquired
                    validate_checkpoint(target, expected_sha256=item.get("checkpoint_sha256"))
                staged["path"] = target

            try:
                result = acquire_cached(url, archive, cache_dir=cache_dir, namespace=catalog_digest,
                    expected_sha256=expected.get("sha256") or known_sha,
                    expected_bytes=expected.get("bytes") if locked else None,
                    max_bytes=item.get("max_bytes", 2 * 1024**3), download_fn=download,
                    validate_fn=validate_acquired, event=record_event, timeout=args.timeout, deadline=args.deadline)
            except CacheError as exc:
                raise AssetError(str(exc)) from None
            record = {"id": item_id, "url": url, "revision": revision, **result}
            downloads.append(record)
            if revision:
                revisions[item_id] = revision
            if item.get("source_kind"):
                destination = source if item["root"] == "source" else rooted(output, item["path"])
                shutil.copytree(staged["path"], destination, dirs_exist_ok=True)
            else:
                destination = rooted(output, item["path"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(staged["path"], destination)
        for link in catalog.get("links", []):
            path, target = rooted(output, link["path"]), rooted(output, link["target"])
            if not target.is_file():
                raise AssetError("symlink target checkpoint missing")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(os.path.relpath(target, path.parent))
    files = manifest_entries(output, source)
    manifest = {"schema_version": 1, "status": "COMPLETE_ASSET_LOCK", "created_utc": datetime.now(timezone.utc).isoformat(), "catalog_sha256": catalog_digest, "revisions": revisions, "downloads": downloads, "files": files}
    if locked:
        verify_manifest(locked, output, source)
        manifest = locked  # Preserve original trust-on-first-use timestamp and bytes.
    write_json(evidence / "assets.lock.json", manifest)
    license_files = []
    for item in files:
        if PurePosixPath(item["path"]).name.lower() in LICENSE_NAMES:
            root = output if item["root"] == "assets" else source
            destination = evidence / "licenses" / item["root"] / item["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(rooted(root, item["path"]), destination)
            license_files.append(item)
    write_json(evidence / "asset-licenses.json", {"status": "LICENSE_EVIDENCE_NOT_LEGAL_APPROVAL", "catalog_entries": [{"id": x["id"], "license": x.get("license"), "license_url": x.get("license_url"), "checkpoint_terms": x.get("checkpoint_terms")} for x in catalog["sources"] + catalog["weights"]], "copied_license_files": license_files})
    result = verify_manifest(manifest, output, source)
    write_json(evidence / "asset-verification.json", result)
    print(json.dumps({"phase": args.phase, **result}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "publish"), required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--lock")
    parser.add_argument("--output", required=True)
    parser.add_argument("--vbench-source", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--cache-dir", help="Optional BuildKit cache mount; never part of the runtime assets")
    parser.add_argument("--timeout", type=int, default=90, help="Per-socket-operation timeout in seconds")
    parser.add_argument("--deadline", type=int, default=1800, help="Per-download elapsed-time limit in seconds")
    args = parser.parse_args()
    try:
        run(args)
    except (AssetError, OSError, ValueError, zipfile.BadZipFile, KeyError) as exc:
        print(f"Asset preparation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
