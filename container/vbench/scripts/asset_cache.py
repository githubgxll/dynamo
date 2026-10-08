"""Verified per-asset download cache; its receipts are never a trust anchor.

The caller supplies the download and content-validation implementations. BuildKit's
sharing=locked cache mount serializes access; this module does not take a second
lock. Every reuse hashes the bytes again and enforces the caller's current lock.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from urllib.parse import urlsplit


class CacheError(RuntimeError):
    pass


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FIELDS = {"schema_version", "namespace", "url", "sha256", "bytes"}
_CHUNK = 1024 * 1024


def _no_links(path):
    """Reject symlinks and Windows junctions, including existing ancestors."""
    path = Path(os.path.abspath(path))
    for candidate in (path, *path.parents):
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024):
            raise CacheError("asset cache path contains a link or reparse point")
    return path


def _regular(path):
    _no_links(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise CacheError("asset cache entry is not a regular file")


def _fingerprint(path, *, expected_sha256, expected_bytes, max_bytes):
    _regular(path)
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            size += len(chunk)
            if size > max_bytes:
                raise CacheError("asset exceeds configured size limit")
            digest.update(chunk)
    if not size:
        raise CacheError("empty asset")
    checksum = digest.hexdigest()
    if expected_sha256 is not None and checksum != expected_sha256:
        raise CacheError("asset SHA256 mismatch")
    if expected_bytes is not None and size != expected_bytes:
        raise CacheError("asset byte count mismatch")
    return {"sha256": checksum, "bytes": size}


def _remove_entry(path):
    # unlink removes a leaf symlink itself; never recurse or follow its target.
    _no_links(path.parent)
    path.unlink(missing_ok=True)


def _atomic_copy(source, destination):
    _no_links(source)
    _no_links(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".asset-copy-", delete=False) as output:
            temporary = Path(output.name)
            with source.open("rb") as source_file:
                shutil.copyfileobj(source_file, output, length=_CHUNK)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CacheError("asset cache receipt has duplicate fields")
        result[key] = value
    return result


def _read_receipt(path, *, namespace, url):
    _regular(path)
    if path.stat().st_size > 16384:
        raise CacheError("asset cache receipt exceeds size limit")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise CacheError("asset cache receipt schema mismatch")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["namespace"] != namespace or value["url"] != url:
        raise CacheError("asset cache receipt identity mismatch")
    if not isinstance(value["sha256"], str) or not _SHA256.fullmatch(value["sha256"]) or type(value["bytes"]) is not int or value["bytes"] < 1:
        raise CacheError("asset cache receipt fingerprint is invalid")
    return value


def _write_receipt(path, value):
    _no_links(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".asset-receipt-", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def acquire_cached(url, destination, *, cache_dir=None, namespace,
                   expected_sha256=None, expected_bytes=None,
                   max_bytes=2 * 1024**3, download_fn, validate_fn, event=None,
                   **download_options):
    """Download/validate one asset, preserving earlier successes across failed RUNs.

``event`` receives a sanitized dict. ``validate_fn(Path)`` must be idempotent:
an invalid cached entry is removed and the same official URL is downloaded once.
Download/validation exceptions from that fresh attempt propagate unchanged.
"""
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or any(character.isspace() for character in url):
        raise CacheError("asset URL must be credential-free canonical HTTPS")
    if not isinstance(namespace, str) or not _SHA256.fullmatch(namespace):
        raise CacheError("asset cache namespace must be a catalog SHA256")
    if expected_sha256 is not None and (not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256)):
        raise CacheError("invalid expected asset SHA256")
    if type(max_bytes) is not int or max_bytes < 1 or (expected_bytes is not None and (type(expected_bytes) is not int or expected_bytes < 1 or expected_bytes > max_bytes)):
        raise CacheError("invalid expected asset byte count or size limit")
    identity = json.dumps({"namespace": namespace, "url": url}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    destination = _no_links(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def emit(name, info=None):
        if event is not None:
            event({"event": name, "host": parsed.hostname, "key": key, **(info or {})})

    def verify(path):
        return _fingerprint(path, expected_sha256=expected_sha256, expected_bytes=expected_bytes, max_bytes=max_bytes)

    blob = receipt = None
    if cache_dir is not None:
        root = _no_links(cache_dir)
        root.mkdir(parents=True, exist_ok=True)
        blob, receipt = root / (key + ".blob"), root / (key + ".receipt.json")
        if destination == blob or destination == receipt:
            raise CacheError("asset destination must be separate from cache entry")
        if os.path.lexists(blob) or os.path.lexists(receipt):
            try:
                cached = _read_receipt(receipt, namespace=namespace, url=url)
                info = verify(blob)
                if info != {"sha256": cached["sha256"], "bytes": cached["bytes"]}:
                    raise CacheError("asset cache receipt does not match bytes")
                _atomic_copy(blob, destination)
                # Validate the independent copy consumed by the caller, not a link.
                if verify(destination) != info:
                    raise CacheError("copied asset changed")
                validate_fn(destination)
                if verify(destination) != info:
                    raise CacheError("asset changed during validation")
            except Exception:
                # Never print the exception: a validator or JSON parser may include
                # untrusted text. Leave other assets' cache entries untouched.
                _remove_entry(blob)
                _remove_entry(receipt)
                _remove_entry(destination)
                emit("invalid")
            else:
                emit("hit", info)
                return info
        emit("miss")

    try:
        download_fn(url, destination, expected_sha256=expected_sha256, max_bytes=max_bytes, event=event, **download_options)
        info = verify(destination)
        validate_fn(destination)
        # Validators may extract to other paths, but must not alter downloaded bytes.
        if verify(destination) != info:
            raise CacheError("asset changed during validation")
    except Exception:
        _remove_entry(destination)
        raise
    if blob is not None:
        _atomic_copy(destination, blob)
        _write_receipt(receipt, {"schema_version": 1, "namespace": namespace, "url": url, **info})
        emit("stored", info)
    return info
