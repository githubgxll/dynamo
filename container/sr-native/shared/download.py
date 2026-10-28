"""Hash-verified HTTPS downloads; bounded retries, resumable partials, no fallback."""
from __future__ import annotations

import hashlib
import http.client
import os
from pathlib import Path, PurePosixPath
import re
import time
import urllib.error
import urllib.parse
import urllib.request


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def relative(name):
    p = PurePosixPath(name)
    if not name or '\\' in name or ':' in name or p.is_absolute() or '..' in p.parts:
        raise ValueError('Unsafe relative path: ' + name)
    return p


def valid(path, digest, size=None):
    return (path.is_file() and not path.is_symlink()
            and (size is None or path.stat().st_size == size) and sha256(path) == digest)


def acquire(url, target, digest, size=None, attempts=4, timeout=120):
    target = Path(target)
    if not re.fullmatch(r'[0-9a-f]{64}', digest):
        raise ValueError('A complete SHA256 is required')
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != 'https' or parsed.username or parsed.password:
        raise ValueError('Only HTTPS sources without embedded credentials are supported')
    if size is not None and (not isinstance(size, int) or size < 1):
        raise ValueError('Invalid expected size')
    if target.exists():
        if valid(target, digest, size):
            return target
        raise ValueError('Existing target differs from lock; preserve and inspect it: ' + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.partial')
    if partial.is_symlink():
        raise ValueError('Partial download cannot be a symlink')
    for attempt in range(1, attempts + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset and valid(partial, digest, size):
            os.replace(partial, target)
            return target
        if size is not None and offset >= size:
            partial.unlink()
            offset = 0
        headers = {'User-Agent': 'dingo-sr-native/1', 'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        print(f'Acquiring {target.name}: attempt {attempt}/{attempts}, retained bytes {offset}', flush=True)
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if urllib.parse.urlsplit(response.url).scheme != 'https':
                    raise ValueError('HTTPS redirect downgraded')
                status = response.status
                if status == 206:
                    content_range = response.headers.get('Content-Range', '')
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+|\*)', content_range)
                    if not match or int(match[1]) != offset:
                        raise ValueError('Unexpected resume range')
                    mode = 'ab'
                elif status == 200:
                    mode = 'wb'  # Server ignored Range; never append a full body.
                else:
                    raise ValueError('Unexpected HTTP status: ' + str(status))
                with partial.open(mode) as output:
                    for chunk in iter(lambda: response.read(4 * 1024 * 1024), b''):
                        output.write(chunk)
            if not valid(partial, digest, size):
                if size is not None and partial.stat().st_size < size:
                    raise OSError('Short response; retain partial for retry')
                partial.unlink()  # Complete but corrupt body cannot be resumed.
                raise OSError('Downloaded bytes fail the locked SHA256/size')
            os.replace(partial, target)
            return target
        except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
            # Do not print signed redirect URLs or proxy credentials in errors.
            print(f'Download failed for {parsed.hostname}: {type(exc).__name__}', flush=True)
            if attempt == attempts:
                raise RuntimeError(f'Download exhausted for {parsed.hostname}; partial retained at {partial}') from None
            time.sleep(min(2 ** attempt, 20))
    raise AssertionError('unreachable')
