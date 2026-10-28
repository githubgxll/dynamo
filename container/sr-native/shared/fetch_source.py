"""Install fixed source archives; reject path traversal and archive links."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import zipfile
from download import acquire, relative


def extract(archive, destination, strip_components=1):
    destination = Path(destination)
    seen = set()

    def target(name):
        parts = relative(name).parts[strip_components:]
        if not parts:
            return None
        result = destination.joinpath(*parts)
        if result in seen:
            raise ValueError('Duplicate archive path: ' + name)
        seen.add(result)
        return result

    def copy(name, stream, mode):
        output = target(name)
        if output is None:
            return
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('xb') as handle:
            shutil.copyfileobj(stream, handle)
        if os.name != 'nt':
            output.chmod(0o755 if mode & 0o111 else 0o644)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for item in bundle.infolist():
                if item.is_dir():
                    continue
                mode = item.external_attr >> 16
                if mode & 0o170000 not in (0, 0o100000):
                    raise ValueError('Archive links/devices are not accepted')
                with bundle.open(item) as source:
                    copy(item.filename, source, mode)
    else:
        with tarfile.open(archive) as bundle:
            for item in bundle:
                if item.isdir():
                    continue
                if not item.isfile():
                    raise ValueError('Archive links/devices are not accepted: ' + item.name)
                with bundle.extractfile(item) as source:
                    copy(item.name, source, item.mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, default=Path('/var/cache/sr-native-sources'))
    args = parser.parse_args()
    for item in json.loads(args.lock.read_text(encoding='utf-8'))['sources']:
        archive = acquire(item['url'], args.cache_dir / item['sha256'], item['sha256'])
        destination = args.output_root.joinpath(*relative(item['destination']).parts)
        if not destination.resolve().is_relative_to(args.output_root.resolve()):
            raise ValueError('Destination escapes source root')
        marker = destination / '.sr-native-source.json'
        if destination.exists():
            if marker.is_file() and json.loads(marker.read_text()) == item:
                continue
            raise ValueError('Destination already exists: ' + str(destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='source-', dir=destination.parent) as temp:
            extracted = Path(temp) / 'content'
            extracted.mkdir()
            extract(archive, extracted, item.get('strip_components', 1))
            (extracted / '.sr-native-source.json').write_text(json.dumps(item, indent=2) + '\n', encoding='utf-8')
            os.replace(extracted, destination)


if __name__ == '__main__':
    main()
