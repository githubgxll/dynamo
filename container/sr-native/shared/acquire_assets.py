"""Download explicitly locked model assets on a writable acquisition host only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from download import acquire, relative, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--select', nargs='+', help='Exact asset IDs; default all')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    catalog = json.loads(args.lock.read_text(encoding='utf-8'))
    assets = catalog['assets']
    ids = [item['id'] for item in assets]
    names = [item['filename'] for item in assets]
    if len(ids) != len(set(ids)) or len(names) != len(set(names)):
        raise ValueError('Duplicate asset IDs or destinations')
    selected = set(args.select or ids)
    if selected - set(ids):
        raise ValueError('Unknown asset IDs: ' + str(sorted(selected - set(ids))))
    for item in assets:
        if item['id'] not in selected:
            continue
        target = args.output.joinpath(*relative(item['filename']).parts)
        if not target.resolve().is_relative_to(args.output.resolve()):
            raise ValueError('Asset destination escapes the output directory')
        if args.verify_only:
            if not valid(target, item['sha256'], item['size']):
                raise ValueError('Asset does not match locked bytes: ' + item['id'])
        else:
            acquire(item['url'], target, item['sha256'], item['size'])
        print('PASS ' + item['id'])


if __name__ == '__main__':
    main()
