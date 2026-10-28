"""Pure-Python identity and media admission for the native reference runner."""
from __future__ import annotations
import hashlib
import json
import math
import re
import subprocess
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = Path('/opt/sources/flashvsr-plus')
UPSTREAM_REVISION = 'b548a2cc2c929d3d4676f2973abe27876ec4dee7'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def verify_source(root=SOURCE):
    lock = json.loads((HERE / 'upstream-files.json').read_text(encoding='utf-8'))
    errors = [name for name, expected in lock['files'].items()
              if not (root / name).is_file() or sha256(root / name) != expected]
    if errors:
        raise RuntimeError('Pinned FlashVSR+ source differs: ' + ', '.join(errors))
    return {'revision': lock['revision'], 'verified_files': len(lock['files'])}


def verify_assets(root):
    lock = json.loads((HERE / 'assets.lock.json').read_text(encoding='utf-8'))
    evidence = []
    for item in lock['assets']:
        path = Path(root) / item['filename']
        if not path.is_file() or path.stat().st_size != item['size']:
            raise RuntimeError('Missing/wrong-size official asset: ' + str(path))
        actual = sha256(path)
        if actual != item['sha256']:
            raise RuntimeError('Official asset SHA256 differs: ' + str(path))
        evidence.append({**item, 'local_path': str(path), 'verified_sha256': actual})
    return evidence


def validate_geometry(source_width, source_height, width, height):
    if min(source_width, source_height, width, height) <= 0:
        raise ValueError('Dimensions must be positive')
    if width % 128 or height % 128:
        raise ValueError('FlashVSR+ target width AND height must be divisible by 128')
    if source_width * height != source_height * width:
        raise ValueError('Reference runner refuses the official center-crop path: preserve full aspect ratio')
    # Mirror the exact int(float(...)) geometry, not merely rational ratios.
    crop_height = int(source_width / width * height)
    if crop_height != source_height:
        raise ValueError('Official floating-point crop arithmetic would remove source rows')
    return {'source_width': source_width, 'source_height': source_height,
            'target_width': width, 'target_height': height, 'crop': None,
            'spatial_preprocessing': 'official run_stream PIL bicubic'}


def media_probe(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_frames', '-show_streams', '-show_frames', '-show_entries',
        'stream=width,height,nb_read_frames,avg_frame_rate,r_frame_rate:frame=best_effort_timestamp_time',
        '-of', 'json', str(path)], capture_output=True, text=True, check=True)
    value = json.loads(result.stdout)
    if len(value.get('streams', [])) != 1:
        raise ValueError('Expected one selected video stream')
    stream = value['streams'][0]
    timestamps = [float(frame['best_effort_timestamp_time']) for frame in value.get('frames', [])]
    return {'width': int(stream['width']), 'height': int(stream['height']),
            'frames': int(stream['nb_read_frames']), 'fps': str(Fraction(stream['avg_frame_rate'])),
            'timestamps': timestamps}


def validate_timeline(media, frames, fps):
    if frames <= 0 or fps <= 0 or not math.isfinite(fps):
        raise ValueError('Expected frames and FPS must be positive')
    if media['frames'] != frames or len(media['timestamps']) != frames:
        raise ValueError('Frame count differs from the frozen window; truncation is not allowed')
    if abs(float(Fraction(media['fps'])) - fps) > 1e-5:
        raise ValueError('FPS differs from the frozen window')
    if any(abs((b-a) - 1/fps) > 2e-5 for a,b in zip(media['timestamps'],media['timestamps'][1:])):
        raise ValueError('Input/output PTS is not the required constant cadence')


def require_digest(value):
    if not re.fullmatch('[a-f0-9]{64}', value):
        raise ValueError('A full lowercase SHA256 is required')
    return value
