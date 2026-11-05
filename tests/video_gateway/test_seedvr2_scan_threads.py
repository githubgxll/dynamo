import importlib.util
import json
from pathlib import Path

import pytest
from dingo.vllm.omni import seedvr2_media as media

@pytest.mark.asyncio
@pytest.mark.parametrize('threads', [1, 16])
async def test_complete_video_scan_uses_bounded_video_threads(monkeypatch, threads):
    calls = []
    async def command(args):
        calls.append(args)
        if '-count_frames' in args:
            return json.dumps({'streams': [
                {'codec_type':'video','codec_name':'h264','pix_fmt':'yuv420p','width':1440,'height':2560,'nb_read_frames':'362','avg_frame_rate':'24/1'},
                {'codec_type':'audio','codec_name':'aac'},
            ]}).encode()
        return json.dumps({'packets': []}).encode()
    monkeypatch.setattr(media, 'PROBE_VIDEO_THREADS', threads)
    monkeypatch.setattr(media, 'file_sha256', lambda path:'digest')
    monkeypatch.setattr(media, 'command', command)
    result = await media.probe(Path('video.mp4'))
    assert result['frames']==362 and result['audio']['codec']=='aac'
    assert calls[0][calls[0].index('-threads:v:0')+1]==str(threads)
    assert '-count_frames' in calls[0]
    assert '-threads:v:0' not in calls[1] and '-show_packets' in calls[1]

@pytest.mark.parametrize('value', ['0','33','bad'])
def test_probe_thread_config_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv('DINGO_VIDEO_SR_PROBE_THREADS', value)
    spec = importlib.util.spec_from_file_location('invalid_probe_config', media.__file__)
    with pytest.raises(ValueError):
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
