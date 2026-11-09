import pytest

from dingo.vllm.omni import seedvr2_media as media


@pytest.mark.asyncio
@pytest.mark.parametrize('corruption', [None, 'before', 'during', 'frames', 'audio'])
async def test_probe_reuse_keeps_integrity_and_final_validation(monkeypatch, tmp_path, corruption):
    base, raw, final = (tmp_path / name for name in ('base.mp4', 'raw.mp4', 'final.mp4'))
    base.write_bytes(b'base')
    raw.write_bytes(b'raw')
    common = {'frames': 362, 'start_time': '0', 'audio': {'codec': 'aac', 'packets': []},
              'copy_compatible': True, 'probe_timings': {'video_scan_s': 9.0}}
    original = dict(common, width=768, height=1344, sha256=media.file_sha256(base))
    restored = dict(common, width=1440, height=2560, sha256=media.file_sha256(raw))
    probes, commands = [], []

    async def probe(path):
        probes.append(path)
        assert path == final, 'Already-probed inputs must not be rescanned'
        result = dict(restored)
        if corruption == 'frames':
            result['frames'] = 361
        if corruption == 'audio':
            result['audio'] = None
        return result

    async def command(args):
        commands.append(args)
        final.write_bytes(b'final')
        if corruption == 'during':
            raw.write_bytes(b'changed during mux')
        return b''

    monkeypatch.setattr(media, 'probe', probe)
    monkeypatch.setattr(media, 'command', command)
    if corruption == 'before':
        raw.write_bytes(b'changed before mux')
    kwargs = dict(width=1440, height=2560, require_stream_copy=True,
                  expected_base_size=(768, 1344), base_info=original, raw_info=restored)
    if corruption:
        with pytest.raises(ValueError):
            await media.finalize(raw, base, final, **kwargs)
    else:
        result = await media.finalize(raw, base, final, **kwargs)
        assert probes == [final]
        assert result['postprocess_mode'] == 'stream_copy'
        assert commands[0][commands[0].index('-c:v') + 1] == 'copy'
        assert result['postprocess_timings']['base_video_scan_s'] == 0
        assert result['postprocess_timings']['raw_video_scan_s'] == 0
    if corruption == 'before':
        assert not commands
    if corruption == 'during':
        assert not probes
