# Linux preparation output

The first configured push exports a preparation artifact without publishing an image.
Import its base.lock.json, requirements.lock, assets.lock.json and input-fingerprint.json
using delivery/vbench/ops.py. Do not invent hashes or copy Windows package versions here.
Policy reports are preserved in the local delivery evidence directory.
