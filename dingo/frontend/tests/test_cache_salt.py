# SPDX-License-Identifier: Apache-2.0
"""Backend-independent cache salt contract tests."""

import copy

import pytest

from dingo.frontend.cache_salt import project_cache_salt, resolve_cache_salt

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.mark.parametrize(
    "request_data, expected",
    [
        ({}, None),
        ({"nvext": None}, None),
        ({"nvext": {"cache_salt": "tenant-a"}}, "tenant-a"),
        ({"cache_salt": "legacy"}, "legacy"),
        ({"nvext": {"cache_salt": "canonical"}, "cache_salt": "legacy"}, "canonical"),
        ({"nvext": {"cache_salt": ""}, "cache_salt": "legacy"}, "legacy"),
        ({"nvext": {"cache_salt": None}, "cache_salt": "legacy"}, "legacy"),
        ({"nvext": {"cache_salt": " "}, "cache_salt": "legacy"}, " "),
        ({"cache_salt": ""}, None),
        ({"cache_salt": 123}, None),
        ({"cache_salt": False}, None),
        ({"cache_salt": []}, None),
    ],
)
def test_resolve_cache_salt(request_data, expected):
    assert resolve_cache_salt(request_data) == expected


@pytest.mark.parametrize("salt", [123, False, [], {}])
def test_invalid_canonical_salt_is_rejected(salt):
    with pytest.raises(ValueError, match="nvext.cache_salt"):
        resolve_cache_salt({"nvext": {"cache_salt": salt}, "cache_salt": "legacy"})


def test_projection_preserves_metadata_and_does_not_mutate_shared_input():
    original = {
        "routing": {"dp_rank": 0, "cache_salt": "old"},
        "extra_args": {"mm_hashes": [123], "nvext": {"metadata_upload": {"x": 1}}},
    }
    snapshot = copy.deepcopy(original)
    preproc = dict(original)
    project_cache_salt(preproc, "resolved")
    assert preproc["routing"] == {"dp_rank": 0, "cache_salt": "resolved"}
    assert preproc["extra_args"] == {
        "mm_hashes": [123],
        "nvext": {"metadata_upload": {"x": 1}, "cache_salt": "resolved"},
    }
    assert original == snapshot


def test_absent_salt_keeps_payload_unchanged():
    preproc = {"routing": None}
    project_cache_salt(preproc, None)
    assert preproc == {"routing": None}
