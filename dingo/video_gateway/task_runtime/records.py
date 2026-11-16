# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Existing business-record codecs over shared exact-key CAS persistence.

This layer deliberately does not serialize a VideoTask as a ContextIRTask,
nor change their key layout. Occupancy/queue changes belong in a larger txn.
"""


class RecordRepository:
    def __init__(self, client, key_for, decode):
        self.client, self.key_for, self.decode = client, key_for, decode

    async def load(self, task_id):
        value = await self.client.get(self.key_for(task_id))
        return self.decode(value) if value is not None else None

    async def update(self, task_id, revision, serialized, *, compares=(), writes=()):
        key = self.key_for(task_id)
        return await self.client.txn(
            [self.client.compare_mod(key, revision), *compares],
            [self.client.put(key, serialized), *writes],
        )
