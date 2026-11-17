# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only lifecycle tests using real receiver/allocator code and a fake transport.

Run directly with Python (no CUDA, torch or NIXL installation required). AST
extraction avoids importing optional runtime dependencies; bytearray views model
shared storage and the fake transport explicitly controls late remote WRITEs.
These are not native NIXL/RDMA integration tests.
"""

import ast
import asyncio
import base64
import json
import logging
import math
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


class _Tensor:
    def __init__(self, storage, start=0, end=None):
        self.storage = storage
        self.start = start
        self.end = len(storage) if end is None else end
        self.device = "cpu"

    def __getitem__(self, index):
        return _Tensor(self.storage, self.start + index.start, self.start + index.stop)

    def view(self, shape=None, *, dtype=None):
        if shape is not None and math.prod(shape) != self.end - self.start:
            raise ValueError("Invalid shape")
        return self

    def data_ptr(self):
        return self.start

    def get_device(self):
        return -1

    def payload(self):
        return bytes(self.storage[self.start : self.end])


class _Request:
    @staticmethod
    def model_validate_json(value):
        return SimpleNamespace(**json.loads(value))


def _load_receiver():
    path = Path(__file__).resolve().parents[2] / "multimodal/embedding_transfer.py"
    source = ast.parse(path.read_text())
    names = {"MonolithicCounter", "RingBuffer", "NixlWriteEmbeddingReceiver"}
    selected = [
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name in names
    ]
    assert len(selected) == len(names)
    namespace = {
        "asyncio": asyncio,
        "base64": base64,
        "time": time,
        "logger": logging.getLogger(__name__),
        "AbstractEmbeddingReceiver": object,
        "TransferRequest": object,
        "NixlTransferRequest": _Request,
        "torch_dtype_from_string": lambda value: value,
        "msgspec": SimpleNamespace(msgpack=SimpleNamespace(encode=lambda value: value)),
        "torch": SimpleNamespace(
            Tensor=_Tensor,
            int8="int8",
            zeros=lambda size, dtype: _Tensor(bytearray(size)),
        ),
    }
    code = compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec")
    exec(code, namespace)
    return namespace


_CODE = _load_receiver()


class _Agent:
    def __init__(self):
        self.notifs = {}
        self.handshakes = []
        self.on_send = None
        self.poll_error = False

    def update_notifs(self):
        if self.poll_error:
            raise RuntimeError("Notification poll failed")
        return self.notifs

    def send_notif(self, sender, *, notif_msg):
        self.handshakes.append((sender, notif_msg))
        if self.on_send:
            self.on_send(sender, notif_msg)

    def done(self, sender, tensor_id):
        self.notifs.setdefault(sender, []).append(str(tensor_id).encode())


def _receiver(size=8):
    receiver = _CODE["NixlWriteEmbeddingReceiver"].__new__(
        _CODE["NixlWriteEmbeddingReceiver"]
    )
    receiver.ring_buffer = _CODE["RingBuffer"](size)
    receiver.nixl_agent = _Agent()
    receiver.remote_agents = {"A": object(), "B": object()}
    receiver.agent_metadata = b"metadata"
    receiver.id_counter = _CODE["MonolithicCounter"]()
    receiver.to_buffer_id = {}
    receiver._quarantined_buffers = {}
    return receiver


def _request(sender="A", size=4, shape=None):
    return SimpleNamespace(
        serialized_request=json.dumps(
            dict(
                sender_agent_id=sender,
                agent_metadata=None,
                tensor_id=0,
                tensor_size=size,
            )
        ),
        embeddings_shape=[size] if shape is None else shape,
        embedding_dtype_str="int8",
    )


class TestNixlWriteBufferLifetime(unittest.IsolatedAsyncioTestCase):
    async def test_late_write_cannot_overwrite_other_receive(self):
        receiver = _receiver()
        agent = receiver.nixl_agent
        with self.assertRaises(TimeoutError):
            await receiver.receive_embeddings(_request(), receive_timeout=0)
        self.assertEqual(receiver._quarantined_buffers, {("A", 0): 0})

        def complete_b(sender, handshake):
            _, (address, size, _, _), tensor_id, _ = handshake
            self.assertEqual(address, 4)  # A still owns [0, 4).
            storage = receiver.ring_buffer.buffer_tensor.storage
            storage[address : address + size] = b"BBBB"
            agent.done(sender, tensor_id)
            # A's WRITE arrives after B's WRITE but before B copies/consumes it.
            storage[0:4] = b"AAAA"

        agent.on_send = complete_b
        tensor_id, tensor = await receiver.receive_embeddings(_request("B"))
        self.assertEqual(tensor.payload(), b"BBBB")
        receiver.release_tensor(tensor_id)
        self.assertEqual(receiver.ring_buffer.allocated_buffer_id_to_range, {0: (0, 4)})
        agent.done("A", 0)
        receiver._poll_write_completions()
        self.assertEqual(receiver.ring_buffer.allocated_buffer_id_to_range, {})
        buffer_id, entire_ring = receiver.ring_buffer.get_buffer(8)
        self.assertEqual(entire_ring.data_ptr(), 0)
        receiver.ring_buffer.release_buffer(buffer_id)

    async def test_full_quarantine_reaped_while_waiting_for_allocation(self):
        receiver = _receiver(size=4)
        agent = receiver.nixl_agent
        with self.assertRaises(TimeoutError):
            await receiver.receive_embeddings(_request(), receive_timeout=0)
        agent.on_send = lambda sender, handshake: agent.done(sender, handshake[2])
        pending = asyncio.create_task(receiver.receive_embeddings(_request("B")))
        await asyncio.sleep(0)
        self.assertEqual(len(agent.handshakes), 1)  # No B address advertised yet.
        agent.done("A", 0)
        tensor_id, tensor = await asyncio.wait_for(pending, timeout=1)
        self.assertEqual(tensor.data_ptr(), 0)
        self.assertEqual(receiver._quarantined_buffers, {})
        receiver.release_tensor(tensor_id)

    async def test_missing_completion_never_reuses_address(self):
        receiver = _receiver(size=4)
        with self.assertRaises(TimeoutError):
            await receiver.receive_embeddings(_request(), receive_timeout=0)
        for _ in range(3):
            with self.assertRaisesRegex(TimeoutError, "available buffer"):
                await receiver.receive_embeddings(_request("B"), receive_timeout=0)
        self.assertEqual(len(receiver.nixl_agent.handshakes), 1)
        self.assertEqual(receiver.ring_buffer.allocated_buffer_id_to_range, {0: (0, 4)})

    async def test_cancel_quarantines_until_late_completion(self):
        receiver = _receiver(size=4)
        pending = asyncio.create_task(receiver.receive_embeddings(_request()))
        await asyncio.sleep(0)
        self.assertEqual(len(receiver.nixl_agent.handshakes), 1)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertEqual(receiver._quarantined_buffers, {("A", 0): 0})
        self.assertEqual(receiver.to_buffer_id, {})
        receiver.nixl_agent.done("A", 0)
        receiver._poll_write_completions()
        self.assertEqual(receiver.ring_buffer.allocated_buffer_id_to_range, {})

    async def test_invalid_shape_releases_unpublished_address(self):
        receiver = _receiver(size=4)
        with self.assertRaises(ValueError):
            await receiver.receive_embeddings(_request(shape=[5]))
        self.assertEqual(receiver.ring_buffer.allocated_buffer_id_to_range, {})
        self.assertEqual(receiver._quarantined_buffers, {})
        self.assertEqual(receiver.nixl_agent.handshakes, [])

    async def test_handshake_error_may_already_have_published_address(self):
        receiver = _receiver(size=4)

        def raise_after_send(sender, handshake):
            raise RuntimeError("Transport failed after accepting notification")

        receiver.nixl_agent.on_send = raise_after_send
        with self.assertRaises(RuntimeError):
            await receiver.receive_embeddings(_request())
        self.assertEqual(receiver._quarantined_buffers, {("A", 0): 0})
        self.assertEqual(receiver.ring_buffer.allocated_buffer_id_to_range, {0: (0, 4)})
        receiver.nixl_agent.done("A", 0)
        receiver._poll_write_completions()
        self.assertEqual(receiver._quarantined_buffers, {})

    async def test_notification_error_retains_published_slot(self):
        receiver = _receiver(size=4)
        receiver.nixl_agent.on_send = lambda *_: setattr(
            receiver.nixl_agent, "poll_error", True
        )
        with self.assertRaises(RuntimeError):
            await receiver.receive_embeddings(_request())
        self.assertEqual(receiver._quarantined_buffers, {("A", 0): 0})
        receiver.nixl_agent.poll_error = False
        receiver.nixl_agent.done("A", 0)
        receiver._poll_write_completions()
        self.assertEqual(receiver._quarantined_buffers, {})

    async def test_exact_sender_and_id_match_preserves_other_notifications(self):
        receiver = _receiver(size=4)
        with self.assertRaises(TimeoutError):
            await receiver.receive_embeddings(_request(), receive_timeout=0)
        agent = receiver.nixl_agent
        agent.done("B", 0)  # Same ID, different sender.
        agent.done("A", 10)  # Contains "0" but is a different ID.
        receiver._poll_write_completions()
        self.assertEqual(receiver._quarantined_buffers, {("A", 0): 0})
        agent.done("A", 0)
        receiver._poll_write_completions()
        self.assertEqual(receiver._quarantined_buffers, {})
        self.assertEqual(agent.notifs, {"B": [b"0"], "A": [b"10"]})

    async def test_normal_completion_hands_off_without_await_or_timeout(self):
        receiver = _receiver(size=4)
        agent = receiver.nixl_agent
        agent.on_send = lambda sender, handshake: agent.done(sender, handshake[2])
        # Immediate completion wins even with an expired deadline. The old loop
        # awaited once more and then raised TimeoutError despite seeing DONE.
        tensor_id, tensor = await receiver.receive_embeddings(
            _request(), receive_timeout=-1
        )
        self.assertEqual(tensor.payload(), b"\0" * 4)
        self.assertEqual(receiver.to_buffer_id, {tensor_id: 0})
        self.assertEqual(receiver._quarantined_buffers, {})
        receiver.release_tensor(tensor_id)
        self.assertEqual(receiver.ring_buffer.allocated_buffer_id_to_range, {})

    async def test_active_receive_owns_its_notification_while_reaping_other(self):
        receiver = _receiver()
        with self.assertRaises(TimeoutError):
            await receiver.receive_embeddings(_request(), receive_timeout=0)
        pending = asyncio.create_task(receiver.receive_embeddings(_request("B")))
        await asyncio.sleep(0)
        receiver.nixl_agent.done("A", 0)
        receiver.nixl_agent.done("B", 1)
        tensor_id, _ = await asyncio.wait_for(pending, timeout=1)
        self.assertEqual(receiver._quarantined_buffers, {})
        self.assertEqual(receiver.to_buffer_id, {tensor_id: 1})
        receiver.release_tensor(tensor_id)
        self.assertEqual(receiver.ring_buffer.allocated_buffer_id_to_range, {})


if __name__ == "__main__":
    unittest.main()
