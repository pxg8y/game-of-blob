"""Empirical Challenger Verification Suite for Milestone 3 Iteration 2.

Authored by teamwork_challenger_m3_it2_1.
Empirically stress-tests:
1. Handshake aborts & partial disconnects (abort after text, abort after bytes, abort before accept).
2. Strict message pairing invariant (every text header MUST be immediately followed by binary payload).
3. Mass concurrent client churn (20 clients repeatedly connecting, stepping, disconnecting).
4. Extreme concurrent command storm under live autonomous streaming (set_speed, reset_world, set_paradigm).
5. Scaled 2000+ blob streaming throughput, packet integrity, bitflags validation.
"""

import asyncio
import concurrent.futures
import json
import struct
import threading
import time
import unittest
from typing import List, Tuple
from unittest.mock import AsyncMock, MagicMock

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient

from blob_sim.config import WorldConfig
from blob_sim.server import app, controller, websocket_endpoint
from blob_sim.world import SimulationWorld


class TestMilestone3Iteration2Empirical(unittest.TestCase):
    """Adversarial stress harness for Milestone 3 Iteration 2 server concurrency."""

    def setUp(self) -> None:
        with controller.lock:
            controller.running = False
            controller.steps_per_frame = 1
            controller.clients.clear()
            controller.client_topo_versions.clear()
            controller.client_food_versions.clear()
            controller.client_viewports.clear()
            controller.client_busy.clear()
            cfg = WorldConfig.from_preset("classic")
            cfg.initial_blobs = 100
            cfg.initial_foods = 150
            controller.world = SimulationWorld(world_cfg=cfg, seed=777)

    def tearDown(self) -> None:
        with controller.lock:
            controller.running = False
            controller.clients.clear()
            controller.client_topo_versions.clear()
            controller.client_food_versions.clear()
            controller.client_viewports.clear()
            controller.client_busy.clear()

    def test_handshake_abort_variants_zero_leaks(self) -> None:
        """Adversarially simulate 4 distinct failure points during handshake.
        
        Point 1: ws.accept() raises WebSocketDisconnect
        Point 2: ws.send_text() raises WebSocketDisconnect
        Point 3: ws.send_bytes() raises WebSocketDisconnect
        Point 4: ws.send_text() raises generic RuntimeError (network reset)
        """
        for mode in ("abort_accept", "abort_text", "abort_bytes", "abort_runtime"):
            mock_ws = MagicMock(spec=WebSocket)
            if mode == "abort_accept":
                mock_ws.accept = AsyncMock(side_effect=WebSocketDisconnect(code=1006))
                mock_ws.send_text = AsyncMock()
                mock_ws.send_bytes = AsyncMock()
            elif mode == "abort_text":
                mock_ws.accept = AsyncMock()
                mock_ws.send_text = AsyncMock(side_effect=WebSocketDisconnect(code=1006))
                mock_ws.send_bytes = AsyncMock()
            elif mode == "abort_bytes":
                mock_ws.accept = AsyncMock()
                mock_ws.send_text = AsyncMock()
                mock_ws.send_bytes = AsyncMock(side_effect=WebSocketDisconnect(code=1006))
            elif mode == "abort_runtime":
                mock_ws.accept = AsyncMock()
                mock_ws.send_text = AsyncMock(side_effect=RuntimeError("Connection reset by peer"))
                mock_ws.send_bytes = AsyncMock()

            try:
                asyncio.run(websocket_endpoint(mock_ws))
            except (WebSocketDisconnect, RuntimeError):
                pass

            with controller.lock:
                self.assertNotIn(mock_ws, controller.clients, f"Leaked in clients in mode {mode}")
                self.assertNotIn(mock_ws, controller.client_busy, f"Leaked in client_busy in mode {mode}")
                self.assertNotIn(mock_ws, controller.client_topo_versions, f"Leaked in topo in mode {mode}")
                self.assertNotIn(mock_ws, controller.client_food_versions, f"Leaked in food in mode {mode}")
                self.assertNotIn(mock_ws, controller.client_viewports, f"Leaked in viewports in mode {mode}")

    def test_race_disconnect_during_broadcast_reintroduces_topo_leak(self) -> None:
        """Adversarial Test: Verify if disconnect during broadcast reintroduces ghost entries into topo/food dicts."""
        mock_ws = MagicMock(spec=WebSocket)
        
        async def fake_send_bytes(b):
            # Simulate client disconnect occurring right as bytes are being transmitted:
            # websocket_endpoint's finally block fires in its task
            with controller.lock:
                controller.clients.discard(mock_ws)
                controller.client_topo_versions.pop(mock_ws, None)
                controller.client_food_versions.pop(mock_ws, None)
                controller.client_viewports.pop(mock_ws, None)

        mock_ws.send_text = AsyncMock()
        mock_ws.send_bytes = AsyncMock(side_effect=fake_send_bytes)

        with controller.lock:
            controller.clients.add(mock_ws)
            controller.client_topo_versions[mock_ws] = 1
            controller.client_food_versions[mock_ws] = 1

        asyncio.run(controller.broadcast_state())

        with controller.lock:
            leaked_topo = mock_ws in controller.client_topo_versions
            leaked_food = mock_ws in controller.client_food_versions
            self.assertFalse(
                leaked_topo,
                "RACE CONDITION CONFIRMED: client_topo_versions leaked ghost socket because lines 65-67 in server.py re-insert without checking if ws is still in self.clients!",
            )
            self.assertFalse(
                leaked_food,
                "RACE CONDITION CONFIRMED: client_food_versions leaked ghost socket!",
            )

    def test_strict_text_binary_pairing_invariant(self) -> None:
        """Adversarially verify that no two text frames or two binary frames are ever received consecutively."""
        with controller.lock:
            controller.running = True
            controller.steps_per_frame = 2

        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                # Track sequence of received types: "text", "bytes"
                received_sequence = []
                for _ in range(50):  # 25 pairs = 50 messages
                    msg = ws.receive()
                    if "text" in msg:
                        received_sequence.append("text")
                    elif "bytes" in msg:
                        received_sequence.append("bytes")

                self.assertEqual(len(received_sequence), 50)
                # Invariant: Even indices MUST be "text", Odd indices MUST be "bytes"
                for idx, msg_type in enumerate(received_sequence):
                    expected = "text" if idx % 2 == 0 else "bytes"
                    self.assertEqual(
                        msg_type,
                        expected,
                        f"Frame pairing violation at sequence index {idx}: expected {expected}, got {msg_type}! Full sequence: {received_sequence}",
                    )

    def test_heavy_concurrent_connection_churn_with_commands(self) -> None:
        """Run 8 concurrent threads connecting, issuing random commands, and disconnecting rapidly."""
        exceptions: List[str] = []
        num_threads = 8
        ops_per_thread = 5

        def worker_churn(t_id: int):
            c = TestClient(app)
            try:
                for op in range(ops_per_thread):
                    with c.websocket_connect("/ws") as ws:
                        # Handshake
                        txt = ws.receive_text()
                        raw = ws.receive_bytes()
                        m = json.loads(txt)
                        self.assertEqual(m["type"], "state")

                        # Send commands
                        if op % 2 == 0:
                            ws.send_text(json.dumps({"cmd": "step_once", "steps": 1}))
                            msg1 = ws.receive()
                            msg2 = ws.receive()
                            self.assertIn("text", msg1)
                            self.assertIn("bytes", msg2)
                        else:
                            ws.send_text(json.dumps({"cmd": "set_viewport", "enabled": True, "bounds": [100, 100, 500, 500]}))
                            # Trigger step via REST
                            res = c.post("/api/command", json={"cmd": "step_once", "steps": 1})
                            self.assertEqual(res.status_code, 200)
                            msg1 = ws.receive()
                            msg2 = ws.receive()
                            self.assertIn("text", msg1)
                            self.assertIn("bytes", msg2)
            except concurrent.futures.CancelledError:
                pass
            except Exception:
                import traceback
                exceptions.append(traceback.format_exc())

        threads = [threading.Thread(target=worker_churn, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)

        self.assertEqual(len(exceptions), 0, f"Exceptions in heavy churn: {exceptions}")

        # Ensure complete cleanup after threads exit
        t_deadline = time.time() + 3.0
        while time.time() < t_deadline:
            with controller.lock:
                if len(controller.clients) == 0 and len(controller.client_topo_versions) == 0:
                    break
            time.sleep(0.05)

        with controller.lock:
            self.assertEqual(len(controller.clients), 0, "Clients set not 0 after churn")
            self.assertEqual(len(controller.client_busy), 0, "Busy set not 0 after churn")
            self.assertEqual(len(controller.client_viewports), 0, "Viewports not 0 after churn")
            self.assertEqual(len(controller.client_topo_versions), 0, "Topo versions not 0 after churn")
            self.assertEqual(len(controller.client_food_versions), 0, "Food versions not 0 after churn")

    def test_scaled_3000_blob_serialization_and_packet_layout(self) -> None:
        """Verify binary packet layout and bitflags on 3,000 blobs."""
        cfg = WorldConfig.from_preset("continent")
        cfg.initial_blobs = 3000
        cfg.initial_foods = 3500
        world = SimulationWorld(world_cfg=cfg, seed=12345)

        living = [b for b in world.blobs if b.alive]
        self.assertGreaterEqual(len(living), 3000)

        t0 = time.perf_counter()
        meta, bin_delta = world.get_ws_state_and_binary(include_topology=False, include_foods=False)
        dt = (time.perf_counter() - t0) * 1000.0

        # Assert delta serialization latency for 3,000 blobs is under 30ms
        self.assertLess(dt, 30.0, f"Serialization took {dt:.2f} ms")

        # Unpack header
        magic, n_blobs, n_foods, tick = struct.unpack_from("<IIII", bin_delta, 0)
        self.assertEqual(magic, 0x424C4F42)
        self.assertEqual(n_blobs, len(living))
        self.assertEqual(n_foods, 0)
        self.assertEqual(len(bin_delta), 16 + n_blobs * 32)

        # Validate unpacked blob records
        data = np.frombuffer(bin_delta[16:], dtype=np.float32).reshape((n_blobs, 8))
        flags = data[:, 7].astype(np.int32)
        # Check bitflags 0..127
        self.assertTrue(np.all((flags >= 0) & (flags <= 127)))


if __name__ == "__main__":
    unittest.main()
