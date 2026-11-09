"""Empirical Stress Test Suite for Milestone 3: Web Server Concurrency, WebSocket Churn,
Backpressure Handling, and Scaled Streaming Throughput.

Authored by teamwork_challenger_m3_1.
"""

import concurrent.futures
import json
import struct
import threading
import time
import unittest
from typing import List, Optional, Tuple

import numpy as np
from fastapi.testclient import TestClient

from blob_sim.config import WorldConfig, NetworkConfig
from blob_sim.server import app, controller
from blob_sim.world import SimulationWorld


class TestMilestone3AdversarialStress(unittest.TestCase):
    """Adversarial stress and concurrency verification for Milestone 3."""

    def setUp(self) -> None:
        """Reset simulation controller state before each test."""
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
            controller.world = SimulationWorld(world_cfg=cfg, seed=123)

    def tearDown(self) -> None:
        """Cleanup after each test."""
        with controller.lock:
            controller.running = False
            controller.clients.clear()
            controller.client_topo_versions.clear()
            controller.client_food_versions.clear()
            controller.client_viewports.clear()
            controller.client_busy.clear()

    def test_websocket_connection_churn_concurrent(self) -> None:
        """Verify server survives concurrent waves of WebSocket connection churn without leaks or crashes."""
        exceptions: List[Exception] = []
        num_waves = 3
        clients_per_wave = 5

        def client_churn_worker(worker_id: int):
            client = TestClient(app)
            try:
                for wave in range(num_waves):
                    with client.websocket_connect("/ws") as ws:
                        # 1. Receive initial full state
                        txt = ws.receive_text()
                        b_init = ws.receive_bytes()
                        meta = json.loads(txt)
                        self.assertEqual(meta["type"], "state")
                        magic, n_blobs, n_foods, tick = struct.unpack_from("<IIII", b_init, 0)
                        self.assertEqual(magic, 0x424C4F42)
                        self.assertGreater(n_blobs, 0)

                        # 2. Issue a command
                        cmd = {"cmd": "select_blob", "blob_id": 1} if worker_id % 2 == 0 else {"cmd": "set_viewport", "enabled": True, "bounds": [0, 0, 1000, 1000]}
                        ws.send_text(json.dumps(cmd))

                        # 3. Trigger 2 steps and receive frames
                        for step_i in range(2):
                            ws.send_text(json.dumps({"cmd": "step_once", "steps": 1}))
                            msg1 = ws.receive()
                            msg2 = ws.receive()
                            if "bytes" not in msg2:
                                text_snippet = msg2.get("text", "")[:100] if "text" in msg2 else str(msg2)
                                raise ValueError(f"[Worker {worker_id} wave {wave} step {step_i}] Expected bytes in msg2 but got keys {list(msg2.keys())}: {text_snippet}")
                            d_magic, d_blobs, d_foods, d_tick = struct.unpack_from("<IIII", msg2["bytes"], 0)
                            self.assertEqual(d_magic, 0x424C4F42)

                        # 4. Abrupt disconnect by exiting context
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                import traceback
                exceptions.append(traceback.format_exc())

        threads = [threading.Thread(target=client_churn_worker, args=(i,)) for i in range(clients_per_wave)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)

        self.assertEqual(len(exceptions), 0, f"Exceptions occurred during churn: {exceptions}")
        t_deadline = time.time() + 2.0
        while time.time() < t_deadline:
            with controller.lock:
                if len(controller.clients) == 0:
                    break
            time.sleep(0.02)

        with controller.lock:
            self.assertEqual(len(controller.clients), 0, "Clients set not cleaned up after churn")
            self.assertEqual(len(controller.client_viewports), 0, "Viewports map not cleaned up")
            self.assertEqual(len(controller.client_busy), 0, "Busy set not cleaned up")

    def test_slow_client_backpressure_does_not_block_fast_clients(self) -> None:
        """Verify a lagging or paused client does not block the server or starve fast clients."""
        client = TestClient(app)
        fast_frames_received = 0
        slow_client_active = True
        exceptions: List[Exception] = []

        def slow_worker():
            try:
                with client.websocket_connect("/ws") as ws_slow:
                    # Read initial frame
                    ws_slow.receive_text()
                    ws_slow.receive_bytes()

                    # Deliberately stall reading to simulate heavy network backpressure
                    time.sleep(0.4)
                    # Attempt reading after delay or exit
            except Exception as e:
                exceptions.append(e)

        t_slow = threading.Thread(target=slow_worker)
        t_slow.start()

        # Connect fast client
        with client.websocket_connect("/ws") as ws_fast:
            ws_fast.receive_text()
            ws_fast.receive_bytes()

            # Advance simulation multiple times while slow client is stalled
            for _ in range(5):
                ws_fast.send_text(json.dumps({"cmd": "step_once", "steps": 1}))
                txt = ws_fast.receive_text()
                raw = ws_fast.receive_bytes()
                magic, n_b, n_f, _ = struct.unpack_from("<IIII", raw, 0)
                self.assertEqual(magic, 0x424C4F42)
                fast_frames_received += 1

        t_slow.join(timeout=5.0)

        self.assertEqual(fast_frames_received, 5, "Fast client was blocked or starved by slow client")

    def test_rapid_speed_changes_and_resets_under_active_streaming(self) -> None:
        """Verify rapid speed updates and paradigm resets under active streaming preserve data integrity."""
        client = TestClient(app)
        exceptions: List[Exception] = []
        stop_event = threading.Event()

        def command_spammer():
            try:
                speeds = [1, 5, 15, 60, 200, 1, 10, 30]
                paradigms = ["tabula_rasa", "rapid_exodus", "curriculum"]
                for i in range(16):
                    if stop_event.is_set():
                        break
                    res = client.post("/api/command", json={"cmd": "set_speed", "speed": speeds[i % len(speeds)]})
                    self.assertEqual(res.status_code, 200)
                    if i % 5 == 0:
                        res = client.post("/api/command", json={"cmd": "set_paradigm", "paradigm": paradigms[i % len(paradigms)]})
                        self.assertEqual(res.status_code, 200)
                    time.sleep(0.01)
            except Exception as e:
                exceptions.append(e)

        t_spam = threading.Thread(target=command_spammer)
        t_spam.start()

        with client.websocket_connect("/ws") as ws:
            ws.receive_text()
            ws.receive_bytes()

            for _ in range(12):
                ws.send_text(json.dumps({"cmd": "step_once", "steps": 2}))
                txt = ws.receive_text()
                raw_bytes = ws.receive_bytes()
                msg = json.loads(txt)
                self.assertIn("type", msg)
                self.assertEqual(msg["type"], "state")

                magic, n_blobs, n_foods, tick = struct.unpack_from("<IIII", raw_bytes, 0)
                self.assertEqual(magic, 0x424C4F42)
                expected_len = 16 + n_blobs * 32 + n_foods * 12
                self.assertEqual(len(raw_bytes), expected_len)

        stop_event.set()
        t_spam.join(timeout=5.0)
        self.assertEqual(len(exceptions), 0, f"Exceptions occurred during rapid command barrage: {exceptions}")

    def test_scaled_2000_blob_streaming_serialization_performance(self) -> None:
        """Empirically verify 2,000 blobs serialization latency and binary packet validity under extreme load."""
        cfg = WorldConfig.from_preset("continent")
        cfg.initial_blobs = 2000
        cfg.initial_foods = 2500
        world = SimulationWorld(world_cfg=cfg, seed=42)

        living = [b for b in world.blobs if b.alive]
        self.assertGreaterEqual(len(living), 2000, "World did not spawn 2,000 blobs")

        # 1. Measure full frame serialization (initial connect)
        t0 = time.perf_counter()
        meta_full, bin_full = world.get_ws_state_and_binary(include_topology=True, include_foods=True)
        t_full = time.perf_counter() - t0

        magic_f, n_blobs_f, n_foods_f, tick_f = struct.unpack_from("<IIII", bin_full, 0)
        self.assertEqual(magic_f, 0x424C4F42)
        self.assertEqual(n_blobs_f, len(living))
        self.assertEqual(n_foods_f, len(world.foods))
        self.assertEqual(len(bin_full), 16 + n_blobs_f * 32 + n_foods_f * 12)

        # 2. Measure delta frame serialization (stationary food cached)
        # Warmup
        world.get_ws_state_and_binary(include_topology=False, include_foods=False)

        latencies = []
        for _ in range(10):
            t_start = time.perf_counter()
            meta_delta, bin_delta = world.get_ws_state_and_binary(include_topology=False, include_foods=False)
            dt = time.perf_counter() - t_start
            latencies.append(dt)

        avg_delta_ms = float(np.mean(latencies)) * 1000.0
        p95_delta_ms = float(np.percentile(latencies, 95)) * 1000.0

        magic_d, n_blobs_d, n_foods_d, tick_d = struct.unpack_from("<IIII", bin_delta, 0)
        self.assertEqual(magic_d, 0x424C4F42)
        self.assertEqual(n_blobs_d, len(living))
        self.assertEqual(n_foods_d, 0, "Delta binary frame must omit food items (n_foods == 0)")
        self.assertEqual(len(bin_delta), 16 + n_blobs_d * 32)

        # Empirical assertion: Delta serialization for 2,000 blobs must be well within 33.3ms budget (<= 25.0ms)
        self.assertLess(avg_delta_ms, 25.0, f"Delta serialization too slow: {avg_delta_ms:.2f} ms")

        # 3. Verify unpacked blob payload properties
        blob_data = np.frombuffer(bin_delta[16:], dtype=np.float32).reshape((n_blobs_d, 8))
        xs = blob_data[:, 0]
        ys = blob_data[:, 1]
        headings = blob_data[:, 2]
        radii = blob_data[:, 3]
        flags = blob_data[:, 7]

        self.assertTrue(np.all(np.isfinite(xs)), "Non-finite blob x coordinate detected")
        self.assertTrue(np.all(np.isfinite(ys)), "Non-finite blob y coordinate detected")
        self.assertTrue(np.all((xs >= 0.0) & (xs <= world.world_cfg.width)), "Blob x out of arena bounds")
        self.assertTrue(np.all((ys >= 0.0) & (ys <= world.world_cfg.height)), "Blob y out of arena bounds")
        self.assertTrue(np.all(np.isfinite(headings)), "Non-finite heading detected")
        self.assertTrue(np.all(radii > 0.0), "Non-positive body radius detected")

        # Verify bitflags are valid 7-bit integers [0, 127]
        flag_ints = flags.astype(np.int32)
        self.assertTrue(np.all((flag_ints >= 0) & (flag_ints <= 127)), f"Invalid flag bitmask: {flag_ints}")

    def test_frustum_viewport_culling_boundary_conditions(self) -> None:
        """Verify viewport culling handles boundary cases: empty frustum, inverted bounds, point frustum."""
        cfg = WorldConfig.from_preset("continent")
        cfg.initial_blobs = 500
        world = SimulationWorld(world_cfg=cfg, seed=99)
        n_total = len([b for b in world.blobs if b.alive])

        # Boundary Case 1: Inverted bounds (min_x > max_x) with include_foods=False
        meta, bin_inv = world.get_ws_state_and_binary(include_foods=False, viewport=(2000.0, 2000.0, 1000.0, 1000.0))
        _, n_blobs_inv, n_foods_inv, _ = struct.unpack_from("<IIII", bin_inv, 0)
        self.assertEqual(n_blobs_inv, 0)
        self.assertEqual(n_foods_inv, 0)
        self.assertEqual(len(bin_inv), 16)  # Exactly 16 bytes header

        # Boundary Case 2: Out-of-bounds frustum with include_foods=False
        meta, bin_oob = world.get_ws_state_and_binary(include_foods=False, viewport=(-5000.0, -5000.0, -1000.0, -1000.0))
        _, n_blobs_oob, n_foods_oob, _ = struct.unpack_from("<IIII", bin_oob, 0)
        self.assertEqual(n_blobs_oob, 0)
        self.assertEqual(n_foods_oob, 0)
        self.assertEqual(len(bin_oob), 16)

        # Boundary Case 3: Massive frustum covering entire arena with include_foods=True
        meta, bin_all = world.get_ws_state_and_binary(include_foods=True, viewport=(-100.0, -100.0, world.world_cfg.width + 100.0, world.world_cfg.height + 100.0))
        _, n_blobs_all, n_foods_all, _ = struct.unpack_from("<IIII", bin_all, 0)
        self.assertEqual(n_blobs_all, n_total)
        self.assertEqual(n_foods_all, len(world.foods))
        self.assertEqual(len(bin_all), 16 + n_total * 32 + len(world.foods) * 12)

    def test_autonomous_background_simulation_loop_live_streaming(self) -> None:
        """Verify autonomous background simulation loop (lifespan) streams at ~30 FPS across multiple clients."""
        with controller.lock:
            controller.running = True
            controller.steps_per_frame = 1

        with TestClient(app) as live_client:
            # Client 1: Full-world observer
            with live_client.websocket_connect("/ws") as ws1:
                # Receive initial frame
                txt1 = ws1.receive_text()
                bin1 = ws1.receive_bytes()
                m1 = json.loads(txt1)
                self.assertEqual(m1["type"], "state")

                # Client 2: Zoomed-in culling observer
                with live_client.websocket_connect("/ws") as ws2:
                    txt2 = ws2.receive_text()
                    bin2 = ws2.receive_bytes()

                    ws2.send_text(json.dumps({
                        "cmd": "set_viewport",
                        "enabled": True,
                        "bounds": [1000.0, 1000.0, 2000.0, 2000.0],
                    }))

                    # Read 10 streaming frames autonomously from ws1
                    ticks_seen = []
                    for _ in range(10):
                        t = ws1.receive_text()
                        b = ws1.receive_bytes()
                        meta = json.loads(t)
                        ticks_seen.append(meta["data"]["tick"])
                        magic, nb, nf, tick = struct.unpack_from("<IIII", b, 0)
                        self.assertEqual(magic, 0x424C4F42)

                    # Ticks must strictly advance monotonically
                    self.assertEqual(len(ticks_seen), 10)
                    for i in range(len(ticks_seen) - 1):
                        self.assertGreater(ticks_seen[i+1], ticks_seen[i], "Simulation ticks did not advance")

                    # Read frame from ws2 (culled)
                    t_cull = ws2.receive_text()
                    b_cull = ws2.receive_bytes()
                    magic_c, nb_c, nf_c, tick_c = struct.unpack_from("<IIII", b_cull, 0)
                    self.assertEqual(magic_c, 0x424C4F42)

                    # Reset speed dynamically to 5x turbo
                    res = live_client.post("/api/command", json={"cmd": "set_speed", "speed": 5})
                    self.assertEqual(res.status_code, 200)

                    # Read 3 more frames on ws1 and verify step jump is >= 5
                    post_ticks = []
                    for _ in range(3):
                        t_p = ws1.receive_text()
                        b_p = ws1.receive_bytes()
                        m_p = json.loads(t_p)
                        post_ticks.append(m_p["data"]["tick"])
                    step_jumps = [post_ticks[i+1] - post_ticks[i] for i in range(len(post_ticks) - 1)]
                    self.assertTrue(
                        any(jump >= 5 for jump in step_jumps),
                        f"Expected at least one step jump >= 5, got jumps: {step_jumps}",
                    )

    def test_handshake_disconnect_causes_permanent_client_leak(self) -> None:
        """Adversarial Test: Verify whether a client disconnecting during initial handshake leaks in controller.clients."""
        from unittest.mock import AsyncMock, MagicMock
        from fastapi import WebSocket, WebSocketDisconnect
        import asyncio
        from blob_sim.server import websocket_endpoint

        mock_ws = MagicMock(spec=WebSocket)
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock(side_effect=WebSocketDisconnect(code=1006))
        mock_ws.send_bytes = AsyncMock()

        try:
            asyncio.run(websocket_endpoint(mock_ws))
        except WebSocketDisconnect:
            pass

        with controller.lock:
            leaked = mock_ws in controller.clients
            self.assertFalse(
                leaked,
                "CRITICAL BUG CONFIRMED: WebSocket client permanently leaked in controller.clients when handshake fails because lines 424-425 in blob_sim/server.py are outside the try...finally block!",
            )


if __name__ == "__main__":
    unittest.main()

