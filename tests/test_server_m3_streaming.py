"""Tests for Milestone 3: High-Throughput Web Server, WebSocket Streaming & Frontend GUI.

Validates:
- Feature 13: Async Worker Simulation Loop (asyncio.to_thread, threading.RLock)
- Feature 14: Two-Tier WebSocket Protocol (concurrent broadcast, topology caching, frame dropping)
- Feature 15: Stationary Food Caching (food_version, eaten_foods tracking, bandwidth savings)
- Feature 16: Vectorized Binary Buffer Packing (pack_binary_blobs_kernel JIT, bitflag encoding)
- Feature 17: Viewport Culling Support (set_viewport command, client frustum filtering)
- Feature 18: Frontend Protocol Invariants & Typed Array Contracts
"""

import json
import struct
import threading
import time
import unittest
from typing import Any, Dict, List

import numpy as np
from fastapi.testclient import TestClient

from blob_sim.config import WorldConfig
from blob_sim.server import app, controller
from blob_sim.spatial import pack_binary_blobs_kernel
from blob_sim.world import SimulationWorld


class TestMilestone3Streaming(unittest.TestCase):
    """Milestone 3 comprehensive test suite."""

    def setUp(self) -> None:
        """Reset simulation controller state before each test."""
        with controller.lock:
            controller.running = False
            controller.steps_per_frame = 1
            controller.client_topo_versions.clear()
            controller.client_food_versions.clear()
            controller.client_viewports.clear()
            controller.client_busy.clear()
            # Reset to a known clean world preset
            cfg = WorldConfig.from_preset("continent")
            cfg.initial_blobs = 120
            cfg.initial_foods = 200
            controller.world = SimulationWorld(world_cfg=cfg, seed=42)

    def tearDown(self) -> None:
        """Ensure simulation controller is stopped after each test."""
        with controller.lock:
            controller.running = False
            controller.client_topo_versions.clear()
            controller.client_food_versions.clear()
            controller.client_viewports.clear()
            controller.client_busy.clear()

    # --- Feature 13: Async Worker Simulation Loop & RLock ---

    def test_controller_lock_is_rlock_and_protects_world_state(self) -> None:
        """Verify controller.lock is a reentrant lock and prevents data race errors."""
        self.assertIsInstance(controller.lock, type(threading.RLock()))

        # Reentrant lock verification
        with controller.lock:
            with controller.lock:
                self.assertIsNotNone(controller.world)

    def test_step_worker_execution_and_precomputation(self) -> None:
        """Verify controller._step_worker executes off-thread and precomputes state."""
        with controller.lock:
            tick_before = controller.world.tick_count
            client_needs_topo = True
            client_needs_food = True

        # Run step worker directly
        text_payload, binary_payload, topo_ver, food_ver = controller._step_worker(
            steps=3,
            need_topo=client_needs_topo,
            need_food=client_needs_food,
            running=True,
            speed=3,
        )

        with controller.lock:
            self.assertEqual(controller.world.tick_count, tick_before + 3)

        msg = json.loads(text_payload)
        self.assertIn("data", msg)
        self.assertIn("obstacles", msg["data"])
        self.assertGreater(len(binary_payload), 16)
        magic, n_blobs, n_foods, tick = struct.unpack_from("<IIII", binary_payload, 0)
        self.assertEqual(magic, 0x424C4F42)
        self.assertEqual(n_blobs, len(controller.world.blobs))
        self.assertGreater(n_foods, 0)

    def test_concurrent_rest_commands_under_lock(self) -> None:
        """Verify concurrent REST API command execution does not race with simulation stepping."""
        client = TestClient(app)
        exceptions: List[Exception] = []

        def worker_commands():
            try:
                for speed in [1, 2, 5, 1, 15]:
                    res = client.post("/api/command", json={"cmd": "set_speed", "speed": speed})
                    self.assertEqual(res.status_code, 200)
                    time.sleep(0.005)
            except Exception as e:
                exceptions.append(e)

        def worker_state_polling():
            try:
                for _ in range(8):
                    res = client.get("/api/state")
                    self.assertEqual(res.status_code, 200)
                    time.sleep(0.005)
            except Exception as e:
                exceptions.append(e)

        def worker_stepping():
            try:
                for _ in range(5):
                    with controller.lock:
                        controller.world.step()
                    time.sleep(0.006)
            except Exception as e:
                exceptions.append(e)

        t1 = threading.Thread(target=worker_commands)
        t2 = threading.Thread(target=worker_state_polling)
        t3 = threading.Thread(target=worker_stepping)

        t1.start()
        t2.start()
        t3.start()

        t1.join(timeout=3.0)
        t2.join(timeout=3.0)
        t3.join(timeout=3.0)

        self.assertEqual(len(exceptions), 0, f"Concurrent exceptions occurred: {exceptions}")

    # --- Feature 14 & 15: Two-Tier Protocol & Stationary Food Caching ---

    def test_two_tier_websocket_protocol_and_food_caching(self) -> None:
        """Verify two-tier WebSocket negotiation: full frame on connect, delta frame on subsequent ticks."""
        client = TestClient(app)

        with client.websocket_connect("/ws") as ws:
            # 1. Initial Frame (Tier 1: Full Topology + Full Food Buffer)
            raw_meta_1 = ws.receive_text()
            meta_1 = json.loads(raw_meta_1)
            self.assertEqual(meta_1["type"], "state")
            self.assertIn("obstacles", meta_1["data"])
            self.assertIn("oases", meta_1["data"])
            self.assertIn("homes", meta_1["data"])
            self.assertIn("zones", meta_1["data"]["world"])

            raw_bin_1 = ws.receive_bytes()
            magic1, n_blobs1, n_foods1, tick1 = struct.unpack_from("<IIII", raw_bin_1, 0)
            self.assertEqual(magic1, 0x424C4F42)
            self.assertGreater(n_blobs1, 0)
            self.assertGreater(n_foods1, 0)
            expected_full_len = 16 + n_blobs1 * 32 + n_foods1 * 12
            self.assertEqual(len(raw_bin_1), expected_full_len)

            # 2. Trigger step_once command over WebSocket
            ws.send_text(json.dumps({"cmd": "step_once", "steps": 1}))

            # 3. Delta Frame (Tier 2: Omits static obstacles and zones; n_foods == 0)
            raw_meta_2 = ws.receive_text()
            meta_2 = json.loads(raw_meta_2)
            self.assertEqual(meta_2["type"], "state")
            self.assertNotIn("obstacles", meta_2["data"])
            self.assertNotIn("zones", meta_2["data"]["world"])

            raw_bin_2 = ws.receive_bytes()
            magic2, n_blobs2, n_foods2, tick2 = struct.unpack_from("<IIII", raw_bin_2, 0)
            self.assertEqual(magic2, 0x424C4F42)
            # Food count is strictly 0 in delta binary buffer
            self.assertEqual(n_foods2, 0)
            # Binary length strictly contains only blobs payload (16 header + n_blobs * 32)
            expected_delta_len = 16 + n_blobs2 * 32
            self.assertEqual(len(raw_bin_2), expected_delta_len)
            # Bandwidth savings: delta buffer omits n_foods1 * 12 bytes
            self.assertEqual(len(raw_bin_1) - len(raw_bin_2), n_foods1 * 12 + (n_blobs1 - n_blobs2) * 32)

    def test_stationary_food_cache_invalidation_on_food_version_change(self) -> None:
        """Verify that when food_version changes (e.g. daily respawn), server streams fresh food buffer."""
        client = TestClient(app)

        with client.websocket_connect("/ws") as ws:
            # Consume initial frame
            ws.receive_text()
            ws.receive_bytes()

            # Advance 1 step: delta frame with n_foods == 0
            ws.send_text(json.dumps({"cmd": "step_once", "steps": 1}))
            ws.receive_text()
            raw_bin_delta = ws.receive_bytes()
            _, _, n_foods_delta, _ = struct.unpack_from("<IIII", raw_bin_delta, 0)
            self.assertEqual(n_foods_delta, 0)

            # Invalidate food version by spawning a custom food cluster
            with controller.lock:
                controller.world.spawn_custom_food_cluster(1000.0, 1000.0, count=10)

            # Next step must deliver fresh full food array because client_food_version is stale
            ws.send_text(json.dumps({"cmd": "step_once", "steps": 1}))
            ws.receive_text()
            raw_bin_refresh = ws.receive_bytes()
            _, _, n_foods_refresh, _ = struct.unpack_from("<IIII", raw_bin_refresh, 0)
            self.assertGreater(n_foods_refresh, 0)
            self.assertEqual(n_foods_refresh, len(controller.world.foods))

    def test_eaten_foods_metadata_tracking(self) -> None:
        """Verify eaten foods are reported in JSON delta metadata."""
        world = controller.world
        with controller.lock:
            world.recent_eaten_indices.append(17)
            world.recent_eaten_indices.append(42)

            meta, _ = world.get_ws_state_and_binary(include_topology=False, include_foods=False)
            self.assertIn("eaten_foods", meta)
            self.assertEqual(meta["eaten_foods"], [17, 42])

            # Recent eaten indices are cleared upon generation
            self.assertEqual(len(world.recent_eaten_indices), 0)

    # --- Feature 16: Vectorized Binary Buffer Packing Kernel ---

    def test_pack_binary_blobs_kernel_parity_and_bitflags(self) -> None:
        """Verify pack_binary_blobs_kernel packs kinematics, IDs, RGBs, and 7-bit status flags accurately."""
        n = 10
        out_buffer = np.zeros((n, 8), dtype=np.float32)
        xs = np.linspace(100.0, 1000.0, n, dtype=np.float64)
        ys = np.linspace(200.0, 2000.0, n, dtype=np.float64)
        headings = np.full(n, 1.23, dtype=np.float32)
        radii = np.full(n, 12.5, dtype=np.float32)
        foods_today = np.full(n, 1.5, dtype=np.float32)
        ids = np.arange(1, n + 1, dtype=np.float32)
        gids = np.full(n, -1, dtype=np.int32)
        gids[0] = 1
        group_rgb_table = np.zeros(4, dtype=np.uint32)
        group_rgb_table[1] = 0x10B981
        gene_rgbs = np.full(n, 0x38BDF8, dtype=np.uint32)
        is_at_homes = np.zeros(n, dtype=np.bool_)
        is_at_homes[0] = True
        in_swamps = np.zeros(n, dtype=np.bool_)
        in_swamps[1] = True
        surplus_foods = np.zeros(n, dtype=np.float32)
        surplus_foods[2] = 0.2
        sprints = np.zeros(n, dtype=np.float32)
        sprints[3] = 0.5
        reputations = np.zeros(n, dtype=np.float32)
        reputations[5] = -0.5  # outlaw
        reputations[6] = 0.5   # honorable

        rep_sum, outlaws, honorable = pack_binary_blobs_kernel(
            out_buffer, xs, ys, headings, radii, foods_today, ids,
            gids, group_rgb_table, gene_rgbs, is_at_homes, in_swamps,
            surplus_foods, sprints, reputations, n
        )

        self.assertEqual(outlaws, 1)
        self.assertEqual(honorable, 1)
        self.assertAlmostEqual(rep_sum, 0.0, places=4)

        # Check blob 0: group RGB + is_at_home flag (bit 0) + group member flag (bit 4)
        flags_0 = int(out_buffer[0, 7])
        self.assertEqual(flags_0 & 1, 1)
        self.assertEqual(flags_0 & 16, 16)
        self.assertEqual(int(out_buffer[0, 6]), 0x10B981)

        # Check blob 1: gene RGB + in_swamps flag (bit 1)
        flags_1 = int(out_buffer[1, 7])
        self.assertEqual(flags_1 & 2, 2)
        self.assertEqual(int(out_buffer[1, 6]), 0x38BDF8)

        # Check blob 2: surplus_foods flag (bit 2)
        flags_2 = int(out_buffer[2, 7])
        self.assertEqual(flags_2 & 4, 4)

        # Check blob 3: sprint flag (bit 3)
        flags_3 = int(out_buffer[3, 7])
        self.assertEqual(flags_3 & 8, 8)

        # Check blob 5: outlaw flag (bit 5)
        flags_5 = int(out_buffer[5, 7])
        self.assertEqual(flags_5 & 32, 32)

        # Check blob 6: honorable flag (bit 6)
        flags_6 = int(out_buffer[6, 7])
        self.assertEqual(flags_6 & 64, 64)

        # Check blob coordinates
        self.assertAlmostEqual(out_buffer[0, 0], 100.0, places=4)
        self.assertAlmostEqual(out_buffer[0, 1], 200.0, places=4)

    # --- Feature 17: Viewport Culling Support ---

    def test_viewport_culling_command_and_filtering(self) -> None:
        """Verify client set_viewport command culls blobs on server without affecting stats."""
        client = TestClient(app)

        with client.websocket_connect("/ws") as ws:
            # Ingest initial full frame
            ws.receive_text()
            raw_bin = ws.receive_bytes()
            _, initial_blobs, _, _ = struct.unpack_from("<IIII", raw_bin, 0)
            self.assertGreater(initial_blobs, 0)

            # Move some blobs inside bounds and some outside in _kin_xs and _kin_ys
            with controller.lock:
                controller.world._kin_xs[:5] = 1500.0
                controller.world._kin_ys[:5] = 1500.0
                controller.world._kin_xs[5:] = 35000.0
                controller.world._kin_ys[5:] = 25000.0

            # Send set_viewport command covering [1000, 1000, 2000, 2000]
            ws.send_text(json.dumps({
                "cmd": "set_viewport",
                "enabled": True,
                "bounds": [1000.0, 1000.0, 2000.0, 2000.0]
            }))

            # Trigger step
            ws.send_text(json.dumps({"cmd": "step_once", "steps": 1}))

            # Receive culled delta frame
            raw_meta = ws.receive_text()
            meta = json.loads(raw_meta)
            culled_bin = ws.receive_bytes()
            _, n_culled_blobs, _, _ = struct.unpack_from("<IIII", culled_bin, 0)

            # Only the 5 blobs inside [1000, 1000, 2000, 2000] are streamed
            self.assertEqual(n_culled_blobs, 5)
            # Worldwide population statistic remains initial_blobs (uncullled global KPI)
            self.assertEqual(meta["data"]["stats"]["population"], initial_blobs)

            # Disable viewport culling
            ws.send_text(json.dumps({"cmd": "set_viewport", "enabled": False}))
            ws.send_text(json.dumps({"cmd": "step_once", "steps": 1}))
            ws.receive_text()
            restored_bin = ws.receive_bytes()
            _, n_restored_blobs, _, _ = struct.unpack_from("<IIII", restored_bin, 0)
            self.assertEqual(n_restored_blobs, initial_blobs)

    # --- Multi-Client Concurrency & Disconnection Cleanup ---

    def test_multi_client_concurrent_streaming_and_disconnect(self) -> None:
        """Verify multiple WebSocket clients stream independently and disconnect without leaks."""
        client = TestClient(app)

        with client.websocket_connect("/ws") as ws1:
            ws1.receive_text()
            ws1.receive_bytes()

            with client.websocket_connect("/ws") as ws2:
                ws2.receive_text()
                ws2.receive_bytes()

                with controller.lock:
                    self.assertEqual(len(controller.clients), 2)

                # Send step to both
                ws1.send_text(json.dumps({"cmd": "step_once", "steps": 1}))
                t1 = ws1.receive_text()
                b1 = ws1.receive_bytes()
                self.assertIsNotNone(t1)
                self.assertIsNotNone(b1)

            # ws2 disconnected, ws1 continues
            with controller.lock:
                # ws2 entry cleaned up
                self.assertEqual(len(controller.clients), 1)

            ws1.send_text(json.dumps({"cmd": "step_once", "steps": 1}))
            t2 = ws1.receive_text()
            b2 = ws1.receive_bytes()
            self.assertIsNotNone(t2)
            self.assertIsNotNone(b2)

        with controller.lock:
            self.assertEqual(len(controller.clients), 0)
            self.assertEqual(len(controller.client_viewports), 0)
            self.assertEqual(len(controller.client_food_versions), 0)

    # --- Map Scale Preset Switching Tests ---

    def test_switch_map_scale_presets(self) -> None:
        """Verify seamless switching between classic, regional, and continent map scale presets."""
        client = TestClient(app)

        # 1. Start from default continent preset
        with controller.lock:
            self.assertEqual(controller.world.world_cfg.scale_preset, "continent")
            self.assertEqual(controller.world.world_cfg.width, 40000.0)
            self.assertEqual(controller.world.world_cfg.height, 30000.0)
            self.assertEqual(controller.world.world_cfg.num_cradles, 12)

        # 2. Switch to 'regional' using `preset` key (frontend format)
        res = client.post("/api/command", json={"cmd": "set_scale_preset", "preset": "regional"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("ok"))
        self.assertEqual(data["world_cfg"]["scale_preset"], "regional")
        with controller.lock:
            self.assertEqual(controller.world.world_cfg.scale_preset, "regional")
            self.assertEqual(controller.world.world_cfg.width, 16000.0)
            self.assertEqual(controller.world.world_cfg.height, 12000.0)
            self.assertEqual(controller.world.world_cfg.num_cradles, 5)

        # 3. Switch to 'classic' using `scale_preset` key (backend format)
        res = client.post("/api/command", json={"cmd": "set_scale_preset", "scale_preset": "classic"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("ok"))
        self.assertEqual(data["world_cfg"]["scale_preset"], "classic")
        with controller.lock:
            self.assertEqual(controller.world.world_cfg.scale_preset, "classic")
            self.assertEqual(controller.world.world_cfg.width, 4800.0)
            self.assertEqual(controller.world.world_cfg.height, 3600.0)
            self.assertEqual(controller.world.world_cfg.num_cradles, 1)

        # 4. Switch back to 'continent' via `update_world`
        res = client.post("/api/command", json={"cmd": "update_world", "world": {"scale_preset": "continent"}})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("ok"))
        self.assertEqual(data["world_cfg"]["scale_preset"], "continent")
        with controller.lock:
            self.assertEqual(controller.world.world_cfg.scale_preset, "continent")
            self.assertEqual(controller.world.world_cfg.width, 40000.0)
            self.assertEqual(controller.world.world_cfg.height, 30000.0)
            self.assertEqual(controller.world.world_cfg.num_cradles, 12)

        # 5. Verify over WebSocket connection as well
        with client.websocket_connect("/ws") as ws:
            ws.receive_text()
            ws.receive_bytes()

            # Send preset switch to regional
            ws.send_text(json.dumps({"cmd": "set_scale_preset", "preset": "regional"}))
            # State broadcast after command execution
            t = ws.receive_text()
            b = ws.receive_bytes()
            state_msg = json.loads(t)
            self.assertEqual(state_msg["data"]["world"]["scale_preset"], "regional")
            self.assertEqual(state_msg["data"]["world"]["width"], 16000.0)


if __name__ == "__main__":
    unittest.main()
