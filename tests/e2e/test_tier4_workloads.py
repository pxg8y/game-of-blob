"""Tier 4 - Real-World Application Workloads E2E Tests.

Tests full long-horizon application workloads:
1. Multi-day ecosystem evolution and telemetry integrity across seasonal cycles
2. Scaled 2,000 concurrent blob simulation throughput & cohort scheduling
3. WebSocket continuous streaming under load & real-time command dispatch
4. Full lifecycle population state serialization & deserialization roundtrip
"""

import json
import struct
import unittest
import numpy as np
from fastapi.testclient import TestClient

from blob_sim.config import NetworkConfig, WorldConfig
from blob_sim.server import app, controller
from blob_sim.world import SimulationWorld


class TestTier4RealWorldWorkloads(unittest.TestCase):
    """Tier 4: End-to-end long-horizon application workloads."""

    def tearDown(self) -> None:
        if hasattr(self, "world") and self.world is not None:
            if hasattr(self.world, "close"):
                self.world.close()
            self.world = None
        if hasattr(self, "world_orig") and self.world_orig is not None:
            if hasattr(self.world_orig, "close"):
                self.world_orig.close()
            self.world_orig = None
        if hasattr(self, "world_restored") and self.world_restored is not None:
            if hasattr(self.world_restored, "close"):
                self.world_restored.close()
            self.world_restored = None
        import gc
        gc.collect()

    def test_multi_day_ecosystem_evolution_and_telemetry_integrity(self) -> None:
        """Verify 5-day continuous ecosystem evolution with daily history snapshots and telemetry logs."""
        world = SimulationWorld(
            world_cfg=WorldConfig(initial_blobs=20, ticks_per_day=50, survival_food_need=1.0),
            seed=101,
        )
        self.world = world
        initial_day = world.day_number

        # Run 5 full simulated days (5 * 50 = 250 ticks)
        world.advance_ticks(250)

        self.assertEqual(world.day_number, initial_day + 5)
        self.assertEqual(world.tick_count, 250)
        self.assertEqual(len(world.history), 5)

        # Telemetry verification
        self.assertGreaterEqual(len(world.telemetry.daily_logs), 5)
        latest_log = world.telemetry.daily_logs[-1]
        self.assertIn("community", latest_log)
        self.assertIn("actions", latest_log)
        self.assertIn("genetics", latest_log)
        self.assertIn("weights", latest_log)

        # Coupling matrix dimensions: [26, 6]
        coupling = latest_log["weights"]["coupling_matrix"]
        self.assertEqual(len(coupling), 26)
        self.assertEqual(len(coupling[0]), 6)

        # Analytics payload validation
        analytics = world.telemetry.get_analytics_payload(world, since_day=0)
        self.assertIn("summary", analytics)
        self.assertIn("daily_logs", analytics)
        self.assertIn("input_labels", analytics)
        self.assertIn("output_labels", analytics)
        self.assertEqual(len(analytics["input_labels"]), 26)
        self.assertEqual(len(analytics["output_labels"]), 6)

        # Population should be viable
        living = [b for b in world.blobs if b.alive]
        self.assertGreater(len(living), 0)

    def test_scaled_2000_blob_continuous_simulation(self) -> None:
        """Verify 2,000 concurrent blobs execute batched kinematics, 3-cohort GEMV, and foraging stably."""
        world = SimulationWorld(
            world_cfg=WorldConfig(
                width=16000.0,
                height=12000.0,
                initial_blobs=2000,
                ticks_per_day=120,
                max_population_cap=5000,
            ),
            seed=202,
        )
        self.world = world
        self.assertEqual(len(world.blobs), 2000)

        # Advance 6 steps so all 3 cohorts (0, 1, 2) execute twice
        for _ in range(6):
            world.step(1)

        self.assertEqual(world.tick_count, 6)
        living = [b for b in world.blobs if b.alive]
        n_living = len(living)
        self.assertGreaterEqual(n_living, 1950)

        # All living slots in brain pool must have valid cached action outputs
        self.assertTrue(np.all(world._brain_pool._out_valid_cap[:n_living]))

        # Positions and energies must remain non-NaN and bounded
        for b in living[:200]:
            self.assertFalse(np.isnan(b.x))
            self.assertFalse(np.isnan(b.y))
            self.assertFalse(np.isnan(b.energy))
            self.assertGreaterEqual(b.energy, 0.0)

    def test_websocket_streaming_under_continuous_load(self) -> None:
        """Verify continuous WebSocket streaming with two-tier protocol and real-time command handling."""
        client = TestClient(app)

        with client.websocket_connect("/ws") as ws:
            # 1. Receive initial topology frame
            raw_meta = ws.receive_text()
            meta_json = json.loads(raw_meta)
            self.assertEqual(meta_json["type"], "state")
            self.assertIn("data", meta_json)
            self.assertIn("zones", meta_json["data"]["world"])
            self.assertIn("obstacles", meta_json["data"])

            raw_binary = ws.receive_bytes()
            magic, n_blobs, n_foods, tick = struct.unpack_from("<IIII", raw_binary, 0)
            self.assertEqual(magic, 0x424C4F42)
            expected_len = 16 + n_blobs * 32 + n_foods * 12
            self.assertEqual(len(raw_binary), expected_len)

            # 2. Send commands over the active WebSocket channel
            ws.send_text(json.dumps({"cmd": "step_once", "steps": 2}))
            ws.send_text(json.dumps({"cmd": "set_speed", "speed": 5}))
            ws.send_text(json.dumps({"cmd": "god_action", "tool": "spawn_food", "x": 1200.0, "y": 900.0}))

            # 3. Receive delta frame response
            delta_meta_raw = ws.receive_text()
            delta_meta = json.loads(delta_meta_raw)
            self.assertEqual(delta_meta["type"], "state")

            delta_bin = ws.receive_bytes()
            d_magic, d_blobs, d_foods, _ = struct.unpack_from("<IIII", delta_bin, 0)
            self.assertEqual(d_magic, 0x424C4F42)
            self.assertEqual(len(delta_bin), 16 + d_blobs * 32 + d_foods * 12)

    def test_full_lifecycle_population_state_persistence_roundtrip(self) -> None:
        """Verify complete population export and import roundtrip preserves day, brain configurations, and state."""
        world_orig = SimulationWorld(
            world_cfg=WorldConfig(initial_blobs=20, ticks_per_day=40, survival_food_need=1.0),
            seed=303,
        )
        self.world_orig = world_orig
        # Advance 2 days so entities acquire rewards, history, and stats
        world_orig.advance_ticks(80)
        self.assertEqual(world_orig.day_number, 3)

        # Export population state to JSON
        exported_json = world_orig.export_population_json()
        self.assertIn("day", exported_json)
        self.assertIn("blobs", exported_json)
        self.assertEqual(exported_json["day"], 3)
        self.assertGreater(len(exported_json["blobs"]), 0)

        # Create a fresh simulation world with different seed
        world_restored = SimulationWorld(seed=999)
        self.world_restored = world_restored
        self.assertNotEqual(world_restored.day_number, world_orig.day_number)

        # Import saved state
        world_restored.import_population_json(exported_json)
        self.assertEqual(world_restored.day_number, world_orig.day_number)
        self.assertEqual(len(world_restored.blobs), len(world_orig.blobs))

        # Check entity attributes and brain preservation
        orig_first = world_orig.blobs[0]
        restored_first = world_restored.blobs[0]
        self.assertEqual(restored_first.generation, exported_json["blobs"][0]["generation"])
        self.assertEqual(restored_first.brain.layer_dims, orig_first.brain.layer_dims)
        self.assertTrue(np.allclose(restored_first.brain.weights[0], orig_first.brain.weights[0]))

        # Step restored world forward to verify seamless continued simulation
        world_restored.step(2)
        self.assertGreater(world_restored.tick_count, 0)


if __name__ == "__main__":
    unittest.main()
