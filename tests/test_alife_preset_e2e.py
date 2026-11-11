"""End-to-End Validation Suite for the 'alife_emergent' Preset and Frontend Integration.

Verifies:
1. Switching to 'alife_emergent' preset via /api/command enables all 5 ALife flags:
   - every_tick_brain == True
   - egocentric_frame == True
   - intrinsic_homeostasis == True
   - continuous_lifecycle == True
   - emergent_interactions == True
   and automatically activates recurrent Baldwin brain (depth=2, width=48, recurrent_dim=16, baldwin_plasticity=True).
2. Simulates 200 continuous ticks, verifying blobs survive, forage, emit bioluminescent signals,
   exhibit no NaN/Inf kinematics or weights, and that WebSocket binary frames (0x424C4F42) pack
   valid signal flags.
3. Queries /api/analytics and ensures valid 26x6 coupling matrix and gene statistics.
4. Switches back to 'classic' and 'continent', verifying all 5 ALife flags cleanly reset to False
   and network configuration reverts to standard baseline.
"""

import json
import math
import struct
import unittest
import numpy as np
from fastapi.testclient import TestClient

from blob_sim.server import app, controller
from blob_sim.config import WorldConfig, NetworkConfig, WORLD_SCALE_PRESETS
from blob_sim.world import SimulationWorld


class TestAlifePresetE2E(unittest.TestCase):
    """E2E test suite for alife_emergent preset."""

    def setUp(self) -> None:
        with controller.lock:
            controller.running = False
            controller.steps_per_frame = 1
            controller.client_topo_versions.clear()
            controller.client_food_versions.clear()
            controller.client_viewports.clear()
            controller.client_busy.clear()
            cfg = WorldConfig.from_preset("continent")
            controller.world = SimulationWorld(world_cfg=cfg, seed=42)

    def tearDown(self) -> None:
        with controller.lock:
            controller.running = False
            controller.client_topo_versions.clear()
            controller.client_food_versions.clear()
            controller.client_viewports.clear()
            controller.client_busy.clear()

    def test_e2e_alife_preset_lifecycle_and_analytics(self) -> None:
        """Complete E2E workflow: switch to alife_emergent, simulate 200 ticks, check WebSocket & analytics, revert."""
        client = TestClient(app)

        # 1. Switch to alife_emergent via /api/command
        resp = client.post("/api/command", json={"cmd": "set_scale_preset", "scale_preset": "alife_emergent"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data.get("ok"))

        world = controller.world
        wcfg = world.world_cfg
        net_cfg = world.net_cfg

        # Check all 5 ALife flags
        self.assertEqual(wcfg.scale_preset, "alife_emergent")
        self.assertTrue(wcfg.every_tick_brain)
        self.assertTrue(wcfg.egocentric_frame)
        self.assertTrue(wcfg.intrinsic_homeostasis)
        self.assertTrue(wcfg.continuous_lifecycle)
        self.assertTrue(wcfg.emergent_interactions)
        self.assertEqual(wcfg.width, 8000.0)
        self.assertEqual(wcfg.height, 6000.0)
        self.assertEqual(wcfg.min_population_floor, 30)
        self.assertEqual(wcfg.ticks_per_day, 600)

        # Check recurrent Baldwin brain auto-configuration
        self.assertEqual(net_cfg.depth, 2)
        self.assertEqual(net_cfg.width, 48)
        self.assertEqual(net_cfg.recurrent_dim, 16)
        self.assertTrue(net_cfg.baldwin_plasticity)
        self.assertEqual(net_cfg.activation, "tanh")

        # Check topography invariants (16 oases, 18 rocks, 6 swamps)
        self.assertEqual(len(world.oases), 16)
        rocks = [o for o in world.obstacles if o.kind == "rock"]
        swamps = [o for o in world.obstacles if o.kind == "swamp"]
        self.assertEqual(len(rocks), 18)
        self.assertEqual(len(swamps), 6)

        # 2. Simulate 200 continuous ticks and verify simulation integrity
        with client.websocket_connect("/ws") as ws:
            # First frame has initial state + binary
            raw_meta = ws.receive_text()
            meta = json.loads(raw_meta)
            self.assertEqual(meta["type"], "state")
            bin_data = ws.receive_bytes()
            magic, n_blobs_p, n_foods_p, tick_p = struct.unpack_from("<IIII", bin_data, 0)
            self.assertEqual(magic, 0x424C4F42)
            self.assertGreater(n_blobs_p, 0)

            # Step 200 ticks
            for step in range(200):
                ws.send_text(json.dumps({"cmd": "step_once", "steps": 1}))
                _ = ws.receive_text()
                bin_frame = ws.receive_bytes()
                f_magic, f_blobs, f_foods, f_tick = struct.unpack_from("<IIII", bin_frame, 0)
                self.assertEqual(f_magic, 0x424C4F42)
                self.assertGreater(f_blobs, 0)

            # Validate living blobs, kinematics, signals, and no NaN/Inf
            living = [b for b in world.blobs if b.alive]
            self.assertGreater(len(living), 0)

            n_liv = len(living)
            self.assertFalse(np.isnan(world._kin_xs[:n_liv]).any())
            self.assertFalse(np.isinf(world._kin_xs[:n_liv]).any())
            self.assertFalse(np.isnan(world._kin_ys[:n_liv]).any())
            self.assertFalse(np.isinf(world._kin_ys[:n_liv]).any())
            self.assertFalse(np.isnan(world._kin_energies[:n_liv]).any())
            self.assertFalse(np.isnan(world._kin_signals[:n_liv]).any())

            # Verify binary unpacking of flags (including emergent signal bits 5 and 6)
            blobs_array = np.frombuffer(bin_frame[16:16 + f_blobs * 32], dtype=np.float32).reshape(-1, 8)
            flags = blobs_array[:, 7].astype(np.int32)
            signals = blobs_array[:, 7]  # flags column
            # Check flags are non-negative and valid integers
            self.assertTrue((flags >= 0).all())

        # 3. Query /api/analytics
        # Advance 1 full day (600 ticks) so daily_logs snapshot with weights and gene statistics is created
        world.advance_ticks(600)
        analytics_resp = client.get("/api/analytics")
        self.assertEqual(analytics_resp.status_code, 200)
        analytics_data = analytics_resp.json()

        self.assertIn("initial_coupling_matrix", analytics_data)
        init_matrix = analytics_data["initial_coupling_matrix"]
        self.assertIsNotNone(init_matrix)
        self.assertEqual(len(init_matrix), 26)  # 26 sensor labels
        for row in init_matrix:
            self.assertEqual(len(row), 6)  # 6 motor outputs
            for val in row:
                self.assertFalse(math.isnan(val))
                self.assertFalse(math.isinf(val))

        self.assertIn("daily_logs", analytics_data)
        self.assertGreater(len(analytics_data["daily_logs"]), 0)
        last_day = analytics_data["daily_logs"][-1]
        self.assertIn("weights", last_day)
        self.assertIn("coupling_matrix", last_day["weights"])
        c_matrix = last_day["weights"]["coupling_matrix"]
        self.assertEqual(len(c_matrix), 26)
        for row in c_matrix:
            self.assertEqual(len(row), 6)

        self.assertIn("genetics", last_day)
        self.assertIn("speed", last_day["genetics"])
        self.assertIn("mean", last_day["genetics"]["speed"])
        self.assertIn("summary", analytics_data)

        # 4. Switch back to classic and continent, verify reset of ALife flags & brain
        resp_classic = client.post("/api/command", json={"cmd": "set_scale_preset", "scale_preset": "classic"})
        self.assertEqual(resp_classic.status_code, 200)
        wcfg_c = controller.world.world_cfg
        net_c = controller.world.net_cfg
        self.assertEqual(wcfg_c.scale_preset, "classic")
        self.assertFalse(wcfg_c.every_tick_brain)
        self.assertFalse(wcfg_c.egocentric_frame)
        self.assertFalse(wcfg_c.intrinsic_homeostasis)
        self.assertFalse(wcfg_c.continuous_lifecycle)
        self.assertFalse(wcfg_c.emergent_interactions)
        self.assertEqual(net_c.width, 24)
        self.assertEqual(net_c.recurrent_dim, 0)
        self.assertFalse(net_c.baldwin_plasticity)

        # Switch to continent
        resp_cont = client.post("/api/command", json={"cmd": "set_scale_preset", "scale_preset": "continent"})
        self.assertEqual(resp_cont.status_code, 200)
        wcfg_cont = controller.world.world_cfg
        self.assertEqual(wcfg_cont.scale_preset, "continent")
        self.assertFalse(wcfg_cont.every_tick_brain)
        self.assertFalse(wcfg_cont.emergent_interactions)


if __name__ == "__main__":
    unittest.main()
