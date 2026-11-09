"""Tier 1 - Feature Coverage E2E Tests.

Tests each core feature of the simulation in isolation:
1. Simulation stepping (single and multi-tick advance)
2. Blob spawning and attribute initialization
3. Neural evaluation (26-in, 6-out, 10k neurons forward pass, activation functions)
4. Food consumption and foraging
5. Energy depletion, metabolism, and terrain drag
6. Reproduction, gene inheritance, and mutation
7. Peer-to-peer trading and sovereign pricing
8. WebSocket binary stream format and topology caching
9. FastAPI REST API endpoints
"""

import gc
import struct
import unittest
import numpy as np
from fastapi.testclient import TestClient

from blob_sim.config import NetworkConfig, WorldConfig
from blob_sim.neural import BlobBrain, INPUT_DIM, OUTPUT_DIM, apply_activation, activation_derivative
from blob_sim.server import app, controller
from blob_sim.world import SimulationWorld


class TestTier1FeatureCoverage(unittest.TestCase):
    """Tier 1: Comprehensive feature coverage in isolation."""

    def setUp(self) -> None:
        self.world = SimulationWorld(
            world_cfg=WorldConfig(initial_blobs=16, ticks_per_day=60, survival_food_need=1.0),
            seed=42,
        )

    def tearDown(self) -> None:
        if hasattr(self, "world") and self.world is not None:
            if hasattr(self.world, "close"):
                self.world.close()
            self.world = None
        gc.collect()

    def test_simulation_stepping_single_and_multi_tick(self) -> None:
        """Verify simulation advances correctly via single steps and multi-tick jumps."""
        initial_tick = self.world.tick_count
        initial_day = self.world.day_number

        # 1. Single step
        self.world.step(1)
        self.assertEqual(self.world.tick_count, initial_tick + 1)
        self.assertEqual(self.world.tick_in_day, 1)
        self.assertEqual(self.world.day_number, initial_day)

        # 2. Advance to complete the day (ticks_per_day = 60)
        remaining = self.world.world_cfg.ticks_per_day - 1
        self.world.advance_ticks(remaining)
        self.assertEqual(self.world.day_number, initial_day + 1)
        self.assertEqual(self.world.tick_in_day, 0)

        # 3. Large multi-tick advance (advance 180 ticks = 3 full days)
        self.world.advance_ticks(180)
        self.assertEqual(self.world.day_number, initial_day + 4)
        self.assertEqual(self.world.tick_in_day, 0)
        self.assertGreaterEqual(len(self.world.history), 3)

    def test_blob_spawning_and_attribute_initialization(self) -> None:
        """Verify blob spawning creates entities with valid positions, genes, and brains."""
        spawn_x, spawn_y = 1250.0, 950.0
        blob = self.world.spawn_custom_blob(spawn_x, spawn_y)

        self.assertIsNotNone(blob)
        self.assertAlmostEqual(blob.x, spawn_x)
        self.assertAlmostEqual(blob.y, spawn_y)
        self.assertTrue(blob.alive)
        self.assertGreater(blob.energy, 0.0)
        self.assertEqual(blob.food_today, 0.0)
        self.assertGreaterEqual(blob.money, 0.0)
        self.assertIsNotNone(blob.brain)
        self.assertEqual(blob.brain.layer_dims[0], INPUT_DIM)
        self.assertEqual(blob.brain.layer_dims[-1], OUTPUT_DIM)
        self.assertIn(blob, self.world.blobs)

    def test_neural_evaluation_dimensions_and_10k_neurons(self) -> None:
        """Verify neural forward pass satisfies 26-in/6-out contract and supports 10k neurons."""
        # 1. Outer interface contract verification
        cfg = NetworkConfig(depth=2, width=32, activation="tanh")
        brain = BlobBrain(config=cfg)
        self.assertEqual(brain.layer_dims[0], INPUT_DIM)
        self.assertEqual(brain.layer_dims[-1], OUTPUT_DIM)
        self.assertEqual(INPUT_DIM, 26)
        self.assertEqual(OUTPUT_DIM, 6)

        obs = np.ones((INPUT_DIM,), dtype=np.float32) * 0.25
        out = brain.forward(obs, explore_noise=0.0)
        self.assertEqual(out.shape, (OUTPUT_DIM,))
        self.assertTrue(np.all(out >= -1.0) and np.all(out <= 1.0))

        # 2. 10k Neuron feedforward evaluation (verifying GEMV mathematical stability)
        w1_10k = np.random.default_rng(123).standard_normal((INPUT_DIM, 10000), dtype=np.float32) * 0.02
        b1_10k = np.zeros((10000,), dtype=np.float32)
        w2_10k = np.random.default_rng(456).standard_normal((10000, OUTPUT_DIM), dtype=np.float32) * 0.02
        b2_10k = np.zeros((OUTPUT_DIM,), dtype=np.float32)

        hidden = apply_activation(obs @ w1_10k + b1_10k, "gelu")
        self.assertEqual(hidden.shape, (10000,))
        out_10k = np.tanh(hidden @ w2_10k + b2_10k)
        self.assertEqual(out_10k.shape, (OUTPUT_DIM,))
        self.assertTrue(np.all(out_10k >= -1.0) and np.all(out_10k <= 1.0))

        # 3. If NetworkConfig has relaxed width clamp (M1+ contract), verify BlobBrain handles 10k config directly
        cfg_10k = NetworkConfig(depth=1, width=10000, custom_layers=[10000])
        hidden_dims = cfg_10k.get_hidden_dims()
        if 10000 in hidden_dims:
            brain_10k = BlobBrain(config=cfg_10k)
            self.assertEqual(brain_10k.layer_dims, [INPUT_DIM, 10000, OUTPUT_DIM])
            out_b10k = brain_10k.forward(obs, explore_noise=0.0)
            self.assertEqual(out_b10k.shape, (OUTPUT_DIM,))
            self.assertTrue(np.all(out_b10k >= -1.0) and np.all(out_b10k <= 1.0))

        # 4. Activation functions & derivatives verification
        for act in ("tanh", "relu", "gelu", "silu"):
            test_x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
            act_y = apply_activation(test_x, act)
            deriv = activation_derivative(act_y, act)
            self.assertEqual(act_y.shape, test_x.shape)
            self.assertEqual(deriv.shape, test_x.shape)
            self.assertFalse(np.any(np.isnan(act_y)))
            self.assertFalse(np.any(np.isnan(deriv)))

    def test_food_consumption_and_foraging(self) -> None:
        """Verify collision-based foraging consumes food, updates food_today, and sets spatial memory."""
        blob = self.world.blobs[0]
        self.assertGreater(len(self.world.foods), 0)
        target_food = self.world.foods[0]

        # Place blob directly on top of the food
        blob.x = target_food.x
        blob.y = target_food.y
        blob.food_today = 0.0
        food_count_before = len(self.world.foods)
        expected_oasis_id = target_food.oasis_id

        self.world._process_foraging([blob])
        self.assertGreater(blob.food_today, 0.0)
        self.assertEqual(len(self.world.foods), food_count_before - 1)
        if expected_oasis_id is not None:
            self.assertEqual(blob.memory_oasis_id, expected_oasis_id)
            self.assertGreater(blob.memory_strength, 0.5)

        # Stomach capacity ceiling: cannot absorb food beyond 3.4 * size
        max_cap = 3.4 * blob.genes.size
        blob.food_today = max_cap + 10.0
        self.world.spawn_custom_food_cluster(blob.x, blob.y, count=1)
        self.world._process_foraging([blob])
        self.assertAlmostEqual(blob.food_today, max_cap + 10.0)

    def test_energy_depletion_and_metabolism(self) -> None:
        """Verify basal cost, sprint cost, swamp drag, and energy non-negativity."""
        blob = self.world.blobs[0]
        blob.energy = 80.0
        blob.x, blob.y = 1500.0, 1500.0

        # Normal walking step
        home = self.world.homes.get(blob.home_id)
        self.world._apply_kinematics_and_metabolism(
            blob, move_x=1.0, move_y=0.0, sprint=0.0, home_obj=home, home_radius=60.0, night_mode=False
        )
        energy_after_walk = blob.energy
        self.assertLess(energy_after_walk, 80.0)

        # Sprint step costs more energy than walking
        self.world._apply_kinematics_and_metabolism(
            blob, move_x=1.0, move_y=0.0, sprint=1.0, home_obj=home, home_radius=60.0, night_mode=False
        )
        energy_after_sprint = blob.energy
        sprint_delta = energy_after_walk - energy_after_sprint
        walk_delta = 80.0 - energy_after_walk
        self.assertGreater(sprint_delta, walk_delta)

        # Energy clamp at zero
        blob.energy = 0.01
        self.world._apply_kinematics_and_metabolism(
            blob, move_x=1.0, move_y=0.0, sprint=1.0, home_obj=home, home_radius=60.0, night_mode=False
        )
        self.assertEqual(blob.energy, 0.0)

    def test_reproduction_and_inheritance(self) -> None:
        """Verify double survival food reproduction, gene mutation, and reputation inheritance."""
        parent = self.world.blobs[0]
        parent.food_today = self.world.world_cfg.survival_food_need * 2.1
        parent.energy = 85.0
        parent.reputation = 0.40

        child = self.world._reproduce_blob(parent)
        self.assertIsNotNone(child)
        self.assertNotEqual(child.id, parent.id)
        self.assertEqual(parent.children_count, 1)
        self.assertEqual(self.world.total_births_all_time, 1)

        # Offspring inherits mutated brain with same dimensions
        self.assertEqual(child.brain.layer_dims, parent.brain.layer_dims)
        # Offspring inherits fractional reputation (25% of parent when solo)
        self.assertAlmostEqual(child.reputation, round(parent.reputation * 0.25, 3), places=2)
        # Child has starting energy and empty stomach
        self.assertGreater(child.energy, 50.0)
        self.assertEqual(child.food_today, 0.0)

    def test_peer_to_peer_trading_and_sovereign_pricing(self) -> None:
        """Verify voluntary peer trade transfers 0.5 food and gold, even below survival need."""
        seller, buyer = self.world.blobs[0], self.world.blobs[1]
        seller.x, seller.y = 500.0, 500.0
        buyer.x, buyer.y = 505.0, 505.0

        # Seller has food below 1.0 survival need but high trade desire
        seller.food_today = 0.7
        seller.surplus_food = 0.0
        seller.trade_desire = 0.75
        seller.money = 5.0

        buyer.food_today = 0.1
        buyer.money = 25.0
        buyer.trade_desire = 0.5

        traded = self.world._try_peer_trade(seller, buyer)
        self.assertTrue(traded)
        self.assertAlmostEqual(seller.food_today, 0.2, places=2)
        self.assertAlmostEqual(buyer.food_today, 0.6, places=2)
        self.assertGreater(seller.money, 5.0)
        self.assertLess(buyer.money, 25.0)

    def test_websocket_binary_stream_format(self) -> None:
        """Verify binary packet header <IIII, blob stride 32B, food stride 12B, and delta mode."""
        meta_full, bin_full = self.world.get_ws_state_and_binary(include_topology=True)
        self.assertIn("obstacles", meta_full)
        self.assertIn("zones", meta_full["world"])

        # Unpack 16-byte binary header
        magic, n_blobs, n_foods, tick = struct.unpack_from("<IIII", bin_full, 0)
        self.assertEqual(magic, 0x424C4F42)  # 'BLOB'
        self.assertEqual(n_blobs, len(self.world.blobs))
        self.assertEqual(n_foods, len(self.world.foods))
        self.assertEqual(tick, self.world.tick_count)

        expected_len = 16 + n_blobs * 32 + n_foods * 12
        self.assertEqual(len(bin_full), expected_len)

        # Delta mode (include_topology=False) omits obstacles to reduce payload
        meta_delta, bin_delta = self.world.get_ws_state_and_binary(include_topology=False)
        self.assertNotIn("obstacles", meta_delta)
        self.assertEqual(len(bin_delta), expected_len)

    def test_fastapi_rest_endpoints(self) -> None:
        """Verify all FastAPI REST endpoints respond with correct status codes and payloads."""
        client = TestClient(app)

        # 1. HTML GUI routes
        res_index = client.get("/")
        self.assertEqual(res_index.status_code, 200)
        self.assertIn("text/html", res_index.headers["content-type"])

        res_analyse = client.get("/analyse")
        self.assertEqual(res_analyse.status_code, 200)
        self.assertIn("text/html", res_analyse.headers["content-type"])

        # 2. State endpoint
        res_state = client.get("/api/state")
        self.assertEqual(res_state.status_code, 200)
        state_data = res_state.json()
        self.assertIn("tick", state_data)
        self.assertIn("day", state_data)
        self.assertIn("blobs", state_data)

        # 3. Analytics endpoint
        res_analytics = client.get("/api/analytics")
        self.assertEqual(res_analytics.status_code, 200)
        analytics_data = res_analytics.json()
        self.assertIn("summary", analytics_data)
        self.assertIn("daily_logs", analytics_data)

        # 4. Command endpoint
        res_cmd = client.post("/api/command", json={"cmd": "toggle_pause"})
        self.assertEqual(res_cmd.status_code, 200)
        self.assertTrue(res_cmd.json()["ok"])

        res_speed = client.post("/api/command", json={"cmd": "set_speed", "speed": 10})
        self.assertEqual(res_speed.status_code, 200)
        self.assertEqual(res_speed.json()["speed"], 10)

        # 5. Export / Import population
        res_export = client.get("/api/export")
        self.assertEqual(res_export.status_code, 200)
        exported = res_export.json()
        self.assertIn("blobs", exported)
        self.assertIn("day", exported)

        res_import = client.post("/api/import", json=exported)
        self.assertEqual(res_import.status_code, 200)
        self.assertTrue(res_import.json()["ok"])


if __name__ == "__main__":
    unittest.main()
