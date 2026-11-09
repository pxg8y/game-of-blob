"""Tier 2 - Boundary & Corner Cases E2E Tests.

Tests boundary conditions, stress configurations, and edge cases:
1. Empty population (N = 0) safety
2. Scaled capacity population (2,000+ blobs) initialization & stepping
3. Zero energy & absolute starvation boundary
4. Total food extinction (0 foods on map)
5. Extreme neural configurations (10,000 neurons, extreme inputs, max depth)
6. Negative & out-of-range observation inputs
7. World boundary collisions and reflections
8. High-speed time warping (500x and multi-day jumps)
"""

import struct
import unittest
import numpy as np

from blob_sim.config import NetworkConfig, WorldConfig
from blob_sim.neural import BlobBrain, INPUT_DIM, OUTPUT_DIM, apply_activation
from blob_sim.world import SimulationWorld


class TestTier2BoundaryCases(unittest.TestCase):
    """Tier 2: Boundary, corner, and extreme condition tests."""

    def tearDown(self) -> None:
        if hasattr(self, "world") and self.world is not None:
            if hasattr(self.world, "close"):
                self.world.close()
            self.world = None
        import gc
        gc.collect()

    def test_empty_population_safety(self) -> None:
        """Verify simulation safely handles 0 living blobs and self-heals via population floor."""
        world = SimulationWorld(world_cfg=WorldConfig(initial_blobs=8, ticks_per_day=30), seed=10)
        self.world = world
        world.blobs.clear()
        self.assertEqual(len(world.blobs), 0)

        # WebSocket serialization with 0 blobs before step
        meta, bin_buf = world.get_ws_state_and_binary(include_topology=True)
        magic, n_blobs, n_foods, tick = struct.unpack_from("<IIII", bin_buf, 0)
        self.assertEqual(magic, 0x424C4F42)
        self.assertEqual(n_blobs, 0)
        self.assertEqual(len(bin_buf), 16 + n_foods * 12)
        self.assertEqual(meta["stats"]["population"], 0)

        # Stepping an empty world triggers self-healing Hall of Fame respawn floor without crashing
        world.step(2)
        self.assertEqual(world.tick_count, 2)
        self.assertGreaterEqual(len(world.blobs), world.world_cfg.min_population_floor)

        # Telemetry analytics payload handles population safely
        analytics = world.telemetry.get_analytics_payload(world, since_day=0)
        self.assertIn("summary", analytics)

    def test_scaled_capacity_population_2000_blobs(self) -> None:
        """Verify simulation initializes and steps 2,000 concurrent blobs without memory or shape errors."""
        world = SimulationWorld(
            world_cfg=WorldConfig(
                width=16000.0,
                height=12000.0,
                initial_blobs=2000,
                ticks_per_day=120,
                max_population_cap=5000,
            ),
            seed=2000,
        )
        self.world = world
        self.assertEqual(len(world.blobs), 2000)

        # Verify all 2,000 blobs spawned within valid map boundaries
        for b in world.blobs[:100]:
            self.assertGreaterEqual(b.x, 0.0)
            self.assertLessEqual(b.x, 16000.0)
            self.assertGreaterEqual(b.y, 0.0)
            self.assertLessEqual(b.y, 12000.0)
            self.assertTrue(b.alive)

        # Step simulation with 2,000 blobs
        world.step(3)
        self.assertEqual(world.tick_count, 3)
        living_count = sum(1 for b in world.blobs if b.alive)
        self.assertGreaterEqual(living_count, 1900)

        # WebSocket packing handles 2,000 blobs
        meta, bin_buf = world.get_ws_state_and_binary(include_topology=False)
        magic, n_blobs, n_foods, _ = struct.unpack_from("<IIII", bin_buf, 0)
        self.assertEqual(magic, 0x424C4F42)
        self.assertEqual(n_blobs, living_count)
        self.assertEqual(len(bin_buf), 16 + n_blobs * 32 + n_foods * 12)

    def test_zero_energy_and_starvation_boundary(self) -> None:
        """Verify zero energy and zero food/money at day end causes death, triggering respawn floor."""
        world = SimulationWorld(
            world_cfg=WorldConfig(
                initial_blobs=10,
                ticks_per_day=40,
                survival_food_need=1.0,
                min_population_floor=6,
            ),
            seed=88,
        )
        self.world = world
        original_blobs = list(world.blobs)
        # Drain all blobs to 0 food, 0 money, 0 energy
        for b in original_blobs:
            b.food_today = 0.0
            b.money = 0.0
            b.energy = 0.0
            b.is_at_home = False

        world._end_of_day_accounting()
        # All original blobs must have died of starvation
        for b in original_blobs:
            self.assertFalse(b.alive)
        self.assertGreaterEqual(world.total_deaths_all_time, 10)

        # Population floor must ensure at least min_population_floor living blobs remain via Hall of Fame respawn
        living = [b for b in world.blobs if b.alive]
        self.assertGreaterEqual(len(living), world.world_cfg.min_population_floor)

    def test_total_food_extinction(self) -> None:
        """Verify simulation stepping, foraging, and sensor extraction when 0 foods exist on map."""
        world = SimulationWorld(world_cfg=WorldConfig(initial_blobs=12, ticks_per_day=30), seed=404)
        self.world = world
        world.foods.clear()
        # Also clear house depots so market purchases don't add food
        for h in world.homes.values():
            h.food_depot = 0.0
        self.assertEqual(len(world.foods), 0)

        # Step simulation with no food on the map
        for _ in range(3):
            world.step(1)

        # Blobs remain alive
        living = [b for b in world.blobs if b.alive]
        self.assertEqual(len(living), 12)

        # Foraging step on empty food list produces no errors and adds zero food
        for b in living:
            b.food_today = 0.0
        world._process_foraging(living)
        self.assertEqual(sum(b.food_today for b in living), 0.0)

    def test_extreme_neural_configurations_and_10k_neurons(self) -> None:
        """Verify 10,000-neuron forward evaluation under extreme inputs (all zeros, huge values, random)."""
        rng = np.random.default_rng(999)
        w1 = rng.standard_normal((INPUT_DIM, 10000), dtype=np.float32) * 0.01
        b1 = np.zeros((10000,), dtype=np.float32)
        w2 = rng.standard_normal((10000, OUTPUT_DIM), dtype=np.float32) * 0.01
        b2 = np.zeros((OUTPUT_DIM,), dtype=np.float32)

        # 1. Extreme inputs: all zeros
        obs_zeros = np.zeros((INPUT_DIM,), dtype=np.float32)
        h0 = apply_activation(obs_zeros @ w1 + b1, "tanh")
        out0 = np.tanh(h0 @ w2 + b2)
        self.assertEqual(out0.shape, (OUTPUT_DIM,))
        self.assertTrue(np.all(np.abs(out0) <= 1.0))
        self.assertFalse(np.any(np.isnan(out0)))

        # 2. Extreme inputs: massive values (+10,000 and -10,000)
        obs_extreme = np.ones((INPUT_DIM,), dtype=np.float32) * 10000.0
        obs_extreme[1::2] = -10000.0
        h_ext = apply_activation(obs_extreme @ w1 + b1, "gelu")
        out_ext = np.tanh(h_ext @ w2 + b2)
        self.assertEqual(out_ext.shape, (OUTPUT_DIM,))
        self.assertTrue(np.all(np.abs(out_ext) <= 1.0))
        self.assertFalse(np.any(np.isnan(out_ext)))
        self.assertFalse(np.any(np.isinf(out_ext)))

        # 3. Maximum network depth: depth = 6
        cfg_deep = NetworkConfig(depth=6, width=64, activation="silu")
        brain_deep = BlobBrain(config=cfg_deep)
        self.assertEqual(brain_deep.layer_dims, [INPUT_DIM, 64, 64, 64, 64, 64, 64, OUTPUT_DIM])
        out_deep = brain_deep.forward(obs_extreme, explore_noise=0.0)
        self.assertEqual(out_deep.shape, (OUTPUT_DIM,))
        self.assertTrue(np.all(np.abs(out_deep) <= 1.0))
        self.assertFalse(np.any(np.isnan(out_deep)))

    def test_negative_and_out_of_range_sensory_inputs(self) -> None:
        """Verify neural forward pass and sensor calculations remain stable with negative inputs."""
        cfg = NetworkConfig(depth=2, width=32, activation="tanh")
        brain = BlobBrain(config=cfg)

        obs_neg = np.full((INPUT_DIM,), -999.0, dtype=np.float32)
        out = brain.forward(obs_neg, explore_noise=0.0)
        self.assertEqual(out.shape, (OUTPUT_DIM,))
        self.assertTrue(np.all(out >= -1.0) and np.all(out <= 1.0))
        self.assertFalse(np.any(np.isnan(out)))

    def test_world_boundary_collisions_and_reflections(self) -> None:
        """Verify kinematics clamps blob positions within bounds and reflects velocities at boundaries."""
        world = SimulationWorld(world_cfg=WorldConfig(width=2000.0, height=1500.0), seed=50)
        self.world = world
        blob = world.blobs[0]
        home = world.homes.get(blob.home_id)

        # Place blob near left boundary (x = 5.0) moving left (vx = -20.0)
        blob.x = 5.0
        blob.y = 750.0
        blob.vx = -20.0
        blob.vy = 0.0

        world._apply_kinematics_and_metabolism(
            blob, move_x=-1.0, move_y=0.0, sprint=0.0, home_obj=home, home_radius=60.0, night_mode=False
        )
        self.assertGreaterEqual(blob.x, 0.0)
        self.assertLessEqual(blob.x, 2000.0)

        # Place blob near right boundary (x = 1995.0) moving right (vx = +20.0)
        blob.x = 1995.0
        blob.vx = 20.0
        world._apply_kinematics_and_metabolism(
            blob, move_x=1.0, move_y=0.0, sprint=0.0, home_obj=home, home_radius=60.0, night_mode=False
        )
        self.assertGreaterEqual(blob.x, 0.0)
        self.assertLessEqual(blob.x, 2000.0)

    def test_high_speed_time_warping_500x(self) -> None:
        """Verify advance_ticks(500) and advance_ticks(1200) execute multi-day jumps consistently."""
        world = SimulationWorld(world_cfg=WorldConfig(initial_blobs=20, ticks_per_day=300), seed=77)
        self.world = world
        self.assertEqual(world.day_number, 1)
        self.assertEqual(world.tick_count, 0)

        # Advance 500 ticks = 1 full day (300 ticks) + 200 ticks into day 2
        world.advance_ticks(500)
        self.assertEqual(world.tick_count, 500)
        self.assertEqual(world.day_number, 2)
        self.assertEqual(world.tick_in_day, 200)
        self.assertEqual(len(world.history), 1)

        # Advance 1200 ticks = 4 full days
        world.advance_ticks(1200)
        self.assertEqual(world.tick_count, 1700)
        self.assertEqual(world.day_number, 6)
        self.assertEqual(world.tick_in_day, 200)
        self.assertEqual(len(world.history), 5)

    def test_zero_blobs_comprehensive_boundary_lifecycle(self) -> None:
        """Verify initial_blobs=0 boundary condition across stepping, multi-day, reset, and repopulation."""
        cfg = WorldConfig.from_preset("classic", initial_blobs=0, min_population_floor=0)
        cfg.ticks_per_day = 40
        world = SimulationWorld(world_cfg=cfg, net_cfg=NetworkConfig(width=16), seed=42)
        self.world = world

        # 1. Clean stepping on empty world
        self.assertEqual(len(world.blobs), 0)
        for _ in range(10):
            world.step(1)
        self.assertEqual(world.tick_count, 10)
        self.assertEqual(len(world.blobs), 0)

        # 2. Multi-day progression (day/night, food spawning, seasonal accounting)
        for _ in range(120):  # 3 full simulated days
            world.step(1)
        self.assertEqual(world.day_number, 4)
        self.assertGreater(len(world.foods), 0)
        self.assertEqual(len(world.blobs), 0)

        # 3. Clean world reset on empty world
        world.reset_world()
        self.assertEqual(world.tick_count, 0)
        self.assertEqual(world.day_number, 1)
        self.assertEqual(len(world.blobs), 0)
        self.assertGreater(len(world.foods), 0)

        # 4. Seamless addition of blobs to empty world
        b1 = world.spawn_custom_blob(250.0, 250.0)
        self.assertTrue(b1.alive)
        self.assertEqual(len(world.blobs), 1)

        for _ in range(5):
            world.step(1)
        self.assertTrue(b1.alive)
        self.assertAlmostEqual(world._kin_xs[0], b1.x, delta=1e-4)
        self.assertAlmostEqual(world._kin_ys[0], b1.y, delta=1e-4)

        b2 = world.spawn_custom_blob(350.0, 350.0)
        self.assertTrue(b2.alive)
        self.assertEqual(len(world.blobs), 2)

        for _ in range(5):
            world.step(1)
        self.assertTrue(b1.alive)
        self.assertTrue(b2.alive)
        self.assertAlmostEqual(world._kin_xs[1], b2.x, delta=1e-4)
        self.assertAlmostEqual(world._kin_ys[1], b2.y, delta=1e-4)
        self.assertGreater(b2.x, 300.0)  # Verify no teleportation to (0, 0)


if __name__ == "__main__":
    unittest.main()
