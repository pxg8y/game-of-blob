"""Unit and integration tests for Blob Life simulation mechanics."""

import unittest
import numpy as np
from blob_sim.config import NetworkConfig, TrainingPreferences, WorldConfig
from blob_sim.neural import BlobBrain, INPUT_DIM, OUTPUT_DIM
from blob_sim.world import SimulationWorld


class TestBlobSimulation(unittest.TestCase):
    def test_custom_neural_network_depth_width_and_activation(self) -> None:
        cfg = NetworkConfig(depth=4, width=32, activation="gelu")
        brain = BlobBrain(config=cfg)
        self.assertEqual(brain.layer_dims, [INPUT_DIM, 32, 32, 32, 32, OUTPUT_DIM])

        obs = np.zeros((INPUT_DIM,), dtype=np.float32)
        out = brain.forward(obs, explore_noise=0.0)
        self.assertEqual(out.shape, (OUTPUT_DIM,))
        self.assertTrue(np.all(out >= -1.0) and np.all(out <= 1.0))

        # Test live architecture adaptation to custom layer list
        new_cfg = NetworkConfig(depth=2, width=16, custom_layers=[20, 12], activation="silu")
        adapted = brain.adapt_to_config(new_cfg)
        self.assertEqual(adapted.layer_dims, [INPUT_DIM, 20, 12, OUTPUT_DIM])

    def test_online_learning_and_mutation(self) -> None:
        cfg = NetworkConfig(depth=2, width=16, activation="tanh")
        brain = BlobBrain(config=cfg)
        obs = np.ones((INPUT_DIM,), dtype=np.float32) * 0.5
        _ = brain.forward(obs)
        before_w0 = brain.weights[0].copy()
        brain.apply_online_learning(reward=2.0, learning_rate=0.05)
        self.assertFalse(np.allclose(before_w0, brain.weights[0]))

        child = brain.clone_and_mutate(mutation_rate=0.5, mutation_strength=0.2)
        self.assertEqual(child.layer_dims, brain.layer_dims)
        self.assertFalse(np.allclose(child.weights[0], brain.weights[0]))

    def test_reproduction_on_double_survival_food(self) -> None:
        world = SimulationWorld(
            world_cfg=WorldConfig(initial_blobs=10, ticks_per_day=50, survival_food_need=1.0),
            seed=123,
        )
        parent = world.blobs[0]
        # Give parent >= 2x survival food (2.0) and place at home
        parent.food_today = 2.2
        parent.x = parent.home_x
        parent.y = parent.home_y
        parent.is_at_home = True
        parent.energy = 90.0

        # Ensure other blobs also survive so we can count exact births
        for b in world.blobs[1:]:
            b.food_today = 1.0
            b.energy = 80.0
            b.is_at_home = True

        world._end_of_day_accounting()
        self.assertGreaterEqual(parent.children_count, 1)
        self.assertGreaterEqual(world.total_births_all_time, 1)

    def test_starvation_when_no_food_and_no_money(self) -> None:
        world = SimulationWorld(
            world_cfg=WorldConfig(initial_blobs=10, ticks_per_day=50, survival_food_need=1.0),
            seed=99,
        )
        victim = world.blobs[0]
        victim.food_today = 0.0
        victim.money = 0.0
        victim.energy = 20.0
        victim.is_at_home = False

        world._end_of_day_accounting()
        self.assertFalse(victim.alive)

    def test_group_formation_and_food_for_money_trade(self) -> None:
        world = SimulationWorld(seed=7)
        a, b = world.blobs[0], world.blobs[1]
        a.x, a.y = 300.0, 300.0
        b.x, b.y = 305.0, 305.0

        # Form group
        world._handle_social_encounter(a, b)
        self.assertIsNotNone(a.group_id)
        self.assertEqual(a.group_id, b.group_id)

        # Give A surplus food and B money + hunger
        a.surplus_food = 1.0
        a_money_before = a.money
        b.food_today = 0.5
        b.money = 20.0

        traded = world._try_peer_trade(a, b)
        self.assertTrue(traded)
        self.assertGreater(a.money, a_money_before)
        self.assertGreater(b.food_today, 0.5)


if __name__ == "__main__":
    unittest.main()
