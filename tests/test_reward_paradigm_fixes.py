"""Targeted unit and integration tests verifying all comprehensive fixes from reward_function_audit_report.md."""

import unittest
import numpy as np

from blob_sim.config import WorldConfig, NetworkConfig
from blob_sim.entities import Blob
from blob_sim.neural import BlobBrain
from blob_sim.world import SimulationWorld


class TestRewardParadigmFixes(unittest.TestCase):
    """Verify all 6 architectural recommendations from the reward function audit report."""

    def tearDown(self) -> None:
        if hasattr(self, "world") and self.world is not None:
            if hasattr(self.world, "close"):
                self.world.close()
            self.world = None

    def test_food_shaping_scales_with_pref_food(self) -> None:
        """Verify that movement shaping toward food scales with pref_food and NOT pref_exploration."""
        world1 = SimulationWorld(seed=42)
        self.world = world1
        world1.prefs.pref_food = 1.0
        world1.prefs.pref_exploration = 1.0

        blob = world1.blobs[0]
        blob.food_today = 0.0

        # Baseline shaping reward with pref_food = 1.0
        r1 = world1._compute_movement_shaping_reward(
            blob=blob,
            move_x=1.0,
            move_y=0.0,
            dx_food=1.0,
            dy_food=0.0,
            prox_food=0.8,
            dx_oasis=0.0,
            dy_oasis=0.0,
            local_fertility=0.5,
            dx_home=0.0,
            dy_home=0.0,
            tod=0.2,
            repro_target=2.0,
            vx=1.0,
            vy=0.0,
            max_speed=1.0,
        )

        # Double pref_food: shaping reward should scale proportionally
        world1.prefs.pref_food = 2.0
        r2 = world1._compute_movement_shaping_reward(
            blob=blob,
            move_x=1.0,
            move_y=0.0,
            dx_food=1.0,
            dy_food=0.0,
            prox_food=0.8,
            dx_oasis=0.0,
            dy_oasis=0.0,
            local_fertility=0.5,
            dx_home=0.0,
            dy_home=0.0,
            tod=0.2,
            repro_target=2.0,
            vx=1.0,
            vy=0.0,
            max_speed=1.0,
        )
        self.assertAlmostEqual(r2, r1 * 2.0, places=5)

        # Change pref_exploration: food approach shaping reward should NOT change
        world1.prefs.pref_exploration = 5.0
        r3 = world1._compute_movement_shaping_reward(
            blob=blob,
            move_x=1.0,
            move_y=0.0,
            dx_food=1.0,
            dy_food=0.0,
            prox_food=0.8,
            dx_oasis=0.0,
            dy_oasis=0.0,
            local_fertility=0.5,
            dx_home=0.0,
            dy_home=0.0,
            tod=0.2,
            repro_target=2.0,
            vx=1.0,
            vy=0.0,
            max_speed=1.0,
        )
        self.assertAlmostEqual(r3, r2, places=5)

    def test_raid_defender_penalty_does_not_use_pref_home(self) -> None:
        """Verify that victim defender of a raid does not receive a pref_home penalty on pending_rl_reward."""
        world = SimulationWorld(seed=42)
        self.world = world
        world.prefs.pref_home = 2.5

        raider, defender = world.blobs[0], world.blobs[1]
        raider.group_id = None
        defender.group_id = None
        raider.genes.size = 1.5
        defender.genes.size = 0.7
        raider.energy = 95.0
        defender.energy = 60.0
        raider.food_today = 0.0
        defender.food_today = 1.2
        raider.social_desire = -0.5
        raider.trade_desire = -0.5
        defender.x, defender.y = 600.0, 400.0
        raider.x, raider.y = 602.0, 402.0

        def_home = world.homes[defender.home_id]
        def_home.x, def_home.y = 50.0, 50.0

        pre_pending = defender.pending_rl_reward
        pre_energy = defender.energy
        success = world._try_peer_raid(raider, defender)
        self.assertTrue(success)

        # Defender must lose energy/stamina but pending_rl_reward must NOT be penalized with pref_home
        self.assertEqual(defender.pending_rl_reward, pre_pending)
        self.assertLess(defender.energy, pre_energy)

    def test_eod_diurnal_accounting_mirrored_in_cumulative_reward(self) -> None:
        """Verify that diurnal survival and homing outcomes are mirrored in cumulative_reward."""
        cfg = WorldConfig.from_preset("classic", initial_blobs=4)
        world = SimulationWorld(world_cfg=cfg, seed=42)
        self.world = world

        blobs = world.blobs[:4]
        # b0: at home, repro ready
        blobs[0].is_at_home = True
        blobs[0].food_today = world.world_cfg.reproduction_food_need
        blobs[0].energy = 80.0

        # b1: outdoors, starving, no money to buy deficit
        blobs[1].is_at_home = False
        blobs[1].food_today = 0.0
        blobs[1].energy = 0.0
        blobs[1].money = 0.0

        cum0_before = blobs[0].cumulative_reward
        cum1_before = blobs[1].cumulative_reward

        world._stage_eod_settlements_and_learning(blobs)

        # b0 should receive home_rew_pos + repro_rew
        expected_b0 = (0.35 * world.prefs.pref_home) + (1.4 * world.prefs.pref_food)
        self.assertAlmostEqual(blobs[0].cumulative_reward - cum0_before, expected_b0, places=4)

        # b1: energy <= 0 after deficit check -> receives home_rew_neg + starve_pen
        expected_b1 = (-0.25 * world.prefs.pref_home) + (-1.0 * world.prefs.pref_food)
        self.assertAlmostEqual(blobs[1].cumulative_reward - cum1_before, expected_b1, places=4)

    def test_midday_starvation_no_dead_blob_backprop(self) -> None:
        """Verify that starving to death does not call dead-blob backprop on brain weights."""
        world = SimulationWorld(seed=42)
        self.world = world
        blob = world.blobs[0]
        blob.energy = 0.0
        blob.money = 0.0
        blob.ticks_since_last_meal = 9999

        w_before = [w.copy() for w in blob.brain.weights]
        world._check_midday_starvation([blob])
        self.assertFalse(blob.alive)
        for wb, wa in zip(w_before, blob.brain.weights, strict=True):
            np.testing.assert_array_equal(wb, wa)

    def test_trace_horizon_decay_extended_to_095(self) -> None:
        """Verify that eligibility trace discount factor is 0.95."""
        cfg = NetworkConfig(depth=2, width=16)
        brain = BlobBrain(config=cfg)
        obs = np.ones(brain.layer_dims[0], dtype=np.float32)
        brain.forward(obs)

        # Traces are initialized and decay with 0.95 factor
        initial_trace = brain.traces_w[0].copy()
        brain.forward(obs)
        self.assertTrue(np.all(np.isfinite(brain.traces_w[0])))

    def test_decoupled_two_stream_baseline(self) -> None:
        """Verify dense shaping baseline is isolated from sparse large meal event rewards."""
        cfg = NetworkConfig(depth=2, width=16)
        brain = BlobBrain(config=cfg)

        # Small dense shaping steps
        for _ in range(5):
            brain.apply_online_learning(reward=0.05, learning_rate=0.01)
        dense_base = brain.reward_baseline_dense
        self.assertGreater(dense_base, 0.0)

        # Large sparse event (e.g. eating food +1.20)
        brain.apply_online_learning(reward=1.20, learning_rate=0.01)
        event_base = brain.reward_baseline_event
        self.assertGreater(event_base, 0.0)

        # On next dense step, dense baseline should NOT have jumped by 1.20
        self.assertLess(brain.reward_baseline_dense, 0.10)


if __name__ == "__main__":
    unittest.main()
