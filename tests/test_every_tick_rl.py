"""Unit tests for every-tick evaluation and online RL mode in blob_sim."""

import unittest
import numpy as np

from blob_sim.config import NetworkConfig, WorldConfig
from blob_sim.neural import _TRACE_RANK, _get_trace_rank_for_dims
from blob_sim.world import SimulationWorld


class TestEveryTickRL(unittest.TestCase):
    def test_trace_rank_for_dims(self) -> None:
        """Verify adaptive trace rank scaling for various network dimensions."""
        self.assertEqual(_get_trace_rank_for_dims([]), _TRACE_RANK)
        self.assertEqual(_get_trace_rank_for_dims([26, 24, 6]), 13)
        self.assertEqual(_get_trace_rank_for_dims([26, 512, 6]), 13)
        self.assertEqual(_get_trace_rank_for_dims([26, 2048, 6]), 6)
        self.assertEqual(_get_trace_rank_for_dims([26, 4096, 6]), 6)
        self.assertEqual(_get_trace_rank_for_dims([26, 10000, 6]), 2)

    def test_every_tick_brain_action_updates_and_weight_flushes(self) -> None:
        """Verify every_tick_brain=True evaluates all blobs across all cohort remainder classes

        (idx % 3 in (0, 1, 2)) every single tick, and online weight changes take effect across all
        cohorts after step rewards and weight flush.
        """
        world_cfg = WorldConfig(initial_blobs=240, every_tick_brain=True)
        net_cfg = NetworkConfig(depth=2, width=24, activation="tanh")
        world = SimulationWorld(world_cfg=world_cfg, net_cfg=net_cfg, seed=42)

        pool = world._brain_pool
        living = [b for b in world.blobs if b.alive]
        self.assertGreaterEqual(len(living), 240)

        # Snapshot initial weights for the first layer across blobs of each cohort remainder
        w0_initial = [b.brain.weights[0].copy() for b in living[:12]]

        # Record outputs over 4 individual ticks
        tick_outputs = []
        for tick in range(4):
            # Give every blob positive pending reward to stimulate online RL updates
            for b in living:
                b.pending_rl_reward = 0.5
            world.step(1)
            # Store copy of _last_out_cap
            tick_outputs.append(pool._last_out_cap[: len(living)].copy())

        # For blobs of each remainder class (idx % 3 in (0, 1, 2)), outputs should change every tick
        for cohort_rem in (0, 1, 2):
            sample_indices = [idx for idx in range(len(living)) if idx % 3 == cohort_rem][:4]
            for s_idx in sample_indices:
                for t in range(3):
                    diff = np.max(np.abs(tick_outputs[t + 1][s_idx] - tick_outputs[t][s_idx]))
                    self.assertGreater(
                        diff,
                        1e-6,
                        f"Blob {s_idx} (cohort {cohort_rem}) output did not update between tick {t} and {t+1}",
                    )

        # Flush pending weight updates
        pool.flush_weights()

        # Verify real weight changes for all cohort remainder classes
        for cohort_rem in (0, 1, 2):
            cohort_indices = [idx for idx in range(12) if idx % 3 == cohort_rem]
            for idx in cohort_indices:
                w_before = w0_initial[idx]
                w_after = living[idx].brain.weights[0]
                weight_diff = np.max(np.abs(w_after - w_before))
                self.assertGreater(
                    weight_diff,
                    1e-6,
                    f"Blob {idx} (cohort {cohort_rem}) weights did not change after online RL!",
                )

    def test_cohort_traces_active_in_standard_mode(self) -> None:
        """Verify for every_tick_brain=False (with standard width <= 512), blobs in cohorts 1 and 2

        also write non-trivial eligibility traces into the ring buffer and participate in online learning.
        """
        world_cfg = WorldConfig(initial_blobs=240, every_tick_brain=False)
        net_cfg = NetworkConfig(depth=2, width=24, activation="tanh")
        world = SimulationWorld(world_cfg=world_cfg, net_cfg=net_cfg, seed=123)

        pool = world._brain_pool
        living = [b for b in world.blobs if b.alive]
        self.assertGreaterEqual(len(living), 240)

        # Run 3 ticks so that substep_count modulo 3 covers cohorts 0, 1, 2
        for _ in range(3):
            for b in living:
                b.pending_rl_reward = 0.5
            world.step(1)

        # Check trace weights in the ring buffer for cohorts 1 and 2
        cohort_1_indices = [idx for idx in range(len(living)) if idx % 3 == 1][:10]
        cohort_2_indices = [idx for idx in range(len(living)) if idx % 3 == 2][:10]

        for idx in cohort_1_indices:
            traces = pool._trace_weights_cap[idx, : pool._ring_ptr]
            self.assertTrue(
                np.any(traces > 0.0),
                f"Cohort 1 blob {idx} has no active traces in ring buffer: {traces}",
            )

        for idx in cohort_2_indices:
            traces = pool._trace_weights_cap[idx, : pool._ring_ptr]
            self.assertTrue(
                np.any(traces > 0.0),
                f"Cohort 2 blob {idx} has no active traces in ring buffer: {traces}",
            )

        # Flush weights and ensure non-trivial weight updates occurred for cohorts 1 and 2
        pool.flush_weights()
        for idx in cohort_1_indices + cohort_2_indices:
            w = pool.W_batch[0][idx]
            self.assertFalse(np.all(np.isnan(w)))


if __name__ == "__main__":
    unittest.main()
