"""Empirical Challenger Multi-Cull Survivor Compaction Stress Suite for M2 It7."""

import gc
import unittest
import numpy as np

from blob_sim.config import WorldConfig
from blob_sim.world import SimulationWorld
from blob_sim.entities import Blob


class TestChallengerM2It7CompactionStress(unittest.TestCase):
    """Stress-test survivor compaction across arbitrary cull sizes, multiple cycles,
    and verify exact is_at_home, neural sensory inputs, and scalar properties.
    """

    def tearDown(self) -> None:
        gc.collect()

    def _run_compaction_matrix(self, cull_pct: float, cycles: int = 5, seed: int = 42) -> None:
        world = SimulationWorld(WorldConfig(initial_blobs=400, ticks_per_day=50), seed=seed)
        try:
            # Step 15 ticks to disperse blobs and create rich internal state
            for _ in range(15):
                world.step(1)

            for cycle in range(cycles):
                survivors_before = [b for b in world.blobs if b.alive]
                n_surv = len(survivors_before)
                if n_surv <= 5:
                    break

                # Snapshot ground truth state for all survivors prior to culling
                gt_state = {}
                gt_int = {}
                for b in survivors_before:
                    gt_state[b.id] = {
                        "is_at_home": b.is_at_home,
                        "x": b.x,
                        "y": b.y,
                        "vx": b.vx,
                        "vy": b.vy,
                        "heading": b.heading,
                        "energy": b.energy,
                        "food_today": b.food_today,
                        "money": b.money,
                        "reputation": b.reputation,
                        "group_id": b.group_id,
                        "home_id": b.home_id,
                        "in_swamp": b.in_swamp,
                    }
                    gt_int[b.id] = int(world._kin_is_at_home_int[b._kin_idx])

                # Cull arbitrary fraction
                k_cull = max(1, int(n_surv * cull_pct))
                if k_cull >= n_surv:
                    k_cull = n_surv - 1
                to_kill = set(world.rng.choice(survivors_before, size=k_cull, replace=False))
                for b in to_kill:
                    b.alive = False

                expected_survivors = [b for b in survivors_before if b not in to_kill]
                expected_ids = {b.id for b in expected_survivors}

                # Execute survivor compaction
                compacted = world._compact_living_blobs()

                # 1. Structural Invariants
                actual_ids = {b.id for b in compacted}
                self.assertEqual(actual_ids, expected_ids, f"Cycle {cycle}: ID set mismatch")
                self.assertEqual(len(compacted), len(expected_survivors), f"Cycle {cycle}: length mismatch")

                # 2. Individual Blob State Invariants & Kin Buffer Alignment
                for idx, b in enumerate(compacted):
                    gt = gt_state[b.id]
                    self.assertEqual(
                        b.is_at_home,
                        gt["is_at_home"],
                        f"Cycle {cycle}, Blob {b.id} at index {idx}: is_at_home corrupted! got {b.is_at_home}, expected {gt['is_at_home']}",
                    )
                    self.assertEqual(
                        bool(world._kin_is_at_homes[idx]),
                        gt["is_at_home"],
                        f"Cycle {cycle}, Blob {b.id} at index {idx}: _kin_is_at_homes buffer out of sync!",
                    )
                    self.assertEqual(
                        int(world._kin_is_at_home_int[idx]),
                        gt_int[b.id],
                        f"Cycle {cycle}, Blob {b.id} at index {idx}: _kin_is_at_home_int buffer out of sync!",
                    )
                    self.assertAlmostEqual(b.x, gt["x"], places=5)
                    self.assertAlmostEqual(b.y, gt["y"], places=5)
                    self.assertAlmostEqual(world._kin_xs[idx], gt["x"], places=5)
                    self.assertAlmostEqual(world._kin_ys[idx], gt["y"], places=5)
                    self.assertAlmostEqual(b.vx, gt["vx"], places=5)
                    self.assertAlmostEqual(b.vy, gt["vy"], places=5)
                    self.assertAlmostEqual(b.heading, gt["heading"], places=5)
                    self.assertAlmostEqual(b.energy, gt["energy"], places=5)
                    self.assertAlmostEqual(b.food_today, gt["food_today"], places=5)
                    self.assertAlmostEqual(b.money, gt["money"], places=5)
                    self.assertAlmostEqual(b.reputation, gt["reputation"], places=5)
                    self.assertEqual(b.group_id, gt["group_id"])
                    self.assertEqual(b.home_id, gt["home_id"])
                    self.assertEqual(b.in_swamp, gt["in_swamp"])

                # 3. Post-compaction Simulation Step Invariant
                # Stepping after compaction must execute cleanly and update living kinematics without NaN/inf
                world.step(1)
                for b in world.blobs:
                    if b.alive:
                        self.assertTrue(np.isfinite(b.x) and np.isfinite(b.y))
                        self.assertTrue(np.isfinite(b.energy))
        finally:
            world.close()

    def test_cull_10_percent_light_attrition(self) -> None:
        """Light attrition: cull 10% across 5 sequential cycles."""
        self._run_compaction_matrix(0.10, cycles=5, seed=101)

    def test_cull_50_percent_heavy_cull(self) -> None:
        """Heavy attrition: cull 50% across 4 sequential cycles."""
        self._run_compaction_matrix(0.50, cycles=4, seed=202)

    def test_cull_90_percent_catastrophic_cull(self) -> None:
        """Catastrophic attrition: cull 90% across 2 sequential cycles."""
        self._run_compaction_matrix(0.90, cycles=2, seed=303)

    def test_arbitrary_variable_cull_sequence(self) -> None:
        """Variable cull sequence: 25%, 70%, 15%, 85% culling."""
        world = SimulationWorld(WorldConfig(initial_blobs=500, ticks_per_day=50), seed=404)
        try:
            for _ in range(12):
                world.step(1)

            cull_rates = [0.25, 0.70, 0.15, 0.85]
            for cycle, rate in enumerate(cull_rates):
                survivors = [b for b in world.blobs if b.alive]
                if len(survivors) <= 5:
                    break
                gt_iah = {b.id: b.is_at_home for b in survivors}
                gt_int = {b.id: int(world._kin_is_at_home_int[b._kin_idx]) for b in survivors}
                k_cull = max(1, int(len(survivors) * rate))
                to_kill = set(world.rng.choice(survivors, size=k_cull, replace=False))
                for b in to_kill:
                    b.alive = False

                compacted = world._compact_living_blobs()
                for idx, b in enumerate(compacted):
                    self.assertEqual(
                        b.is_at_home,
                        gt_iah[b.id],
                        f"Cycle {cycle} (rate {rate}): Blob {b.id} is_at_home mismatch!",
                    )
                    self.assertEqual(bool(world._kin_is_at_homes[idx]), gt_iah[b.id])
                    self.assertEqual(int(world._kin_is_at_home_int[idx]), gt_int[b.id])
        finally:
            world.close()


if __name__ == "__main__":
    unittest.main()
