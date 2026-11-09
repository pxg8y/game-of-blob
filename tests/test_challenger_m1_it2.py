"""Adversarial Verification Suite for Milestone 1 Iteration 2: Slot Invalidation Remediation Stress Test.

Focus areas:
1. Multi-death stress tests (5, 15, 30, 50 deaths) in SimulationWorld.
2. Exact boundary behavior at 31, 32, 33 invalid slots.
3. Verification of union evaluation when num_invalid <= 32 vs fallback when num_invalid > 32.
4. Non-persistence of invalid slots across multi-tick simulation waves.
5. Direct BatchedBrainPool adversarial harness testing inspect_idx, cohort residue classes, and trace updates.
"""

import unittest
import numpy as np

from blob_sim.config import WorldConfig, NetworkConfig
from blob_sim.world import SimulationWorld
from blob_sim.neural import BlobBrain, BatchedBrainPool, INPUT_DIM, OUTPUT_DIM


class TestChallengerM1It2MultiDeathStress(unittest.TestCase):
    """Adversarial stress testing of slot invalidation and multi-death compaction in SimulationWorld."""

    def tearDown(self) -> None:
        if hasattr(self, "world") and self.world is not None:
            if hasattr(self.world, "close"):
                self.world.close()
            self.world = None
        import gc
        gc.collect()

    def _setup_world(self, initial_blobs: int = 360, seed: int = 42) -> SimulationWorld:
        cfg = WorldConfig.from_preset("regional")
        cfg.initial_blobs = initial_blobs
        world = SimulationWorld(world_cfg=cfg, seed=seed)
        self.world = world
        # Advance 4 ticks so all 3 cohorts (0, 1, 2) run and initialize valid slots
        for _ in range(4):
            world.step(1)
        self.assertEqual(len(world.blobs), initial_blobs)
        self.assertTrue(
            np.all(world._brain_pool._out_valid_cap[:initial_blobs]),
            "All slots must be valid after warmup",
        )
        return world

    def test_multi_death_cohort_union_and_fallback(self) -> None:
        """Adversarially test killing 5, 15, 30, and 50 blobs in a single tick.

        Verifies:
        - When <= 32 invalid slots: cohort evaluation unions invalid slots.
          Relocated blobs recompute, active cohort recomputes, off-cohort intact blobs keep old outputs.
        - When > 32 invalid slots: full-population fallback occurs.
          ALL blobs recompute their outputs.
        - In all cases: all living slots are restored to valid, and no invalid slots persist into subsequent ticks.
        """
        death_counts = [5, 15, 30, 50]
        seeds = [101, 202, 303]

        for k in death_counts:
            for s in seeds:
                with self.subTest(death_count=k, seed=s):
                    world = self._setup_world(initial_blobs=360, seed=s)
                    try:
                        rng = np.random.default_rng(s + k)
                        n_pre = len(world.blobs)

                        # Snapshot state before deaths
                        pre_outs = world._brain_pool._last_out_cap[:n_pre].copy()

                        # Pick k distinct blobs to kill, ensuring some are head, middle, tail
                        kill_indices = sorted(rng.choice(n_pre, size=k, replace=False))
                        dead_set = set(kill_indices)

                        # Kill selected blobs
                        for idx in kill_indices:
                            world._kill_blob(world.blobs[idx], reason="adversarial_test")

                        # Run compaction manually to inspect intermediate invalidation state
                        living_blobs = world._compact_living_blobs()
                        n_post = len(living_blobs)
                        self.assertEqual(n_post, n_pre - k)

                        # Determine which slots were relocated by swap-remove:
                        # Slots write_idx < n_post that were killed received a survivor from the tail.
                        relocated_slots = [i for i in kill_indices if i < n_post]
                        num_relocated = len(relocated_slots)
                        self.assertGreater(num_relocated, 0, "At least some slots should be relocated")

                        # Verify that relocated slots are marked invalid prior to forward pass
                        for slot in relocated_slots:
                            self.assertFalse(
                                bool(world._brain_pool._out_valid_cap[slot]),
                                f"Relocated slot {slot} must be invalid before forward pass",
                            )

                        # Number of invalid slots
                        num_invalid = int(np.sum(~world._brain_pool._out_valid_cap[:n_post]))
                        self.assertEqual(num_invalid, num_relocated)

                        # Next tick cohort
                        next_cohort_idx = (world._substep_count + 1) % 3

                        # Advance 1 step
                        world.step(1)

                        # 1. Verify all living slots are valid
                        self.assertTrue(
                            np.all(world._brain_pool._out_valid_cap[:n_post]),
                            f"All {n_post} slots must be valid after forward pass for k={k}",
                        )

                        post_outs = world._brain_pool._last_out_cap[:n_post].copy()

                        # 2. Verify relocated blobs recomputed their outputs
                        for slot in relocated_slots:
                            # Relocated blob must not have identical output to pre-swap output at that slot
                            # (pre_outs[slot] belonged to the dead blob)
                            self.assertFalse(
                                np.array_equal(post_outs[slot], pre_outs[slot]),
                                f"Slot {slot} must have newly computed output, not dead blob output",
                            )

                        # 3. Check union vs fallback behavior
                        if num_invalid <= 32:
                            # <= 32 invalid slots -> 3-cohort scheduling was used with union
                            # Blobs that are NOT in next_cohort_idx AND NOT in relocated_slots MUST retain their output!
                            unaffected_slots = [
                                i for i in range(n_post)
                                if (i % 3 != next_cohort_idx) and (i not in relocated_slots)
                            ]
                            self.assertGreater(len(unaffected_slots), 0)
                            for slot in unaffected_slots:
                                self.assertTrue(
                                    np.array_equal(post_outs[slot], pre_outs[slot]),
                                    f"Slot {slot} (off-cohort, non-relocated) should have retained output under union eval",
                                )
                        else:
                            # > 32 invalid slots (e.g. k=50 where relocated > 32) -> fallback to full evaluation!
                            # Under full evaluation, ALL slots recompute!
                            # Blobs off-cohort also recompute their outputs with new observations/actions.
                            # Check that at least the vast majority of off-cohort intact blobs recomputed
                            off_cohort_intact = [
                                i for i in range(n_post)
                                if (i % 3 != next_cohort_idx) and (i not in relocated_slots)
                            ]
                            recomputed_count = sum(
                                not np.array_equal(post_outs[s], pre_outs[s])
                                for s in off_cohort_intact
                            )
                            self.assertEqual(
                                recomputed_count,
                                len(off_cohort_intact),
                                "In full-population fallback (>32 invalid), all slots must be recomputed",
                            )

                        # 4. Multi-tick persistence check: step 3 more ticks
                        for step_i in range(3):
                            world.step(1)
                            cur_n = len(world.blobs)
                            self.assertTrue(
                                np.all(world._brain_pool._out_valid_cap[:cur_n]),
                                f"Tick +{step_i + 1}: No invalid slots must persist into subsequent ticks",
                            )
                    finally:
                        if hasattr(world, "close"):
                            world.close()
                        self.world = None

    def test_exact_boundary_31_32_33_invalid_slots(self) -> None:
        """Adversarially challenge the exact threshold at 32 invalid slots:
        - Exactly 31 invalid slots -> union evaluation (use_cohort=True)
        - Exactly 32 invalid slots -> union evaluation (use_cohort=True)
        - Exactly 33 invalid slots -> fallback evaluation (use_cohort=False)
        """
        for target_invalid in [31, 32, 33]:
            with self.subTest(target_invalid=target_invalid):
                world = self._setup_world(initial_blobs=360, seed=777)
                try:
                    n_pre = len(world.blobs)

                    # Kill exactly target_invalid blobs at indices 0 .. target_invalid-1
                    # When compacted, tail blobs 360-target_invalid .. 359 will swap into 0 .. target_invalid-1
                    # giving exactly target_invalid relocated slots!
                    for idx in range(target_invalid):
                        world._kill_blob(world.blobs[idx], reason="boundary_test")

                    world._compact_living_blobs()
                    n_post = len(world.blobs)
                    self.assertEqual(n_post, n_pre - target_invalid)

                    num_invalid = int(np.sum(~world._brain_pool._out_valid_cap[:n_post]))
                    self.assertEqual(num_invalid, target_invalid)

                    pre_outs = world._brain_pool._last_out_cap[:n_post].copy()
                    next_cohort = (world._substep_count + 1) % 3

                    # Advance 1 step
                    world.step(1)

                    # All living slots must be valid
                    self.assertTrue(
                        np.all(world._brain_pool._out_valid_cap[:n_post]),
                        f"All slots must be valid after boundary step for invalid={target_invalid}",
                    )

                    post_outs = world._brain_pool._last_out_cap[:n_post].copy()

                    # Off-cohort non-relocated slots
                    off_cohort_intact = [
                        i for i in range(target_invalid, n_post)
                        if i % 3 != next_cohort
                    ]
                    self.assertGreater(len(off_cohort_intact), 0)

                    recomputed = [
                        not np.array_equal(post_outs[s], pre_outs[s])
                        for s in off_cohort_intact
                    ]

                    if target_invalid <= 32:
                        # Union evaluation: off-cohort intact blobs must NOT be recomputed
                        self.assertFalse(
                            any(recomputed),
                            f"target_invalid={target_invalid} <= 32 should use cohort union, but some off-cohort slots recomputed",
                        )
                    else:
                        # Fallback evaluation: all off-cohort intact blobs MUST be recomputed
                        self.assertTrue(
                            all(recomputed),
                            f"target_invalid={target_invalid} > 32 should fallback to full eval, but some slots did not recompute",
                        )
                finally:
                    if hasattr(world, "close"):
                        world.close()
                    self.world = None

    def test_consecutive_multi_wave_mortality_cascade(self) -> None:
        """Adversarially simulate rapid, successive multi-death ticks:
        Tick 1: kill 5
        Tick 2: kill 15
        Tick 3: kill 30
        Tick 4: kill 50 (fallback)
        Tick 5: kill 0
        Tick 6: kill 32
        Tick 7: kill 33
        Verify no crashes, no invalid slot retention, and full validity across the wave.
        """
        world = self._setup_world(initial_blobs=400, seed=888)
        waves = [5, 15, 30, 50, 0, 32, 33]

        for wave_idx, count in enumerate(waves):
            n_cur = len(world.blobs)
            if count > 0:
                rng = np.random.default_rng(wave_idx * 100)
                kill_indices = sorted(rng.choice(n_cur, size=count, replace=False))
                for idx in kill_indices:
                    world._kill_blob(world.blobs[idx], reason=f"wave_{wave_idx}")

            world.step(1)
            n_after = len(world.blobs)
            self.assertTrue(
                np.all(world._brain_pool._out_valid_cap[:n_after]),
                f"Wave {wave_idx} (kill {count}): All slots must be valid after step",
            )


class TestChallengerM1It2BatchedPoolAdversarialHarness(unittest.TestCase):
    """Direct adversarial unit testing on BatchedBrainPool slot invalidation and cohort scheduling."""

    def tearDown(self) -> None:
        if hasattr(self, "world") and self.world is not None:
            if hasattr(self.world, "close"):
                self.world.close()
            self.world = None
        if hasattr(self, "pool") and self.pool is not None:
            if hasattr(self.pool, "close"):
                self.pool.close()
            self.pool = None
        import gc
        gc.collect()

    def test_pool_union_and_fallback_direct_matrix(self) -> None:
        """Directly verify BatchedBrainPool.forward_and_trace behavior for invalid slot counts 0..40."""
        cfg = NetworkConfig(depth=1, width=32)
        n = 240
        brains = [BlobBrain(config=cfg, rng=np.random.default_rng(i)) for i in range(n)]
        pool = BatchedBrainPool()
        self.pool = pool
        pool.bind(brains)
        obs = np.ones((n, INPUT_DIM), dtype=np.float32)

        # Warm up pool
        pool.forward_and_trace(brains, obs, explore_noise=0.0, rng=np.random.default_rng(0), cohort_idx=-1)
        self.assertTrue(np.all(pool._out_valid_cap[:n]))

        iter_count = 0
        for num_invalid in [0, 1, 5, 16, 31, 32, 33, 40]:
            for cohort_idx in [0, 1, 2]:
                iter_count += 1
                with self.subTest(num_invalid=num_invalid, cohort_idx=cohort_idx):
                    # Mark all valid
                    pool._out_valid_cap[:n] = True
                    # Invalidate specific slots
                    invalid_slots = list(range(num_invalid))
                    for s in invalid_slots:
                        pool._out_valid_cap[s] = False

                    # Snapshot old outputs
                    old_outs = pool._last_out_cap[:n].copy()

                    # Provide dynamically fresh observations so newly evaluated slots have distinctly different outputs
                    obs_varied = np.sin(obs * (iter_count + 1.0) + 0.1 * np.arange(1, n + 1)[:, None]).astype(np.float32)

                    # Forward pass
                    pool.forward_and_trace(
                        brains,
                        obs_varied,
                        explore_noise=0.0,
                        rng=np.random.default_rng(1),
                        cohort_idx=cohort_idx,
                    )

                    # 1. All slots must be valid
                    self.assertTrue(
                        np.all(pool._out_valid_cap[:n]),
                        f"Failed validity restoration for num_invalid={num_invalid}, cohort_idx={cohort_idx}",
                    )

                    # 2. Check which slots changed
                    changed = [
                        not np.array_equal(pool._last_out_cap[i], old_outs[i])
                        for i in range(n)
                    ]

                    cohort_slots = set(range(cohort_idx, n, 3))
                    expected_union = cohort_slots.union(invalid_slots)

                    if num_invalid <= 32:
                        # Should evaluate exactly the union
                        for i in range(n):
                            if i in expected_union:
                                self.assertTrue(
                                    changed[i],
                                    f"Slot {i} was in union but not evaluated (num_invalid={num_invalid}, cohort={cohort_idx})",
                                )
                            else:
                                self.assertFalse(
                                    changed[i],
                                    f"Slot {i} was NOT in union but evaluated (num_invalid={num_invalid}, cohort={cohort_idx})",
                                )
                    else:
                        # Fallback: all slots must be evaluated
                        self.assertTrue(
                            all(changed),
                            f"Fallback failed: some slots were not evaluated for num_invalid={num_invalid}",
                        )

    def test_10k_neurons_multi_death_stress(self) -> None:
        """Adversarially stress test 200 blobs with 10,000 neurons under 5, 15, 30, and 50 deaths."""
        cfg = NetworkConfig(depth=1, width=10000)
        n = 200
        brains = [BlobBrain(config=cfg, rng=np.random.default_rng(i)) for i in range(n)]
        pool = BatchedBrainPool()
        pool.bind(brains)
        obs = np.ones((n, INPUT_DIM), dtype=np.float32)

        # Warmup
        pool.forward_and_trace(brains, obs, explore_noise=0.0, rng=np.random.default_rng(0), cohort_idx=-1)
        self.assertTrue(np.all(pool._out_valid_cap[:n]))

        for k in [5, 15, 30, 50]:
            with self.subTest(death_count=k):
                pool._out_valid_cap[:n] = True
                pool._out_valid_cap[:k] = False
                obs_k = obs * (1.1 + k * 0.01)
                pool.forward_and_trace(brains, obs_k, explore_noise=0.0, rng=np.random.default_rng(k), cohort_idx=1)
                self.assertTrue(
                    np.all(pool._out_valid_cap[:n]),
                    f"10k neuron validity restoration failed for k={k}",
                )

    def test_inspect_idx_parity_under_union_and_fallback(self) -> None:
        """Verify inspect_idx activation extraction parity when inspecting invalid vs off-cohort slots."""
        cfg = NetworkConfig(depth=2, width=24)
        n = 200
        brains = [BlobBrain(config=cfg, rng=np.random.default_rng(i)) for i in range(n)]
        pool = BatchedBrainPool()
        pool.bind(brains)
        obs = np.random.default_rng(42).standard_normal((n, INPUT_DIM)).astype(np.float32)

        # Case 1: inspect_idx is in invalid set
        pool._out_valid_cap[:n] = True
        pool._out_valid_cap[5] = False
        pool.forward_and_trace(brains, obs, explore_noise=0.0, rng=np.random.default_rng(0), inspect_idx=5, cohort_idx=0)
        a = obs[5]
        for idx in range(len(brains[5].weights)):
            iz = a @ brains[5].weights[idx] + brains[5].biases[idx]
            a = np.tanh(iz)
        self.assertTrue(np.allclose(brains[5].last_activations[-1], a, atol=1e-4))

        # Case 2: inspect_idx is off-cohort and not in invalid set
        pool._out_valid_cap[:n] = True
        pool._out_valid_cap[0] = False
        # cohort_idx = 0 -> slot 4 is in cohort 1 (off-cohort) and valid
        pool.forward_and_trace(brains, obs, explore_noise=0.0, rng=np.random.default_rng(0), inspect_idx=4, cohort_idx=0)
        a = obs[4]
        for idx in range(len(brains[4].weights)):
            iz = a @ brains[4].weights[idx] + brains[4].biases[idx]
            a = np.tanh(iz)
        self.assertTrue(np.allclose(brains[4].last_activations[-1], a, atol=1e-4))

    def test_extreme_population_catastrophe(self) -> None:
        """Verify mass extinction down to 10 blobs and 0 blobs transitions cleanly and restores valid slots."""
        cfg = WorldConfig.from_preset("regional")
        cfg.initial_blobs = 360
        world = SimulationWorld(world_cfg=cfg, seed=999)
        self.world = world
        for _ in range(4):
            world.step(1)

        # Kill 350 blobs (leaving 10 blobs, population drops below 180 cohort threshold)
        for idx in range(350):
            world._kill_blob(world.blobs[idx], reason="catastrophe")

        world.step(1)
        self.assertEqual(len(world.blobs), 10)
        self.assertTrue(np.all(world._brain_pool._out_valid_cap[:10]))

        # Kill remaining 10 blobs (total extinction -> triggers hall of fame respawn)
        for b in list(world.blobs):
            world._kill_blob(b, reason="extinction")

        world.step(1)
        self.assertGreaterEqual(len(world.blobs), 2)
        self.assertTrue(np.all(world._brain_pool._out_valid_cap[:len(world.blobs)]))
        world.close()
        self.world = None


if __name__ == "__main__":
    unittest.main()

