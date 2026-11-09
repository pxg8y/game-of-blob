"""Adversarial Verification Suite for Milestone 1: Neural Architecture & Batched Execution Engine.

Tests:
1. Extreme inputs on 10k neurons: NaN, inf, 1e10, 1e38, zero vectors, alternating patterns across tanh, relu, gelu, silu.
2. Numerical equivalence between BlobBrain.forward and BatchedBrainPool.forward_and_trace across seeds, activations, and topologies.
3. Dynamic capacity resizing, odd cohort boundaries, and swap-remove slot invalidation semantics.
4. Memory budget verification and multi-tick RL stress on 2,000 blobs x 10,000 neurons.
"""

import gc
import unittest
import numpy as np

from blob_sim.config import NetworkConfig
from blob_sim.neural import BlobBrain, BatchedBrainPool, INPUT_DIM, OUTPUT_DIM


class TestChallengerM1ExtremeInputs(unittest.TestCase):
    """Adversarial stress testing of extreme, boundary, and non-finite sensory inputs."""

    def tearDown(self) -> None:
        if hasattr(self, "pool") and self.pool is not None:
            self.pool.close()
            self.pool = None
        gc.collect()

    def test_extreme_magnitude_floats_across_activations(self) -> None:
        """Verify 10k neuron topologies evaluate massive inputs without exploding out of [-1, 1]."""
        for act in ["tanh", "relu", "gelu", "silu"]:
            cfg = NetworkConfig(depth=1, width=10000, activation=act)
            brain = BlobBrain(config=cfg, rng=np.random.default_rng(42))
            pool = BatchedBrainPool()
            self.pool = pool
            pool.bind([brain])

            test_cases = [
                ("all_zeros", np.zeros(INPUT_DIM, dtype=np.float32)),
                ("pos_1e10", np.full(INPUT_DIM, 1e10, dtype=np.float32)),
                ("neg_1e10", np.full(INPUT_DIM, -1e10, dtype=np.float32)),
                ("pos_1e38", np.full(INPUT_DIM, 1e38, dtype=np.float32)),
                ("alternating", np.array([1e10 if i % 2 == 0 else -1e10 for i in range(INPUT_DIM)], dtype=np.float32)),
            ]

            for case_name, obs in test_cases:
                with self.subTest(activation=act, case=case_name):
                    out_single = brain.forward(obs, explore_noise=0.0)
                    out_pool = pool.forward_and_trace(
                        [brain], obs[None, :], explore_noise=0.0, rng=np.random.default_rng(0)
                    )[0]

                    self.assertEqual(out_single.shape, (OUTPUT_DIM,))
                    self.assertEqual(out_pool.shape, (OUTPUT_DIM,))
                    self.assertTrue(
                        np.all(out_single >= -1.0) and np.all(out_single <= 1.0),
                        f"Single brain output out of bounds for {act}/{case_name}: {out_single}",
                    )
                    self.assertTrue(
                        np.all(out_pool >= -1.0) and np.all(out_pool <= 1.0),
                        f"Pool output out of bounds for {act}/{case_name}: {out_pool}",
                    )
                    self.assertFalse(np.any(np.isnan(out_single)))
                    self.assertFalse(np.any(np.isnan(out_pool)))
                    self.assertTrue(
                        np.allclose(out_single, out_pool, atol=1e-4),
                        f"Equivalence failed for {act}/{case_name}",
                    )
            pool.close()
            self.pool = None

    def test_non_finite_inputs_propagation(self) -> None:
        """Verify handling of NaN and +/- Inf inputs does not crash the batched execution engine."""
        cfg = NetworkConfig(depth=1, width=1000, activation="tanh")
        brains = [BlobBrain(config=cfg) for _ in range(10)]
        pool = BatchedBrainPool()
        self.pool = pool
        pool.bind(brains)

        # 1. NaN inputs
        obs_nan = np.full((10, INPUT_DIM), np.nan, dtype=np.float32)
        out_nan = pool.forward_and_trace(brains, obs_nan, explore_noise=0.0, rng=np.random.default_rng(0))
        self.assertEqual(out_nan.shape, (10, OUTPUT_DIM))
        self.assertTrue(np.all(np.isnan(out_nan)))

        # 2. Inf inputs
        obs_inf = np.full((10, INPUT_DIM), np.inf, dtype=np.float32)
        out_inf = pool.forward_and_trace(brains, obs_inf, explore_noise=0.0, rng=np.random.default_rng(0))
        self.assertEqual(out_inf.shape, (10, OUTPUT_DIM))
        self.assertTrue(np.all(np.isnan(out_inf)))
        pool.close()
        self.pool = None


class TestChallengerM1NumericalEquivalence(unittest.TestCase):
    """Equivalence testing between single-brain forward and batched execution."""

    def tearDown(self) -> None:
        if hasattr(self, "pool") and self.pool is not None:
            self.pool.close()
            self.pool = None
        gc.collect()

    def test_single_vs_batched_across_seeds_and_architectures(self) -> None:
        """Verify strict numerical parity across seeds, activations, and layer topologies."""
        seeds = [101, 202, 303]
        activations = ["tanh", "relu", "gelu", "silu"]
        topologies = [
            ("default_24", {"depth": 2, "width": 24}),
            ("small_32", {"depth": 1, "width": 32}),
            ("numba_sub96", {"depth": 1, "width": 48}),  # N=40
            ("numba_super96", {"depth": 1, "width": 48}),  # N=100
            ("bottleneck_10k", {"custom_layers": [32, 10000, 32]}),
            ("wide_10k", {"depth": 1, "width": 10000}),
        ]

        for seed in seeds:
            for act in activations:
                for name, kwargs in topologies:
                    cfg = NetworkConfig(activation=act, **kwargs)
                    n_blobs = 100 if "super96" in name or "10k" in name else 40
                    rng = np.random.default_rng(seed)
                    brains = [BlobBrain(config=cfg, rng=np.random.default_rng(seed + i)) for i in range(n_blobs)]
                    pool = BatchedBrainPool()
                    try:
                        obs = rng.standard_normal((n_blobs, INPUT_DIM), dtype=np.float32)

                        single_outs = np.zeros((n_blobs, OUTPUT_DIM), dtype=np.float32)
                        for i in range(n_blobs):
                            single_outs[i] = brains[i].forward(obs[i], explore_noise=0.0)

                        pool_outs = pool.forward_and_trace(
                            brains, obs, explore_noise=0.0, rng=np.random.default_rng(seed), cohort_idx=-1
                        )

                        max_diff = float(np.max(np.abs(single_outs - pool_outs)))
                        self.assertLess(
                            max_diff,
                            1e-4,
                            f"Parity broken for {name} with {act} (seed={seed}, N={n_blobs}): max_diff={max_diff}",
                        )
                    finally:
                        for b in brains:
                            b.weights.clear()
                            b.biases.clear()
                        brains.clear()
                        pool.close()
                        del pool

    def test_deterministic_noise_injection_parity(self) -> None:
        """Verify exact parity between single brain and batched pool under identical noise arrays."""
        class MockRNG:
            def __init__(self, arr: np.ndarray) -> None:
                self.arr = arr

            def standard_normal(self, shape: tuple, dtype: np.dtype = np.float32) -> np.ndarray:
                return self.arr

        cfg = NetworkConfig(depth=1, width=64, activation="tanh")
        n_blobs = 100
        brains = [BlobBrain(config=cfg) for _ in range(n_blobs)]
        obs = np.ones((n_blobs, INPUT_DIM), dtype=np.float32)
        noise = np.random.default_rng(999).standard_normal((n_blobs, OUTPUT_DIM), dtype=np.float32)

        pool = BatchedBrainPool()
        self.pool = pool
        out_pool = pool.forward_and_trace(brains, obs, explore_noise=0.15, rng=MockRNG(noise)).copy()

        out_single = np.zeros((n_blobs, OUTPUT_DIM), dtype=np.float32)
        for i, br in enumerate(brains):
            br.rng = MockRNG(noise[i])
            out_single[i] = br.forward(obs[i], explore_noise=0.15)

        diff = float(np.max(np.abs(out_pool - out_single)))
        self.assertLess(diff, 1e-4, f"Parity broken with explore noise: diff={diff}")
        pool.close()
        self.pool = None


class TestChallengerM1ResizingAndSwapRemoves(unittest.TestCase):
    """Stress testing capacity resizing, cohort thresholds, and swap-remove invalidation."""

    def tearDown(self) -> None:
        if hasattr(self, "pool") and self.pool is not None:
            self.pool.close()
            self.pool = None
        gc.collect()

    def test_rapid_capacity_expansions_and_contractions(self) -> None:
        """Verify capacity growth and contraction across boundary thresholds preserves weight integrity."""
        cfg = NetworkConfig(depth=1, width=32)
        pool = BatchedBrainPool()
        self.pool = pool
        sizes = [1, 2, 10, 31, 32, 33, 63, 64, 65, 95, 96, 97, 179, 180, 181, 703, 704, 705, 1000, 500, 100, 1]
        all_brains = [BlobBrain(config=cfg, rng=np.random.default_rng(i)) for i in range(1200)]
        orig_w0 = {id(b): b.weights[0].copy() for b in all_brains}

        for sz in sizes:
            sub = all_brains[:sz]
            obs = np.ones((sz, INPUT_DIM), dtype=np.float32)
            out = pool.forward_and_trace(sub, obs, explore_noise=0.0, rng=np.random.default_rng(0))
            self.assertEqual(out.shape, (sz, OUTPUT_DIM))
            for b in sub:
                self.assertTrue(np.array_equal(b.weights[0], orig_w0[id(b)]))
        pool.close()
        self.pool = None

    def test_swap_remove_arbitrary_indices_and_monotonicity(self) -> None:
        """Verify swap-remove at index 0, middle, last, and over 300 random birth/kill cycles."""
        cfg = NetworkConfig(depth=1, width=32)
        pool = BatchedBrainPool()
        self.pool = pool

        def swap_remove(b_list: list, idx: int) -> None:
            last = len(b_list) - 1
            if idx < last:
                b_list[idx] = b_list[last]
            b_list.pop()

        living = [BlobBrain(config=cfg, rng=np.random.default_rng(i)) for i in range(150)]
        orig_w = {id(b): b.weights[0].copy() for b in living}
        pool.bind(living)

        # 1. Remove 0 (head)
        swap_remove(living, 0)
        pool.bind(living)
        for idx, b in enumerate(living):
            self.assertEqual(pool._slot_map[id(b)], idx)
            self.assertTrue(np.array_equal(b.weights[0], orig_w[id(b)]))

        # 2. Remove last
        swap_remove(living, len(living) - 1)
        pool.bind(living)
        for idx, b in enumerate(living):
            self.assertEqual(pool._slot_map[id(b)], idx)
            self.assertTrue(np.array_equal(b.weights[0], orig_w[id(b)]))

        # 3. Remove middle
        swap_remove(living, len(living) // 2)
        pool.bind(living)
        for idx, b in enumerate(living):
            self.assertEqual(pool._slot_map[id(b)], idx)
            self.assertTrue(np.array_equal(b.weights[0], orig_w[id(b)]))

        # 4. Stress cycles
        rng = np.random.default_rng(1234)
        for step in range(300):
            if len(living) > 10 and rng.random() < 0.6:
                swap_remove(living, int(rng.integers(0, len(living))))
            else:
                new_b = BlobBrain(config=cfg, rng=np.random.default_rng(10000 + step))
                orig_w[id(new_b)] = new_b.weights[0].copy()
                living.append(new_b)
            pool.bind(living)

        obs = np.ones((len(living), INPUT_DIM), dtype=np.float32)
        out = pool.forward_and_trace(living, obs, explore_noise=0.0, rng=rng)
        self.assertEqual(out.shape, (len(living), OUTPUT_DIM))
        for idx, b in enumerate(living):
            self.assertEqual(pool._slot_map[id(b)], idx)
            self.assertTrue(np.array_equal(b.weights[0], orig_w[id(b)]))
        pool.close()
        self.pool = None

    def test_cohort_slot_invalidation_and_recovery(self) -> None:
        """Verify invalid slots are prioritized in 3-cohort scheduling and recovered to True."""
        cfg = NetworkConfig(depth=1, width=32)
        n = 200
        brains = [BlobBrain(config=cfg, rng=np.random.default_rng(i)) for i in range(n)]
        pool = BatchedBrainPool()
        self.pool = pool
        pool.bind(brains)
        obs = np.ones((n, INPUT_DIM), dtype=np.float32)

        # Initialize slots to valid
        pool.forward_and_trace(brains, obs, explore_noise=0.0, rng=np.random.default_rng(0), cohort_idx=-1)
        self.assertTrue(np.all(pool._out_valid_cap[:n]))

        # Invalidate 4 slots
        targets = [0, 50, 100, 199]
        for t in targets:
            pool._out_valid_cap[t] = False

        # Run with cohort_idx = 1
        pool.forward_and_trace(brains, obs, explore_noise=0.0, rng=np.random.default_rng(1), cohort_idx=1)
        for t in targets:
            self.assertTrue(pool._out_valid_cap[t], f"Slot {t} was not recovered!")
            single_out = brains[t].forward(obs[t], explore_noise=0.0)
            self.assertTrue(np.allclose(single_out, pool._last_out_cap[t], atol=1e-4))
        pool.close()
        self.pool = None


class TestChallengerM1TenThousandNeuronsRLStress(unittest.TestCase):
    """Stress testing 2,000 blobs x 10,000 neurons memory footprint and Lamarckian online RL."""

    def setUp(self) -> None:
        gc.collect()

    def tearDown(self) -> None:
        if hasattr(self, "pool") and self.pool is not None:
            self.pool.close()
            self.pool = None
        if hasattr(self, "brains") and self.brains is not None:
            for b in self.brains:
                b._pool_ref = None
                b.weights.clear()
                b.biases.clear()
            self.brains.clear()
            self.brains = None
        gc.collect()

    def test_2000_blobs_10k_neurons_memory_budget_and_rl_stability(self) -> None:
        """Verify RAM usage is strictly <3.1 GB and multi-step RL keeps weights clipped."""
        cfg = NetworkConfig(depth=1, width=10000, activation="tanh")
        n_blobs = 2000
        rng = np.random.default_rng(42)

        brains = [BlobBrain(config=cfg, rng=np.random.default_rng(i)) for i in range(n_blobs)]
        pool = BatchedBrainPool()
        pool.bind(brains)
        self.brains = brains
        self.pool = pool

        mem_w = sum(w.nbytes for w in pool.W_batch)
        mem_b = sum(b.nbytes for b in pool.B_batch)
        mem_a = sum(a.nbytes for a in pool._A_hist_cap)
        mem_d = sum(d.nbytes for d in pool._D_hist_cap)
        total_mb = (mem_w + mem_b + mem_a + mem_d) / (1024 * 1024)

        self.assertLess(total_mb, 3100.0, f"Memory exceeded: {total_mb:.2f} MB")

        obs = rng.standard_normal((n_blobs, INPUT_DIM), dtype=np.float32)

        for tick in range(6):
            c_idx = tick % 3
            out = pool.forward_and_trace(
                brains, obs, explore_noise=0.04, rng=rng, cohort_idx=c_idx, update_traces=(c_idx == 0)
            )
            self.assertEqual(out.shape, (n_blobs, OUTPUT_DIM))
            rewards = rng.uniform(-1.0, 2.0, size=n_blobs).astype(np.float32)
            pool.apply_online_learning_batch(brains, rewards, learning_rate=0.015, flush=False)

        pool.flush_weights()
        for w in pool.W_batch:
            self.assertTrue(np.all(w >= -3.5) and np.all(w <= 3.5))
        for b in pool.B_batch:
            self.assertTrue(np.all(b >= -2.5) and np.all(b <= 2.5))

        pool.close()
        for b in brains:
            b._pool_ref = None
            b.weights.clear()
            b.biases.clear()
        brains.clear()
        self.brains = None
        self.pool = None
        gc.collect()


if __name__ == "__main__":
    unittest.main()
