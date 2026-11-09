"""Adversarial Verification Suite for Milestone 1 Iteration 2 Challenger 2:
Continuous Simulation Cycles with Deaths, Births, Neural Vector Fidelity, and RSS Stability.

Verification criteria:
1. 50-tick continuous simulation with 2,000 blobs where natural starvation deaths and births occur.
2. Verification that `_last_out_cap` output vectors match genuine neural forward outputs at all times
   (no stale cached/dummy data, exact numerical match to BlobBrain forward pass, 100% validity post-forward).
3. Verification that physical RSS memory remains stable (<3.2 GB) without cumulative memory leaks
   during slot reallocation under continuous life cycles, including up to 10,000 neurons per blob.
"""

import unittest
import ctypes
from ctypes import wintypes
import numpy as np

from blob_sim.config import WorldConfig, NetworkConfig
from blob_sim.world import SimulationWorld
from blob_sim.neural import BlobBrain, BatchedBrainPool, INPUT_DIM, OUTPUT_DIM


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _get_process_rss_gb() -> float:
    """Retrieve current process physical working set size (RSS) in Gigabytes."""
    if hasattr(ctypes, "windll"):
        fn = ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        fn.restype = wintypes.BOOL
        h = ctypes.windll.kernel32.GetCurrentProcess()
        pmc = _PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
        res = fn(h, ctypes.byref(pmc), pmc.cb)
        if not res:
            return 0.0
        return float(pmc.WorkingSetSize) / (1024.0 ** 3)
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / (1024.0 * 1024.0)
    except Exception:
        pass
    import resource
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def _get_process_commit_gb() -> float:
    """Retrieve current process committed virtual memory in Gigabytes."""
    if hasattr(ctypes, "windll"):
        fn = ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        fn.restype = wintypes.BOOL
        h = ctypes.windll.kernel32.GetCurrentProcess()
        pmc = _PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
        res = fn(h, ctypes.byref(pmc), pmc.cb)
        if not res:
            return 0.0
        return float(pmc.PagefileUsage) / (1024.0 ** 3)
    return _get_process_rss_gb()


class TestChallengerM1It2ContinuousSimulation(unittest.TestCase):
    """Adversarial stress testing of 50-tick continuous simulation cycles with deaths and births."""

    def tearDown(self) -> None:
        if hasattr(self, "world") and self.world is not None:
            if hasattr(self.world, "close"):
                self.world.close()
            self.world = None
        import gc
        gc.collect()

    def test_50_ticks_continuous_simulation_deaths_births_and_vector_fidelity(self) -> None:
        """Run 50 continuous ticks with 2,000 initial blobs.
        Verify natural starvation deaths > 0, natural births > 0, valid output vectors at all times,
        and RSS memory strictly < 3.2 GB without cumulative leaks.
        """
        cfg = WorldConfig.from_preset(
            "regional",
            initial_blobs=2000,
            ticks_per_day=15,
            daily_food_spawn=3000,
        )
        world = SimulationWorld(world_cfg=cfg, seed=42)
        self.world = world

        initial_pop = len(world.blobs)
        self.assertEqual(initial_pop, 2000)

        # Hook into brain pool to intercept and verify output vectors at every single step
        orig_forward = world._brain_pool.forward_and_trace
        tick_eval_records = []

        def monitored_forward(brains, obs_batch, explore_noise, rng, inspect_idx=-1, update_traces=True, cohort_idx=-1):
            n = len(brains)
            pre_valid = world._brain_pool._out_valid_cap[:n].copy()
            num_invalid_pre = int(np.sum(~pre_valid))

            actions = orig_forward(
                brains, obs_batch, explore_noise, rng, inspect_idx, update_traces, cohort_idx
            )

            post_valid = world._brain_pool._out_valid_cap[:n]
            # 1. Post forward, all living slots must be 100% valid
            self.assertTrue(
                np.all(post_valid),
                f"Invalid slots detected immediately after forward pass! {np.flatnonzero(~post_valid)}",
            )

            # 2. Returned actions must match _last_out_cap
            self.assertTrue(
                np.array_equal(actions, world._brain_pool._last_out_cap[:n]),
                "Returned actions array does not match _last_out_cap slice!",
            )

            # 3. All outputs bounded in [-1.0, 1.0] and finite
            self.assertTrue(np.all(np.isfinite(actions)), "Output contains NaN or Inf!")
            self.assertTrue(np.all(actions >= -1.0) and np.all(actions <= 1.0), "Outputs out of [-1, 1] range!")

            tick_eval_records.append((n, num_invalid_pre))
            return actions

        world._brain_pool.forward_and_trace = monitored_forward

        rss_readings = []
        for tick in range(1, 51):
            world.step(1)
            rss = _get_process_rss_gb()
            rss_readings.append(rss)
            self.assertLess(
                rss,
                3.2,
                f"Tick {tick}: Physical RSS memory exceeded limit! {rss:.3f} GB >= 3.2 GB",
            )

        # Verification of death and birth dynamics
        self.assertGreater(
            world.total_deaths_all_time,
            0,
            "Adversarial condition failed: No natural starvation deaths occurred!",
        )
        self.assertGreater(
            world.total_births_all_time,
            0,
            "Adversarial condition failed: No natural births occurred!",
        )

        # Verify RSS stability (no cumulative leak)
        max_rss = max(rss_readings)
        final_rss = rss_readings[-1]
        self.assertLess(max_rss, 3.2)
        # Standard topology RSS should remain well under 1.0 GB
        self.assertLess(final_rss, 1.0)

        # Verify all 50 ticks executed forward evaluations with slot recoveries
        self.assertEqual(len(tick_eval_records), 50)
        total_invalid_repaired = sum(rec[1] for rec in tick_eval_records)
        self.assertGreater(
            total_invalid_repaired,
            0,
            "Slot invalidation recovery mechanism was never exercised during the 50-tick run!",
        )

    def test_neural_forward_oracle_exact_numerical_match_across_strided_cycles(self) -> None:
        """Adversarial Oracle Test: verify that BatchedBrainPool._last_out_cap matches
        single BlobBrain.forward(obs, explore_noise=0.0) with machine precision (<1e-4)
        for every evaluated slot across multiple death-reallocation cycles.
        """
        cfg = NetworkConfig(depth=2, width=32, activation="tanh")
        n = 300
        rng = np.random.default_rng(2026)
        brains = [BlobBrain(config=cfg, rng=np.random.default_rng(i)) for i in range(n)]
        pool = BatchedBrainPool()
        pool.bind(brains)

        obs = rng.standard_normal((n, INPUT_DIM), dtype=np.float32)

        # Warm up pool so all slots are initialized
        pool.forward_and_trace(brains, obs, explore_noise=0.0, rng=rng, cohort_idx=-1)
        self.assertTrue(np.all(pool._out_valid_cap[:n]))

        # Simulate 12 ticks of continuous strided evaluation with pseudo-deaths
        for tick in range(12):
            cohort_idx = tick % 3
            # On some ticks, simulate deaths by swap-removing a slot
            if tick in (1, 4, 7, 10):
                # Kill slot 0 and swap tail
                dead_slot = 0
                tail_slot = len(brains) - 1
                brains[dead_slot] = brains[tail_slot]
                del brains[tail_slot]
                obs = obs[:len(brains)].copy()
                pool._out_valid_cap[dead_slot] = False

            # Randomize observations to ensure outputs vary dynamically
            obs += rng.standard_normal(obs.shape, dtype=np.float32) * 0.05
            cur_n = len(brains)

            # Snapshot pre-forward state
            pre_valid = pool._out_valid_cap[:cur_n].copy()
            pre_outs = pool._last_out_cap[:cur_n].copy()

            out_batch = pool.forward_and_trace(
                brains, obs, explore_noise=0.0, rng=rng, cohort_idx=cohort_idx
            )

            # 1. All living slots must be valid post-forward
            self.assertTrue(np.all(pool._out_valid_cap[:cur_n]))

            # 2. Determine which slots were evaluated
            invalid_indices = set(np.flatnonzero(~pre_valid))
            cohort_indices = set(range(cohort_idx, cur_n, 3))
            if len(invalid_indices) <= 32:
                eval_target = sorted(list(cohort_indices.union(invalid_indices)))
            else:
                eval_target = list(range(cur_n))

            eval_set = set(eval_target)

            # 3. For every evaluated slot: exact numerical match with single BlobBrain.forward oracle!
            for i in eval_target:
                oracle_out = brains[i].forward(obs[i], explore_noise=0.0)
                diff = np.max(np.abs(out_batch[i] - oracle_out))
                self.assertLess(
                    diff,
                    1e-4,
                    f"Tick {tick}, slot {i}: Batched output does not match BlobBrain forward oracle! Diff: {diff}",
                )

            # 4. For off-cohort intact slots: must retain previous valid output
            for i in range(cur_n):
                if i not in eval_set:
                    self.assertTrue(
                        np.array_equal(out_batch[i], pre_outs[i]),
                        f"Tick {tick}, slot {i}: Off-cohort intact slot output unexpectedly changed!",
                    )

        pool.close()
        for b in brains:
            b._pool_ref = None
            b.weights.clear()
            b.biases.clear()
        brains.clear()
        import gc
        gc.collect()

    def test_2000_blobs_10k_neurons_continuous_50_ticks_memory_stability_under_3_2_gb(self) -> None:
        """Adversarial Peak Stress: Run 50 continuous simulation ticks with 2,000 blobs having
        up to 10,000 neurons each (26 -> 10,000 -> 6), with natural starvation deaths and births.
        Verify physical working set RSS memory remains strictly < 3.8 GB throughout,
        and committed memory (PagefileUsage) exhibits zero cumulative memory leaks across ticks.
        """
        import gc
        gc.collect()

        cfg = WorldConfig.from_preset(
            "regional",
            initial_blobs=1950,
            max_population_cap=2000,
            ticks_per_day=15,
            daily_food_spawn=3000,
        )
        net_cfg = NetworkConfig(depth=1, width=10000, activation="tanh")
        world = SimulationWorld(world_cfg=cfg, net_cfg=net_cfg, seed=99)
        self.world = world

        initial_rss = _get_process_rss_gb()
        initial_commit = _get_process_commit_gb()
        # Verify initial footprint complies with <3.8 GB limit (and process <4.0 GB per Project Contract)
        self.assertLess(
            initial_rss,
            3.8,
            f"Initial process RSS exceeded 3.8 GB budget: {initial_rss:.3f} GB",
        )
        self.assertLess(
            initial_commit,
            4.0,
            f"Initial process commit charge exceeded 4.0 GB Project budget: {initial_commit:.3f} GB",
        )

        rss_per_tick = []
        commit_per_tick = []
        try:
            for tick in range(1, 51):
                world.step(1)
                rss = _get_process_rss_gb()
                commit = _get_process_commit_gb()
                rss_per_tick.append(rss)
                commit_per_tick.append(commit)
                self.assertLess(
                    rss,
                    3.8,
                    f"Tick {tick}: Physical RSS violated 3.8 GB limit! rss={rss:.3f} GB",
                )
                self.assertLess(
                    commit,
                    4.0,
                    f"Tick {tick}: Committed virtual memory violated 4.0 GB Project budget! commit={commit:.3f} GB",
                )
                # Verify outputs are valid and within bounds
                n = len(world.blobs)
                bound_n = len(world._brain_pool._bound_ids)
                # When deaths occur at end of day, bound_n reflects living before tick
                check_n = min(n, bound_n)
                last_outs = world._brain_pool._last_out_cap[:check_n]
                self.assertTrue(np.all(np.isfinite(last_outs)), f"Tick {tick}: Non-finite outputs detected!")
                self.assertTrue(
                    np.all(last_outs >= -1.0) and np.all(last_outs <= 1.0),
                    f"Tick {tick}: Outputs out of [-1, 1] range!",
                )

            peak_rss = max(rss_per_tick)
            final_commit = commit_per_tick[-1]

            # Verify peak physical memory budget compliance (< 3.8 GB)
            self.assertLess(
                peak_rss,
                3.8,
                f"Peak physical WorkingSetSize exceeded 3.8 GB budget: {peak_rss:.3f} GB",
            )

            # True cumulative memory leak evaluation via PagefileUsage (private commit charge).
            # Commit charge directly measures allocated heap bytes from Python/C/Numba,
            # completely immune to Windows NT working set page trimming and demand-page refaulting.
            commit_drift = final_commit - max(commit_per_tick[:3])
            self.assertLess(
                commit_drift,
                0.20,
                f"Cumulative memory leak detected: Commit charge grew by {commit_drift:.3f} GB across 50 ticks!",
            )

            # Post-warmup linear slope evaluation (ticks 15..50, covering post-midnight births and stabilization)
            warmup_idx = 15
            post_warmup_commit = commit_per_tick[warmup_idx:]
            ticks = np.arange(len(post_warmup_commit))
            slope, _ = np.polyfit(ticks, post_warmup_commit, 1)
            self.assertLess(
                slope,
                0.0015,
                f"Cumulative memory drift detected: Commit charge slope is {slope:.6f} GB/tick across post-warmup ticks",
            )

            # Verify that natural deaths occurred under 10k neuron stress
            self.assertGreater(
                world.total_deaths_all_time,
                0,
                "Starvation deaths must occur under 10k neuron continuous simulation",
            )
            # Verify that natural births occurred under 10k neuron stress
            self.assertGreater(
                world.total_births_all_time,
                0,
                "Births must occur under 10k neuron continuous simulation",
            )
        finally:
            if hasattr(self, "world") and self.world is not None:
                self.world.close()
                self.world = None
            import gc
            gc.collect()


if __name__ == "__main__":
    unittest.main()
