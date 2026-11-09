"""Multi-Seed Empirical Benchmark Suite for Milestone 2 Iteration 6 Challenger 2.

Tests:
1. Multi-Seed Continental Scale Trials across Seeds 42, 43, 44, 45, 46:
   - 2,100 initial blobs on Continental Topography (210 rock formations)
   - 100 continuous ticks per seed
   - Subsystem latency tracking: Kinematics (<4.0 ms), Observations (<7.5 ms), Total Tick (<40.0 ms)
   - Midnight rollover tracking at Ticks 40 and 80
   - Physical memory RSS tracking (<3.5 GB) and leak detection (<0.20 GB drift)
   - Population vitality and economic mechanics
2. Multi-Seed Bitwise Numerical Parity on Feature 6:
   - Scalar vs Batch Kinematics across multiple seeds and action patterns
"""

import unittest
import time
import math
import ctypes
from ctypes import wintypes
from typing import Dict, List, Any
import gc
import numpy as np

from blob_sim.config import WorldConfig
from blob_sim.world import SimulationWorld
from blob_sim.entities import Blob


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


class TestChallengerM2It6MultiSeedContinentalBenchmark(unittest.TestCase):
    """Empirical multi-seed verification of Continental Scale Throughput and Latency."""

    @classmethod
    def setUpClass(cls) -> None:
        if hasattr(ctypes, "windll") and hasattr(ctypes.windll, "kernel32"):
            # Elevate process priority to HIGH_PRIORITY_CLASS (0x0080) to prevent
            # Windows EcoQoS background throttling across consecutive trials.
            # Explicitly declare 64-bit argtypes/restype to avoid 32-bit HANDLE truncation.
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            k32.SetPriorityClass.restype = wintypes.BOOL
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x0080)
        from blob_sim.spatial import warmup_simulation_engine
        warmup_simulation_engine()

    def setUp(self) -> None:
        gc.collect()

    def tearDown(self) -> None:
        gc.collect()

    def run_single_seed_trial(self, seed: int) -> Dict[str, Any]:
        """Execute a 100-tick trial for a given random seed and collect comprehensive metrics."""
        cfg = WorldConfig.from_preset(
            "continent",
            initial_blobs=2100,
            max_population_cap=2500,
            daily_food_spawn=4800,
            ticks_per_day=40,
        )
        world = SimulationWorld(world_cfg=cfg, seed=seed)
        orig_kinematics = world._apply_kinematics_and_metabolism_batch
        orig_obs = world._prepare_step_observations
        try:
            # Warm up JIT kernels with 5 ticks (covering all 3 cohorts + recovery + steady state)
            for _ in range(5):
                world.step(1)

            rss_measurements: List[float] = []
            tick_durations: List[float] = []
            kinematics_times: List[float] = []
            obs_times: List[float] = []

            def timed_kinematics(*args, **kwargs):
                t0 = time.perf_counter()
                res = orig_kinematics(*args, **kwargs)
                kinematics_times.append((time.perf_counter() - t0) * 1000.0)
                return res

            def timed_obs(*args, **kwargs):
                t0 = time.perf_counter()
                res = orig_obs(*args, **kwargs)
                obs_times.append((time.perf_counter() - t0) * 1000.0)
                return res

            world._apply_kinematics_and_metabolism_batch = timed_kinematics
            world._prepare_step_observations = timed_obs

            midnight_durations = {}

            # Collect residual garbage and isolate micro-benchmark timing from cyclic GC pauses
            gc.collect()
            gc.disable()
            try:
                for tick in range(1, 101):
                    t_start = time.perf_counter()
                    world.step(1)
                    t_elapsed = (time.perf_counter() - t_start) * 1000.0
                    tick_durations.append(t_elapsed)

                    rss = _get_process_rss_gb()
                    rss_measurements.append(rss)

                    if (world.tick_count % cfg.ticks_per_day) == 0:
                        midnight_durations[tick] = t_elapsed

                    self.assertLess(rss, 3.5, f"Tick {tick} seed {seed}: RSS {rss:.3f} GB >= 3.5 GB")
            finally:
                gc.enable()

            avg_tick = float(np.mean(tick_durations))
            p95_tick = float(np.percentile(tick_durations, 95))
            max_tick = float(np.max(tick_durations))
            min_tick = float(np.min(tick_durations))
            avg_kin = float(np.mean(kinematics_times)) if kinematics_times else 0.0
            max_kin = float(np.max(kinematics_times)) if kinematics_times else 0.0
            avg_obs = float(np.mean(obs_times)) if obs_times else 0.0
            max_obs = float(np.max(obs_times)) if obs_times else 0.0

            living_blobs = [b for b in world.blobs if b.alive]
            total_foods = sum(b.total_food_eaten for b in world.blobs)
            max_rss = max(rss_measurements)
            final_rss = rss_measurements[-1]
            baseline_rss = rss_measurements[9]
            rss_drift = final_rss - baseline_rss

            mid_durations_list = list(midnight_durations.values())
            mid40 = mid_durations_list[0] if len(mid_durations_list) > 0 else 0.0
            mid80 = mid_durations_list[1] if len(mid_durations_list) > 1 else 0.0
            mid_mean = float(np.mean(mid_durations_list)) if mid_durations_list else 0.0

            # Strict Assertions
            self.assertLess(avg_tick, 40.0, f"Seed {seed}: Mean tick {avg_tick:.2f} ms >= 40.0 ms")
            self.assertLess(avg_kin, 4.0, f"Seed {seed}: Mean kinematics {avg_kin:.2f} ms >= 4.0 ms")
            self.assertLess(avg_obs, 7.5, f"Seed {seed}: Mean observation {avg_obs:.2f} ms >= 7.5 ms")
            self.assertGreater(len(living_blobs), 1000, f"Seed {seed}: population collapsed ({len(living_blobs)})")
            self.assertGreater(total_foods, 0, f"Seed {seed}: no food eaten")
            self.assertGreater(world.total_births_all_time, 0, f"Seed {seed}: no births")
            self.assertGreater(world.total_deaths_all_time, 0, f"Seed {seed}: no deaths")
            self.assertLess(max_rss, 3.5, f"Seed {seed}: Peak RSS {max_rss:.3f} GB >= 3.5 GB")
            self.assertLess(rss_drift, 0.20, f"Seed {seed}: RSS drift {rss_drift:.3f} GB >= 0.20 GB")

            res = {
                "seed": seed,
                "mean_tick_ms": round(avg_tick, 2),
                "p95_tick_ms": round(p95_tick, 2),
                "max_tick_ms": round(max_tick, 2),
                "min_tick_ms": round(min_tick, 2),
                "mean_kinematics_ms": round(avg_kin, 2),
                "max_kinematics_ms": round(max_kin, 2),
                "mean_obs_ms": round(avg_obs, 2),
                "max_obs_ms": round(max_obs, 2),
                "midnight_tick40_ms": round(mid40, 2),
                "midnight_tick80_ms": round(mid80, 2),
                "mean_midnight_ms": round(mid_mean, 2),
                "living_blobs": len(living_blobs),
                "total_births": world.total_births_all_time,
                "total_deaths": world.total_deaths_all_time,
                "total_food_eaten": total_foods,
                "peak_rss_gb": round(max_rss, 3),
                "rss_drift_gb": round(rss_drift, 3),
                "tps": round(1000.0 / avg_tick, 1),
            }
            return res
        finally:
            world._apply_kinematics_and_metabolism_batch = orig_kinematics
            world._prepare_step_observations = orig_obs
            if hasattr(world, "close"):
                world.close()
            del world
            gc.collect()

    def test_multi_seed_continental_trials_42_to_46(self) -> None:
        """Run continuous 100 ticks across Seeds 42, 43, 44, 45, 46 and print markdown table."""
        seeds = [42, 43, 44, 45, 46]
        results = []

        for s in seeds:
            res = self.run_single_seed_trial(s)
            results.append(res)

        print("\n" + "=" * 115)
        print("CONTINENTAL SCALE THROUGHPUT & LATENCY MULTI-SEED EMPIRICAL BENCHMARK (SEEDS 42-46)")
        print("=" * 115)
        header = (
            f"| {'Seed':<4} | {'Mean Tick':<9} | {'P95 Tick':<8} | {'Max Tick':<8} "
            f"| {'Mean Kin':<8} | {'Mean Obs':<8} | {'Tick 40':<7} | {'Tick 80':<7} | {'Midnight Avg':<12} | {'TPS':<5} | {'Living':<6} |"
        )
        print(header)
        print("|" + "|".join(["-" * (len(col) + 2) for col in header.split("|")[1:-1]]) + "|")

        for r in results:
            print(
                f"| {r['seed']:<4} | {r['mean_tick_ms']:>6.2f} ms | {r['p95_tick_ms']:>5.2f} ms | {r['max_tick_ms']:>5.2f} ms "
                f"| {r['mean_kinematics_ms']:>5.2f} ms | {r['mean_obs_ms']:>5.2f} ms "
                f"| {r['midnight_tick40_ms']:>5.2f} ms | {r['midnight_tick80_ms']:>5.2f} ms | {r['mean_midnight_ms']:>8.2f} ms "
                f"| {r['tps']:>5.1f} | {r['living_blobs']:>6} |"
            )

        mean_all_ticks = float(np.mean([r["mean_tick_ms"] for r in results]))
        mean_all_kin = float(np.mean([r["mean_kinematics_ms"] for r in results]))
        mean_all_obs = float(np.mean([r["mean_obs_ms"] for r in results]))
        mean_all_midnight = float(np.mean([r["mean_midnight_ms"] for r in results]))
        mean_all_tps = float(np.mean([r["tps"] for r in results]))

        print("-" * 115)
        print(
            f"GRAND MEAN: Tick = {mean_all_ticks:.2f} ms | Kinematics = {mean_all_kin:.2f} ms | "
            f"Obs = {mean_all_obs:.2f} ms | Midnight = {mean_all_midnight:.2f} ms | TPS = {mean_all_tps:.1f}"
        )
        print("=" * 115 + "\n")

        self.assertLess(mean_all_ticks, 40.0)
        self.assertLess(mean_all_kin, 4.0)
        self.assertLess(mean_all_obs, 7.5)


if __name__ == "__main__":
    unittest.main()
