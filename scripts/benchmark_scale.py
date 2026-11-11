"""Canonical Automated Scaling Benchmark Script for Game of Blob (Milestone 4 / R1 Verification).

Measures and asserts:
1. Continuous simulation throughput (>30 TPS / <33.33 ms per tick).
2. Physical RSS memory stability (<3.8 GB working set) and zero cumulative leaks (<0.20 GB drift).
3. Subsystem latency profiling (kinematics, neural inference, observation extraction, foraging, interactions).
4. Population vitality and economic mechanics under scaled concurrent blob loads.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import gc
import json
import os
import sys
import time

# Auto-discover and insert project root into sys.path
_search_dir = os.path.abspath(os.path.dirname(__file__))
while _search_dir and _search_dir != os.path.dirname(_search_dir):
    if os.path.isdir(os.path.join(_search_dir, "blob_sim")):
        if _search_dir not in sys.path:
            sys.path.insert(0, _search_dir)
        break
    _search_dir = os.path.dirname(_search_dir)

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from blob_sim.config import NetworkConfig, WorldConfig, WORLD_SCALE_PRESETS
from blob_sim.neural import BatchedBrainPool
from blob_sim.spatial import warmup_simulation_engine
from blob_sim.world import SimulationWorld


# ---------------------------------------------------------------------------
# Windows Process Memory Profiler
# ---------------------------------------------------------------------------

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


def get_process_memory_info() -> Tuple[float, float, float]:
    """Retrieve Windows process memory metrics in Gigabytes: (RSS_GB, Peak_RSS_GB, Commit_GB).

    Falls back to cross-platform standard library / resource if not on Windows.
    """
    if sys.platform == "win32" and hasattr(ctypes, "windll"):
        try:
            fn = ctypes.windll.psapi.GetProcessMemoryInfo
            fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
            fn.restype = wintypes.BOOL
            h = ctypes.windll.kernel32.GetCurrentProcess()
            pmc = _PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
            if fn(h, ctypes.byref(pmc), pmc.cb):
                rss = float(pmc.WorkingSetSize) / (1024.0 ** 3)
                peak_rss = float(pmc.PeakWorkingSetSize) / (1024.0 ** 3)
                commit = float(pmc.PagefileUsage) / (1024.0 ** 3)
                return rss, peak_rss, commit
        except Exception:
            pass

    # Cross-platform fallback
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # Linux ru_maxrss is in KB, macOS is in Bytes
        rss_gb = (ru.ru_maxrss * 1024.0 if sys.platform != "darwin" else ru.ru_maxrss) / (1024.0 ** 3)
        return rss_gb, rss_gb, rss_gb
    except ImportError:
        return 0.0, 0.0, 0.0


def set_high_process_priority() -> bool:
    """Elevate Windows process priority class to HIGH_PRIORITY_CLASS (0x0080).

    Prevents Windows 11 EcoQoS background throttling and thread demotion during benchmarks.
    """
    if sys.platform == "win32" and hasattr(ctypes, "windll"):
        try:
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            k32.SetPriorityClass.restype = wintypes.BOOL
            return bool(k32.SetPriorityClass(k32.GetCurrentProcess(), 0x0080))
        except Exception:
            return False
    return False


# ---------------------------------------------------------------------------
# Benchmark Execution Engine
# ---------------------------------------------------------------------------

def run_scale_benchmark(
    blobs: int = 2000,
    neurons: int = 10000,
    ticks: int = 100,
    preset: str = "continent",
    seed: int = 42,
    warmup: int = 5,
    ticks_per_day: Optional[int] = None,
    output_json: Optional[str] = None,
    strict_assert: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Execute the scaling benchmark and return comprehensive performance metrics."""
    set_high_process_priority()
    warmup_simulation_engine()
    gc.collect()

    init_rss, init_peak_rss, init_commit = get_process_memory_info()

    # 1. World and Neural Configuration
    net_cfg = NetworkConfig(depth=1, width=neurons, activation="tanh")
    world_cfg = WorldConfig.from_preset(
        preset,
        initial_blobs=blobs,
        max_population_cap=max(blobs + 500, int(blobs * 1.25)),
    )
    if ticks_per_day is not None and ticks_per_day > 0:
        world_cfg.ticks_per_day = int(ticks_per_day)

    print("=" * 80)
    print(f"GAME OF BLOB: SCALING BENCHMARK (M4 / R1)")
    print(f"Configuration: {blobs} Blobs | {neurons} Neurons/Blob | {ticks} Ticks | Preset: {preset} | Seed: {seed}")
    print(f"Initial Memory: RSS = {init_rss:.3f} GB | Commit = {init_commit:.3f} GB")
    print("=" * 80)

    t0_world_init = time.perf_counter()
    world = SimulationWorld(world_cfg=world_cfg, net_cfg=net_cfg, seed=seed)
    dur_world_init_s = time.perf_counter() - t0_world_init

    # Contract verification: BatchedBrainPool must back all brains
    assert isinstance(world._brain_pool, BatchedBrainPool), "SimulationWorld brain pool is not BatchedBrainPool!"
    assert len(world.blobs) == blobs, f"World initialized with {len(world.blobs)} blobs instead of {blobs}!"

    post_init_rss, _, post_init_commit = get_process_memory_info()
    print(f"Initialization completed in {dur_world_init_s:.2f}s.")
    print(f"Post-Init Memory: RSS = {post_init_rss:.3f} GB | Commit = {post_init_commit:.3f} GB")

    # 2. Warmup Execution (Warm up JIT, cohorts, and ring buffers)
    if warmup > 0:
        print(f"Running {warmup} unmeasured warmup ticks for JIT compilation & cohort cache residency...")
        for _ in range(warmup):
            world.step(1)
        gc.collect()

    warmup_rss, _, warmup_commit = get_process_memory_info()
    print(f"Warmup complete. Baseline Memory: RSS = {warmup_rss:.3f} GB | Commit = {warmup_commit:.3f} GB")

    # 3. Subsystem Instrumentation
    kinematics_times: List[float] = []
    observation_times: List[float] = []
    neural_times: List[float] = []
    forage_times: List[float] = []
    learning_times: List[float] = []

    orig_kin = world._apply_kinematics_and_metabolism_batch
    orig_obs = world._prepare_step_observations
    orig_fwd = world._brain_pool.forward_and_trace
    orig_forage = world._process_foraging
    orig_learn = world._brain_pool.apply_online_learning_batch

    def timed_kinematics(*args, **kwargs):
        t0 = time.perf_counter()
        res = orig_kin(*args, **kwargs)
        kinematics_times.append((time.perf_counter() - t0) * 1000.0)
        return res

    def timed_obs(*args, **kwargs):
        t0 = time.perf_counter()
        res = orig_obs(*args, **kwargs)
        observation_times.append((time.perf_counter() - t0) * 1000.0)
        return res

    def timed_neural(*args, **kwargs):
        t0 = time.perf_counter()
        res = orig_fwd(*args, **kwargs)
        neural_times.append((time.perf_counter() - t0) * 1000.0)
        return res

    def timed_foraging(*args, **kwargs):
        t0 = time.perf_counter()
        res = orig_forage(*args, **kwargs)
        forage_times.append((time.perf_counter() - t0) * 1000.0)
        return res

    def timed_learning(*args, **kwargs):
        t0 = time.perf_counter()
        res = orig_learn(*args, **kwargs)
        learning_times.append((time.perf_counter() - t0) * 1000.0)
        return res

    world._apply_kinematics_and_metabolism_batch = timed_kinematics
    world._prepare_step_observations = timed_obs
    world._brain_pool.forward_and_trace = timed_neural
    world._process_foraging = timed_foraging
    world._brain_pool.apply_online_learning_batch = timed_learning

    # 4. Measured Simulation Loop
    tick_durations: List[float] = []
    regular_tick_durations: List[float] = []
    midnight_ticks: Dict[int, float] = {}
    rss_history: List[float] = []
    commit_history: List[float] = []
    living_history: List[int] = []

    print("-" * 80)
    print(f"Beginning continuous measurement of {ticks} simulation ticks...")

    try:
        for t in range(1, ticks + 1):
            is_midnight = (world.tick_count % world_cfg.ticks_per_day == 0)

            t_start = time.perf_counter()
            world.step(1)
            t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

            cur_rss, _, cur_commit = get_process_memory_info()
            cur_living = len([b for b in world.blobs if b.alive])

            tick_durations.append(t_elapsed_ms)
            rss_history.append(cur_rss)
            commit_history.append(cur_commit)
            living_history.append(cur_living)

            if is_midnight:
                midnight_ticks[t] = t_elapsed_ms
                if verbose:
                    print(f"  [Midnight Tick {t}]: {t_elapsed_ms:.2f} ms | Living: {cur_living} | RSS: {cur_rss:.3f} GB")
            else:
                regular_tick_durations.append(t_elapsed_ms)
                if verbose and (t <= 10 or t % 20 == 0 or t == ticks):
                    print(f"  [Tick {t:3d}]: {t_elapsed_ms:.2f} ms | Kin: {kinematics_times[-1]:.2f}ms | Neural: {neural_times[-1]:.2f}ms | RSS: {cur_rss:.3f} GB")

    finally:
        world._apply_kinematics_and_metabolism_batch = orig_kin
        world._prepare_step_observations = orig_obs
        world._brain_pool.forward_and_trace = orig_fwd
        world._process_foraging = orig_forage
        world._brain_pool.apply_online_learning_batch = orig_learn

    # 5. Statistical Calculations
    tick_arr = np.array(tick_durations)
    reg_arr = np.array(regular_tick_durations) if regular_tick_durations else tick_arr

    all_mean_ms = float(np.mean(tick_arr))
    all_tps = 1000.0 / max(0.001, all_mean_ms)
    all_median_ms = float(np.median(tick_arr))
    all_p90_ms = float(np.percentile(tick_arr, 90))
    all_p95_ms = float(np.percentile(tick_arr, 95))
    all_p99_ms = float(np.percentile(tick_arr, 99))
    all_min_ms = float(np.min(tick_arr))
    all_max_ms = float(np.max(tick_arr))

    reg_mean_ms = float(np.mean(reg_arr))
    reg_tps = 1000.0 / max(0.001, reg_mean_ms)
    reg_p95_ms = float(np.percentile(reg_arr, 95))

    mean_kin_ms = float(np.mean(kinematics_times)) if kinematics_times else 0.0
    p95_kin_ms = float(np.percentile(kinematics_times, 95)) if kinematics_times else 0.0
    mean_obs_ms = float(np.mean(observation_times)) if observation_times else 0.0
    p95_obs_ms = float(np.percentile(observation_times, 95)) if observation_times else 0.0
    mean_neural_ms = float(np.mean(neural_times)) if neural_times else 0.0
    p95_neural_ms = float(np.percentile(neural_times, 95)) if neural_times else 0.0
    mean_forage_ms = float(np.mean(forage_times)) if forage_times else 0.0
    p95_forage_ms = float(np.percentile(forage_times, 95)) if forage_times else 0.0
    mean_learn_ms = float(np.mean(learning_times)) if learning_times else 0.0
    p95_learn_ms = float(np.percentile(learning_times, 95)) if learning_times else 0.0

    mean_subsystems = mean_kin_ms + mean_obs_ms + mean_neural_ms + mean_forage_ms + mean_learn_ms
    mean_interactions_ms = max(0.0, reg_mean_ms - mean_subsystems)

    final_rss = rss_history[-1] if rss_history else warmup_rss
    peak_rss = max(rss_history) if rss_history else warmup_rss
    final_commit = commit_history[-1] if commit_history else warmup_commit
    peak_commit = max(commit_history) if commit_history else warmup_commit
    rss_drift = final_rss - warmup_rss

    final_living = living_history[-1] if living_history else len(world.blobs)
    total_births = world.total_births_all_time
    total_deaths = world.total_deaths_all_time
    total_food_eaten = sum(b.total_food_eaten for b in world.blobs)
    total_trades = world.total_trades_all_time
    total_raids = world.total_raids_all_time

    # Cleanup world resources
    world.close()
    del world
    gc.collect()

    # 6. Contract Verification & Assertions
    pass_tps_reg = reg_mean_ms < 33.33  # > 30 TPS regular ticks
    pass_tps_all = all_mean_ms < 40.0   # > 25 TPS grand mean including midnight rollovers
    pass_peak_rss = peak_rss < 3.8      # strictly < 3.8 GB working set limit
    pass_drift = rss_drift < 0.20       # < 0.20 GB drift across 100 ticks
    pass_pop = final_living >= int(blobs * 0.40)  # at least 40% population survived

    all_passed = pass_tps_reg and pass_tps_all and pass_peak_rss and pass_drift and pass_pop

    # 7. Formatted Results Output
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Ticks Measured:       {ticks}")
    print(f"Regular Tick Mean:          {reg_mean_ms:.2f} ms ({reg_tps:.1f} TPS)  [Contract: <33.33 ms / >30 TPS]")
    print(f"Regular Tick P95:           {reg_p95_ms:.2f} ms")
    print(f"All Ticks Mean (w/ Resets): {all_mean_ms:.2f} ms ({all_tps:.1f} TPS)  [Contract: <40.00 ms]")
    print(f"Median / P95 / P99:         {all_median_ms:.2f} ms / {all_p95_ms:.2f} ms / {all_p99_ms:.2f} ms")
    print(f"Min / Max Tick:             {all_min_ms:.2f} ms / {all_max_ms:.2f} ms")
    if midnight_ticks:
        mid_mean = float(np.mean(list(midnight_ticks.values())))
        print(f"Midnight Rollover Ticks:    {len(midnight_ticks)} resets, Mean = {mid_mean:.2f} ms ({midnight_ticks})")

    print("\n--- Subsystem Latency Breakdown ---")
    print(f"  Kinematics:               Mean = {mean_kin_ms:.2f} ms | P95 = {p95_kin_ms:.2f} ms")
    print(f"  Neural Inference (GEMV):  Mean = {mean_neural_ms:.2f} ms | P95 = {p95_neural_ms:.2f} ms")
    print(f"  Observation Extraction:   Mean = {mean_obs_ms:.2f} ms | P95 = {p95_obs_ms:.2f} ms")
    print(f"  Foraging & Food Grid:     Mean = {mean_forage_ms:.2f} ms | P95 = {p95_forage_ms:.2f} ms")
    print(f"  Online RL Learning Step:  Mean = {mean_learn_ms:.2f} ms | P95 = {p95_learn_ms:.2f} ms")
    print(f"  Social/Market/Other:      Mean = {mean_interactions_ms:.2f} ms")

    print("\n--- Memory & Leak Analysis ---")
    print(f"  Initial RSS:              {init_rss:.3f} GB")
    print(f"  Post-Warmup Baseline RSS: {warmup_rss:.3f} GB")
    print(f"  Peak Working Set (RSS):   {peak_rss:.3f} GB  [Contract Limit: < 3.800 GB]")
    print(f"  Final RSS:                {final_rss:.3f} GB")
    print(f"  RSS Drift:                {rss_drift:.3f} GB  [Contract Limit: < 0.200 GB]")
    print(f"  Peak Commit Charge:       {peak_commit:.3f} GB")

    print("\n--- Population Dynamics ---")
    print(f"  Living Blobs at End:      {final_living} / {blobs}")
    print(f"  Total Births:             {total_births}")
    print(f"  Total Starvation Deaths:  {total_deaths}")
    print(f"  Total Food Eaten:         {total_food_eaten}")
    print(f"  Total Trades / Raids:     {total_trades} / {total_raids}")

    print("\n--- Verification Contracts ---")
    print(f"  [Assertion 1] Regular TPS > 30 (>30 TPS, <33.33 ms):   {'PASSED' if pass_tps_reg else 'FAILED'} ({reg_mean_ms:.2f} ms)")
    print(f"  [Assertion 2] All Ticks Mean (<40.0 ms budget):       {'PASSED' if pass_tps_all else 'FAILED'} ({all_mean_ms:.2f} ms)")
    print(f"  [Assertion 3] Peak Physical RSS (<3.8 GB working set): {'PASSED' if pass_peak_rss else 'FAILED'} ({peak_rss:.3f} GB)")
    print(f"  [Assertion 4] Memory Drift (<0.20 GB across run):      {'PASSED' if pass_drift else 'FAILED'} ({rss_drift:.3f} GB)")
    print(f"  [Assertion 5] Population Viability (Living >= 40%):   {'PASSED' if pass_pop else 'FAILED'} ({final_living} living)")
    print("=" * 80)

    # 8. Structured Payload
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "blobs": blobs,
            "neurons": neurons,
            "ticks": ticks,
            "preset": preset,
            "seed": seed,
            "warmup": warmup,
            "ticks_per_day": world_cfg.ticks_per_day,
        },
        "summary": {
            "all_mean_tick_ms": round(all_mean_ms, 2),
            "all_tps": round(all_tps, 1),
            "regular_mean_tick_ms": round(reg_mean_ms, 2),
            "regular_tps": round(reg_tps, 1),
            "median_tick_ms": round(all_median_ms, 2),
            "p90_tick_ms": round(all_p90_ms, 2),
            "p95_tick_ms": round(all_p95_ms, 2),
            "p99_tick_ms": round(all_p99_ms, 2),
            "min_tick_ms": round(all_min_ms, 2),
            "max_tick_ms": round(all_max_ms, 2),
            "peak_rss_gb": round(peak_rss, 3),
            "rss_drift_gb": round(rss_drift, 3),
            "all_contracts_passed": all_passed,
        },
        "subsystems": {
            "kinematics": {"mean_ms": round(mean_kin_ms, 2), "p95_ms": round(p95_kin_ms, 2)},
            "observation_prep": {"mean_ms": round(mean_obs_ms, 2), "p95_ms": round(p95_obs_ms, 2)},
            "neural_inference": {"mean_ms": round(mean_neural_ms, 2), "p95_ms": round(p95_neural_ms, 2)},
            "foraging": {"mean_ms": round(mean_forage_ms, 2), "p95_ms": round(p95_forage_ms, 2)},
            "learning": {"mean_ms": round(mean_learn_ms, 2), "p95_ms": round(p95_learn_ms, 2)},
            "interactions": {"mean_ms": round(mean_interactions_ms, 2)},
        },
        "memory": {
            "initial_rss_gb": round(init_rss, 3),
            "warmup_rss_gb": round(warmup_rss, 3),
            "final_rss_gb": round(final_rss, 3),
            "peak_rss_gb": round(peak_rss, 3),
            "rss_drift_gb": round(rss_drift, 3),
            "peak_commit_gb": round(peak_commit, 3),
        },
        "population": {
            "initial_blobs": blobs,
            "final_living": final_living,
            "total_births": total_births,
            "total_deaths": total_deaths,
            "total_food_eaten": total_food_eaten,
            "total_trades": total_trades,
            "total_raids": total_raids,
        },
        "assertions": {
            "regular_tps_gt_30": pass_tps_reg,
            "all_ticks_mean_lt_40ms": pass_tps_all,
            "peak_rss_lt_3_8_gb": pass_peak_rss,
            "memory_drift_lt_0_20_gb": pass_drift,
            "population_viable": pass_pop,
            "all_passed": all_passed,
        },
        "ticks": [
            {"tick": i + 1, "duration_ms": round(d, 2), "rss_gb": round(r, 3)}
            for i, (d, r) in enumerate(zip(tick_durations, rss_history))
        ],
    }

    if output_json:
        out_dir = os.path.dirname(output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[OK] Benchmark JSON payload exported to: {output_json}")

    if strict_assert and not all_passed:
        failed_contracts = [k for k, v in payload["assertions"].items() if not v and k != "all_passed"]
        raise AssertionError(f"Scaling benchmark contracts failed: {failed_contracts}")

    return payload


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonical Automated Scaling Benchmark for Game of Blob (Milestone 4 / R1 Verification)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--blobs", type=int, default=2000, help="Initial blob population count.")
    parser.add_argument("--neurons", type=int, default=10000, help="Neurons in hidden layer (up to 10,000).")
    parser.add_argument("--ticks", type=int, default=100, help="Number of continuous ticks to simulate.")
    parser.add_argument("--preset", type=str, default="continent", choices=list(WORLD_SCALE_PRESETS.keys()), help="World map scale preset.")
    parser.add_argument("--seed", type=int, default=42, help="Simulation random seed.")
    parser.add_argument("--output-json", type=str, default=None, help="Output file path for JSON benchmark results.")
    parser.add_argument("--warmup", type=int, default=5, help="Number of unmeasured warmup ticks.")
    parser.add_argument("--ticks-per-day", type=int, default=None, help="Optional override for ticks_per_day.")
    parser.add_argument("--no-assert", action="store_true", help="Disable strict assertion exit code on contract failure.")
    parser.add_argument("--verbose", action="store_true", help="Print per-tick metrics during execution.")

    args = parser.parse_args()

    try:
        run_scale_benchmark(
            blobs=args.blobs,
            neurons=args.neurons,
            ticks=args.ticks,
            preset=args.preset,
            seed=args.seed,
            warmup=args.warmup,
            ticks_per_day=args.ticks_per_day,
            output_json=args.output_json,
            strict_assert=not args.no_assert,
            verbose=args.verbose,
        )
        sys.exit(0)
    except AssertionError as err:
        print(f"\n[FAIL] {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] Unexpected benchmark execution error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
