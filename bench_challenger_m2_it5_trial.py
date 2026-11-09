"""
Empirical Benchmark for Milestone 2 Iteration 5 Challenger 2
Runs 5 seeds (42, 43, 44, 45, 46) for 100 continuous ticks each (500 ticks total)
Continental topography (2100 initial blobs, 210 rocks, 40 ticks/day).
Measures:
- Mean tick duration (<33.3 ms requirement / >30 TPS, strictly <40.0 ms budget)
- P95 tick duration
- Mean kinematics time (<4.0 ms budget)
- Mean observation preparation time (<7.5 ms budget)
- Midnight reset duration (tick 40 and tick 80)
"""

import time
import json
import gc
import numpy as np
from typing import Dict, List, Any

from blob_sim.config import WorldConfig
from blob_sim.world import SimulationWorld


def run_seed_benchmark(seed: int, num_ticks: int = 100) -> Dict[str, Any]:
    # Force garbage collection before run
    gc.collect()

    cfg = WorldConfig.from_preset(
        "continent",
        initial_blobs=2100,
        max_population_cap=2500,
        daily_food_spawn=4800,
        ticks_per_day=40,
    )
    world = SimulationWorld(world_cfg=cfg, seed=seed)

    # Warm up JIT kernels with 3 ticks (consistent with test_challenger_m2.py)
    world.step(1)
    world.step(1)
    world.step(1)

    tick_durations: List[float] = []
    kinematics_times: List[float] = []
    obs_times: List[float] = []
    eval_times: List[float] = []
    forage_times: List[float] = []
    neural_times: List[float] = []
    midnight_resets: Dict[int, float] = {}

    orig_kinematics = world._apply_kinematics_and_metabolism_batch
    orig_obs = world._prepare_step_observations
    orig_eval = world._evaluate_step_interactions
    orig_forage = world._process_foraging
    orig_brain_forward = world._brain_pool.forward_and_trace

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

    def timed_eval(*args, **kwargs):
        t0 = time.perf_counter()
        res = orig_eval(*args, **kwargs)
        eval_times.append((time.perf_counter() - t0) * 1000.0)
        return res

    def timed_forage(*args, **kwargs):
        t0 = time.perf_counter()
        res = orig_forage(*args, **kwargs)
        forage_times.append((time.perf_counter() - t0) * 1000.0)
        return res

    def timed_forward(*args, **kwargs):
        t0 = time.perf_counter()
        res = orig_brain_forward(*args, **kwargs)
        neural_times.append((time.perf_counter() - t0) * 1000.0)
        return res

    world._apply_kinematics_and_metabolism_batch = timed_kinematics
    world._prepare_step_observations = timed_obs
    world._evaluate_step_interactions = timed_eval
    world._process_foraging = timed_forage
    world._brain_pool.forward_and_trace = timed_forward

    regular_tick_durations: List[float] = []

    for tick in range(1, num_ticks + 1):
        # Check if next step triggers midnight reset (tick 40, tick 80)
        # In world, tick_in_day advances by dt and resets when tick_in_day >= ticks_per_day
        is_midnight = (world.tick_in_day + 1 >= world.world_cfg.ticks_per_day)

        t_start = time.perf_counter()
        world.step(1)
        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        tick_durations.append(t_elapsed)

        sim_tick = world.tick_count
        if is_midnight:
            midnight_resets[sim_tick] = t_elapsed
        else:
            regular_tick_durations.append(t_elapsed)

    world.close()
    del world
    gc.collect()

    tick_arr = np.array(tick_durations)
    reg_arr = np.array(regular_tick_durations)
    kin_arr = np.array(kinematics_times)
    obs_arr = np.array(obs_times)

    return {
        "seed": seed,
        "mean_tick_duration_ms": float(np.mean(tick_arr)),
        "median_tick_duration_ms": float(np.median(tick_arr)),
        "p95_tick_duration_ms": float(np.percentile(tick_arr, 95)),
        "p99_tick_duration_ms": float(np.percentile(tick_arr, 99)),
        "min_tick_duration_ms": float(np.min(tick_arr)),
        "max_tick_duration_ms": float(np.max(tick_arr)),
        "mean_regular_tick_ms": float(np.mean(reg_arr)) if len(reg_arr) > 0 else 0.0,
        "p95_regular_tick_ms": float(np.percentile(reg_arr, 95)) if len(reg_arr) > 0 else 0.0,
        "mean_kinematics_ms": float(np.mean(kin_arr)) if len(kin_arr) > 0 else 0.0,
        "p95_kinematics_ms": float(np.percentile(kin_arr, 95)) if len(kin_arr) > 0 else 0.0,
        "mean_obs_prep_ms": float(np.mean(obs_arr)) if len(obs_arr) > 0 else 0.0,
        "p95_obs_prep_ms": float(np.percentile(obs_arr, 95)) if len(obs_arr) > 0 else 0.0,
        "mean_neural_ms": float(np.mean(neural_times)) if len(neural_times) > 0 else 0.0,
        "mean_forage_ms": float(np.mean(forage_times)) if len(forage_times) > 0 else 0.0,
        "mean_eval_ms": float(np.mean(eval_times)) if len(eval_times) > 0 else 0.0,
        "midnight_resets": {str(k): v for k, v in midnight_resets.items()},
    }


def main():
    seeds = [42, 43, 44, 45, 46]
    results = []

    print(f"Starting 5-seed empirical benchmark across seeds {seeds} (100 ticks each, 500 ticks total)...")
    t0_all = time.perf_counter()

    for s in seeds:
        print(f"\n--- Running Seed {s} ---")
        res = run_seed_benchmark(s, 100)
        results.append(res)
        print(f"  Mean Tick:       {res['mean_tick_duration_ms']:.2f} ms ({1000.0/res['mean_tick_duration_ms']:.1f} TPS)")
        print(f"  P95 Tick:        {res['p95_tick_duration_ms']:.2f} ms")
        print(f"  Mean Regular:    {res['mean_regular_tick_ms']:.2f} ms ({1000.0/res['mean_regular_tick_ms']:.1f} TPS)")
        print(f"  Mean Kinematics: {res['mean_kinematics_ms']:.2f} ms (Budget: < 4.0 ms)")
        print(f"  Mean Obs Prep:   {res['mean_obs_prep_ms']:.2f} ms (Budget: < 7.5 ms)")
        print(f"  Midnight Resets: {res['midnight_resets']}")

    total_wall_s = time.perf_counter() - t0_all

    all_means = [r["mean_tick_duration_ms"] for r in results]
    all_p95s = [r["p95_tick_duration_ms"] for r in results]
    all_kins = [r["mean_kinematics_ms"] for r in results]
    all_obs = [r["mean_obs_prep_ms"] for r in results]
    all_reg_means = [r["mean_regular_tick_ms"] for r in results]

    summary = {
        "seeds": seeds,
        "trials": results,
        "grand_mean_tick_ms": float(np.mean(all_means)),
        "grand_mean_regular_tick_ms": float(np.mean(all_reg_means)),
        "grand_p95_tick_ms": float(np.mean(all_p95s)),
        "grand_mean_kinematics_ms": float(np.mean(all_kins)),
        "grand_mean_obs_prep_ms": float(np.mean(all_obs)),
        "total_wall_time_s": total_wall_s,
    }

    print("\n" + "=" * 60)
    print("5-SEED GRAND SUMMARY (500 TICKS TOTAL):")
    print(f"  Grand Mean Tick:       {summary['grand_mean_tick_ms']:.2f} ms")
    print(f"  Grand Mean Regular:    {summary['grand_mean_regular_tick_ms']:.2f} ms")
    print(f"  Grand P95 Tick:        {summary['grand_p95_tick_ms']:.2f} ms")
    print(f"  Grand Mean Kinematics: {summary['grand_mean_kinematics_ms']:.2f} ms")
    print(f"  Grand Mean Obs Prep:   {summary['grand_mean_obs_prep_ms']:.2f} ms")
    print(f"  Total Benchmark Time:  {summary['total_wall_time_s']:.2f} s")
    print("=" * 60)

    with open("benchmark_m2_it5_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    main()
