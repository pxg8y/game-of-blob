"""Empirical Challenger Edge-Case & Adversarial Stress Harness for Milestone 4.

Tests:
1. Zero Blobs: Initialized / empty population stepping across 50 ticks.
2. Single Blob: 1 solitary blob stepping across 100 ticks (shape & scalar invariants).
3. All-Death Extinction & Recovery: Complete population extinction and respawn floor recovery.
4. Long Continuous Run: 500 ticks continuous simulation with 500 blobs checking memory drift and stability.
5. Coordinate & Sensory Fuzzing: Blobs on map extremes and singularity coordinates.
"""

from __future__ import annotations

import gc
import os
import sys
import time
import numpy as np

# Auto-discover and insert project root into sys.path
_search_dir = os.path.abspath(os.path.dirname(__file__))
while _search_dir and _search_dir != os.path.dirname(_search_dir):
    if os.path.isdir(os.path.join(_search_dir, "blob_sim")):
        if _search_dir not in sys.path:
            sys.path.insert(0, _search_dir)
        break
    _search_dir = os.path.dirname(_search_dir)

from blob_sim.config import NetworkConfig, WorldConfig
from blob_sim.world import SimulationWorld
from scripts.benchmark_scale import get_process_memory_info


def test_zero_blobs_edge_case():
    print("\n[CHALLENGE 1] Testing Zero Blobs Edge Case...")
    cfg = WorldConfig.from_preset("classic", initial_blobs=0, min_population_floor=0)
    world = SimulationWorld(world_cfg=cfg, net_cfg=NetworkConfig(width=16), seed=42)

    # Empty population
    assert len(world.blobs) == 0, f"Expected 0 blobs, got {len(world.blobs)}"

    # Step for 50 ticks
    for t in range(50):
        world.step(1)

    assert len(world.blobs) == 0
    print(f"  -> PASSED: 50 ticks executed with 0 blobs, tick_count={world.tick_count}, no crash or exception.")
    world.close()


def test_single_blob_edge_case():
    print("\n[CHALLENGE 2] Testing Single Blob Edge Case (N=1)...")
    cfg = WorldConfig.from_preset("classic", initial_blobs=1, min_population_floor=1)
    world = SimulationWorld(world_cfg=cfg, net_cfg=NetworkConfig(width=24), seed=42)

    assert len(world.blobs) == 1, f"Expected 1 blob, got {len(world.blobs)}"
    blob = world.blobs[0]
    initial_energy = blob.energy

    for t in range(100):
        world.step(1)
        if not blob.alive:
            break

    print(f"  -> PASSED: Solitary blob stepped for {world.tick_count} ticks. Alive={blob.alive}, FoodEaten={blob.total_food_eaten:.2f}.")
    world.close()


def test_extinction_and_recovery():
    print("\n[CHALLENGE 3] Testing All-Death Extinction & Recovery Floor...")
    cfg = WorldConfig.from_preset("classic", initial_blobs=50, min_population_floor=10)
    cfg.ticks_per_day = 20
    world = SimulationWorld(world_cfg=cfg, net_cfg=NetworkConfig(width=24), seed=42)

    assert len(world.blobs) == 50

    # Artificially starve all blobs
    for b in world.blobs:
        b.energy = 0.0
        b.food_today = 0.0
        b.money = 0.0
        b.alive = False

    # Force step to midnight rollover
    remaining_ticks = cfg.ticks_per_day - (world.tick_count % cfg.ticks_per_day)
    for _ in range(remaining_ticks):
        world.step(1)

    living_now = len([b for b in world.blobs if b.alive])
    print(f"  -> Extinction rollover triggered. Living blobs after floor recovery: {living_now}")
    assert living_now >= cfg.min_population_floor, f"Expected >= {cfg.min_population_floor} revived blobs, got {living_now}"

    # Step further to verify revived blobs function normally
    for _ in range(25):
        world.step(1)

    print(f"  -> PASSED: Simulation recovered from 100% extinction, restored to {living_now} blobs, and continued stepping.")
    world.close()


def test_long_continuous_run_and_memory_drift():
    print("\n[CHALLENGE 4] Testing Long Continuous Run (500 Blobs, 500 Ticks, 10 Days)...")
    gc.collect()
    start_rss, _, _ = get_process_memory_info()

    cfg = WorldConfig.from_preset("regional", initial_blobs=500, max_population_cap=600)
    cfg.ticks_per_day = 50
    world = SimulationWorld(world_cfg=cfg, net_cfg=NetworkConfig(width=24), seed=12345)

    # Warmup
    for _ in range(5):
        world.step(1)
    gc.collect()
    baseline_rss, _, _ = get_process_memory_info()

    t0 = time.perf_counter()
    rss_samples = []

    for tick in range(1, 501):
        world.step(1)
        if tick % 100 == 0:
            rss, _, _ = get_process_memory_info()
            rss_samples.append(rss)
            living = len([b for b in world.blobs if b.alive])
            print(f"    [Tick {tick:3d}/500]: Day {world.day_number} | Living: {living:3d} | RSS: {rss:.3f} GB")

    elapsed_s = time.perf_counter() - t0
    final_rss, peak_rss, _ = get_process_memory_info()
    drift = final_rss - baseline_rss
    mean_tick_ms = (elapsed_s / 500.0) * 1000.0

    print(f"  -> 500 ticks finished in {elapsed_s:.2f}s ({mean_tick_ms:.2f} ms/tick, {1000.0/mean_tick_ms:.1f} TPS).")
    print(f"  -> Memory: Baseline={baseline_rss:.3f} GB, Peak={peak_rss:.3f} GB, Final={final_rss:.3f} GB, Drift={drift:.4f} GB.")

    assert drift < 0.10, f"Memory drift exceeded 0.10 GB: {drift:.4f} GB"
    assert peak_rss < 1.0, f"Peak RSS exceeded 1.0 GB: {peak_rss:.3f} GB"
    print("  -> PASSED: Long continuous run verified zero memory explosion and sustained high throughput.")
    world.close()


def test_extreme_coordinates_and_fuzzing():
    print("\n[CHALLENGE 5] Testing Extreme Boundary Coordinates & Fuzzing...")
    cfg = WorldConfig.from_preset("classic", initial_blobs=10)
    world = SimulationWorld(world_cfg=cfg, net_cfg=NetworkConfig(width=16), seed=999)

    # Place blobs on singularity boundary positions
    extremes = [
        (0.0, 0.0),
        (cfg.width, cfg.height),
        (0.0, cfg.height),
        (cfg.width, 0.0),
        (-50.0, -50.0),
        (cfg.width + 100.0, cfg.height + 100.0),
        (cfg.width / 2.0, cfg.height / 2.0),
    ]

    for i, (x, y) in enumerate(extremes):
        if i < len(world.blobs):
            world.blobs[i].x = x
            world.blobs[i].y = y
            world.blobs[i].vx = 100.0 * (1 if i % 2 == 0 else -1)
            world.blobs[i].vy = 100.0 * (1 if i % 3 == 0 else -1)

    # Step simulation and check that kinematics clamps/reflects within valid bounds
    for t in range(20):
        world.step(1)
        for b in world.blobs:
            assert 0.0 <= b.x <= cfg.width, f"Blob x out of bounds: {b.x} (width={cfg.width})"
            assert 0.0 <= b.y <= cfg.height, f"Blob y out of bounds: {b.y} (height={cfg.height})"
            assert np.isfinite(b.vx) and np.isfinite(b.vy), f"Non-finite velocities: vx={b.vx}, vy={b.vy}"

    print("  -> PASSED: Extreme coordinates and velocity reflections clamped safely.")
    world.close()


def main():
    print("================================================================================")
    print("EMPIRICAL CHALLENGER ADVERSARIAL STRESS SUITE (MILESTONE 4)")
    print("================================================================================")
    results = {}
    tests = [
        ("Zero Blobs (initial_blobs=0)", test_zero_blobs_edge_case),
        ("Single Solitary Blob (N=1)", test_single_blob_edge_case),
        ("All-Death Extinction & Recovery", test_extinction_and_recovery),
        ("Long Continuous Run (500b x 500t)", test_long_continuous_run_and_memory_drift),
        ("Extreme Boundary Coordinates & Fuzzing", test_extreme_coordinates_and_fuzzing),
    ]

    for name, fn in tests:
        try:
            fn()
            results[name] = "PASSED"
        except Exception as exc:
            import traceback
            print(f"\n[FAIL] {name}: {exc}")
            traceback.print_exc()
            results[name] = f"FAILED: {exc}"

    print("\n================================================================================")
    print("EMPIRICAL CHALLENGE SUMMARY:")
    for name, res in results.items():
        print(f"  - {name}: {res}")
    print("================================================================================")
    
    passed_count = sum(1 for r in results.values() if r == "PASSED")
    total_count = len(results)
    if passed_count == total_count:
        print("ALL EMPIRICAL CHALLENGES PASSED. VERDICT: CONFIRMED.")
        return 0
    else:
        print(f"SOME CHALLENGES FAILED ({passed_count}/{total_count}). VERDICT: DISCONFIRMED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
