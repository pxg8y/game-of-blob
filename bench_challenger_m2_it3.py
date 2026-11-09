"""
Independent Empirical Challenger Verification Script for Milestone 2 Iteration 3
Author: challenger_m2_it3_2
Date: 2026-11-08

Tasks:
1. Empirically verify 2D CSR Obstacle Grid Culling:
   - Benchmark query_nearest_foods_grid and compute_oasis_and_feelers_numba on Continental scale
     (2,100 blobs, 287 rocks, 72 swamps) with and without CSR culling.
   - Verify bitwise exact numerical parity (max_abs_diff == 0.0).
   - Adversarial boundary testing across 10,000 synthetic test points (cell borders, obstacle boundaries).
2. Empirically verify Real-Time Simulation Throughput:
   - 100-continuous-tick simulation with 2,100 blobs on Continental topography.
   - Per-tick execution times, regular tick mean, P95, midnight reset analysis, >30 TPS (<33.3 ms/tick).
3. Process Memory Verification:
   - Monitor RSS memory across the lifecycle and verify strictly < 4.0 GB.
"""

import time
import os
import sys
import math
import ctypes
from ctypes import wintypes
import numpy as np

from blob_sim.config import WorldConfig
from blob_sim.world import SimulationWorld
from blob_sim.spatial import (
    query_nearest_foods_grid,
    compute_oasis_and_feelers_numba,
    build_spatial_linked_grid,
    INV_SPATIAL_CELL,
)

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

def get_process_memory_mb():
    fn = ctypes.windll.psapi.GetProcessMemoryInfo
    fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
    fn.restype = wintypes.BOOL
    h = ctypes.windll.kernel32.GetCurrentProcess()
    pmc = _PROCESS_MEMORY_COUNTERS()
    pmc.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
    res = fn(h, ctypes.byref(pmc), pmc.cb)
    if not res:
        return 0.0
    return float(pmc.WorkingSetSize) / (1024.0 * 1024.0)


def print_separator(title=""):
    print("=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)

def verify_task1_csr_culling():
    print_separator("TASK 1: 2D CSR OBSTACLE GRID CULLING & BITWISE PARITY")

    cfg = WorldConfig.from_preset(
        "continent",
        initial_blobs=2100,
        max_population_cap=2500,
        daily_food_spawn=4800,
        ticks_per_day=40,
    )
    print(f"Initializing Continental World: width={cfg.width}, height={cfg.height}, initial_blobs={cfg.initial_blobs}...")
    world = SimulationWorld(world_cfg=cfg, seed=42)

    oasis_list = list(world.oases.values())
    n_oases = len(oasis_list)
    ox = np.fromiter((o.x for o in oasis_list), dtype=np.float32, count=n_oases)
    oy = np.fromiter((o.y for o in oasis_list), dtype=np.float32, count=n_oases)
    orad = np.fromiter((o.radius for o in oasis_list), dtype=np.float32, count=n_oases)
    ofert = np.fromiter((o.fertility for o in oasis_list), dtype=np.float32, count=n_oases)
    oring_bonus = np.fromiter(
        (1.22 if getattr(o, "ring", "outer") == "outer" else 0.85 for o in oasis_list),
        dtype=np.float32,
        count=n_oases,
    )

    n_blobs = len(world.blobs)
    n_rocks = len(world._rock_x1)
    n_swamps = len(world._swamp_x1)
    n_foods = len(world.foods)

    print(f"World initialized:")
    print(f"  Blobs:   {n_blobs}")
    print(f"  Rocks:   {n_rocks}")
    print(f"  Swamps:  {n_swamps}")
    print(f"  Oases:   {n_oases}")
    print(f"  Foods:   {n_foods}")
    print(f"  Rock CSR: offsets={len(world._rock_offsets)}, indices={len(world._rock_indices)}, grid={world._rock_gw}x{world._rock_gh}")
    print(f"  Swamp CSR: offsets={len(world._swamp_offsets)}, indices={len(world._swamp_indices)}, grid={world._swamp_gw}x{world._swamp_gh}")

    blob_xs = np.ascontiguousarray([b.x for b in world.blobs], dtype=np.float32)
    blob_ys = np.ascontiguousarray([b.y for b in world.blobs], dtype=np.float32)
    eff_visions = np.ascontiguousarray([b.vision_radius for b in world.blobs], dtype=np.float32)
    headings = np.ascontiguousarray([b.heading for b in world.blobs], dtype=np.float32)

    food_xs = np.fromiter((f.x for f in world.foods), dtype=np.float32, count=len(world.foods))
    food_ys = np.fromiter((f.y for f in world.foods), dtype=np.float32, count=len(world.foods))
    grid_w = max(1, int(world.world_cfg.width * INV_SPATIAL_CELL) + 2)
    grid_h = max(1, int(world.world_cfg.height * INV_SPATIAL_CELL) + 2)
    food_head, food_next = build_spatial_linked_grid(food_xs, food_ys, INV_SPATIAL_CELL, grid_w, grid_h)

    # 1. Warm-up calls to ensure Numba JIT compilation
    _ = query_nearest_foods_grid(
        blob_xs, blob_ys, eff_visions, food_xs, food_ys, food_head, food_next,
        INV_SPATIAL_CELL, grid_w, grid_h,
        world._rock_x1, world._rock_y1, world._rock_dx, world._rock_dy,
        world._rock_inv_len_sq, world._rock_r,
        world._rock_offsets, world._rock_indices, world._rock_inv_cell,
        world._rock_gw, world._rock_gh
    )
    _ = query_nearest_foods_grid(
        blob_xs, blob_ys, eff_visions, food_xs, food_ys, food_head, food_next,
        INV_SPATIAL_CELL, grid_w, grid_h,
        world._rock_x1, world._rock_y1, world._rock_dx, world._rock_dy,
        world._rock_inv_len_sq, world._rock_r,
        None, None, 0.0, 0, 0
    )

    _ = compute_oasis_and_feelers_numba(
        blob_xs, blob_ys, headings,
        ox, oy, orad, ofert, oring_bonus,
        world._rock_x1, world._rock_y1, world._rock_dx, world._rock_dy, world._rock_inv_len_sq, world._rock_r, world._rock_feeler_r2,
        world._swamp_x1, world._swamp_y1, world._swamp_dx, world._swamp_dy, world._swamp_inv_len_sq, world._swamp_r, world._swamp_feeler_r2,
        world._rock_offsets, world._rock_indices, world._rock_inv_cell, world._rock_gw, world._rock_gh,
        world._swamp_offsets, world._swamp_indices, world._swamp_inv_cell, world._swamp_gw, world._swamp_gh
    )
    _ = compute_oasis_and_feelers_numba(
        blob_xs, blob_ys, headings,
        ox, oy, orad, ofert, oring_bonus,
        world._rock_x1, world._rock_y1, world._rock_dx, world._rock_dy, world._rock_inv_len_sq, world._rock_r, world._rock_feeler_r2,
        world._swamp_x1, world._swamp_y1, world._swamp_dx, world._swamp_dy, world._swamp_inv_len_sq, world._swamp_r, world._swamp_feeler_r2,
        None, None, 0.0, 0, 0,
        None, None, 0.0, 0, 0
    )

    # 2. Benchmark query_nearest_foods_grid
    num_runs = 50
    t0 = time.perf_counter()
    for _ in range(num_runs):
        res_foods_csr = query_nearest_foods_grid(
            blob_xs, blob_ys, eff_visions, food_xs, food_ys, food_head, food_next,
            INV_SPATIAL_CELL, grid_w, grid_h,
            world._rock_x1, world._rock_y1, world._rock_dx, world._rock_dy,
            world._rock_inv_len_sq, world._rock_r,
            world._rock_offsets, world._rock_indices, world._rock_inv_cell,
            world._rock_gw, world._rock_gh
        )
    dur_foods_csr = (time.perf_counter() - t0) * 1000.0 / num_runs

    t0 = time.perf_counter()
    for _ in range(num_runs):
        res_foods_lin = query_nearest_foods_grid(
            blob_xs, blob_ys, eff_visions, food_xs, food_ys, food_head, food_next,
            INV_SPATIAL_CELL, grid_w, grid_h,
            world._rock_x1, world._rock_y1, world._rock_dx, world._rock_dy,
            world._rock_inv_len_sq, world._rock_r,
            None, None, 0.0, 0, 0
        )
    dur_foods_lin = (time.perf_counter() - t0) * 1000.0 / num_runs

    foods_max_diff = float(np.max(np.abs(res_foods_csr - res_foods_lin)))
    foods_bitwise_match = bool(np.array_equal(res_foods_csr, res_foods_lin) or foods_max_diff == 0.0)

    print("\n--- Benchmark: query_nearest_foods_grid (2,100 blobs, 287 rocks) ---")
    print(f"  With CSR Culling:   {dur_foods_csr:.4f} ms/call")
    print(f"  Without CSR (Scan): {dur_foods_lin:.4f} ms/call")
    print(f"  Speedup:            {dur_foods_lin / max(1e-6, dur_foods_csr):.2f}x")
    print(f"  Max Absolute Diff:  {foods_max_diff:.8e}")
    print(f"  Bitwise Exact:      {foods_bitwise_match}")

    # 3. Benchmark compute_oasis_and_feelers_numba
    t0 = time.perf_counter()
    for _ in range(num_runs):
        oasis_csr, feeler_csr = compute_oasis_and_feelers_numba(
            blob_xs, blob_ys, headings,
            ox, oy, orad, ofert, oring_bonus,
            world._rock_x1, world._rock_y1, world._rock_dx, world._rock_dy, world._rock_inv_len_sq, world._rock_r, world._rock_feeler_r2,
            world._swamp_x1, world._swamp_y1, world._swamp_dx, world._swamp_dy, world._swamp_inv_len_sq, world._swamp_r, world._swamp_feeler_r2,
            world._rock_offsets, world._rock_indices, world._rock_inv_cell, world._rock_gw, world._rock_gh,
            world._swamp_offsets, world._swamp_indices, world._swamp_inv_cell, world._swamp_gw, world._swamp_gh
        )
    dur_feelers_csr = (time.perf_counter() - t0) * 1000.0 / num_runs

    t0 = time.perf_counter()
    for _ in range(num_runs):
        oasis_lin, feeler_lin = compute_oasis_and_feelers_numba(
            blob_xs, blob_ys, headings,
            ox, oy, orad, ofert, oring_bonus,
            world._rock_x1, world._rock_y1, world._rock_dx, world._rock_dy, world._rock_inv_len_sq, world._rock_r, world._rock_feeler_r2,
            world._swamp_x1, world._swamp_y1, world._swamp_dx, world._swamp_dy, world._swamp_inv_len_sq, world._swamp_r, world._swamp_feeler_r2,
            None, None, 0.0, 0, 0,
            None, None, 0.0, 0, 0
        )
    dur_feelers_lin = (time.perf_counter() - t0) * 1000.0 / num_runs

    oasis_max_diff = float(np.max(np.abs(oasis_csr - oasis_lin)))
    feeler_max_diff = float(np.max(np.abs(feeler_csr - feeler_lin)))
    oasis_bitwise_match = bool(np.array_equal(oasis_csr, oasis_lin) or oasis_max_diff == 0.0)
    feeler_bitwise_match = bool(np.array_equal(feeler_csr, feeler_lin) or feeler_max_diff == 0.0)

    print("\n--- Benchmark: compute_oasis_and_feelers_numba (2,100 blobs, 287 rocks, 72 swamps) ---")
    print(f"  With CSR Culling:   {dur_feelers_csr:.4f} ms/call")
    print(f"  Without CSR (Scan): {dur_feelers_lin:.4f} ms/call")
    print(f"  Speedup:            {dur_feelers_lin / max(1e-6, dur_feelers_csr):.2f}x (reduction: {dur_feelers_lin - dur_feelers_csr:.2f} ms)")
    print(f"  Oasis Max Diff:     {oasis_max_diff:.8e} (Bitwise: {oasis_bitwise_match})")
    print(f"  Feeler Max Diff:    {feeler_max_diff:.8e} (Bitwise: {feeler_bitwise_match})")

    # 4. Adversarial Stress Test: 10,000 synthetic test points covering the entire arena
    print("\n--- Adversarial Stress: 10,000 Synthetic Blobs Across Boundary & Obstacle Zones ---")
    rng = np.random.RandomState(1337)
    n_stress = 10000

    # Mix of uniform random, cell-boundary adjacent, and obstacle-adjacent points
    stress_xs = np.empty(n_stress, dtype=np.float32)
    stress_ys = np.empty(n_stress, dtype=np.float32)

    # 4000 uniform random points
    stress_xs[:4000] = rng.uniform(0.0, cfg.width, size=4000).astype(np.float32)
    stress_ys[:4000] = rng.uniform(0.0, cfg.height, size=4000).astype(np.float32)

    # 3000 points specifically around cell boundaries (multiples of 256 +/- jitter)
    cell_boundaries_x = np.arange(0, cfg.width, 256.0, dtype=np.float32)
    cell_boundaries_y = np.arange(0, cfg.height, 256.0, dtype=np.float32)
    sel_bx = rng.choice(cell_boundaries_x, size=3000)
    sel_by = rng.choice(cell_boundaries_y, size=3000)
    stress_xs[4000:7000] = np.clip(sel_bx + rng.uniform(-5.0, 5.0, size=3000), 0.0, cfg.width).astype(np.float32)
    stress_ys[4000:7000] = np.clip(sel_by + rng.uniform(-5.0, 5.0, size=3000), 0.0, cfg.height).astype(np.float32)

    # 3000 points right next to rocks and swamps (within feeler reach <= 72.0)
    rock_indices_sample = rng.choice(len(world._rock_x1), size=1500)
    rx = world._rock_x1[rock_indices_sample] + rng.uniform(0.0, 1.0, size=1500).astype(np.float32) * world._rock_dx[rock_indices_sample]
    ry = world._rock_y1[rock_indices_sample] + rng.uniform(0.0, 1.0, size=1500).astype(np.float32) * world._rock_dy[rock_indices_sample]
    stress_xs[7000:8500] = np.clip(rx + rng.uniform(-60.0, 60.0, size=1500), 0.0, cfg.width).astype(np.float32)
    stress_ys[7000:8500] = np.clip(ry + rng.uniform(-60.0, 60.0, size=1500), 0.0, cfg.height).astype(np.float32)

    swamp_indices_sample = rng.choice(len(world._swamp_x1), size=1500)
    sx = world._swamp_x1[swamp_indices_sample] + rng.uniform(0.0, 1.0, size=1500).astype(np.float32) * world._swamp_dx[swamp_indices_sample]
    sy = world._swamp_y1[swamp_indices_sample] + rng.uniform(0.0, 1.0, size=1500).astype(np.float32) * world._swamp_dy[swamp_indices_sample]
    stress_xs[8500:] = np.clip(sx + rng.uniform(-50.0, 50.0, size=1500), 0.0, cfg.width).astype(np.float32)
    stress_ys[8500:] = np.clip(sy + rng.uniform(-50.0, 50.0, size=1500), 0.0, cfg.height).astype(np.float32)

    stress_headings = rng.uniform(-math.pi, math.pi, size=n_stress).astype(np.float32)
    stress_visions = rng.uniform(50.0, 180.0, size=n_stress).astype(np.float32)

    # Test food queries under 10k stress points
    adv_food_csr = query_nearest_foods_grid(
        stress_xs, stress_ys, stress_visions, food_xs, food_ys, food_head, food_next,
        INV_SPATIAL_CELL, grid_w, grid_h,
        world._rock_x1, world._rock_y1, world._rock_dx, world._rock_dy,
        world._rock_inv_len_sq, world._rock_r,
        world._rock_offsets, world._rock_indices, world._rock_inv_cell,
        world._rock_gw, world._rock_gh
    )
    adv_food_lin = query_nearest_foods_grid(
        stress_xs, stress_ys, stress_visions, food_xs, food_ys, food_head, food_next,
        INV_SPATIAL_CELL, grid_w, grid_h,
        world._rock_x1, world._rock_y1, world._rock_dx, world._rock_dy,
        world._rock_inv_len_sq, world._rock_r,
        None, None, 0.0, 0, 0
    )
    adv_food_max_diff = float(np.max(np.abs(adv_food_csr - adv_food_lin)))

    # Test feeler queries under 10k stress points
    _, adv_feeler_csr = compute_oasis_and_feelers_numba(
        stress_xs, stress_ys, stress_headings,
        ox, oy, orad, ofert, oring_bonus,
        world._rock_x1, world._rock_y1, world._rock_dx, world._rock_dy, world._rock_inv_len_sq, world._rock_r, world._rock_feeler_r2,
        world._swamp_x1, world._swamp_y1, world._swamp_dx, world._swamp_dy, world._swamp_inv_len_sq, world._swamp_r, world._swamp_feeler_r2,
        world._rock_offsets, world._rock_indices, world._rock_inv_cell, world._rock_gw, world._rock_gh,
        world._swamp_offsets, world._swamp_indices, world._swamp_inv_cell, world._swamp_gw, world._swamp_gh
    )
    _, adv_feeler_lin = compute_oasis_and_feelers_numba(
        stress_xs, stress_ys, stress_headings,
        ox, oy, orad, ofert, oring_bonus,
        world._rock_x1, world._rock_y1, world._rock_dx, world._rock_dy, world._rock_inv_len_sq, world._rock_r, world._rock_feeler_r2,
        world._swamp_x1, world._swamp_y1, world._swamp_dx, world._swamp_dy, world._swamp_inv_len_sq, world._swamp_r, world._swamp_feeler_r2,
        None, None, 0.0, 0, 0,
        None, None, 0.0, 0, 0
    )
    adv_feeler_max_diff = float(np.max(np.abs(adv_feeler_csr - adv_feeler_lin)))

    print(f"  10k Stress Points Food Query Max Diff:   {adv_food_max_diff:.8e} (Bitwise: {adv_food_max_diff == 0.0})")
    print(f"  10k Stress Points Feeler Query Max Diff: {adv_feeler_max_diff:.8e} (Bitwise: {adv_feeler_max_diff == 0.0})")

    assert foods_max_diff == 0.0, f"query_nearest_foods_grid diverged: diff={foods_max_diff}"
    assert feeler_max_diff == 0.0, f"compute_oasis_and_feelers_numba diverged: diff={feeler_max_diff}"
    assert adv_food_max_diff == 0.0, f"10k adversarial food query diverged: diff={adv_food_max_diff}"
    assert adv_feeler_max_diff == 0.0, f"10k adversarial feeler query diverged: diff={adv_feeler_max_diff}"
    print("Task 1 Check: PASSED with 100% Bitwise Exact Numerical Parity (max_abs_diff == 0.0)!")
    return {
        "dur_foods_csr": dur_foods_csr,
        "dur_foods_lin": dur_foods_lin,
        "dur_feelers_csr": dur_feelers_csr,
        "dur_feelers_lin": dur_feelers_lin,
        "foods_max_diff": foods_max_diff,
        "feeler_max_diff": feeler_max_diff,
        "adv_food_max_diff": adv_food_max_diff,
        "adv_feeler_max_diff": adv_feeler_max_diff,
    }

def verify_task2_simulation_throughput():
    print_separator("TASK 2: 100-CONTINUOUS-TICK SIMULATION THROUGHPUT (CONTINENTAL SCALE)")

    cfg = WorldConfig.from_preset(
        "continent",
        initial_blobs=2100,
        max_population_cap=2500,
        daily_food_spawn=4800,
        ticks_per_day=40,
    )
    world = SimulationWorld(world_cfg=cfg, seed=42)
    mem_init_mb = get_process_memory_mb()
    print(f"Initial population: {len(world.blobs)} blobs, initial RSS: {mem_init_mb:.2f} MB")

    # Warmup 3 ticks
    print("Executing 3 warmup ticks for JIT compilation...")
    world.step(3)
    mem_after_warmup_mb = get_process_memory_mb()
    print(f"Warmup complete. RSS after warmup: {mem_after_warmup_mb:.2f} MB")

    total_ticks = 100
    tick_durations = []
    regular_tick_durations = []
    midnight_ticks = {}
    population_history = []
    mem_peak_mb = mem_after_warmup_mb

    print(f"Executing {total_ticks} continuous simulation ticks...")
    for t in range(1, total_ticks + 1):
        is_midnight = (world.tick_count % cfg.ticks_per_day == 0)
        t0 = time.perf_counter()
        world.step(1)
        dur_ms = (time.perf_counter() - t0) * 1000.0

        tick_durations.append(dur_ms)
        population_history.append(len(world.blobs))
        curr_mem = get_process_memory_mb()
        if curr_mem > mem_peak_mb:
            mem_peak_mb = curr_mem

        if is_midnight:
            midnight_ticks[world.tick_count] = dur_ms
            day_val = getattr(world, "day_count", getattr(world, "day", 0))
            print(f"  [Midnight Day Reset] Tick {world.tick_count} (Day {day_val}): {dur_ms:.2f} ms, Blobs: {len(world.blobs)}")
        else:
            regular_tick_durations.append(dur_ms)

    tick_durations = np.array(tick_durations)
    regular_tick_durations = np.array(regular_tick_durations)

    reg_mean = float(np.mean(regular_tick_durations))
    reg_median = float(np.median(regular_tick_durations))
    reg_p90 = float(np.percentile(regular_tick_durations, 90))
    reg_p95 = float(np.percentile(regular_tick_durations, 95))
    reg_p99 = float(np.percentile(regular_tick_durations, 99))
    reg_min = float(np.min(regular_tick_durations))
    reg_max = float(np.max(regular_tick_durations))
    reg_tps = 1000.0 / reg_mean

    all_mean = float(np.mean(tick_durations))
    all_tps = 1000.0 / all_mean

    mem_final_mb = get_process_memory_mb()

    print("\n--- 100-Tick Throughput Results ---")
    print(f"Total Ticks:            {total_ticks}")
    print(f"Regular Ticks Count:    {len(regular_tick_durations)}")
    print(f"Regular Tick Mean:      {reg_mean:.2f} ms ({reg_tps:.2f} TPS) [Target: < 33.3 ms / > 30 TPS]")
    print(f"Regular Tick Median:    {reg_median:.2f} ms")
    print(f"Regular Tick P90:       {reg_p90:.2f} ms")
    print(f"Regular Tick P95:       {reg_p95:.2f} ms")
    print(f"Regular Tick P99:       {reg_p99:.2f} ms")
    print(f"Regular Tick Min/Max:   {reg_min:.2f} ms / {reg_max:.2f} ms")
    print(f"All Ticks Mean (w/ Resets): {all_mean:.2f} ms ({all_tps:.2f} TPS)")
    print(f"Midnight Resets:        {midnight_ticks}")
    print(f"Final Population:       {len(world.blobs)} blobs")

    print("\n--- Memory Stability Results ---")
    print(f"Initial RSS:            {mem_init_mb:.2f} MB")
    print(f"Warmup RSS:             {mem_after_warmup_mb:.2f} MB")
    print(f"Final RSS:              {mem_final_mb:.2f} MB")
    print(f"Peak RSS:               {mem_peak_mb:.2f} MB (Contract limit: < 4096.0 MB)")

    assert reg_mean < 33.33, f"Regular tick mean {reg_mean:.2f} ms exceeds 33.33 ms (<30 TPS) threshold!"
    assert mem_peak_mb < 4096.0, f"Peak memory {mem_peak_mb:.2f} MB exceeds 4.0 GB contract!"

    print("Task 2 & 3 Simulation Check: PASSED (>30 TPS on regular ticks, RAM stable & compliant)!")
    return {
        "reg_mean": reg_mean,
        "reg_tps": reg_tps,
        "reg_median": reg_median,
        "reg_p95": reg_p95,
        "reg_p99": reg_p99,
        "reg_min": reg_min,
        "reg_max": reg_max,
        "all_mean": all_mean,
        "all_tps": all_tps,
        "midnight_ticks": midnight_ticks,
        "mem_init_mb": mem_init_mb,
        "mem_final_mb": mem_final_mb,
        "mem_peak_mb": mem_peak_mb,
        "population_end": len(world.blobs),
    }

if __name__ == "__main__":
    t_start = time.perf_counter()
    r1 = verify_task1_csr_culling()
    r2 = verify_task2_simulation_throughput()
    total_time = time.perf_counter() - t_start
    print_separator(f"ALL BENCHMARKS COMPLETED IN {total_time:.2f} SECONDS")
