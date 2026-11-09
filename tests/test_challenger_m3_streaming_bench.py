"""Empirical benchmark and stress verification suite for Milestone 3 streaming features.

Authored by teamwork_challenger_m3_2.

Features tested:
- Feature 16: Binary buffer packing latency (<0.5 ms at Continental scale)
- Feature 15: Stationary food caching bandwidth savings (~65-75 KB/frame)
- Feature 17: Viewport culling payload reductions and boundary robustness
- Protocol invariants & bitflag packing precision under high scale
"""

import gc
import json
import struct
import time
import unittest
from typing import Any, Dict, List, Tuple

import numpy as np

from blob_sim.config import WorldConfig
from blob_sim.spatial import pack_binary_blobs_kernel, warmup_all_jit_kernels
from blob_sim.world import SimulationWorld


class TestChallengerM3EmpiricalBenchmarks(unittest.TestCase):
    """Empirical challenger test suite for Milestone 3 high-throughput streaming."""

    @classmethod
    def setUpClass(cls) -> None:
        """Pre-warm JIT compilation before benchmarking."""
        warmup_all_jit_kernels()

    def tearDown(self) -> None:
        """Clean up simulation instances after each test."""
        if hasattr(self, "world") and self.world is not None:
            if hasattr(self.world, "close"):
                self.world.close()
            self.world = None
        gc.collect()

    def test_continental_buffer_packing_latency(self) -> None:
        """Feature 16 Empirical Benchmark: Assert buffer packing latency is strictly <0.5 ms.
        
        Tests both:
        1. Continental scale (2,000 blobs, 5,400 foods)
        2. Maximum Continental scale (3,600 blobs, 6,210 foods)
        3. Super-Continental stress scale (5,000 blobs, 8,000 foods)
        """
        scales = [
            ("Continental-2000", 2000, 5400),
            ("Continental-3600", 3600, 6210),
            ("Stress-5000", 5000, 8000),
        ]

        for scale_name, n_blobs_target, n_foods_target in scales:
            with self.subTest(scale=scale_name):
                cfg = WorldConfig.from_preset("continent")
                cfg.initial_blobs = n_blobs_target
                cfg.daily_food_spawn = n_foods_target
                self.world = SimulationWorld(world_cfg=cfg, seed=123)

                actual_blobs = len(self.world.blobs)
                actual_foods = len(self.world.foods)
                self.assertGreaterEqual(actual_blobs, n_blobs_target)
                self.assertGreaterEqual(actual_foods, n_foods_target)

                # Warmup iterations
                for _ in range(10):
                    _ = self.world.get_ws_state_and_binary(include_topology=False, include_foods=False)

                # 1. Benchmark isolated JIT packing kernel
                out_buf = np.empty((actual_blobs, 8), dtype=np.float32)
                kernel_times_ms: List[float] = []
                for _ in range(100):
                    t0 = time.perf_counter()
                    rep_sum, outlaws, honorable = pack_binary_blobs_kernel(
                        out_buf,
                        self.world._kin_xs,
                        self.world._kin_ys,
                        self.world._kin_headings,
                        self.world._kin_body_radii,
                        self.world._kin_foods_today,
                        self.world._kin_ids,
                        self.world._kin_gids,
                        self.world._group_rgb_table,
                        self.world._kin_gene_rgbs,
                        self.world._kin_is_at_homes,
                        self.world._kin_in_swamps,
                        self.world._kin_surplus_foods,
                        self.world._kin_sprints,
                        self.world._kin_reputations,
                        actual_blobs,
                    )
                    t1 = time.perf_counter()
                    kernel_times_ms.append((t1 - t0) * 1000.0)

                kernel_mean = float(np.mean(kernel_times_ms))
                kernel_p95 = float(np.percentile(kernel_times_ms, 95))
                kernel_p99 = float(np.percentile(kernel_times_ms, 99))
                kernel_min = float(np.min(kernel_times_ms))
                kernel_max = float(np.max(kernel_times_ms))

                # 2. Benchmark complete binary serialization pipeline in SimulationWorld
                # Delta frame (blobs only, food cached)
                delta_ser_times_ms: List[float] = []
                for _ in range(100):
                    t0 = time.perf_counter()
                    _, bin_payload = self.world.get_ws_state_and_binary(
                        include_topology=False,
                        include_foods=False,
                    )
                    t1 = time.perf_counter()
                    delta_ser_times_ms.append((t1 - t0) * 1000.0)

                delta_mean = float(np.mean(delta_ser_times_ms))
                delta_p95 = float(np.percentile(delta_ser_times_ms, 95))
                delta_p99 = float(np.percentile(delta_ser_times_ms, 99))
                delta_max = float(np.max(delta_ser_times_ms))

                # Verify payload structure
                magic, packed_blobs, packed_foods, tick = struct.unpack_from("<IIII", bin_payload, 0)
                self.assertEqual(magic, 0x424C4F42)
                self.assertEqual(packed_blobs, actual_blobs)
                self.assertEqual(packed_foods, 0)
                self.assertEqual(len(bin_payload), 16 + actual_blobs * 32)

                print(f"\n--- Benchmark [{scale_name}] ({actual_blobs} blobs, {actual_foods} foods) ---")
                print(f"JIT Kernel Packing Latency: Mean={kernel_mean:.4f} ms, P95={kernel_p95:.4f} ms, P99={kernel_p99:.4f} ms (Min={kernel_min:.4f} ms, Max={kernel_max:.4f} ms)")
                print(f"Full Delta Serialization: Mean={delta_mean:.4f} ms, P95={delta_p95:.4f} ms, P99={delta_p99:.4f} ms, Max={delta_max:.4f} ms")

                # EMPIRICAL ASSERTION: Buffer packing latency must strictly be <0.5 ms
                self.assertLess(
                    kernel_mean,
                    0.5,
                    f"Kernel packing latency mean {kernel_mean:.4f} ms exceeded 0.5 ms threshold at {scale_name}"
                )
                self.assertLess(
                    kernel_p95,
                    0.5,
                    f"Kernel packing latency P95 {kernel_p95:.4f} ms exceeded 0.5 ms threshold at {scale_name}"
                )

    def test_stationary_food_caching_bandwidth_savings(self) -> None:
        """Feature 15 Empirical Benchmark: Measure bandwidth reduction from stationary food caching."""
        cfg = WorldConfig.from_preset("continent")
        cfg.initial_blobs = 3600
        cfg.daily_food_spawn = 6200
        self.world = SimulationWorld(world_cfg=cfg, seed=42)

        n_blobs = len(self.world.blobs)
        n_foods = len(self.world.foods)

        # 1. Full Frame (Tier 1: Connect or Daily Food Invalidation)
        meta_full, bin_full = self.world.get_ws_state_and_binary(
            include_topology=True,
            include_foods=True,
        )
        magic_f, blobs_f, foods_f, tick_f = struct.unpack_from("<IIII", bin_full, 0)
        self.assertEqual(magic_f, 0x424C4F42)
        self.assertEqual(blobs_f, n_blobs)
        self.assertEqual(foods_f, n_foods)
        expected_full_bin_len = 16 + n_blobs * 32 + n_foods * 12
        self.assertEqual(len(bin_full), expected_full_bin_len)

        # 2. Delta Frame (Tier 2: Stationary Food Caching Active)
        meta_delta, bin_delta = self.world.get_ws_state_and_binary(
            include_topology=False,
            include_foods=False,
        )
        magic_d, blobs_d, foods_d, tick_d = struct.unpack_from("<IIII", bin_delta, 0)
        self.assertEqual(magic_d, 0x424C4F42)
        self.assertEqual(blobs_d, n_blobs)
        self.assertEqual(foods_d, 0)
        expected_delta_bin_len = 16 + n_blobs * 32
        self.assertEqual(len(bin_delta), expected_delta_bin_len)

        # Bandwidth Savings Calculation
        saved_bytes_per_frame = len(bin_full) - len(bin_delta)
        expected_saved_bytes = n_foods * 12
        self.assertEqual(saved_bytes_per_frame, expected_saved_bytes)

        kb_saved = saved_bytes_per_frame / 1024.0
        mb_per_sec_saved_at_30fps = (saved_bytes_per_frame * 30) / (1024.0 * 1024.0)

        # Delta savings percentage on binary payload
        pct_savings = (saved_bytes_per_frame / len(bin_full)) * 100.0

        print(f"\n--- Stationary Food Caching Benchmark ({n_blobs} blobs, {n_foods} foods) ---")
        print(f"Full binary payload size:  {len(bin_full):,} bytes ({len(bin_full)/1024.0:.1f} KB)")
        print(f"Delta binary payload size: {len(bin_delta):,} bytes ({len(bin_delta)/1024.0:.1f} KB)")
        print(f"Bytes saved per frame:     {saved_bytes_per_frame:,} bytes ({kb_saved:.1f} KB)")
        print(f"Payload reduction:         {pct_savings:.2f}%")
        print(f"Bandwidth saved @ 30 FPS:  {mb_per_sec_saved_at_30fps:.2f} MB/s per connected client")

        # ASSERTIONS
        # Continental scale has ~5,400 to 7,200 foods -> ~64.8 KB to ~85 KB saved per frame
        self.assertGreaterEqual(kb_saved, 60.0)
        self.assertLessEqual(kb_saved, 95.0)
        self.assertGreaterEqual(pct_savings, 35.0)  # >35% binary buffer bandwidth reduction

    def test_viewport_culling_payload_reduction(self) -> None:
        """Feature 17 Empirical Benchmark: Assert client viewport culling reduces payload proportionally."""
        cfg = WorldConfig.from_preset("continent")
        cfg.initial_blobs = 3600
        cfg.daily_food_spawn = 6200
        self.world = SimulationWorld(world_cfg=cfg, seed=999)

        total_blobs = len(self.world.blobs)
        full_meta, full_bin = self.world.get_ws_state_and_binary(
            include_topology=False,
            include_foods=False,
            viewport=None,
        )
        _, full_blobs_packed, _, _ = struct.unpack_from("<IIII", full_bin, 0)
        self.assertEqual(full_blobs_packed, total_blobs)
        full_size = len(full_bin)

        # Find center of first starter cradle home
        h0 = list(self.world.homes.values())[0]

        # Test viewports of different zoom levels:
        # 1. Close zoom (Frustum: 1,000 x 1,000 centered on home 0) -> ~0.08% of arena area
        # 2. Medium zoom (Frustum: 4,000 x 4,000 centered on home 0) -> ~1.3% of arena area
        # 3. Macro view (Frustum: 20,000 x 15,000) -> 25% of arena area
        # 4. Out-of-bounds / Empty corner (Frustum: 39,500 to 40,000 x 29,500 to 30,000)
        test_cases = [
            ("Close Zoom (1k x 1k at Cradle 0)", (h0.x - 500.0, h0.y - 500.0, h0.x + 500.0, h0.y + 500.0)),
            ("Medium Zoom (4k x 4k at Cradle 0)", (h0.x - 2000.0, h0.y - 2000.0, h0.x + 2000.0, h0.y + 2000.0)),
            ("Macro Zoom (20k x 15k)", (10000.0, 7500.0, 30000.0, 22500.0)),
            ("Empty Outer Corner (500 x 500)", (39500.0, 29500.0, 40000.0, 30000.0)),
        ]

        print(f"\n--- Viewport Culling Benchmark (Full: {total_blobs} blobs, {full_size:,} bytes) ---")
        for label, vp in test_cases:
            meta_culled, bin_culled = self.world.get_ws_state_and_binary(
                include_topology=False,
                include_foods=False,
                viewport=vp,
            )
            _, culled_blobs, _, _ = struct.unpack_from("<IIII", bin_culled, 0)
            culled_size = len(bin_culled)
            expected_culled_len = 16 + culled_blobs * 32
            self.assertEqual(culled_size, expected_culled_len)

            reduction_pct = ((full_size - culled_size) / full_size) * 100.0

            # CRITICAL INVARIANT: Worldwide population statistic must remain the full count
            self.assertEqual(meta_culled["stats"]["population"], total_blobs)

            print(f"[{label}] Blobs visible: {culled_blobs}/{total_blobs} ({culled_blobs/total_blobs*100:.1f}%) | Payload: {culled_size:,} bytes | Reduction: {reduction_pct:.2f}%")

            if label.startswith("Close Zoom"):
                # A close zoom centered on cradle 0 contains a focused subset (<15% of total blobs)
                self.assertGreater(culled_blobs, 0)
                self.assertLess(culled_blobs, total_blobs * 0.20)
                self.assertGreater(reduction_pct, 80.0)

            if label.startswith("Empty Outer Corner"):
                # Outer empty corner should have 0 blobs
                self.assertEqual(culled_blobs, 0)
                self.assertGreater(reduction_pct, 99.0)



    def test_binary_unpack_roundtrip_precision_and_bitflags(self) -> None:
        """Stress verification: Assert roundtrip unpack matches raw world float arrays exactly."""
        cfg = WorldConfig.from_preset("continent")
        cfg.initial_blobs = 1000
        cfg.daily_food_spawn = 2000
        self.world = SimulationWorld(world_cfg=cfg, seed=777)

        # Step 3 ticks to create varied movement, headings, and metabolism
        for _ in range(3):
            self.world.step()

        n_blobs = len(self.world.blobs)
        n_foods = len(self.world.foods)

        _, bin_payload = self.world.get_ws_state_and_binary(
            include_topology=True,
            include_foods=True,
        )

        magic, packed_blobs, packed_foods, tick = struct.unpack_from("<IIII", bin_payload, 0)
        self.assertEqual(magic, 0x424C4F42)
        self.assertEqual(packed_blobs, n_blobs)
        self.assertEqual(packed_foods, n_foods)

        # Unpack blobs
        blob_bytes = bin_payload[16 : 16 + n_blobs * 32]
        blob_array = np.frombuffer(blob_bytes, dtype=np.float32).reshape((n_blobs, 8))

        # Compare float32 arrays
        np.testing.assert_allclose(
            blob_array[:, 0],
            self.world._kin_xs[:n_blobs].astype(np.float32),
            rtol=1e-5,
            err_msg="Packed X coordinates do not match world kinematic arrays",
        )
        np.testing.assert_allclose(
            blob_array[:, 1],
            self.world._kin_ys[:n_blobs].astype(np.float32),
            rtol=1e-5,
            err_msg="Packed Y coordinates do not match world kinematic arrays",
        )
        np.testing.assert_allclose(
            blob_array[:, 2],
            self.world._kin_headings[:n_blobs],
            rtol=1e-5,
            err_msg="Packed headings do not match",
        )
        np.testing.assert_allclose(
            blob_array[:, 3],
            self.world._kin_body_radii[:n_blobs],
            rtol=1e-5,
            err_msg="Packed body radii do not match",
        )
        np.testing.assert_allclose(
            blob_array[:, 4],
            self.world._kin_foods_today[:n_blobs],
            rtol=1e-5,
            err_msg="Packed foods_today do not match",
        )
        np.testing.assert_allclose(
            blob_array[:, 5],
            self.world._kin_ids[:n_blobs],
            rtol=1e-5,
            err_msg="Packed blob IDs do not match",
        )

        # Verify bitflags decoding
        flags = blob_array[:, 7].astype(np.int32)
        for i in range(min(50, n_blobs)):
            fl = flags[i]
            # Bit 0: is_at_home
            self.assertEqual(bool(fl & 1), bool(self.world._kin_is_at_homes[i]))
            # Bit 1: in_swamp
            self.assertEqual(bool(fl & 2), bool(self.world._kin_in_swamps[i]))
            # Bit 2: surplus_food (>0.05)
            self.assertEqual(bool(fl & 4), bool(self.world._kin_surplus_foods[i] > 0.05))
            # Bit 3: sprint (>0.35)
            self.assertEqual(bool(fl & 8), bool(self.world._kin_sprints[i] > 0.35))
            # Bit 4: group_member (gid >= 0)
            self.assertEqual(bool(fl & 16), bool(self.world._kin_gids[i] >= 0))
            # Bit 5: outlaw (rep <= -0.35)
            self.assertEqual(bool(fl & 32), bool(self.world._kin_reputations[i] <= -0.35))
            # Bit 6: honorable (rep >= 0.35)
            self.assertEqual(bool(fl & 64), bool(self.world._kin_reputations[i] >= 0.35))

        # Unpack foods
        food_bytes = bin_payload[16 + n_blobs * 32 :]
        food_array = np.frombuffer(food_bytes, dtype=np.float32).reshape((n_foods, 3))
        np.testing.assert_allclose(
            food_array[:, 0],
            self.world._food_xs[:n_foods].astype(np.float32),
            rtol=1e-5,
            err_msg="Packed food X coords do not match",
        )
        np.testing.assert_allclose(
            food_array[:, 1],
            self.world._food_ys[:n_foods].astype(np.float32),
            rtol=1e-5,
            err_msg="Packed food Y coords do not match",
        )

    def test_empty_and_boundary_edge_cases(self) -> None:
        """Adversarial stress testing: Empty world, zero blobs, out-of-bounds viewports."""
        cfg = WorldConfig.from_preset("continent")
        cfg.initial_blobs = 0
        cfg.daily_food_spawn = 0
        self.world = SimulationWorld(world_cfg=cfg, seed=1)

        # 1. Empty blobs with food caching disabled (delta mode: 0 blobs, 0 foods)
        meta_delta, bin_delta = self.world.get_ws_state_and_binary(include_foods=False)
        self.assertEqual(len(bin_delta), 16)
        magic, n_b, n_f, _ = struct.unpack_from("<IIII", bin_delta, 0)
        self.assertEqual(magic, 0x424C4F42)
        self.assertEqual(n_b, 0)
        self.assertEqual(n_f, 0)

        # 2. Total world extinction: 0 blobs, 0 foods
        self.world.foods.clear()
        self.world._sync_food_arrays()
        meta_ext, bin_ext = self.world.get_ws_state_and_binary(include_foods=True)
        self.assertEqual(len(bin_ext), 16)
        magic_ext, n_b_ext, n_f_ext, _ = struct.unpack_from("<IIII", bin_ext, 0)
        self.assertEqual(magic_ext, 0x424C4F42)
        self.assertEqual(n_b_ext, 0)
        self.assertEqual(n_f_ext, 0)

        # 3. Viewport with min > max (inverted coordinates)
        cfg.initial_blobs = 100
        self.world = SimulationWorld(world_cfg=cfg, seed=2)
        meta_inv, bin_inv = self.world.get_ws_state_and_binary(
            include_foods=False,
            viewport=(5000.0, 5000.0, 1000.0, 1000.0),
        )
        _, n_b_inv, _, _ = struct.unpack_from("<IIII", bin_inv, 0)
        # Inverted coords should cull to 0 without throwing an exception
        self.assertEqual(n_b_inv, 0)
        self.assertEqual(len(bin_inv), 16)


if __name__ == "__main__":
    unittest.main()
