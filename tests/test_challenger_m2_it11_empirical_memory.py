"""
Adversarial Empirical Verification Suite for Milestone 2 Iteration 11 Challenger:
Cumulative Memory RSS Drift and Dynamic Buffer Slice Allocation/Deallocation Integrity.

Verification criteria:
1. Memory RSS does not exhibit cumulative drift across ticks under continuous simulation.
2. Dynamic allocations and deallocations leave zero orphaned buffer slices during life cycles.
3. SimulationWorld.close() and BatchedBrainPool.close() leave zero pinned numpy buffers or brain weight views.
"""

import gc
import ctypes
from ctypes import wintypes
import unittest
import numpy as np

from blob_sim.config import WorldConfig, NetworkConfig
from blob_sim.world import SimulationWorld
from blob_sim.entities import Group
from blob_sim.neural import BatchedBrainPool, BlobBrain, INPUT_DIM, OUTPUT_DIM


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


class TestChallengerM2It11EmpiricalMemoryAndBufferIntegrity(unittest.TestCase):
    """Adversarial empirical tests for memory RSS drift and buffer slice lifecycle invariants."""

    def tearDown(self) -> None:
        if hasattr(self, "world") and self.world is not None:
            if hasattr(self.world, "close"):
                self.world.close()
            self.world = None
        gc.collect()

    def test_continuous_simulation_rss_cumulative_drift_and_stability(self) -> None:
        """Adversarial check: run 60 continuous ticks with 2,000 blobs.
        Verify that RSS memory after 3-cohort initialization exhibits zero cumulative drift
        (linear regression slope across ticks 10-60 is < 0.001 GB/tick and total drift < 0.20 GB).
        """
        gc.collect()
        baseline_rss = _get_process_rss_gb()

        cfg = WorldConfig.from_preset(
            "regional",
            initial_blobs=2000,
            ticks_per_day=20,
            daily_food_spawn=3500,
        )
        world = SimulationWorld(world_cfg=cfg, seed=777)
        self.world = world

        rss_measurements = []
        for tick in range(1, 61):
            world.step(1)
            rss = _get_process_rss_gb()
            rss_measurements.append(rss)

        # Evaluate drift post-warmup (ticks 10 to 60, covering 2+ complete day rollovers)
        warmup_idx = 10
        post_warmup_rss = rss_measurements[warmup_idx:]
        ticks = np.arange(len(post_warmup_rss))
        # Linear regression slope (GB / tick)
        slope, intercept = np.polyfit(ticks, post_warmup_rss, 1)

        final_rss = rss_measurements[-1]
        baseline_warmup_rss = rss_measurements[warmup_idx]
        total_drift = final_rss - baseline_warmup_rss
        peak_rss = max(rss_measurements)

        # 1. Total process RSS must remain strictly < 4.0 GB Project budget
        self.assertLess(peak_rss, 4.0, f"Peak process RSS exceeded 4.0 GB budget: {peak_rss:.3f} GB")

        # 2. Linear slope across post-warmup ticks must be <= 0.001 GB/tick (effectively zero drift)
        self.assertLess(
            slope,
            0.001,
            f"Cumulative memory drift detected! RSS slope is {slope:.6f} GB/tick across 50 post-warmup ticks",
        )

        # 3. Net drift across 50 ticks must be under 0.15 GB
        self.assertLess(
            total_drift,
            0.15,
            f"Net post-warmup RSS drift exceeded 0.15 GB: {total_drift:.3f} GB",
        )

    def test_dynamic_allocations_and_deallocations_zero_orphaned_buffer_slices(self) -> None:
        """Adversarial check: across continuous birth and death churn cycles,
        verify 1:1 bijection between living blobs, kinematics arrays, and neural pool slots.
        Verify zero orphaned slots and zero orphaned slice views.
        """
        cfg = WorldConfig.from_preset(
            "regional",
            initial_blobs=600,
            ticks_per_day=15,
            daily_food_spawn=1200,
            survival_food_need=1.0,
            reproduction_food_Multiplier=2.0,
        )
        world = SimulationWorld(world_cfg=cfg, seed=888)
        self.world = world

        # Run 46 ticks (3 full days + morning step) to generate turnover and execute morning rebind
        for _ in range(46):
            world.step(1)

        self.assertGreater(world.total_deaths_all_time, 0, "Turnover stress must cause deaths")
        self.assertGreater(world.total_births_all_time, 0, "Turnover stress must cause births")

        living = world.blobs
        n_living = len(living)
        pool = world._brain_pool

        # 1. Verify exact 1:1 bijection between world.blobs and living entities
        self.assertEqual(len(living), len(pool._bound_ids), "Living count mismatch with pool._bound_ids")
        self.assertEqual(len(living), len(pool._slot_map), "Living count mismatch with pool._slot_map")

        for idx, blob in enumerate(living):
            self.assertTrue(blob.alive, f"Dead blob {blob.id} found in living world.blobs list at index {idx}")
            self.assertEqual(blob._sim_idx, idx, f"Blob {blob.id} _sim_idx {blob._sim_idx} != {idx}")
            self.assertEqual(blob._kin_idx, idx, f"Blob {blob.id} _kin_idx {blob._kin_idx} != {idx}")

            # Check pool slot bijection
            brain_id = id(blob.brain)
            self.assertIn(brain_id, pool._slot_map, f"Blob {blob.id} brain ID not in pool._slot_map")
            self.assertEqual(pool._slot_map[brain_id], idx, f"Pool slot map for blob {blob.id} != {idx}")
            self.assertEqual(pool._bound_ids[idx], brain_id, f"Pool bound ID at {idx} != brain ID")
            self.assertIs(blob.brain._pool_ref, pool, f"Brain._pool_ref != pool")

            # Check kinematics buffer values match blob state
            self.assertAlmostEqual(world._kin_xs[idx], blob.x, places=5)
            self.assertAlmostEqual(world._kin_ys[idx], blob.y, places=5)
            self.assertAlmostEqual(world._kin_energies[idx], blob.energy, places=5)
            self.assertAlmostEqual(world._kin_reputations[idx], blob.reputation, places=5)

        # 2. Verify all unallocated slots up to capacity are unmapped
        for unused_slot in range(n_living, pool._capacity):
            self.assertNotIn(unused_slot, pool._slot_map.values(), f"Unused slot {unused_slot} has mapped brain!")

        # 3. Test deallocation cleanliness on world.close()
        world.close()
        self.world = None
        gc.collect()

        # Verify that all blob brains had their weights/biases emptied
        for blob in living:
            self.assertEqual(len(blob.brain.weights), 0, f"Blob {blob.id} weights were not cleared by close()")
            self.assertEqual(len(blob.brain.biases), 0, f"Blob {blob.id} biases were not cleared by close()")
            self.assertIsNone(blob.brain._pool_ref, f"Blob {blob.id} _pool_ref was not unlinked")
            self.assertIsNone(blob._kin_ref, f"Blob {blob.id} _kin_ref was not unlinked")

    def test_reputation_sync_into_kin_and_sensory_channel_25(self) -> None:
        """Adversarial check: verify that adjusting a blob's reputation updates _kin_reputations
        and propagates to observation channel 25 on subsequent simulation steps.
        """
        world = SimulationWorld(WorldConfig(initial_blobs=20), seed=42)
        self.world = world
        world.step(1)

        b = world.blobs[0]
        b.adjust_reputation(0.40)
        self.assertAlmostEqual(b.reputation, 0.40, places=3)

        # 1. Check in-place buffer synchronization
        self.assertAlmostEqual(
            float(world._kin_reputations[b._kin_idx]),
            b.reputation,
            places=3,
            msg=f"_kin_reputations buffer out of sync! got {world._kin_reputations[b._kin_idx]}, expected {b.reputation}",
        )

        # 2. Check neural sensory observation channel 25
        obs_captured = []
        orig_forward = world._brain_pool.forward_and_trace

        def mon_forward(brains, obs_batch, *args, **kwargs):
            obs_captured.append(obs_batch.copy())
            return orig_forward(brains, obs_batch, *args, **kwargs)

        world._brain_pool.forward_and_trace = mon_forward
        world.step(1)

        self.assertGreater(len(obs_captured), 0, "Forward pass was not executed")
        obs = obs_captured[0]
        obs_rep = float(obs[b._sim_idx, 25])
        self.assertAlmostEqual(
            obs_rep,
            b.reputation,
            places=3,
            msg=f"Observation channel 25 out of sync! got {obs_rep}, expected {b.reputation}",
        )

    def test_midnight_reputation_decay_synchronization_multi_day_normal_and_famine(self) -> None:
        """Adversarial check: across multiple midnight rollovers (normal and famine conditions),
        verify:
        1. abs(world._kin_reputations[idx] - np.float32(blob.reputation)) == 0.0 for all living blobs.
        2. Sensory channel 25 contains the exact decayed reputation value during morning forward pass.
        3. Group cached average reputation matches ground truth calculation.
        4. Dead blob snapshot in _snapshot_blob_kin_state captures the exact decayed reputation.
        """
        # =========================================================================
        # SCENARIO 1: Normal Conditions with Clan Formation (2 full day cycles)
        # =========================================================================
        cfg_normal = WorldConfig(initial_blobs=30, ticks_per_day=10, daily_food_spawn=500)
        world_n = SimulationWorld(world_cfg=cfg_normal, seed=1337)
        self.world = world_n

        grp_n = Group(id=999, name="ClanAlpha", color="#00ff00", home_id=1, member_ids=[])
        world_n.groups[grp_n.id] = grp_n
        for i, b in enumerate(world_n.blobs[:15]):
            b.group_id = grp_n.id
            b.reputation = round(0.8 - 0.1 * i, 3)
        world_n._recompute_group_reputation(grp_n)

        obs_log_n = []
        orig_fwd_n = world_n._brain_pool.forward_and_trace

        def mon_fwd_n(brains, obs_batch, *args, **kwargs):
            reps = {b._sim_idx: b.reputation for b in world_n.blobs if b.alive}
            obs_log_n.append((world_n.tick_count, obs_batch.copy(), reps))
            return orig_fwd_n(brains, obs_batch, *args, **kwargs)

        world_n._brain_pool.forward_and_trace = mon_fwd_n

        for day in range(1, 3):
            # Step to midnight rollover
            for _ in range(10):
                world_n.step(1)

            midnight_tick = world_n.tick_count
            living = [b for b in world_n.blobs if b.alive]
            self.assertGreater(len(living), 0, f"Day {day} must have living blobs")

            # 1. Kin reputations sync invariant: abs(kin_rep - np.float32(blob.rep)) == 0.0
            for b in living:
                diff = abs(world_n._kin_reputations[b._kin_idx] - np.float32(b.reputation))
                self.assertEqual(
                    diff,
                    0.0,
                    f"Day {day} (tick {midnight_tick}) desync for blob {b.id}: kin={world_n._kin_reputations[b._kin_idx]}, rep={b.reputation}",
                )

            # 3. Group cached average reputation matches ground truth calculation
            living_grp = [b for b in living if b.group_id == grp_n.id]
            if living_grp:
                gt_avg = round(sum(b.reputation for b in living_grp) / len(living_grp), 3)
                self.assertAlmostEqual(
                    grp_n.avg_reputation,
                    gt_avg,
                    places=3,
                    msg=f"Day {day} cached grp.avg_reputation mismatch! cached={grp_n.avg_reputation}, gt={gt_avg}",
                )
                self.assertAlmostEqual(
                    world_n._get_group_avg_reputation(grp_n),
                    gt_avg,
                    places=3,
                    msg=f"Day {day} queried grp.avg_reputation mismatch! queried={world_n._get_group_avg_reputation(grp_n)}, gt={gt_avg}",
                )

            # Step 1 morning tick to verify sensory channel 25
            world_n.step(1)
            fwd_tick, obs, reps_at_fwd = obs_log_n[-1]
            # 2. Sensory channel 25 contains exact decayed reputation
            for b in world_n.blobs:
                if b.alive and b._sim_idx in reps_at_fwd:
                    obs_val = obs[b._sim_idx, 25]
                    exp_val = np.float32(reps_at_fwd[b._sim_idx])
                    self.assertEqual(
                        obs_val,
                        exp_val,
                        f"Sensory channel 25 out of sync on tick {fwd_tick} for blob {b.id}: got {obs_val}, expected {exp_val}",
                    )

        # 4. Dead blob snapshot in _snapshot_blob_kin_state captures exact decayed reputation
        target_n = [b for b in world_n.blobs if b.alive and b.reputation != 0.0][0]
        exp_decayed_rep = target_n.reputation
        target_n.alive = False
        world_n._compact_living_blobs()
        self.assertAlmostEqual(
            target_n.reputation,
            exp_decayed_rep,
            places=3,
            msg=f"Dead blob snapshot reverted decayed reputation! got {target_n.reputation}, expected {exp_decayed_rep}",
        )

        world_n.close()
        self.world = None

        # =========================================================================
        # SCENARIO 2: Multi-day Famine Conditions (3 full day cycles = 30 ticks)
        # =========================================================================
        cfg_famine = WorldConfig(initial_blobs=30, ticks_per_day=10, daily_food_spawn=0)
        world_f = SimulationWorld(world_cfg=cfg_famine, seed=777)
        self.world = world_f

        grp_f = Group(id=888, name="ClanFamine", color="#ff0000", home_id=1, member_ids=[])
        world_f.groups[grp_f.id] = grp_f
        for i, b in enumerate(world_f.blobs[:20]):
            b.group_id = grp_f.id
            b.reputation = round(0.75 - 0.08 * i, 3)
        world_f._recompute_group_reputation(grp_f)

        obs_log_f = []
        orig_fwd_f = world_f._brain_pool.forward_and_trace

        def mon_fwd_f(brains, obs_batch, *args, **kwargs):
            reps = {b._sim_idx: b.reputation for b in world_f.blobs if b.alive}
            obs_log_f.append((world_f.tick_count, obs_batch.copy(), reps))
            return orig_fwd_f(brains, obs_batch, *args, **kwargs)

        world_f._brain_pool.forward_and_trace = mon_fwd_f

        for day in range(1, 4):
            for _ in range(10):
                world_f.step(1)

            midnight_tick = world_f.tick_count
            living = [b for b in world_f.blobs if b.alive]
            self.assertGreater(len(living), 0, f"Famine Day {day} must have living blobs")

            # 1. Kin reputations sync invariant under famine
            for b in living:
                diff = abs(world_f._kin_reputations[b._kin_idx] - np.float32(b.reputation))
                self.assertEqual(
                    diff,
                    0.0,
                    f"Famine Day {day} (tick {midnight_tick}) desync for blob {b.id}: kin={world_f._kin_reputations[b._kin_idx]}, rep={b.reputation}",
                )

            # 3. Group cached average reputation matches ground truth calculation
            living_grp = [b for b in living if b.group_id == grp_f.id]
            if living_grp:
                gt_avg = round(sum(b.reputation for b in living_grp) / len(living_grp), 3)
                self.assertAlmostEqual(
                    grp_f.avg_reputation,
                    gt_avg,
                    places=3,
                    msg=f"Famine Day {day} cached grp.avg_reputation mismatch! cached={grp_f.avg_reputation}, gt={gt_avg}",
                )
                self.assertAlmostEqual(
                    world_f._get_group_avg_reputation(grp_f),
                    gt_avg,
                    places=3,
                    msg=f"Famine Day {day} queried grp.avg_reputation mismatch! queried={world_f._get_group_avg_reputation(grp_f)}, gt={gt_avg}",
                )

            # Step 1 morning tick to verify sensory channel 25
            world_f.step(1)
            fwd_tick, obs, reps_at_fwd = obs_log_f[-1]
            # 2. Sensory channel 25 contains exact decayed reputation
            for b in world_f.blobs:
                if b.alive and b._sim_idx in reps_at_fwd:
                    obs_val = obs[b._sim_idx, 25]
                    exp_val = np.float32(reps_at_fwd[b._sim_idx])
                    self.assertEqual(
                        obs_val,
                        exp_val,
                        f"Famine sensory channel 25 out of sync on tick {fwd_tick} for blob {b.id}: got {obs_val}, expected {exp_val}",
                    )

        # 4. Dead blob snapshot in _snapshot_blob_kin_state under famine
        target_f = [b for b in world_f.blobs if b.alive and b.reputation != 0.0][0]
        exp_decayed_rep_f = target_f.reputation
        target_f.alive = False
        world_f._compact_living_blobs()
        self.assertAlmostEqual(
            target_f.reputation,
            exp_decayed_rep_f,
            places=3,
            msg=f"Famine dead blob snapshot reverted decayed reputation! got {target_f.reputation}, expected {exp_decayed_rep_f}",
        )

        world_f.close()
        self.world = None


if __name__ == "__main__":
    unittest.main()

