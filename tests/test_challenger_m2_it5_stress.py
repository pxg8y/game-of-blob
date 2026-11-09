"""Challenger Empirical Stress Verification Suite for Milestone 2 Iteration 5.

Adversarial stress testing of:
1. Initial Population Boundary: N=1 blob continuous multi-tick survival, day/night cycles,
   mortality to N=0, extinction self-healing respawn, and post-respawn stability.
2. Population Bottleneck: Abrupt mass culling down to N=1 survivor at mid-day, night, and
   midnight rollover; repeated oscillation cycles between high population and single survivor.
3. High Population Scale N=2,500 Blobs with Active Relocations and Conquests:
   Continuous simulation with concurrent per-tick relocations across homes of varying radii,
   faction house conquests, reputation mutations, and sensor/reputation oracle parity checks.
"""

import math
import time
import unittest
from typing import Dict, List, Optional, Set, Tuple
import gc
import numpy as np

from blob_sim.config import WorldConfig
from blob_sim.entities import Blob, Group, Home
from blob_sim.world import SimulationWorld


class TestChallengerM2It5PopulationBoundaryStress(unittest.TestCase):
    """Empirical adversarial stress testing of population boundaries and high scale dynamics."""

    def setUp(self) -> None:
        gc.collect()

    def tearDown(self) -> None:
        if hasattr(self, "world") and self.world is not None:
            if hasattr(self.world, "close"):
                self.world.close()
            self.world = None
        gc.collect()

    def _extract_ground_truth_observations(
        self,
        world: SimulationWorld,
        living_blobs: List[Blob],
        food_vecs: np.ndarray,
        peer_vecs: np.ndarray,
        oasis_vecs: np.ndarray,
        feeler_vecs: np.ndarray,
        tod: float,
        surv_target: float,
        repro_target: float,
        nearest_peers: Optional[List[Optional[Blob]]] = None,
    ) -> np.ndarray:
        """Independent ground-truth calculation for all 26 observation channels."""
        n = len(living_blobs)
        gt_obs = np.zeros((n, 26), dtype=np.float32)
        if n == 0:
            return gt_obs

        food_gain = np.float32(0.85 + 0.25 * world.prefs.pref_food)
        home_gain = np.float32(0.85 + 0.35 * world.prefs.pref_home)
        inv_900 = np.float32(1.0 / 900.0)
        inv_repro = np.float32(1.0 / repro_target)

        # Vectorized channels
        gt_obs[:, 0] = food_vecs[:, 0] * food_gain
        gt_obs[:, 1] = food_vecs[:, 1] * food_gain
        gt_obs[:, 2] = food_vecs[:, 2]
        gt_obs[:, 6:12] = peer_vecs
        gt_obs[:, 15] = np.float32(tod * 2.0 - 1.0)
        gt_obs[:, 16:19] = oasis_vecs
        gt_obs[:, 21:24] = feeler_vecs

        # Scalar ground-truth channels per blob
        for idx, blob in enumerate(living_blobs):
            h_obj = blob.home_obj
            if h_obj is None:
                h_obj = world.homes.get(blob.home_id)
            home_radius = h_obj.radius if h_obj is not None else 48.0

            dhx = blob.home_x - blob.x
            dhy = blob.home_y - blob.y
            dist_home_sq = dhx * dhx + dhy * dhy
            at_home = dist_home_sq <= (home_radius * home_radius)

            if dist_home_sq > 1e-8:
                dist_home = math.sqrt(dist_home_sq)
                inv_dh = 1.0 / dist_home
                raw_dx_home = dhx * inv_dh
                raw_dy_home = dhy * inv_dh
            else:
                dist_home = 0.0
                raw_dx_home = 0.0
                raw_dy_home = 0.0

            if at_home:
                gt_obs[idx, 3] = 0.0
                gt_obs[idx, 4] = 0.0
            else:
                gt_obs[idx, 3] = np.float32(raw_dx_home * home_gain)
                gt_obs[idx, 4] = np.float32(raw_dy_home * home_gain)

            dh_norm = dist_home * inv_900
            gt_obs[idx, 5] = np.float32(0.0 if dh_norm >= 1.0 else (1.0 - dh_norm))

            sat = blob.food_today * inv_repro
            gt_obs[idx, 12] = np.float32(1.5 if sat > 1.5 else sat)
            gt_obs[idx, 13] = np.float32((blob.energy / (85.0 + 15.0 * blob.genes.size)) * 2.0 - 1.0)
            gt_obs[idx, 14] = np.float32(math.tanh(blob.money * 0.05))

            # Spatial memory (channels 19, 20)
            if blob.memory_strength > 0.02:
                dmx = blob.memory_x - blob.x
                dmy = blob.memory_y - blob.y
                dist_mem = math.hypot(dmx, dmy)
                if dist_mem > 1e-4:
                    inv_dm = blob.memory_strength / dist_mem
                    gt_obs[idx, 19] = np.float32(dmx * inv_dm)
                    gt_obs[idx, 20] = np.float32(dmy * inv_dm)

            # Peer reputation (channel 24)
            if nearest_peers is not None and nearest_peers[idx] is not None:
                gt_obs[idx, 24] = np.float32(nearest_peers[idx].reputation)

            # Own reputation (channel 25)
            gt_obs[idx, 25] = np.float32(blob.reputation)

        return gt_obs

    def _unoptimized_group_avg_reputation(
        self, world: SimulationWorld, grp: Group, exclude_blob_id: Optional[int] = None
    ) -> float:
        """Unoptimized O(N) linear scan ground truth for group reputation."""
        living_reps = []
        seen = set()
        for mid in grp.member_ids:
            if mid in seen:
                continue
            seen.add(mid)
            b = world._blobs_by_id.get(mid)
            if b is not None and b.alive and (exclude_blob_id is None or b.id != exclude_blob_id):
                living_reps.append(b.reputation)
        if not living_reps:
            return 0.0
        return round(sum(living_reps) / len(living_reps), 3)

    # =========================================================================
    # 1. INITIAL POPULATION N=1 BLOB STRESS TESTS
    # =========================================================================

    def test_initial_population_n1_continuous_multi_tick_stability(self) -> None:
        """Adversarially test N=1 blob across 60 continuous ticks covering full day/night cycle.
        With min_population_floor=1, the population remains strictly solitary without unneeded respawn.
        """
        world = SimulationWorld(WorldConfig(initial_blobs=1, ticks_per_day=30, min_population_floor=1), seed=42)
        self.assertEqual(len(world.blobs), 1)

        # Give the solitary blob ample energy to survive full 60 ticks
        b = world.blobs[0]
        b.energy = 500.0

        for tick in range(1, 61):
            world.step(1)
            living = [x for x in world.blobs if x.alive]
            self.assertEqual(len(living), 1, f"Solitary blob population changed at tick {tick}: {len(living)}")

            # Verify spatial perception buffers under N=1
            self.assertTrue(hasattr(world, "_last_min_p_idxs"))
            self.assertEqual(len(world._last_min_p_idxs), 1)
            # Solitary blob has no peer, so nearest peer index must be -1
            self.assertEqual(world._last_min_p_idxs[0], -1)

            # Check coordinate bounds and finiteness
            b_curr = living[0]
            self.assertTrue(0.0 <= b_curr.x <= world.world_cfg.width)
            self.assertTrue(0.0 <= b_curr.y <= world.world_cfg.height)
            self.assertTrue(np.isfinite(b_curr.x))
            self.assertTrue(np.isfinite(b_curr.y))
            self.assertTrue(np.isfinite(b_curr.energy))

    def test_initial_population_n1_extinction_and_empty_world_stepping(self) -> None:
        """Adversarially test N=1 blob dying (N=0), triggering immediate extinction self-healing."""
        world = SimulationWorld(WorldConfig(initial_blobs=1, ticks_per_day=20, min_population_floor=14), seed=99)
        world.step(1)

        # Force mortality to N=0
        solitary = world.blobs[0]
        solitary.alive = False
        solitary.energy = 0.0

        # Step 1 tick: Simulation detects living_blobs is empty and immediately executes _respawn_from_hall_of_fame
        world.step(1)

        # Self-healing must have respawned blobs up to min_population_floor without crash or dirty buffer errors
        living_after = [b for b in world.blobs if b.alive]
        self.assertEqual(
            len(living_after),
            world.world_cfg.min_population_floor,
            "Extinction self-healing failed to restore population to min_population_floor",
        )

        # Step another 20 ticks to confirm the respawned population operates cleanly across midnight
        world.step(20)
        living_running = [b for b in world.blobs if b.alive]
        self.assertGreaterEqual(len(living_running), 1)
        for b in living_running:
            self.assertTrue(np.isfinite(b.x) and np.isfinite(b.y) and np.isfinite(b.energy))

    # =========================================================================
    # 2. POPULATION BOTTLENECK DOWN TO 1 SURVIVOR STRESS TESTS
    # =========================================================================

    def test_bottleneck_at_critical_diurnal_moments(self) -> None:
        """Cull population down to N=1 survivor at mid-day, night, and immediately before midnight."""
        world = SimulationWorld(
            WorldConfig(initial_blobs=100, ticks_per_day=30, min_population_floor=1), seed=123
        )

        # 1. Bottleneck at Mid-day (tick 10)
        world.step(10)
        survivor = world.blobs[0]
        survivor.energy = 300.0
        for b in world.blobs[1:]:
            b.alive = False
            b.energy = 0.0

        # Step through mid-day with 1 survivor
        world.step(5)
        living = [b for b in world.blobs if b.alive]
        self.assertEqual(len(living), 1)
        self.assertIs(living[0], survivor)

        # 2. Bottleneck during Night (tick 20)
        world.step(5)  # reach tick 20
        # If any reproduced, cull back to survivor
        for b in world.blobs:
            if b is not survivor:
                b.alive = False
                b.energy = 0.0
        world.step(5)  # step to tick 25 (nighttime)
        living = [b for b in world.blobs if b.alive]
        self.assertEqual(len(living), 1)

        # 3. Bottleneck immediately before midnight (tick 29)
        ticks_until_29 = 29 - (world.tick_count % 30)
        if ticks_until_29 > 0:
            world.step(ticks_until_29)

        for b in world.blobs:
            if b is not survivor:
                b.alive = False
                b.energy = 0.0

        # Step over midnight (ticks 30 -> 31)
        world.step(2)
        # Check that midnight accounting succeeded without crash
        self.assertGreater(world.day_number, 0)
        living_after = [b for b in world.blobs if b.alive]
        self.assertEqual(len(living_after), 1)
        self.assertIs(living_after[0], survivor)

    def test_repeated_bottleneck_oscillation_cycles(self) -> None:
        """Repeat 5 cycles of high population (N=60) -> culling to 1 survivor -> verification."""
        world = SimulationWorld(
            WorldConfig(initial_blobs=60, ticks_per_day=40, min_population_floor=1), seed=456
        )

        for cycle in range(5):
            world.step(3)

            living = [b for b in world.blobs if b.alive]
            self.assertGreater(len(living), 0)
            survivor = living[0]
            survivor.energy = 200.0

            # Cull all others
            for b in living[1:]:
                b.alive = False
                b.energy = 0.0

            world._compact_living_blobs()
            self.assertEqual(len(world.blobs), 1)

            # Step with solitary survivor
            world.step(2)
            self.assertTrue(survivor.alive)

            # Adversarially test sensor extraction on solitary survivor
            n = 1
            food_vecs = np.zeros((n, 3), np.float32)
            peer_vecs = np.zeros((n, 6), np.float32)
            oasis_vecs = np.zeros((n, 3), np.float32)
            feeler_vecs = np.zeros((n, 3), np.float32)

            obs_batch, _, _, _, _, _, _, _ = world._prepare_step_observations(
                [survivor], food_vecs, peer_vecs, oasis_vecs, feeler_vecs,
                0.5, 1.0, 2.0, 1, nearest_peers=[None]
            )

            gt_obs = self._extract_ground_truth_observations(
                world, [survivor], food_vecs, peer_vecs, oasis_vecs, feeler_vecs,
                0.5, 1.0, 2.0, nearest_peers=[None]
            )

            max_err = float(np.max(np.abs(obs_batch - gt_obs)))
            self.assertLessEqual(max_err, 1e-6, f"Cycle {cycle} sensor error {max_err} > 1e-6")

            # Spawn 20 new blobs for the next cycle
            for i in range(20):
                new_b = Blob(
                    blob_id=world._next_blob_id,
                    x=world.rng.uniform(100.0, 900.0),
                    y=world.rng.uniform(100.0, 900.0),
                    home_id=survivor.home_id,
                    home_x=survivor.home_x,
                    home_y=survivor.home_y,
                    brain=survivor.brain,
                    genes=survivor.genes,
                    group_id=survivor.group_id,
                )
                new_b.energy = 80.0
                world._next_blob_id += 1
                world._register_blob(new_b)

    # =========================================================================
    # 3. HIGH POPULATION SCALE N=2,500 BLOBS WITH ACTIVE RELOCATIONS & CONQUESTS
    # =========================================================================

    def test_scale_2500_blobs_with_continuous_relocations_and_conquests(self) -> None:
        """Continuous simulation of 2,500 blobs with concurrent per-tick relocations,
        conquests, reputation modifications, and zero sensor / reputation drift.
        """
        world = SimulationWorld(WorldConfig(initial_blobs=2500, ticks_per_day=50), seed=777)
        self.world = world
        living = [b for b in world.blobs if b.alive]
        self.assertEqual(len(living), 2500)

        # Ensure we have multiple homes and groups
        if len(world.groups) < 4:
            for g_i in range(10, 15):
                grp = Group(id=g_i, name=f"Faction_{g_i}", color=f"#{g_i:06x}", home_id=1, member_ids=[])
                world.groups[g_i] = grp
                # Assign some blobs to this faction
                start_idx = (g_i - 10) * 100
                for b in world.blobs[start_idx : start_idx + 100]:
                    b.group_id = g_i

        # Create additional homes with diverse radii to test relocation sensor sensitivity
        ref_home = list(world.homes.values())[0]
        h_radii_options = [38.0, 48.0, 56.0, 64.0, 72.0]
        extra_homes: List[Home] = []
        for h_i, r in enumerate(h_radii_options, start=100):
            new_h = Home(
                id=h_i,
                x=ref_home.x + float(h_i * 30),
                y=ref_home.y + float(h_i * 20),
                radius=r,
                max_residents=500,
                group_id=list(world.groups.keys())[h_i % len(world.groups)],
            )
            world.homes[h_i] = new_h
            extra_homes.append(new_h)
            # Add to group home_ids
            if new_h.group_id in world.groups:
                world.groups[new_h.group_id].home_ids.append(h_i)

        all_homes = list(world.homes.values())
        groups_list = list(world.groups.values())

        # Warm up JIT kernels with 3 ticks (covering all 3 cohorts)
        world.step(1)
        world.step(1)
        world.step(1)

        tick_durations: List[float] = []
        total_relocations = 0
        total_conquests = 0

        # Step 25 continuous ticks at 2,500 scale
        for step_idx in range(1, 26):
            living_current = [b for b in world.blobs if b.alive]

            # 1. Active per-tick relocations: relocate 40 random living blobs
            if len(living_current) >= 40:
                sample_reloc = world.rng.choice(living_current, size=40, replace=False)
                for b in sample_reloc:
                    target_home = all_homes[int(world.rng.integers(0, len(all_homes)))]
                    world._relocate_blob_home(b, target_home.id)
                    total_relocations += 1

            # 2. Active conquests: trigger 2 house conquests per tick
            for _ in range(2):
                h_target = all_homes[int(world.rng.integers(0, len(all_homes)))]
                new_owner_grp = groups_list[int(world.rng.integers(0, len(groups_list)))]
                old_gid = h_target.group_id
                if old_gid != new_owner_grp.id:
                    if old_gid in world.groups:
                        old_grp = world.groups[old_gid]
                        if h_target.id in old_grp.home_ids:
                            old_grp.home_ids.remove(h_target.id)
                        old_grp.rep_dirty = True
                    h_target.group_id = new_owner_grp.id
                    if h_target.id not in new_owner_grp.home_ids:
                        new_owner_grp.home_ids.append(h_target.id)
                    new_owner_grp.rep_dirty = True
                    # Annex current residents to new group
                    for r_id in list(h_target.resident_ids[:5]):
                        r_blob = world._blobs_by_id.get(r_id)
                        if r_blob is not None and r_blob.alive:
                            r_blob.group_id = new_owner_grp.id
                    total_conquests += 1

            # 3. Random reputation perturbations on 30 blobs
            if len(living_current) >= 30:
                sample_rep = world.rng.choice(living_current, size=30, replace=False)
                for b in sample_rep:
                    delta = float(world.rng.uniform(-0.2, 0.2))
                    b.adjust_reputation(delta)

            # Execute simulation tick and measure execution time
            t0 = time.perf_counter()
            world.step(1)
            t_tick = time.perf_counter() - t0
            tick_durations.append(t_tick)

            # Invariant checks every 5 ticks
            if step_idx % 5 == 0:
                survivors = [b for b in world.blobs if b.alive]
                self.assertGreater(len(survivors), 1800, "Population collapsed unexpectedly")

                # Check numerical finiteness and bounds
                for b in survivors[:200]:
                    self.assertTrue(0.0 <= b.x <= world.world_cfg.width)
                    self.assertTrue(0.0 <= b.y <= world.world_cfg.height)
                    self.assertTrue(np.isfinite(b.x) and np.isfinite(b.y))
                    self.assertTrue(np.isfinite(b.vx) and np.isfinite(b.vy))
                    self.assertTrue(np.isfinite(b.energy))
                    self.assertGreaterEqual(b.energy, 0.0)

                # Check Group Reputation Oracle Parity across all groups (strictly 0 drift)
                for grp in world.groups.values():
                    cached_rep = world._get_group_avg_reputation(grp)
                    gt_rep = self._unoptimized_group_avg_reputation(world, grp)
                    self.assertEqual(
                        cached_rep,
                        gt_rep,
                        f"Step {step_idx}: Group {grp.id} reputation drifted! cached={cached_rep}, gt={gt_rep}",
                    )

                # Check Sensor Parity across all living blobs
                n_surv = len(survivors)
                dummy_f = np.zeros((n_surv, 3), np.float32)
                dummy_p = np.zeros((n_surv, 6), np.float32)
                dummy_o = np.zeros((n_surv, 3), np.float32)
                dummy_feel = np.zeros((n_surv, 3), np.float32)

                p_idxs = world._last_min_p_idxs
                nearest_peers = [survivors[p] if 0 <= p < n_surv else None for p in p_idxs]

                obs_sample, _, _, _, _, _, _, _ = world._prepare_step_observations(
                    survivors, dummy_f, dummy_p, dummy_o, dummy_feel,
                    0.5, 1.0, 2.0, 1, nearest_peers=nearest_peers
                )
                gt_sample = self._extract_ground_truth_observations(
                    world, survivors, dummy_f, dummy_p, dummy_o, dummy_feel,
                    0.5, 1.0, 2.0, nearest_peers=nearest_peers
                )
                max_sensor_err = float(np.max(np.abs(obs_sample - gt_sample)))
                self.assertLessEqual(
                    max_sensor_err,
                    1e-6,
                    f"Step {step_idx}: Sensor channel drift {max_sensor_err:.4e} > 1e-6 after relocations/conquests!",
                )

        # Performance & Throughput Assertion (budget < 120 ms at 2,500 scale under continuous adversarial churn)
        mean_tick_ms = (sum(tick_durations) / len(tick_durations)) * 1000.0
        self.assertLess(
            mean_tick_ms,
            120.0,
            f"Mean tick duration at 2500 scale with relocations/conquests was {mean_tick_ms:.2f} ms (budget < 120 ms)",
        )
        self.assertGreaterEqual(total_relocations, 800)
        self.assertGreaterEqual(total_conquests, 40)


if __name__ == "__main__":
    unittest.main()
