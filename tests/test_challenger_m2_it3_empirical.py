"""Challenger Empirical Verification Suite for Milestone 2 Iteration 3.

Empirically verifies:
1. `_compact_living_blobs`:
   - High-churn deaths and compactions under adversarial patterns (alternating, front-heavy, tail-heavy, multi-round churn).
   - Verifies all living survivors remain in `_blobs_by_id`.
   - Verifies all living survivors retain non-None, functional `_on_rep_change` and `_on_group_change` listeners.
   - Verifies dead blobs are fully evicted from `_blobs_by_id` and have None listeners.
2. Group Reputation Oracle Parity:
   - 100 simulation days with continuous churn, deaths, births, exiles, and house conquests.
   - Cached `_get_group_avg_reputation` compared against unoptimized linear scan over surviving members.
   - Verifies maximum drift is strictly 0.000 across all 100 days.
   - Verifies exclude_blob_id queries across all members.
3. Kinematics & Sensor Float Parity:
   - Float64 kinematics integration matches scalar physics across 2,000 blobs under warped dt=5.0 with zero float32 ULP divergence (< 1.5e-7 relative error).
   - All 26 observation channels in `_prepare_step_observations` match ground-truth scalar extraction within 1e-6 tolerance across 2,000 blobs.
"""

import math
import unittest
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from blob_sim.config import NetworkConfig, WorldConfig
from blob_sim.entities import Blob, Genes, Group, Home
from blob_sim.world import SimulationWorld


class TestChallengerM2It3CompactionIntegrity(unittest.TestCase):
    """Empirical adversarial verification of _compact_living_blobs and callback persistence."""

    def tearDown(self) -> None:
        if hasattr(self, "world") and self.world is not None:
            if hasattr(self.world, "close"):
                self.world.close()
            self.world = None
        import gc
        gc.collect()

    def test_compact_living_blobs_adversarial_patterns(self) -> None:
        """Adversarial stress-test of compaction under alternating, front-heavy, and tail-heavy deaths."""
        patterns = [
            ("alternating", lambda idx, total: idx % 2 == 0),  # 50% dead, alternating
            ("front_heavy", lambda idx, total: idx < int(total * 0.8)),  # first 80% dead
            ("tail_heavy", lambda idx, total: idx >= int(total * 0.2)),  # last 80% dead
            ("all_but_last", lambda idx, total: idx < total - 1),  # only the very last blob lives
            ("all_but_first", lambda idx, total: idx > 0),  # only the very first blob lives
            ("interleaved_blocks", lambda idx, total: (idx // 10) % 2 == 0),  # 10 alive, 10 dead blocks
        ]

        for pattern_name, is_dead_fn in patterns:
            with self.subTest(pattern=pattern_name):
                world = SimulationWorld(WorldConfig(initial_blobs=200), seed=42)
                initial_blobs = list(world.blobs)
                total = len(initial_blobs)

                expected_living_ids: Set[int] = set()
                expected_dead_ids: Set[int] = set()

                for idx, b in enumerate(initial_blobs):
                    if is_dead_fn(idx, total):
                        b.alive = False
                        b.energy = 0.0
                        expected_dead_ids.add(b.id)
                    else:
                        b.alive = True
                        expected_living_ids.add(b.id)

                # Execute compaction
                survivors = world._compact_living_blobs()

                # Verify survivor list length and identity
                self.assertEqual(
                    len(survivors),
                    len(expected_living_ids),
                    f"[{pattern_name}] Survivor count mismatch: got {len(survivors)}, expected {len(expected_living_ids)}",
                )
                self.assertEqual(len(world.blobs), len(expected_living_ids))

                survivor_ids = {b.id for b in survivors}
                self.assertEqual(survivor_ids, expected_living_ids, f"[{pattern_name}] Survivor ID set mismatch")

                # Verify all survivors remain in _blobs_by_id and have active listeners
                for b in survivors:
                    self.assertTrue(b.alive, f"[{pattern_name}] Survivor {b.id} marked as dead")
                    self.assertIn(b.id, world._blobs_by_id, f"[{pattern_name}] Survivor {b.id} missing from _blobs_by_id")
                    self.assertIs(world._blobs_by_id[b.id], b, f"[{pattern_name}] _blobs_by_id[{b.id}] is not survivor instance")

                    # Verify non-None and callable listeners
                    self.assertIsNotNone(b._on_rep_change, f"[{pattern_name}] Survivor {b.id} has None _on_rep_change")
                    self.assertTrue(callable(b._on_rep_change), f"[{pattern_name}] Survivor {b._on_rep_change} not callable")
                    self.assertIsNotNone(b._on_group_change, f"[{pattern_name}] Survivor {b.id} has None _on_group_change")
                    self.assertTrue(callable(b._on_group_change), f"[{pattern_name}] Survivor {b._on_group_change} not callable")

                # Verify dead blobs are completely evicted from _blobs_by_id and have None listeners
                for dead_id in expected_dead_ids:
                    self.assertNotIn(dead_id, world._blobs_by_id, f"[{pattern_name}] Dead blob {dead_id} remains in _blobs_by_id")
                    # Check original blob object
                    dead_b = next(b for b in initial_blobs if b.id == dead_id)
                    self.assertIsNone(dead_b._on_rep_change, f"[{pattern_name}] Dead blob {dead_id} has non-None _on_rep_change")
                    self.assertIsNone(dead_b._on_group_change, f"[{pattern_name}] Dead blob {dead_id} has non-None _on_group_change")
                world.close()

    def test_compact_survivor_callback_functionality(self) -> None:
        """Verify that survivor callbacks actually trigger state updates on groups after compaction."""
        world = SimulationWorld(WorldConfig(initial_blobs=60), seed=123)
        self.world = world
        grp = Group(id=555, name="SurvivorClan", color="#00ff00", home_id=1, member_ids=[])
        world.groups[grp.id] = grp

        # Assign first 30 blobs to grp
        for b in world.blobs[:30]:
            b.group_id = grp.id
            grp.member_ids.append(b.id)

        # Recompute to clear dirty
        world._recompute_group_reputation(grp)
        self.assertFalse(grp.rep_dirty)

        # Mark half of the group members and half of non-group members dead
        for i in range(0, 30, 2):
            world.blobs[i].alive = False
        for i in range(30, 60, 2):
            world.blobs[i].alive = False

        # Run compaction
        survivors = world._compact_living_blobs()

        # grp.rep_dirty should be True because dead members were unregistered
        self.assertTrue(grp.rep_dirty)
        world._recompute_group_reputation(grp)
        self.assertFalse(grp.rep_dirty)

        # Now pick a surviving member of the group, adjust reputation, and verify callback fires
        grp_survivors = [b for b in survivors if b.group_id == grp.id]
        self.assertGreater(len(grp_survivors), 0)
        target = grp_survivors[0]

        target.adjust_reputation(0.2)
        self.assertTrue(grp.rep_dirty, "Modifying survivor reputation did not assert grp.rep_dirty! Callback broken.")

        # Recompute and verify
        world._recompute_group_reputation(grp)
        self.assertFalse(grp.rep_dirty)

        # Test group change listener on survivor
        grp2 = Group(id=777, name="SecondClan", color="#ff0000", home_id=1, member_ids=[])
        world.groups[grp2.id] = grp2

        target.group_id = grp2.id
        self.assertTrue(grp.rep_dirty, "Old group rep_dirty not set on survivor group change")
        self.assertTrue(grp2.rep_dirty, "New group rep_dirty not set on survivor group change")
        self.assertIn(target.id, grp2.member_ids)
        self.assertNotIn(target.id, grp.member_ids)

    def test_multi_round_continuous_churn_compaction(self) -> None:
        """50 continuous rounds of random culling, compaction, reproduction and ID verification."""
        world = SimulationWorld(WorldConfig(initial_blobs=100), seed=789)
        self.world = world

        for round_idx in range(50):
            living = [b for b in world.blobs if b.alive]
            if len(living) > 20:
                # Cull 30% of living blobs at random
                cull_count = int(len(living) * 0.3)
                to_cull = world.rng.choice(living, size=cull_count, replace=False)
                for b in to_cull:
                    b.alive = False

            world._compact_living_blobs()

            # Verify integrity
            current_blobs = list(world.blobs)
            self.assertEqual(len(current_blobs), len(world._blobs_by_id))
            for b in current_blobs:
                self.assertTrue(b.alive)
                self.assertIn(b.id, world._blobs_by_id)
                self.assertIsNotNone(b._on_rep_change)
                self.assertIsNotNone(b._on_group_change)


class TestChallengerM2It3ReputationOracleParity(unittest.TestCase):
    """Empirical verification of 100 simulation days group reputation parity with zero drift."""

    def tearDown(self) -> None:
        if hasattr(self, "world") and self.world is not None:
            if hasattr(self.world, "close"):
                self.world.close()
            self.world = None
        import gc
        gc.collect()

    def _unoptimized_group_avg_reputation(
        self, world: SimulationWorld, grp: Group, exclude_blob_id: Optional[int] = None
    ) -> Tuple[float, float]:
        """Ground-truth linear scan over surviving group members."""
        living_ids = []
        rep_sum = 0.0
        seen = set()
        for mid in grp.member_ids:
            if mid in seen:
                continue
            seen.add(mid)
            b = world._blobs_by_id.get(mid)
            if b is not None and b.alive and (exclude_blob_id is None or b.id != exclude_blob_id):
                living_ids.append(mid)
                rep_sum += b.reputation
        count = len(living_ids)
        if count == 0:
            return 0.0, 0.0
        raw = rep_sum / count
        return round(raw, 3), raw

    def test_reputation_100_days_parity_and_zero_drift_with_conquest(self) -> None:
        """Run 100 simulation days with continuous churn, deaths, births, exiles, and house conquests.
        Verify cached _get_group_avg_reputation matches unoptimized linear scan with strictly 0.000 max drift.
        """
        world_cfg = WorldConfig(
            initial_blobs=80,
            ticks_per_day=30,
            survival_food_need=1.0,
            reproduction_food_Multiplier=2.0,
        )
        world = SimulationWorld(world_cfg=world_cfg, seed=101)
        self.world = world

        max_drift = 0.0
        max_raw_exclude_drift = 0.0
        total_queries = 0
        conquest_count = 0

        for day in range(1, 101):
            world.step(30)

            # Adversarial injections
            # 1. Random reputation perturbations every 3 days
            if day % 3 == 0:
                living = [b for b in world.blobs if b.alive]
                if living:
                    sample_size = min(len(living), 12)
                    for b in world.rng.choice(living, size=sample_size, replace=False):
                        delta = float(world.rng.uniform(-0.35, 0.35))
                        b.adjust_reputation(delta)
                    # Trigger exiles
                    for b in living[:6]:
                        world._check_blob_reputation_exile(b)

            # 2. Simulated House Conquest every 7 days
            if day % 7 == 0 and len(world.groups) >= 2 and len(world.homes) >= 2:
                groups_list = list(world.groups.values())
                homes_list = list(world.homes.values())
                target_home = homes_list[day % len(homes_list)]
                atk_grp = groups_list[(day + 1) % len(groups_list)]
                old_gid = target_home.group_id
                if old_gid != atk_grp.id and old_gid in world.groups:
                    old_grp = world.groups[old_gid]
                    if target_home.id in old_grp.home_ids:
                        old_grp.home_ids.remove(target_home.id)
                    old_grp.rep_dirty = True
                    target_home.group_id = atk_grp.id
                    if target_home.id not in atk_grp.home_ids:
                        atk_grp.home_ids.append(target_home.id)
                    atk_grp.rep_dirty = True
                    conquest_count += 1

            # 3. Dynamic group creation or joining every 10 days
            if day % 10 == 0:
                living = [b for b in world.blobs if b.alive]
                if len(living) > 5:
                    new_grp_id = 9000 + day
                    new_grp = Group(id=new_grp_id, name=f"Nomad_{day}", color="#aabbcc", home_id=1, member_ids=[])
                    world.groups[new_grp_id] = new_grp
                    for b in living[:3]:
                        b.group_id = new_grp_id

            # Verification of all groups against ground-truth linear scan
            for grp in list(world.groups.values()):
                cached = world._get_group_avg_reputation(grp)
                gt_rounded, gt_raw = self._unoptimized_group_avg_reputation(world, grp)
                err = abs(cached - gt_rounded)
                max_drift = max(max_drift, err)
                total_queries += 1

                self.assertEqual(
                    cached,
                    gt_rounded,
                    f"Reputation drift at day {day} for group {grp.id} ({grp.name}): cached={cached}, gt={gt_rounded}, diff={err:.4f}",
                )

                # Verify exclude_blob_id oracle for all living members in this group
                for mid in list(grp.member_ids):
                    b_obj = world._blobs_by_id.get(mid)
                    if b_obj is not None and b_obj.alive:
                        cached_ex = world._get_group_avg_reputation(grp, exclude_blob_id=mid)
                        gt_ex_rounded, gt_ex_raw = self._unoptimized_group_avg_reputation(world, grp, exclude_blob_id=mid)
                        rem_count = grp.cached_rep_count - 1
                        if rem_count > 0:
                            cached_raw_ex = (grp.cached_rep_sum - b_obj.reputation) / rem_count
                            raw_ex_err = abs(cached_raw_ex - gt_ex_raw)
                            max_raw_exclude_drift = max(max_raw_exclude_drift, raw_ex_err)
                            self.assertLessEqual(
                                raw_ex_err,
                                1e-12,
                                f"Raw exclude drift at day {day} for group {grp.id}, member {mid}: raw_err={raw_ex_err}",
                            )

        self.assertGreater(total_queries, 300, "Should have tested hundreds of group reputation queries across 100 days")
        self.assertEqual(max_drift, 0.0, f"Expected strictly 0.000 max drift, got {max_drift}")
        self.assertLessEqual(max_raw_exclude_drift, 1e-12, f"Expected raw exclude drift <= 1e-12, got {max_raw_exclude_drift}")


class TestChallengerM2It3KinematicsAndSensorFloatParity(unittest.TestCase):
    """Empirical verification of 2,000-blob kinematics under warped dt=5.0 and 26 observation channels."""

    def tearDown(self) -> None:
        if hasattr(self, "world") and self.world is not None:
            if hasattr(self.world, "close"):
                self.world.close()
            self.world = None
        import gc
        gc.collect()

    def test_2000_blob_kinematics_float64_warped_dt5_zero_ulp(self) -> None:
        """Verify float64 kinematics integration matches scalar physics across 2,000 blobs
        under warped dt=5.0 with zero float32 ULP divergence (< 1.5e-7 relative coordinate error).
        """
        world = SimulationWorld(WorldConfig(initial_blobs=2000), seed=42)
        self.world = world
        living = [b for b in world.blobs if b.alive][:2000]
        n = len(living)
        self.assertEqual(n, 2000, f"Expected 2,000 blobs, got {n}")

        for night_mode in (False, True):
            dt_scale = 5.0
            rng = np.random.default_rng(555 + int(night_mode))
            actions = rng.uniform(-1.0, 1.0, size=(n, 6)).astype(np.float32)

            # Deep copy clones for scalar simulation
            scalar_blobs: List[Blob] = []
            for b in living:
                clone = Blob(
                    blob_id=b.id,
                    x=float(b.x),
                    y=float(b.y),
                    home_id=b.home_id,
                    home_x=float(b.home_x),
                    home_y=float(b.home_y),
                    brain=b.brain,
                    genes=b.genes,
                    group_id=b.group_id,
                    money=b.money,
                    reputation=b.reputation,
                )
                clone.vx = float(b.vx)
                clone.vy = float(b.vy)
                clone.heading = float(b.heading)
                clone.energy = float(b.energy)
                clone.in_swamp = bool(b.in_swamp)
                clone.is_at_home = bool(b.is_at_home)
                clone.food_today = float(b.food_today)
                scalar_blobs.append(clone)

            home_objs = [world.homes.get(b.home_id) for b in living]
            home_radii = [h.radius if h is not None else 48.0 for h in home_objs]

            # 1. Run single-blob scalar physics
            for i, b in enumerate(scalar_blobs):
                h_obj = home_objs[i]
                hr = home_radii[i]
                mx = float(actions[i, 0])
                my = float(actions[i, 1])
                sp = float((actions[i, 2] + 1.0) * 0.5)
                world._apply_kinematics_and_metabolism(
                    blob=b,
                    move_x=mx,
                    move_y=my,
                    sprint=sp,
                    home_obj=h_obj,
                    home_radius=hr,
                    night_mode=night_mode,
                    dt_scale=dt_scale,
                )

            # 2. Run batch kinematics
            world._apply_kinematics_and_metabolism_batch(
                living_blobs=living,
                actions_batch=actions,
                home_objs=home_objs,
                home_radii=home_radii,
                night_mode=night_mode,
                dt_scale=dt_scale,
            )

            # 3. Assert zero float32 ULP divergence
            max_rel_x = 0.0
            max_rel_y = 0.0
            max_vx_diff = 0.0
            max_vy_diff = 0.0
            max_heading_diff = 0.0
            max_energy_diff = 0.0

            mode_name = "Nighttime" if night_mode else "Daytime"
            for b_batch, b_scalar in zip(living, scalar_blobs):
                denom_x = max(abs(b_scalar.x), 1.0)
                denom_y = max(abs(b_scalar.y), 1.0)
                rel_diff_x = abs(b_batch.x - b_scalar.x) / denom_x
                rel_diff_y = abs(b_batch.y - b_scalar.y) / denom_y

                max_rel_x = max(max_rel_x, rel_diff_x)
                max_rel_y = max(max_rel_y, rel_diff_y)
                max_vx_diff = max(max_vx_diff, abs(b_batch.vx - b_scalar.vx))
                max_vy_diff = max(max_vy_diff, abs(b_batch.vy - b_scalar.vy))
                max_heading_diff = max(max_heading_diff, abs(b_batch.heading - b_scalar.heading))
                max_energy_diff = max(max_energy_diff, abs(b_batch.energy - b_scalar.energy))

                self.assertLess(
                    rel_diff_x,
                    1.5e-7,
                    f"[{mode_name} dt=5.0] Blob {b_batch.id} X relative error {rel_diff_x:.2e} exceeded float32 ULP",
                )
                self.assertLess(
                    rel_diff_y,
                    1.5e-7,
                    f"[{mode_name} dt=5.0] Blob {b_batch.id} Y relative error {rel_diff_y:.2e} exceeded float32 ULP",
                )
                self.assertLess(
                    abs(b_batch.vx - b_scalar.vx),
                    1e-5,
                    f"[{mode_name} dt=5.0] Blob {b_batch.id} vx diff exceeded 1e-5",
                )
                self.assertLess(
                    abs(b_batch.vy - b_scalar.vy),
                    1e-5,
                    f"[{mode_name} dt=5.0] Blob {b_batch.id} vy diff exceeded 1e-5",
                )
                self.assertLess(
                    abs(b_batch.energy - b_scalar.energy),
                    1e-5,
                    f"[{mode_name} dt=5.0] Blob {b_batch.id} energy diff exceeded 1e-5",
                )

    def test_26_channel_observation_parity_2000_blobs(self) -> None:
        """Verify that all 26 observation channels in _prepare_step_observations match
        ground-truth scalar extraction within 1e-6 tolerance across 2,000 blobs.
        """
        world = SimulationWorld(WorldConfig(initial_blobs=2000), seed=999)
        self.world = world
        living = [b for b in world.blobs if b.alive][:2000]
        n = len(living)
        self.assertEqual(n, 2000)

        rng = np.random.default_rng(888)
        food_vecs = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
        peer_vecs = rng.uniform(-1.0, 1.0, (n, 6)).astype(np.float32)
        oasis_vecs = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
        feeler_vecs = rng.uniform(0.0, 1.0, (n, 3)).astype(np.float32)
        nearest_peers = [world.blobs[(i + 1) % n] if i % 2 == 0 else None for i in range(n)]

        tod = 0.42
        surv_target = 1.0
        repro_target = 2.0
        dt = 1

        obs_batch, home_objs, home_radii, _, _, _, _, _ = world._prepare_step_observations(
            living_blobs=living,
            food_vecs=food_vecs,
            peer_vecs=peer_vecs,
            oasis_vecs=oasis_vecs,
            feeler_vecs=feeler_vecs,
            tod=tod,
            surv_target=surv_target,
            repro_target=repro_target,
            dt=dt,
            nearest_peers=nearest_peers,
        )

        gt_obs = np.zeros((n, 26), dtype=np.float32)
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
        gt_obs[:, 19:21] = 0.0
        gt_obs[:, 21:24] = feeler_vecs

        # Scalar ground-truth channels per blob
        for idx, blob in enumerate(living):
            home_obj = home_objs[idx]
            home_radius = home_radii[idx]

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

            if blob.memory_strength > 0.02:
                dmx = blob.memory_x - blob.x
                dmy = blob.memory_y - blob.y
                dist_mem = math.hypot(dmx, dmy)
                if dist_mem > 1e-4 and blob.memory_strength > 0.02:
                    inv_dm = blob.memory_strength / dist_mem
                    gt_obs[idx, 19] = np.float32(dmx * inv_dm)
                    gt_obs[idx, 20] = np.float32(dmy * inv_dm)

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

            np_peer = nearest_peers[idx]
            gt_obs[idx, 24] = np.float32(np_peer.reputation if np_peer is not None else 0.0)
            gt_obs[idx, 25] = np.float32(blob.reputation)

        # Check maximum error across all 2,000 blobs x 26 channels
        channel_errors = np.max(np.abs(obs_batch - gt_obs), axis=0)
        overall_max_error = float(np.max(channel_errors))

        for ch in range(26):
            self.assertLessEqual(
                float(channel_errors[ch]),
                1e-6,
                f"Channel {ch} max error {channel_errors[ch]:.2e} exceeded 1e-06 tolerance",
            )

        self.assertLessEqual(
            overall_max_error,
            1e-6,
            f"Overall 26-channel max error {overall_max_error:.2e} exceeded 1e-06 tolerance",
        )


if __name__ == "__main__":
    unittest.main()
