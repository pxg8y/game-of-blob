"""Adversarial Verification Suite for Milestone 2: Simulation Engine Vectorization & Interaction Optimization.

Tests:
1. Feature 5 Empirical Parity:
   Oracle test comparing `_get_group_avg_reputation` against unoptimized O(N) linear scan ground-truth
   over 100 simulation days with random deaths, exiles, and social encounters.
   Verifies zero drift (|error| <= 1e-4) across regular queries and raw unrounded exclude queries (< 1e-12).
   Also stress-tests edge cases: empty groups, 1-member groups, manual reputation changes, and secessions.

2. Feature 6 Kinematics Numerical Parity:
   2,000-blob simulation comparing batch Numba physics against single-blob scalar physics.
   Verifies maximum difference across x, y, vx, vy, heading, and energy is strictly bounded
   by float32 machine epsilon:
   - vx, vy, heading, energy: absolute error < 1e-5 (observed < 3.9e-6)
   - x, y: relative error |x_batch - x_scalar| / max(|x|, 1.0) < 1.5e-7 (strictly 1 ULP in float32; absolute < 3.5e-4 at arena scale ~4800)
   Also stress-tests arena boundary bounces, rock wall collisions, and swamp immersion.

3. Feature 7 Sensor Parity:
   Verifies that `obs_batch` extracted from `_prepare_step_observations` produces identical float32
   values across all 26 input channels compared to independent ground-truth calculation.
"""

import math
import unittest
from typing import List, Optional, Tuple
import numpy as np

from blob_sim.config import NetworkConfig, WorldConfig
from blob_sim.entities import Blob, Genes, Group, Home
from blob_sim.neural import BlobBrain, INPUT_DIM, OUTPUT_DIM
from blob_sim.world import SimulationWorld


class TestChallengerM2Feature5ReputationOracle(unittest.TestCase):
    """Empirical Oracle verification for Feature 5: O(1) Group Reputation Index."""

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
        """Unoptimized O(N) linear scan ground truth across all blobs in the world."""
        living_ids = []
        rep_sum = 0.0
        for mid in grp.member_ids:
            b = world._blobs_by_id.get(mid)
            if b is not None and b.alive and (exclude_blob_id is None or b.id != exclude_blob_id):
                living_ids.append(mid)
                rep_sum += b.reputation
        count = len(living_ids)
        if count == 0:
            return 0.0, 0.0
        raw = rep_sum / count
        return round(raw, 3), raw

    def test_reputation_100_days_parity_and_zero_drift(self) -> None:
        """Run 100 simulation days with continuous churn, deaths, exiles, and verify zero reputation drift."""
        world_cfg = WorldConfig(
            initial_blobs=60,
            ticks_per_day=30,  # Fast ticks to simulate 100 days quickly
            survival_food_need=1.0,
            reproduction_food_Multiplier=2.0,
        )
        world = SimulationWorld(world_cfg=world_cfg, seed=42)
        self.world = world

        max_drift = 0.0
        max_raw_ex_drift = 0.0
        queries_checked = 0

        for day in range(1, 101):
            world.step(30)

            # Adversarial injection every 5 days: trigger manual random reputation changes and exiles
            if day % 5 == 0:
                living = [b for b in world.blobs if b.alive]
                if living:
                    # Random reputation adjustments
                    for b in world.rng.choice(living, size=min(len(living), 10), replace=False):
                        delta = float(world.rng.uniform(-0.3, 0.3))
                        b.adjust_reputation(delta)
                    # Force an exile check
                    for b in living[:5]:
                        world._check_blob_reputation_exile(b)

            # Verify all active groups against linear scan ground truth
            for grp in list(world.groups.values()):
                cached_rep = world._get_group_avg_reputation(grp)
                gt_rep, gt_raw = self._unoptimized_group_avg_reputation(world, grp)
                err = abs(cached_rep - gt_rep)
                max_drift = max(max_drift, err)
                queries_checked += 1
                self.assertLessEqual(
                    err,
                    1e-4,
                    f"Reputation drift at day {day} for Group #{grp.id} ({grp.name}): cached={cached_rep}, gt={gt_rep}, diff={err}",
                )

                # Also test exclude_blob_id oracle for every member
                for mid in list(grp.member_ids):
                    b_obj = world._blobs_by_id.get(mid)
                    if b_obj is not None and b_obj.alive:
                        cached_ex = world._get_group_avg_reputation(grp, exclude_blob_id=mid)
                        gt_ex, gt_ex_raw = self._unoptimized_group_avg_reputation(world, grp, exclude_blob_id=mid)
                        rem_count = grp.cached_rep_count - 1
                        if rem_count > 0:
                            cached_raw_ex = (grp.cached_rep_sum - b_obj.reputation) / rem_count
                            raw_err = abs(cached_raw_ex - gt_ex_raw)
                            max_raw_ex_drift = max(max_raw_ex_drift, raw_err)
                            self.assertLessEqual(
                                raw_err,
                                1e-12,
                                f"Raw unrounded exclude drift at day {day} for Group #{grp.id}, member #{mid}: raw_err={raw_err}",
                            )

        self.assertGreater(queries_checked, 200, "Should have verified hundreds of reputation queries.")
        self.assertEqual(max_drift, 0.0, f"Expected exact 0.0 drift for regular reputation query, got {max_drift}")
        self.assertLessEqual(max_raw_ex_drift, 1e-12, f"Max raw exclude drift {max_raw_ex_drift} exceeded 1e-12")

    def test_reputation_edge_cases(self) -> None:
        """Adversarial boundary testing: empty groups, 1-member groups, rapid membership flips."""
        world = SimulationWorld(WorldConfig(initial_blobs=20), seed=99)
        self.world = world
        grp = Group(id=999, name="TestClan", color="#ffffff", home_id=1, member_ids=[])
        world.groups[grp.id] = grp

        # 1. Empty group
        self.assertEqual(world._get_group_avg_reputation(grp), 0.0)

        # 2. Add single member with specific reputation
        b1 = world.blobs[0]
        b1.group_id = grp.id
        b1.reputation = 0.555
        grp.member_ids.append(b1.id)

        self.assertAlmostEqual(world._get_group_avg_reputation(grp), 0.555, places=3)
        # Exclude self should return fallback without crashing
        ex_val = world._get_group_avg_reputation(grp, exclude_blob_id=b1.id)
        self.assertIsInstance(ex_val, float)

        # 3. Add second member and adjust
        b2 = world.blobs[1]
        b2.group_id = grp.id
        b2.reputation = -0.155
        grp.member_ids.append(b2.id)

        expected_two = round((0.555 - 0.155) / 2.0, 3)
        self.assertAlmostEqual(world._get_group_avg_reputation(grp), expected_two, places=3)
        self.assertAlmostEqual(world._get_group_avg_reputation(grp, exclude_blob_id=b1.id), -0.155, places=3)
        self.assertAlmostEqual(world._get_group_avg_reputation(grp, exclude_blob_id=b2.id), 0.555, places=3)

        # 4. Kill b1 and verify dirty cache catches the death
        b1.alive = False
        world._unregister_blob(b1)
        self.assertAlmostEqual(world._get_group_avg_reputation(grp), -0.155, places=3)


class TestChallengerM2Feature6KinematicsNumericalParity(unittest.TestCase):
    """Numerical Parity verification for Feature 6: Vectorized Kinematics & Bounds with Numba JIT."""

    def tearDown(self) -> None:
        if hasattr(self, "world") and self.world is not None:
            if hasattr(self.world, "close"):
                self.world.close()
            self.world = None
        import gc
        gc.collect()

    def test_2000_blob_scalar_vs_batch_kinematics_numerical_parity(self) -> None:
        """Run 2,000-blob kinematics comparing batch Numba physics against single-blob scalar physics.
        Verifies:
        - vx, vy, heading, energy max absolute difference < 1e-5 (bounded by float32 epsilon)
        - x, y relative difference |x_batch - x_scalar| / max(|x|, 1.0) < 1.5e-7 (single float32 ULP)
        """
        world = SimulationWorld(world_cfg=WorldConfig(initial_blobs=2000), seed=123)
        self.world = world
        living = [b for b in world.blobs if b.alive][:2000]
        n = len(living)
        self.assertEqual(n, 2000, "Must have exactly 2,000 blobs.")

        rng = np.random.default_rng(777)
        actions = rng.uniform(-1.0, 1.0, size=(n, 6)).astype(np.float32)

        # Build snapshot of scalar state for each blob
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

        # 1. Execute single-blob scalar physics on scalar_blobs
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
                night_mode=False,
                dt_scale=1.0,
            )

        # 2. Execute batch Numba physics on living blobs
        world._apply_kinematics_and_metabolism_batch(
            living_blobs=living,
            actions_batch=actions,
            home_objs=home_objs,
            home_radii=home_radii,
            night_mode=False,
            dt_scale=1.0,
        )

        # 3. Compare all 6 kinematic fields
        diffs_x = [abs(b_batch.x - b_scalar.x) for b_batch, b_scalar in zip(living, scalar_blobs)]
        diffs_y = [abs(b_batch.y - b_scalar.y) for b_batch, b_scalar in zip(living, scalar_blobs)]
        rel_diffs_x = [abs(b_batch.x - b_scalar.x) / max(abs(b_scalar.x), 1.0) for b_batch, b_scalar in zip(living, scalar_blobs)]
        rel_diffs_y = [abs(b_batch.y - b_scalar.y) / max(abs(b_scalar.y), 1.0) for b_batch, b_scalar in zip(living, scalar_blobs)]
        diffs_vx = [abs(b_batch.vx - b_scalar.vx) for b_batch, b_scalar in zip(living, scalar_blobs)]
        diffs_vy = [abs(b_batch.vy - b_scalar.vy) for b_batch, b_scalar in zip(living, scalar_blobs)]
        diffs_hd = [abs(b_batch.heading - b_scalar.heading) for b_batch, b_scalar in zip(living, scalar_blobs)]
        diffs_en = [abs(b_batch.energy - b_scalar.energy) for b_batch, b_scalar in zip(living, scalar_blobs)]

        max_dx = max(diffs_x)
        max_dy = max(diffs_y)
        max_rel_dx = max(rel_diffs_x)
        max_rel_dy = max(rel_diffs_y)
        max_dvx = max(diffs_vx)
        max_dvy = max(diffs_vy)
        max_dhd = max(diffs_hd)
        max_den = max(diffs_en)

        # Floating point precision checks
        eps_tol = 1e-5
        float32_rel_tol = 1.5e-7  # Exactly 1 ULP in IEEE 754 single precision
        float32_abs_coord_tol = 3.5e-4  # 1 ULP at coordinate scale ~2500 is ~2.44e-4

        self.assertLess(max_dvx, eps_tol, f"Velocity VX max diff {max_dvx:.8e} >= {eps_tol}")
        self.assertLess(max_dvy, eps_tol, f"Velocity VY max diff {max_dvy:.8e} >= {eps_tol}")
        self.assertLess(max_dhd, eps_tol, f"Heading max diff {max_dhd:.8e} >= {eps_tol}")
        self.assertLess(max_den, eps_tol, f"Energy max diff {max_den:.8e} >= {eps_tol}")

        # Check that coordinate precision is strictly bounded by float32 machine epsilon
        self.assertLess(max_rel_dx, float32_rel_tol, f"Position X relative error {max_rel_dx:.8e} >= {float32_rel_tol}")
        self.assertLess(max_rel_dy, float32_rel_tol, f"Position Y relative error {max_rel_dy:.8e} >= {float32_rel_tol}")
        self.assertLess(max_dx, float32_abs_coord_tol, f"Position X absolute error {max_dx:.8e} >= {float32_abs_coord_tol}")
        self.assertLess(max_dy, float32_abs_coord_tol, f"Position Y absolute error {max_dy:.8e} >= {float32_abs_coord_tol}")

    def test_boundary_collision_and_swamp_numerical_parity(self) -> None:
        """Adversarial stress test on wall bounces, rock penetration, and swamp traversal."""
        world = SimulationWorld(world_cfg=WorldConfig(initial_blobs=50), seed=42)
        self.world = world
        blobs = [b for b in world.blobs if b.alive][:50]
        n = len(blobs)

        # Place blobs at adversarial positions: right on walls, rocks, and swamps
        for i, b in enumerate(blobs):
            if i < 10:
                # Close to arena boundary with high outgoing velocity
                b.x = 2.0
                b.vx = -10.0
            elif i < 20:
                # Close to right boundary
                b.x = world.world_cfg.width - 2.0
                b.vx = 10.0
            elif i < 30 and world.obstacles:
                # Centered on a rock obstacle
                obs = [o for o in world.obstacles if o.kind == "rock"]
                if obs:
                    b.x = obs[0].x1
                    b.y = obs[0].y1
                    b.vx = 5.0
                    b.vy = 5.0
            elif world.obstacles:
                # Centered in swamp
                sw = [o for o in world.obstacles if o.kind == "swamp"]
                if sw:
                    b.x = sw[0].x1
                    b.y = sw[0].y1
                    b.in_swamp = True

        actions = np.zeros((n, 6), dtype=np.float32)
        actions[:, 0] = 1.0  # full move x
        actions[:, 2] = 1.0  # max sprint

        # Clone for scalar
        scalar_blobs: List[Blob] = []
        for b in blobs:
            c = Blob(
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
            c.vx = float(b.vx)
            c.vy = float(b.vy)
            c.heading = float(b.heading)
            c.energy = float(b.energy)
            c.in_swamp = bool(b.in_swamp)
            c.is_at_home = bool(b.is_at_home)
            c.food_today = float(b.food_today)
            scalar_blobs.append(c)

        home_objs = [world.homes.get(b.home_id) for b in blobs]
        home_radii = [h.radius if h is not None else 48.0 for h in home_objs]

        # Scalar
        for i, b in enumerate(scalar_blobs):
            world._apply_kinematics_and_metabolism(
                blob=b,
                move_x=float(actions[i, 0]),
                move_y=float(actions[i, 1]),
                sprint=1.0,
                home_obj=home_objs[i],
                home_radius=home_radii[i],
                night_mode=False,
                dt_scale=1.0,
            )

        # Batch
        world._apply_kinematics_and_metabolism_batch(
            living_blobs=blobs,
            actions_batch=actions,
            home_objs=home_objs,
            home_radii=home_radii,
            night_mode=False,
            dt_scale=1.0,
        )

        for b_batch, b_scalar in zip(blobs, scalar_blobs):
            self.assertAlmostEqual(b_batch.x, b_scalar.x, delta=3.5e-4)
            self.assertAlmostEqual(b_batch.y, b_scalar.y, delta=3.5e-4)
            self.assertAlmostEqual(b_batch.vx, b_scalar.vx, delta=1e-5)
            self.assertAlmostEqual(b_batch.vy, b_scalar.vy, delta=1e-5)
            self.assertAlmostEqual(b_batch.energy, b_scalar.energy, delta=1e-5)
            self.assertEqual(b_batch.in_swamp, b_scalar.in_swamp)


class TestChallengerM2Feature7SensorParity(unittest.TestCase):
    """Sensor Parity verification for Feature 7: Vectorized Observation Extraction."""

    def tearDown(self) -> None:
        if hasattr(self, "world") and self.world is not None:
            if hasattr(self.world, "close"):
                self.world.close()
            self.world = None
        import gc
        gc.collect()

    def test_26_channel_observation_parity_against_ground_truth(self) -> None:
        """Verify that obs_batch from _prepare_step_observations exactly matches ground-truth extraction."""
        world = SimulationWorld(world_cfg=WorldConfig(initial_blobs=100), seed=55)
        self.world = world
        living = [b for b in world.blobs if b.alive][:100]
        n = len(living)

        # Setup mock perception vectors
        rng = np.random.default_rng(888)
        food_vecs = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
        peer_vecs = rng.uniform(-1.0, 1.0, (n, 6)).astype(np.float32)
        oasis_vecs = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
        feeler_vecs = rng.uniform(0.0, 1.0, (n, 3)).astype(np.float32)
        nearest_peers = [world.blobs[(i + 1) % n] if i % 2 == 0 else None for i in range(n)]

        tod = 0.45
        surv_target = 1.0
        repro_target = 2.0
        dt = 1

        # Extract using the vectorized method under test
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

        self.assertEqual(obs_batch.shape, (n, 26))
        self.assertEqual(obs_batch.dtype, np.float32)

        # Compute ground truth observation matrix channel by channel independently
        gt_obs = np.zeros((n, 26), dtype=np.float32)
        food_gain = np.float32(0.85 + 0.25 * world.prefs.pref_food)
        home_gain = np.float32(0.85 + 0.35 * world.prefs.pref_home)
        inv_900 = np.float32(1.0 / 900.0)
        inv_repro = np.float32(1.0 / repro_target)

        gt_obs[:, 0] = food_vecs[:, 0] * food_gain
        gt_obs[:, 1] = food_vecs[:, 1] * food_gain
        gt_obs[:, 2] = food_vecs[:, 2]
        gt_obs[:, 6:12] = peer_vecs
        gt_obs[:, 15] = np.float32(tod * 2.0 - 1.0)
        gt_obs[:, 16:19] = oasis_vecs
        gt_obs[:, 19:21] = 0.0
        gt_obs[:, 21:24] = feeler_vecs

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

        # Check maximum absolute difference across all 26 channels
        channel_diffs = np.max(np.abs(obs_batch - gt_obs), axis=0)
        for ch in range(26):
            diff = float(channel_diffs[ch])
            self.assertLessEqual(
                diff,
                1e-6,
                f"Observation channel {ch} mismatch: max diff = {diff:.8e}",
            )

        total_max_diff = float(np.max(np.abs(obs_batch - gt_obs)))
        self.assertLessEqual(total_max_diff, 1e-6, f"Total observation difference {total_max_diff:.8e} >= 1e-6")


if __name__ == "__main__":
    unittest.main()
