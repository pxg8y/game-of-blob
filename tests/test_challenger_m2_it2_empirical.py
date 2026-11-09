"""Empirical Parity Verification Suite for Milestone 2 Iteration 2 / Iteration 3.

Validates:
1. Feature 5: Dynamic churn and conquest reputation oracle parity across multi-step simulation.
2. Feature 6: Scalar vs batch kinematics exhaustive numerical parity under normal and warped dt.
3. Feature 7: Exact 26-channel observation tensor numerical agreement against ground truth.
"""

import math
import unittest
from typing import List, Optional, Tuple
import numpy as np

from blob_sim.config import NetworkConfig, WorldConfig
from blob_sim.entities import Blob, Group, Home
from blob_sim.world import SimulationWorld


class TestChallengerM2It2EmpiricalParity(unittest.TestCase):
    """Empirical audit and challenger parity verification for Milestone 2 Iteration 2/3."""

    def tearDown(self) -> None:
        if hasattr(self, "world") and self.world is not None:
            if hasattr(self.world, "close"):
                self.world.close()
            self.world = None
        import gc
        gc.collect()

    def _unoptimized_group_avg_reputation(
        self, world: SimulationWorld, grp: Group, exclude_blob_id: Optional[int] = None
    ) -> float:
        """Unoptimized O(N) ground-truth linear scan over group members."""
        living_reps: List[float] = []
        rep_sum = 0.0
        for mid in grp.member_ids:
            b = world._blobs_by_id.get(mid)
            if b is not None and b.alive and (exclude_blob_id is None or b.id != exclude_blob_id):
                living_reps.append(b.reputation)
                rep_sum += b.reputation
        if not living_reps:
            return 0.0
        return round(rep_sum / len(living_reps), 3)

    def test_feature5_oracle_dynamic_churn_and_conquest_parity(self) -> None:
        """Verify dynamic group reputation parity across 50 simulation steps with natural churn,
        deaths, exiles, and joins. Ensures zero discrepancy (specifically at step 25 and beyond).
        """
        world_cfg = WorldConfig(
            initial_blobs=40,
            ticks_per_day=20,
            survival_food_need=1.0,
            reproduction_food_Multiplier=2.0,
        )
        world = SimulationWorld(world_cfg=world_cfg, seed=42)
        self.world = world

        # Ensure active groups exist
        for step in range(1, 51):
            world.step(1)

            # Inject manual reputation adjustments and churn periodically
            if step % 5 == 0:
                living = [b for b in world.blobs if b.alive]
                if living:
                    for b in world.rng.choice(living, size=min(len(living), 8), replace=False):
                        delta = float(world.rng.uniform(-0.25, 0.25))
                        b.adjust_reputation(delta)
                    # Trigger an exile check on a few blobs
                    for b in living[:4]:
                        world._check_blob_reputation_exile(b)

            # Verify every active group against unoptimized ground truth
            for grp in list(world.groups.values()):
                cached = world._get_group_avg_reputation(grp)
                gt = self._unoptimized_group_avg_reputation(world, grp)
                diff = abs(cached - gt)
                self.assertAlmostEqual(
                    cached,
                    gt,
                    delta=0.001,
                    msg=f"Discrepancy at step {step} for group {grp.id}: cached={cached}, gt={gt} (diff={diff:.4f})",
                )

    def test_feature6_scalar_vs_batch_kinematics_exhaustive_parity(self) -> None:
        """Verify numerical equivalence between batch Numba kinematics and scalar physics
        under normal (dt=1.0) and time-warped conditions (dt=2.0, dt=5.0) for day and night.
        Ensures relative coordinate differences remain strictly within single-precision ULP bounds.
        """
        world = SimulationWorld(world_cfg=WorldConfig(initial_blobs=150), seed=101)
        self.world = world
        test_dts = [1.0, 2.0, 5.0]

        for dt in test_dts:
            for night_mode in (False, True):
                living = [b for b in world.blobs if b.alive][:150]
                n = len(living)
                if n == 0:
                    break

                rng = np.random.default_rng(202 + int(dt * 10))
                actions = rng.uniform(-1.0, 1.0, size=(n, 6)).astype(np.float32)

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

                # 1. Run scalar physics
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
                        dt_scale=dt,
                    )

                # 2. Run batch kinematics
                world._apply_kinematics_and_metabolism_batch(
                    living_blobs=living,
                    actions_batch=actions,
                    home_objs=home_objs,
                    home_radii=home_radii,
                    night_mode=night_mode,
                    dt_scale=dt,
                )

                # 3. Verify parity
                for b_batch, b_scalar in zip(living, scalar_blobs):
                    denom_x = max(abs(b_scalar.x), 1.0)
                    denom_y = max(abs(b_scalar.y), 1.0)
                    rel_diff_x = abs(b_batch.x - b_scalar.x) / denom_x
                    rel_diff_y = abs(b_batch.y - b_scalar.y) / denom_y

                    mode_name = "Nighttime" if night_mode else "Daytime"
                    self.assertLess(
                        rel_diff_x,
                        1.5e-7,
                        f"[{mode_name} warped dt={dt}] Blob {b_batch.id} X relative diff {rel_diff_x:.2e} exceeded float32 ULP",
                    )
                    self.assertLess(
                        rel_diff_y,
                        1.5e-7,
                        f"[{mode_name} warped dt={dt}] Blob {b_batch.id} Y relative diff {rel_diff_y:.2e} exceeded float32 ULP",
                    )

    def test_feature7_sensor_channel_exactness_all_26_channels(self) -> None:
        """Verify that observation extraction across all 26 channels matches ground truth
        within 1e-6 maximum numerical error, with zero direction distortion on home vectors.
        """
        world = SimulationWorld(world_cfg=WorldConfig(initial_blobs=120), seed=77)
        self.world = world
        living = [b for b in world.blobs if b.alive][:120]
        n = len(living)

        rng = np.random.default_rng(999)
        food_vecs = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
        peer_vecs = rng.uniform(-1.0, 1.0, (n, 6)).astype(np.float32)
        oasis_vecs = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
        feeler_vecs = rng.uniform(0.0, 1.0, (n, 3)).astype(np.float32)
        nearest_peers = [world.blobs[(i + 1) % n] if i % 2 == 0 else None for i in range(n)]

        tod = 0.35
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

        max_error = float(np.max(np.abs(obs_batch - gt_obs)))
        self.assertLessEqual(
            max_error,
            1e-6,
            f"Observation batch max error {max_error:.2e} exceeded 1e-06",
        )


if __name__ == "__main__":
    unittest.main()
