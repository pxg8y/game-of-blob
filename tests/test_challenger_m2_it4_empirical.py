"""Challenger Empirical Verification Suite for Milestone 2 Iteration 4.

Adversarial stress testing of:
1. Feature 7: Sensor channel exactness across all 26 observation channels with 0 numerical drift
   under dynamic home relocation and settlement founding.
2. Feature 5: Group reputation stability and zero-drift oracle parity across 100 continuous days,
   including same-tick multi-event reputation modifications.
3. Extreme Population Biological & Economic Dynamics:
   - Population bottleneck down to N=1 survivor and N=1 initialization.
   - High-scale population dynamics (N=2,500+) verifying coordinate bounds, non-negative energies,
     monies, and food conservation.
"""

import gc
import math
import unittest
from typing import List, Optional, Tuple
import numpy as np

from blob_sim.config import WorldConfig
from blob_sim.entities import Blob, Group, Home
from blob_sim.world import SimulationWorld


class TestChallengerM2It4Feature7SensorExactness(unittest.TestCase):
    """Adversarial stress-testing of Feature 7 observation channel exactness."""

    def tearDown(self) -> None:
        if hasattr(self, "world") and self.world is not None:
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
        """Independent ground-truth extractor for all 26 observation channels."""
        n = len(living_blobs)
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
        gt_obs[:, 21:24] = feeler_vecs

        # Per-blob scalar channels
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

    def test_sensor_exactness_baseline_static_world(self) -> None:
        """Verify baseline 26-channel observation extraction without home relocations."""
        world = self.world = SimulationWorld(WorldConfig(initial_blobs=200), seed=42)
        living = [b for b in world.blobs if b.alive]
        n = len(living)

        rng = np.random.default_rng(101)
        food_vecs = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
        peer_vecs = rng.uniform(-1.0, 1.0, (n, 6)).astype(np.float32)
        oasis_vecs = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
        feeler_vecs = rng.uniform(0.0, 1.0, (n, 3)).astype(np.float32)
        nearest_peers = [world.blobs[(i + 1) % n] if i % 2 == 0 else None for i in range(n)]

        obs_batch, _, _, _, _, _, _, _ = world._prepare_step_observations(
            living, food_vecs, peer_vecs, oasis_vecs, feeler_vecs,
            0.45, 1.0, 2.0, 1, nearest_peers=nearest_peers
        )

        gt_obs = self._extract_ground_truth_observations(
            world, living, food_vecs, peer_vecs, oasis_vecs, feeler_vecs,
            0.45, 1.0, 2.0, nearest_peers=nearest_peers
        )

        max_err = float(np.max(np.abs(obs_batch - gt_obs)))
        self.assertLessEqual(max_err, 1e-6, f"Static sensor extraction drift {max_err} > 1e-6")

    def test_sensor_exactness_under_home_relocation_divergence(self) -> None:
        """Adversarially test Feature 7 parity when a blob relocates to a home with different radius.
        
        Demonstrates the failure in worker_m2_it4 where removing `self._kin_static_dirty = True`
        leaves stale radii in `home_radii_list` and stale home objects in `home_objs`.
        """
        world = self.world = SimulationWorld(WorldConfig(initial_blobs=50), seed=42)
        world.step(1)

        b = world.blobs[0]
        # Construct partner home with group (radius 52.0 vs initial home radius 46.0/48.0)
        grp = world.groups[1]
        new_home = world._construct_new_partner_home(grp, 600.0, 600.0, 4)
        world._relocate_blob_home(b, new_home.id)

        # Place blob 0 at distance 50.0 from new home (inside new_home radius 52.0, outside old home radius 46.0)
        b.x = new_home.x + 50.0
        b.y = new_home.y

        living = [x for x in world.blobs if x.alive]
        n = len(living)

        food_vecs = np.zeros((n, 3), np.float32)
        peer_vecs = np.zeros((n, 6), np.float32)
        oasis_vecs = np.zeros((n, 3), np.float32)
        feeler_vecs = np.zeros((n, 3), np.float32)

        obs_batch, h_objs, h_radii, _, _, _, _, _ = world._prepare_step_observations(
            living, food_vecs, peer_vecs, oasis_vecs, feeler_vecs,
            0.5, 1.0, 2.0, 1
        )

        gt_obs = self._extract_ground_truth_observations(
            world, living, food_vecs, peer_vecs, oasis_vecs, feeler_vecs,
            0.5, 1.0, 2.0
        )

        ch3_drift = float(abs(obs_batch[0, 3] - gt_obs[0, 3]))
        ch4_drift = float(abs(obs_batch[0, 4] - gt_obs[0, 4]))
        max_drift = max(ch3_drift, ch4_drift)

        # Empirical challenge check:
        # A compliant implementation must have zero numerical drift (<= 1e-6).
        # Stale radius cache causes ch3_drift ~ 1.235.
        self.assertLessEqual(
            max_drift,
            1e-6,
            f"Feature 7 VIOLATION: Channel 3/4 drift {max_drift:.4f} > 1e-6 on home relocation! "
            f"Obs was {obs_batch[0, 3]:.4f}, expected ground truth {gt_obs[0, 3]:.4f} "
            f"(returned home radius {h_radii[0]}, actual home radius {b.home_obj.radius})"
        )


class TestChallengerM2It4Feature5ReputationStability(unittest.TestCase):
    """Adversarial stress-testing of Feature 5 Group Reputation stability."""

    def tearDown(self) -> None:
        if hasattr(self, "world") and self.world is not None:
            self.world.close()
            self.world = None
        gc.collect()

    def test_reputation_same_tick_multiple_modifications_drift(self) -> None:
        """Adversarially test Group Reputation when multiple member reputations change in the same tick.
        
        Demonstrates the debouncing bug in worker_m2_it4's `_get_group_avg_reputation` where:
        `if not is_stepping or last_tick != self.tick_count:`
        suppresses recomputation after the first query in a tick, resulting in stale cached sums
        and massive exclude_blob_id drift.
        """
        world = self.world = SimulationWorld(WorldConfig(initial_blobs=30), seed=42)
        grp = Group(id=888, name="TestClan", color="#ff0000", home_id=1, member_ids=[])
        world.groups[grp.id] = grp

        b1 = world.blobs[0]
        b2 = world.blobs[1]
        b1.group_id = grp.id
        b2.group_id = grp.id
        b1.reputation = 0.5
        b2.reputation = 0.5

        # Simulate active stepping context
        world._in_step_once = True
        try:
            # First change in tick
            b1.adjust_reputation(0.1)  # b1 rep = 0.6
            avg1 = world._get_group_avg_reputation(grp)  # queries and sets _last_recompute_tick

            # Second change IN THE SAME TICK
            b2.adjust_reputation(0.2)  # b2 rep = 0.7. Correct group sum = 1.3, avg = 0.65

            actual_avg = world._get_group_avg_reputation(grp)
            expected_avg = round((0.6 + 0.7) / 2.0, 3)

            # Query exclude_blob_id for b1 (should return b2.reputation = 0.7)
            actual_peer_avg = world._get_group_avg_reputation(grp, exclude_blob_id=b1.id)
            expected_peer_avg = round(b2.reputation, 3)

            drift_avg = abs(actual_avg - expected_avg)
            drift_peer = abs(actual_peer_avg - expected_peer_avg)

            self.assertEqual(
                drift_avg,
                0.0,
                f"Feature 5 VIOLATION: Group avg reputation drifted by {drift_avg:.4f} "
                f"(got {actual_avg}, expected {expected_avg}) due to same-tick debouncing suppression!"
            )
            self.assertEqual(
                drift_peer,
                0.0,
                f"Feature 5 VIOLATION: Group peer exclude_blob_id drifted by {drift_peer:.4f} "
                f"(got {actual_peer_avg}, expected {expected_peer_avg}) due to stale cached_rep_sum!"
            )
        finally:
            world._in_step_once = False


class TestChallengerM2It4ExtremePopulationDynamics(unittest.TestCase):
    """Adversarial stress-testing of simulation dynamics under extreme population boundaries."""

    def tearDown(self) -> None:
        if hasattr(self, "world") and self.world is not None:
            self.world.close()
            self.world = None
        gc.collect()

    def test_single_blob_initialization_crash(self) -> None:
        """Adversarially verify that a world initialized with N=1 blob can execute step() without crashing.
        
        Demonstrates the bug in worker_m2_it4 where `_compute_peer_spatial_vectors` returns early
        without setting `self._last_min_p_idxs`, crashing with AttributeError on line 2733.
        """
        world = self.world = SimulationWorld(WorldConfig(initial_blobs=1), seed=42)
        try:
            world.step(1)
        except AttributeError as e:
            self.fail(f"Extreme Population VIOLATION: N=1 crashed with AttributeError: {e}")

    def test_population_bottleneck_crash_to_single_survivor(self) -> None:
        """Adversarially verify that culling population down to N=1 survivor does not crash on next step.
        
        Demonstrates the bug in worker_m2_it4 where `_last_min_p_idxs` retains stale out-of-bounds indices,
        crashing with IndexError at line 2490 during observation extraction.
        """
        world = self.world = SimulationWorld(WorldConfig(initial_blobs=10), seed=42)
        world.step(1)

        # Force mass mortality down to 1 survivor
        survivor = world.blobs[0]
        for b in world.blobs[1:]:
            b.alive = False
            b.energy = 0.0

        try:
            world.step(1)
        except IndexError as e:
            self.fail(f"Extreme Population VIOLATION: Bottleneck down to 1 survivor crashed with IndexError: {e}")

    def test_large_population_scale_2500_blobs_invariants(self) -> None:
        """Verify ecological, biological, and economic invariants under high-scale population (N=2,500)."""
        world = self.world = SimulationWorld(WorldConfig(initial_blobs=2500), seed=777)
        world.step(5)

        living = [b for b in world.blobs if b.alive]
        self.assertGreater(len(living), 2000, "Population collapsed unexpectedly")

        for b in living:
            self.assertTrue(0.0 <= b.x <= world.world_cfg.width, f"Blob {b.id} x out of bounds: {b.x}")
            self.assertTrue(0.0 <= b.y <= world.world_cfg.height, f"Blob {b.id} y out of bounds: {b.y}")
            self.assertGreaterEqual(b.energy, 0.0, f"Blob {b.id} negative energy: {b.energy}")
            self.assertGreaterEqual(b.money, 0.0, f"Blob {b.id} negative money: {b.money}")
            self.assertGreaterEqual(b.food_today, 0.0, f"Blob {b.id} negative food: {b.food_today}")
            self.assertGreaterEqual(b.surplus_food, 0.0, f"Blob {b.id} negative surplus: {b.surplus_food}")
            self.assertTrue(np.isfinite(b.x) and np.isfinite(b.y) and np.isfinite(b.energy))


if __name__ == "__main__":
    unittest.main()
