"""
Adversarial Empirical Challenge Suite for Milestone 2 Iteration 10:
In-Place Home Radius Updates in _federate_home_into_group(), ULP Drift, Boundary Stability, and Compaction Invariants.
"""

import math
import unittest
import numpy as np

from blob_sim.config import NetworkConfig, TrainingPreferences, WorldConfig
from blob_sim.entities import Blob, Genes, Group, Home
from blob_sim.neural import INPUT_DIM, OUTPUT_DIM
from blob_sim.world import SimulationWorld


class TestChallengerM2It10FederationParityAndULPDrift(unittest.TestCase):
    """Adversarial stress-testing of in-place home radius updates in _federate_home_into_group()."""

    def tearDown(self) -> None:
        if hasattr(self, "world") and self.world is not None:
            if hasattr(self.world, "close"):
                self.world.close()
            self.world = None
        import gc
        gc.collect()

    def test_direct_federation_in_place_exact_parity_with_full_sync(self) -> None:
        """Adversarially verify that in-place updates match _sync_static_blob_kin_arrays with 0 ULP drift."""
        world = SimulationWorld(WorldConfig(initial_blobs=60), seed=42)
        self.world = world
        world.step(1)

        # Pick an ungrouped home with residents
        ungrouped_homes = [h for h in world.homes.values() if h.group_id is None and h.resident_ids]
        self.assertTrue(len(ungrouped_homes) > 0, "Must have ungrouped homes with residents")
        target_home = ungrouped_homes[0]
        self.assertIn(target_home.radius, (46.0, 48.0))

        # Pick or create a group
        grp = world.groups.get(1)
        if grp is None:
            grp = Group(id=99, name="TestClan", color="#ff0000", home_id=target_home.id, home_ids=[target_home.id])
            world.groups[99] = grp

        # Execute in-place federation
        world._federate_home_into_group(target_home, grp)

        self.assertEqual(target_home.group_id, grp.id)
        self.assertEqual(target_home.radius, 52.0)
        self.assertIn(target_home.id, grp.home_ids)

        # For every resident, verify in-place buffers were updated
        n = len(world.blobs)
        for rid in target_home.resident_ids:
            b = world._blobs_by_id[rid]
            idx = b._sim_idx
            self.assertEqual(world._kin_home_radii[idx], 52.0)
            self.assertEqual(world._step_home_radii_list[idx], 52.0)
            self.assertIs(world._step_home_objs[idx], target_home)
            self.assertIs(b.home_obj, target_home)

        # Now save the current in-place arrays
        in_place_kin_radii = np.copy(world._kin_home_radii[:n])
        in_place_step_radii = list(world._step_home_radii_list[:n])

        # Force a full cold _sync_static_blob_kin_arrays
        world._sync_static_blob_kin_arrays(world.blobs)

        cold_sync_kin_radii = np.copy(world._kin_home_radii[:n])
        cold_sync_step_radii = list(world._step_home_radii_list[:n])

        # Check ZERO ULP drift across every single element
        diff = np.max(np.abs(in_place_kin_radii - cold_sync_kin_radii))
        self.assertEqual(diff, 0.0, f"Kinematics home radii ULP drift {diff} != 0.0")
        self.assertEqual(in_place_step_radii, cold_sync_step_radii, "Step home radii list does not match cold sync")

    def test_boundary_distance_stability_transition_at_radius_50(self) -> None:
        """Verify boundary condition when blob is between old radius (48.0) and new federated radius (52.0)."""
        world = SimulationWorld(WorldConfig(initial_blobs=20), seed=123)
        self.world = world
        world.step(1)

        b = world.blobs[0]
        h = world.homes[b.home_id]
        h.radius = 48.0
        h.group_id = None
        world._kin_home_radii[0] = 48.0
        world._step_home_radii_list[0] = 48.0
        world._step_home_objs[0] = h

        # Place blob exactly 50.0 units from home
        b.x = h.x + 50.0
        b.y = h.y
        world._kin_xs[0] = b.x
        world._kin_ys[0] = b.y
        world._kin_home_xs[0] = h.x
        world._kin_home_ys[0] = h.y

        # Under 48.0 radius, dist 50.0 is OUTSIDE home
        d2 = 50.0 * 50.0
        self.assertFalse(d2 <= 48.0 * 48.0, "Dist 50 should be outside radius 48")

        # Now federate home into group
        grp = Group(id=88, name="FrontierClan", color="#00ff00", home_id=h.id, home_ids=[h.id])
        world.groups[88] = grp
        world._federate_home_into_group(h, grp)

        # In-place radius must now be 52.0
        self.assertEqual(world._kin_home_radii[0], 52.0)
        self.assertEqual(world._step_home_radii_list[0], 52.0)

        # Dist 50.0 is now INSIDE home (50.0 <= 52.0)
        self.assertTrue(d2 <= 52.0 * 52.0, "Dist 50 should be inside radius 52")

        # Execute observation extraction and kinematics step
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

        # In-home channel 3 and 4 must be EXACTLY 0.0
        self.assertEqual(obs_batch[0, 3], 0.0, f"Channel 3 must be 0.0 at home, got {obs_batch[0, 3]}")
        self.assertEqual(obs_batch[0, 4], 0.0, f"Channel 4 must be 0.0 at home, got {obs_batch[0, 4]}")
        self.assertEqual(world._kin_is_at_home_int[0], 1, "Blob must be flagged as at_home in buffer")

    def test_federation_resilience_to_dead_or_corrupted_resident_ids(self) -> None:
        """Adversarially verify that dead or stale resident IDs do not crash or corrupt buffer indices."""
        world = SimulationWorld(WorldConfig(initial_blobs=30), seed=777)
        self.world = world
        world.step(1)

        h = world.homes[1]
        # Inject non-existent blob ID, and a dead blob ID
        dead_blob = world.blobs.pop()  # remove last blob
        dead_blob.alive = False
        dead_blob._kin_idx = -1
        dead_blob._sim_idx = -1
        world._blobs_by_id[dead_blob.id] = dead_blob

        h.resident_ids = [dead_blob.id, 999999]  # 999999 does not exist in _blobs_by_id

        grp = Group(id=77, name="GhostClan", color="#0000ff", home_id=h.id, home_ids=[h.id])
        world.groups[77] = grp

        # Should complete without error and without touching any valid buffer slots
        snapshot_radii = np.copy(world._kin_home_radii[:len(world.blobs)])
        world._federate_home_into_group(h, grp)

        self.assertEqual(h.radius, 52.0)
        # Remaining valid blobs must NOT have been corrupted
        diff = np.max(np.abs(world._kin_home_radii[:len(world.blobs)] - snapshot_radii))
        self.assertEqual(diff, 0.0, "Dead/invalid resident must not alter surviving blob buffers")

    def test_all_federation_call_sites_parity(self) -> None:
        """Verify parity and zero drift across social encounters, colonization, and group founding."""
        world = SimulationWorld(WorldConfig(initial_blobs=40), seed=999)
        self.world = world
        world.step(1)

        # 1. Social encounter group founding
        b0 = world.blobs[0]
        b1 = world.blobs[1]
        b0.group_id = None
        b1.group_id = None
        world._handle_social_encounter(b0, b1)

        grp_id = b0.group_id
        self.assertIsNotNone(grp_id)
        grp = world.groups[grp_id]
        h0 = world.homes[b0.home_id]
        self.assertEqual(h0.radius, 52.0)
        idx0 = b0._sim_idx
        self.assertEqual(world._kin_home_radii[idx0], 52.0)
        self.assertEqual(world._step_home_radii_list[idx0], 52.0)

        # 2. Join blob to group
        b2 = world.blobs[2]
        b2.group_id = None
        world._join_blob_to_group(b2, grp_id)
        idx2 = b2._sim_idx
        h2 = world.homes[b2.home_id]
        self.assertEqual(h2.radius, 52.0)
        self.assertEqual(world._kin_home_radii[idx2], 52.0)

        # 3. Construct partner home with group
        new_h = world._construct_new_partner_home(grp, 500.0, 500.0, 4)
        self.assertEqual(new_h.radius, 52.0)
        self.assertEqual(new_h.group_id, grp_id)
        self.assertIn(new_h.id, grp.home_ids)

        # Check full sync parity
        n = len(world.blobs)
        cur_kin_radii = np.copy(world._kin_home_radii[:n])
        world._sync_static_blob_kin_arrays(world.blobs)
        cold_kin_radii = np.copy(world._kin_home_radii[:n])

        ulp_diff = np.max(np.abs(cur_kin_radii - cold_kin_radii))
        self.assertEqual(ulp_diff, 0.0, f"Call-site federation ULP drift {ulp_diff} != 0.0")


if __name__ == "__main__":
    unittest.main()
