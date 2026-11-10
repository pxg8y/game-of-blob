"""Unit tests for egocentric reference frame navigation, sensor transformation, and kinematics parity."""

import math
import unittest
import numpy as np

from blob_sim.config import WorldConfig, NetworkConfig
from blob_sim.entities import Blob, Genes, FoodItem
from blob_sim.neural import BlobBrain
from blob_sim.spatial import (
    apply_kinematics_and_metabolism_batch_numba,
    transform_observations_to_egocentric_numba,
)
from blob_sim.world import SimulationWorld


class TestEgocentricNavigation(unittest.TestCase):
    """Test suite for egocentric reference frame navigation and differential drive kinematics."""

    def test_rotational_invariance(self) -> None:
        """
        Rotationsinvarianz-Test:
        Platziere einen Blob im Modus egocentric_frame=True nacheinander mit heading in [0.0, pi/2, pi, -pi/2].
        Platziere fuer jeden Winkel ein Futterstueck bei (bx + 100 * cos(hd) - 50 * sin(hd), by + 100 * sin(hd) + 50 * cos(hd)).
        Pruefe, dass _prepare_step_observations in allen 4 Himmelsrichtungen identische Werte fuer
        obs_batch[0, 0] (> 0) und obs_batch[0, 1] (> 0) innerhalb atol=1e-5 liefert.
        """
        headings = [0.0, math.pi / 2.0, math.pi, -math.pi / 2.0]
        results = []

        bx, by = 1200.0, 1200.0
        net_cfg = NetworkConfig(depth=2, width=24)
        for hd in headings:
            cfg = WorldConfig.from_preset("classic", initial_blobs=1, egocentric_frame=True)
            world = SimulationWorld(world_cfg=cfg, seed=123)
            world.blobs.clear()
            world.foods.clear()

            brain = BlobBrain(config=net_cfg, rng=np.random.default_rng(123), innate_seed=0.72)
            blob = Blob(
                blob_id=1,
                x=bx,
                y=by,
                home_id=1,
                home_x=bx,
                home_y=by,
                brain=brain,
                genes=Genes(),
            )
            blob.heading = float(hd)
            world.blobs.append(blob)

            fx = bx + 100.0 * math.cos(hd) - 50.0 * math.sin(hd)
            fy = by + 100.0 * math.sin(hd) + 50.0 * math.cos(hd)
            world.foods.append(FoodItem(id=1, x=fx, y=fy, nutrition=1.0))
            world._sync_food_arrays()

            (
                food_vecs,
                peer_vecs,
                nearest_peers,
                nearest_peer_dists,
                oasis_vecs,
                feeler_vecs,
            ) = world._compute_spatial_perception([blob], night_mode=False, dt=1)

            (
                obs_batch,
                home_objs,
                home_radii,
                dx_foods,
                dy_foods,
                dx_homes,
                dy_homes,
                inspect_idx,
            ) = world._prepare_step_observations(
                [blob],
                food_vecs,
                peer_vecs,
                oasis_vecs,
                feeler_vecs,
                tod=0.2,
                surv_target=1.0,
                repro_target=2.0,
                dt=1,
                nearest_peers=nearest_peers,
            )

            ego_fx = float(obs_batch[0, 0])
            ego_fy = float(obs_batch[0, 1])
            self.assertGreater(ego_fx, 0.0, f"Forward food vector must be positive, got {ego_fx}")
            self.assertGreater(ego_fy, 0.0, f"Lateral food vector must be positive, got {ego_fy}")
            results.append((ego_fx, ego_fy))

        ref_fx, ref_fy = results[0]
        for idx, (efx, efy) in enumerate(results[1:], start=1):
            np.testing.assert_allclose(
                efx, ref_fx, atol=1e-5,
                err_msg=f"Forward food component mismatch at heading index {idx}: {efx} vs {ref_fx}"
            )
            np.testing.assert_allclose(
                efy, ref_fy, atol=1e-5,
                err_msg=f"Lateral food component mismatch at heading index {idx}: {efy} vs {ref_fy}"
            )

    def test_raw_transform_observations_isolated(self) -> None:
        """
        Test isolated execution of transform_observations_to_egocentric_numba on unrotated world vectors.
        """
        obs = np.zeros((1, 26), dtype=np.float32)
        # 5 spatial channels in world coordinates:
        # (0, 1) Food: (100, 50)
        # (3, 4) Home: (200, 0)
        # (6, 7) Peer: (0, 80)
        # (16, 17) Oasis: (-40, 30)
        # (19, 20) Memory: (10, -20)
        obs[0, 0], obs[0, 1] = 100.0, 50.0
        obs[0, 3], obs[0, 4] = 200.0, 0.0
        obs[0, 6], obs[0, 7] = 0.0, 80.0
        obs[0, 16], obs[0, 17] = -40.0, 30.0
        obs[0, 19], obs[0, 20] = 10.0, -20.0

        hd = np.array([math.pi / 2.0], dtype=np.float32)
        transform_observations_to_egocentric_numba(obs, hd)

        # At theta = pi/2: c = 0, s = 1.
        # forward = dx * 0 + dy * 1 = dy
        # lateral = -dx * 1 + dy * 0 = -dx
        np.testing.assert_allclose(obs[0, 0], 50.0, atol=1e-5)
        np.testing.assert_allclose(obs[0, 1], -100.0, atol=1e-5)

        np.testing.assert_allclose(obs[0, 3], 0.0, atol=1e-5)
        np.testing.assert_allclose(obs[0, 4], -200.0, atol=1e-5)

        np.testing.assert_allclose(obs[0, 6], 80.0, atol=1e-5)
        np.testing.assert_allclose(obs[0, 7], 0.0, atol=1e-5)

        np.testing.assert_allclose(obs[0, 16], 30.0, atol=1e-5)
        np.testing.assert_allclose(obs[0, 17], 40.0, atol=1e-5)

        np.testing.assert_allclose(obs[0, 19], -20.0, atol=1e-5)
        np.testing.assert_allclose(obs[0, 20], -10.0, atol=1e-5)

    def test_batch_scalar_parity(self) -> None:
        """
        Batch-vs-Skalar-Paritätstest:
        Prüfe für 200 Blobs mit zufälligen Aktionen, Positionen, Felsen und Sümpfen,
        dass _apply_kinematics_and_metabolism_batch und _apply_kinematics_and_metabolism
        bei egocentric_mode=True innerhalb 1e-4 übereinstimmen.
        """
        cfg = WorldConfig.from_preset("classic", initial_blobs=200, egocentric_frame=True)
        world = SimulationWorld(world_cfg=cfg, seed=42)
        n = 200

        rng = np.random.default_rng(999)
        actions = rng.uniform(-1.0, 1.0, size=(n, 6)).astype(np.float32)

        batch_blobs = []
        scalar_blobs = []
        home_objs = []
        home_radii = []

        net_cfg = NetworkConfig(depth=2, width=24)
        for i in range(n):
            bx = float(rng.uniform(100.0, cfg.width - 100.0))
            by = float(rng.uniform(100.0, cfg.height - 100.0))
            b_vx = float(rng.uniform(-2.0, 2.0))
            b_vy = float(rng.uniform(-2.0, 2.0))
            hd = float(rng.uniform(-math.pi, math.pi))
            energy = float(rng.uniform(20.0, 80.0))
            food_val = float(rng.uniform(0.0, 2.0))

            brain = BlobBrain(config=net_cfg, rng=rng, innate_seed=0.0)
            b1 = Blob(
                blob_id=i + 1,
                x=bx,
                y=by,
                home_id=1,
                home_x=bx,
                home_y=by,
                brain=brain,
                genes=Genes(),
            )
            b1.vx = b_vx
            b1.vy = b_vy
            b1.heading = hd
            b1.energy = energy
            b1.food_today = food_val

            b2 = Blob(
                blob_id=i + 1001,
                x=bx,
                y=by,
                home_id=1,
                home_x=bx,
                home_y=by,
                brain=brain,
                genes=Genes(),
            )
            b2.vx = b_vx
            b2.vy = b_vy
            b2.heading = hd
            b2.energy = energy
            b2.food_today = food_val

            h = world.homes.get(b1.home_id)
            hr = h.radius if h is not None else 48.0
            home_objs.append(h)
            home_radii.append(hr)

            batch_blobs.append(b1)
            scalar_blobs.append(b2)

        # 1. Scalar kinematics execution
        for i in range(n):
            b = scalar_blobs[i]
            mx = float(actions[i, 0])
            my = float(actions[i, 1])
            sp = float((actions[i, 2] + 1.0) * 0.5)
            world._apply_kinematics_and_metabolism(
                blob=b,
                move_x=mx,
                move_y=my,
                sprint=sp,
                home_obj=home_objs[i],
                home_radius=home_radii[i],
                night_mode=False,
                dt_scale=1.0,
                egocentric_mode=True,
            )

        # 2. Batch kinematics execution
        world._apply_kinematics_and_metabolism_batch(
            living_blobs=batch_blobs,
            actions_batch=actions,
            home_objs=home_objs,
            home_radii=home_radii,
            night_mode=False,
            dt_scale=1.0,
        )

        # 3. Assert parity within 1e-4 tolerance
        for i in range(n):
            b_batch = batch_blobs[i]
            b_scalar = scalar_blobs[i]

            diff_x = abs(b_batch.x - b_scalar.x)
            diff_y = abs(b_batch.y - b_scalar.y)
            diff_vx = abs(b_batch.vx - b_scalar.vx)
            diff_vy = abs(b_batch.vy - b_scalar.vy)
            diff_hd = abs(math.atan2(math.sin(b_batch.heading - b_scalar.heading), math.cos(b_batch.heading - b_scalar.heading)))
            diff_e = abs(b_batch.energy - b_scalar.energy)

            self.assertLessEqual(diff_x, 1e-4, f"Blob {i} x mismatch: {b_batch.x} vs {b_scalar.x}")
            self.assertLessEqual(diff_y, 1e-4, f"Blob {i} y mismatch: {b_batch.y} vs {b_scalar.y}")
            self.assertLessEqual(diff_vx, 1e-4, f"Blob {i} vx mismatch: {b_batch.vx} vs {b_scalar.vx}")
            self.assertLessEqual(diff_vy, 1e-4, f"Blob {i} vy mismatch: {b_batch.vy} vs {b_scalar.vy}")
            self.assertLessEqual(diff_hd, 1e-4, f"Blob {i} heading mismatch: {b_batch.heading} vs {b_scalar.heading}")
            self.assertLessEqual(diff_e, 1e-4, f"Blob {i} energy mismatch: {b_batch.energy} vs {b_scalar.energy}")

    def test_steering_reflex(self) -> None:
        """
        Lenk-Reflex-Test:
        Prüfe, dass ein Blob mit Standard-BlobBrain (innate_seed=0.72) im egocentric_frame=True-Modus
        ein seitlich vor ihm liegendes Futterstück aktiv ansteuert und die Distanz über 25 Ticks deutlich verringert.
        """
        cfg = WorldConfig.from_preset("training_kindergarten", initial_blobs=1, egocentric_frame=True)
        world = SimulationWorld(world_cfg=cfg, seed=42)
        world.blobs.clear()
        world.foods.clear()

        # Place blob facing positive x (heading = 0.0)
        bx, by = 2000.0, 1800.0
        net_cfg = NetworkConfig(depth=2, width=24)
        brain = BlobBrain(config=net_cfg, rng=np.random.default_rng(123), innate_seed=0.72)
        blob = Blob(
            blob_id=1,
            x=bx,
            y=by,
            home_id=1,
            home_x=bx,
            home_y=by,
            brain=brain,
            genes=Genes(),
        )
        blob.heading = 0.0
        world.blobs.append(blob)

        # Place food ahead and to the right (+lateral): hd=0 -> ahead=x, right=y
        fx, fy = bx + 90.0, by + 45.0
        food = FoodItem(id=1, x=fx, y=fy, nutrition=1.0)
        world.foods.append(food)
        world._sync_food_arrays()

        init_dist = math.hypot(blob.x - fx, blob.y - fy)

        # Step simulation for 25 ticks
        for _ in range(25):
            world.step(1)

        final_dist = math.hypot(blob.x - fx, blob.y - fy)
        # Verify distance decreased significantly (at least by 35% of initial distance)
        dist_reduction = init_dist - final_dist
        self.assertGreater(
            dist_reduction,
            init_dist * 0.35,
            f"Steering reflex failed to significantly approach food! Init dist: {init_dist:.2f}, Final dist: {final_dist:.2f}"
        )


if __name__ == "__main__":
    unittest.main()
