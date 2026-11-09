"""Numerical Parity Multi-Seed Verification across Seeds 42, 43, 44, 45, 46."""

import unittest
from typing import List
import numpy as np

from blob_sim.config import WorldConfig
from blob_sim.entities import Blob
from blob_sim.world import SimulationWorld


class TestFeature6MultiSeedParity(unittest.TestCase):
    def tearDown(self) -> None:
        if hasattr(self, "world") and self.world is not None:
            if hasattr(self.world, "close"):
                self.world.close()
            self.world = None
        import gc
        gc.collect()

    def test_feature6_parity_across_seeds(self):
        seeds = [42, 43, 44, 45, 46]
        parity_results = []

        for seed in seeds:
            world = SimulationWorld(world_cfg=WorldConfig(initial_blobs=2000), seed=seed)
            self.world = world
            scalar_blobs: List[Blob] = []
            try:
                living = [b for b in world.blobs if b.alive][:2000]
                n = len(living)

                rng = np.random.default_rng(seed * 17)
                actions = rng.uniform(-1.0, 1.0, size=(n, 6)).astype(np.float32)

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

                # 1. Scalar physics
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

                # 2. Batch Numba physics
                world._apply_kinematics_and_metabolism_batch(
                    living_blobs=living,
                    actions_batch=actions,
                    home_objs=home_objs,
                    home_radii=home_radii,
                    night_mode=False,
                    dt_scale=1.0,
                )

                # 3. Differences
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

                self.assertLess(max_dvx, 1e-5)
                self.assertLess(max_dvy, 1e-5)
                self.assertLess(max_dhd, 1e-5)
                self.assertLess(max_den, 1e-5)
                self.assertLess(max_rel_dx, 1.5e-7)
                self.assertLess(max_rel_dy, 1.5e-7)
                self.assertLess(max_dx, 3.5e-4)
                self.assertLess(max_dy, 3.5e-4)

                parity_results.append({
                    "seed": seed,
                    "max_rel_dx": max_rel_dx,
                    "max_rel_dy": max_rel_dy,
                    "max_abs_dx": max_dx,
                    "max_abs_dy": max_dy,
                    "max_dvx": max_dvx,
                    "max_dvy": max_dvy,
                    "max_dhd": max_dhd,
                    "max_den": max_den,
                })
            finally:
                if hasattr(world, "close"):
                    world.close()
                self.world = None
                scalar_blobs.clear()
                import gc
                gc.collect()

        print("\n" + "=" * 110)
        print("FEATURE 6 KINEMATICS NUMERICAL PARITY MULTI-SEED VERIFICATION (SEEDS 42-46)")
        print("=" * 110)
        header = f"| {'Seed':<4} | {'Max Rel dX':<12} | {'Max Rel dY':<12} | {'Max Abs dX':<12} | {'Max Abs dY':<12} | {'Max dVX':<11} | {'Max dVY':<11} | {'Max dHD':<11} | {'Max dEN':<11} |"
        print(header)
        print("|" + "|".join(["-" * (len(col) + 2) for col in header.split("|")[1:-1]]) + "|")
        for p in parity_results:
            print(
                f"| {p['seed']:<4} | {p['max_rel_dx']:>10.4e} | {p['max_rel_dy']:>10.4e} | {p['max_abs_dx']:>10.4e} | {p['max_abs_dy']:>10.4e} "
                f"| {p['max_dvx']:>9.4e} | {p['max_dvy']:>9.4e} | {p['max_dhd']:>9.4e} | {p['max_den']:>9.4e} |"
            )
        print("=" * 110 + "\n")


if __name__ == "__main__":
    unittest.main()
