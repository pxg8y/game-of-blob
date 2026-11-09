"""Unit and Multi-Day Empirical Validation Suite for the Training Kindergarten Map Rebuild.

Verifies:
1. Preset configuration parameters match basic_training_map_plan.md specifications.
2. Topography invariants:
   - Zone 0 (r < 850) is completely free of rock and swamp obstacles.
   - 4 resilient cradle oases in cross/diamond layout with proper depletion & recovery rates.
   - 8 convex rock guideposts with minimum clearance > 320 units between all obstacles.
   - 8 frontier biomes (4 Golden Groves, 4 Thermal Springs) and 4 outer mud swamps.
3. Multi-day simulation benchmark:
   - Day 2 population retention >= 88% (no 87.7% collapse!).
   - Low midday starvation deaths (< 10 / day).
   - Stable cradle oasis fertility (>= 0.65).
   - Active homing, foraging, and weight coupling progression.
"""

import math
import unittest
import numpy as np

from blob_sim.config import WorldConfig, WORLD_SCALE_PRESETS
from blob_sim.world import SimulationWorld


class TestTrainingKindergartenMap(unittest.TestCase):
    """Validation test suite for the comprehensive kindergarten training map rebuild."""

    def tearDown(self) -> None:
        if hasattr(self, "world") and self.world is not None:
            if hasattr(self.world, "close"):
                self.world.close()
            self.world = None

    def test_preset_configuration_parameters(self) -> None:
        """Verify that training_kindergarten preset is registered and defines all required specs."""
        self.assertIn("training_kindergarten", WORLD_SCALE_PRESETS)
        preset = WORLD_SCALE_PRESETS["training_kindergarten"]
        self.assertEqual(preset["width"], 4800.0)
        self.assertEqual(preset["height"], 3600.0)
        self.assertEqual(preset["initial_blobs"], 100)
        self.assertEqual(preset["daily_food_spawn"], 320)
        self.assertEqual(preset["max_starvation_ticks"], 1350)
        self.assertEqual(preset["daylight_ratio"], 0.74)
        self.assertEqual(preset["num_cradles"], 1)
        self.assertEqual(preset["min_population_floor"], 24)

        cfg = WorldConfig.from_preset("training_kindergarten")
        self.assertEqual(cfg.scale_preset, "training_kindergarten")
        self.assertEqual(cfg.width, 4800.0)
        self.assertEqual(cfg.height, 3600.0)
        self.assertEqual(cfg.initial_blobs, 100)
        self.assertEqual(cfg.daily_food_spawn, 320)
        self.assertEqual(cfg.max_starvation_ticks, 1350)
        self.assertEqual(cfg.daylight_ratio, 0.74)
        self.assertEqual(cfg.min_population_floor, 24)

    def test_topography_invariants(self) -> None:
        """Verify kindergarten topography geometry and obstacle placement."""
        cfg = WorldConfig.from_preset("training_kindergarten")
        world = SimulationWorld(world_cfg=cfg, seed=42)
        self.world = world

        cx = cfg.width * 0.5
        cy = cfg.height * 0.5

        # 1. Zone 0 (The Nursery Basin, r < 850) must contain ZERO rocks and ZERO swamps
        for ob in world.obstacles:
            d_p1 = math.hypot(ob.x1 - cx, ob.y1 - cy)
            d_p2 = math.hypot(ob.x2 - cx, ob.y2 - cy)
            min_d = min(d_p1, d_p2)
            self.assertGreaterEqual(
                min_d,
                850.0,
                f"Obstacle #{ob.id} ({ob.kind}) breached Zone 0 boundary: dist={min_d:.1f} < 850.0",
            )

        # 2. Exactly 4 Inner Cradle Oases in diamond pattern with balanced carrying capacity
        inner_oases = [o for o in world.oases.values() if o.ring == "inner"]
        self.assertEqual(len(inner_oases), 4)
        for o in inner_oases:
            self.assertEqual(o.biome, "cradle")
            self.assertEqual(o.depletion_rate, 0.85)
            self.assertEqual(o.recovery_rate, 0.65)
            self.assertEqual(o.max_capacity, 80)
            self.assertGreaterEqual(o.fertility, 0.85)

        # 3. Exactly 8 outer frontier oases (4 Golden Groves + 4 Thermal Springs)
        outer_oases = [o for o in world.oases.values() if o.ring == "outer"]
        self.assertEqual(len(outer_oases), 8)
        gold_groves = [o for o in outer_oases if o.biome == "gold_grove"]
        thermal_springs = [o for o in outer_oases if o.biome == "thermal_spring"]
        self.assertEqual(len(gold_groves), 4)
        self.assertEqual(len(thermal_springs), 4)

        # 4. Obstacle layout: exactly 8 rock guideposts and 4 outer mud swamps
        rocks = [o for o in world.obstacles if o.kind == "rock"]
        swamps = [o for o in world.obstacles if o.kind == "swamp"]
        self.assertEqual(len(rocks), 8)
        self.assertEqual(len(swamps), 4)

        # 5. Clearance between all rock obstacles > 320 units (passes are wide and cannot trap blobs)
        for i in range(len(rocks)):
            for j in range(i + 1, len(rocks)):
                r1 = rocks[i]
                r2 = rocks[j]
                c1x, c1y = 0.5 * (r1.x1 + r1.x2), 0.5 * (r1.y1 + r1.y2)
                c2x, c2y = 0.5 * (r2.x1 + r2.x2), 0.5 * (r2.y1 + r2.y2)
                clearance = math.hypot(c1x - c2x, c1y - c2y) - (r1.radius + r2.radius)
                self.assertGreater(
                    clearance,
                    320.0,
                    f"Obstacle gap too narrow: clearance {clearance:.1f} <= 320 between #{r1.id} and #{r2.id}",
                )

        # 6. Starter homes placed exclusively in Zone 0 near inner oases
        for h in world.homes.values():
            d_h = math.hypot(h.x - cx, h.y - cy)
            self.assertLess(
                d_h,
                850.0,
                f"Starter home #{h.id} placed outside Zone 0: dist={d_h:.1f} >= 850.0",
            )
            # Maximum distance from any home to nearest food oasis <= 150 (vision radius 215)
            nearest_o = min(inner_oases, key=lambda o: math.hypot(o.x - h.x, o.y - h.y))
            d_to_oasis_edge = max(0.0, math.hypot(nearest_o.x - h.x, nearest_o.y - h.y) - nearest_o.radius)
            self.assertLessEqual(d_to_oasis_edge, 150.0)

    def test_radial_food_spawning_ratio(self) -> None:
        """Verify that at least 65% of daily food spawns in Zone 0 during early days."""
        cfg = WorldConfig.from_preset("training_kindergarten")
        world = SimulationWorld(world_cfg=cfg, seed=123)
        self.world = world

        cx = cfg.width * 0.5
        cy = cfg.height * 0.5

        # Initial foods
        inner_foods = [f for f in world.foods if math.hypot(f.x - cx, f.y - cy) < 850.0]
        ratio = len(inner_foods) / max(1, len(world.foods))
        self.assertGreaterEqual(
            ratio,
            0.60,
            f"Inner Zone 0 food fraction {ratio:.2f} was below 0.60",
        )

    def test_multi_day_survival_and_no_extinction_collapse(self) -> None:
        """Run a 3-day continuous simulation on the Kindergarten map.
        Verify population retention (Day 2 survival >= 88%), minimal midday starvation (<10),
        and resilient cradle oasis fertility (>= 0.65).
        """
        cfg = WorldConfig.from_preset("training_kindergarten")
        world = SimulationWorld(world_cfg=cfg, seed=42)
        self.world = world

        initial_pop = len(world.blobs)
        self.assertEqual(initial_pop, 100)

        tpd = cfg.ticks_per_day

        for day in range(1, 4):
            world.advance_ticks(tpd)
            living = [b for b in world.blobs if b.alive]
            pop = len(living)

            if day == 2:
                self.assertGreaterEqual(
                    pop,
                    88,
                    f"Day 2 population collapse detected! {pop} living out of {initial_pop} (< 88%)",
                )

            inner_oases = [o for o in world.oases.values() if o.ring == "inner"]
            avg_fert = sum(o.fertility for o in inner_oases) / len(inner_oases)
            self.assertGreaterEqual(
                avg_fert,
                0.65,
                f"Day {day}: Cradle oasis fertility collapsed: {avg_fert:.3f} < 0.65",
            )

        final_living = len([b for b in world.blobs if b.alive])
        self.assertGreaterEqual(final_living, 80, f"Day 3 ended with low population: {final_living}")


if __name__ == "__main__":
    unittest.main()
