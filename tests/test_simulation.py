"""Unit and integration tests for Blob Life simulation mechanics."""

import unittest
import numpy as np
from blob_sim.config import NetworkConfig, TrainingPreferences, WorldConfig
from blob_sim.entities import Group
from blob_sim.neural import BlobBrain, INPUT_DIM, OUTPUT_DIM
from blob_sim.world import SimulationWorld


class TestBlobSimulation(unittest.TestCase):
    def tearDown(self) -> None:
        if hasattr(self, "world") and self.world is not None:
            if hasattr(self.world, "close"):
                self.world.close()
            self.world = None
        import gc
        gc.collect()

    def test_custom_neural_network_depth_width_and_activation(self) -> None:
        cfg = NetworkConfig(depth=4, width=32, activation="gelu")
        brain = BlobBrain(config=cfg)
        self.assertEqual(brain.layer_dims, [INPUT_DIM, 32, 32, 32, 32, OUTPUT_DIM])

        obs = np.zeros((INPUT_DIM,), dtype=np.float32)
        out = brain.forward(obs, explore_noise=0.0)
        self.assertEqual(out.shape, (OUTPUT_DIM,))
        self.assertTrue(np.all(out >= -1.0) and np.all(out <= 1.0))

        # Test live architecture adaptation to custom layer list
        new_cfg = NetworkConfig(depth=2, width=16, custom_layers=[20, 12], activation="silu")
        adapted = brain.adapt_to_config(new_cfg)
        self.assertEqual(adapted.layer_dims, [INPUT_DIM, 20, 12, OUTPUT_DIM])

    def test_innate_instinct_preserved_across_all_depths(self) -> None:
        for depth in (1, 2, 3, 4):
            cfg = NetworkConfig(depth=depth, width=24, activation="tanh")
            diffs = []
            for seed in range(25):
                brain = BlobBrain(config=cfg, rng=np.random.default_rng(seed))
                obs_r = np.zeros((INPUT_DIM,), dtype=np.float32)
                obs_r[0] = 1.0
                obs_l = np.zeros((INPUT_DIM,), dtype=np.float32)
                obs_l[0] = -1.0
                out_r = brain.forward(obs_r, explore_noise=0.0)
                out_l = brain.forward(obs_l, explore_noise=0.0)
                diffs.append(float(out_r[0] - out_l[0]))
            self.assertGreater(float(np.mean(diffs)), 0.15, f"Innate instinct failed at depth={depth}")

    def test_online_learning_and_mutation(self) -> None:
        cfg = NetworkConfig(depth=2, width=16, activation="tanh")
        brain = BlobBrain(config=cfg)
        obs = np.ones((INPUT_DIM,), dtype=np.float32) * 0.5
        _ = brain.forward(obs)
        before_w0 = brain.weights[0].copy()
        brain.apply_online_learning(reward=2.0, learning_rate=0.05)
        self.assertFalse(np.allclose(before_w0, brain.weights[0]))

        # Zero-reward ticks must decay reward_baseline rather than freezing it
        high_baseline = brain.reward_baseline
        for _ in range(20):
            brain.apply_online_learning(reward=0.0, learning_rate=0.05)
        self.assertLess(brain.reward_baseline, high_baseline)

        child = brain.clone_and_mutate(mutation_rate=0.5, mutation_strength=0.2)
        self.assertEqual(child.layer_dims, brain.layer_dims)
        self.assertFalse(np.allclose(child.weights[0], brain.weights[0]))

    def test_reproduction_on_double_survival_food(self) -> None:
        world = SimulationWorld(
            world_cfg=WorldConfig(initial_blobs=10, ticks_per_day=50, survival_food_need=1.0),
            seed=123,
        )
        parent = world.blobs[0]
        parent.food_today = 2.2
        parent.x = parent.home_x
        parent.y = parent.home_y
        parent.is_at_home = True
        parent.energy = 90.0

        for b in world.blobs[1:]:
            b.food_today = 1.0
            b.energy = 80.0
            b.is_at_home = True

        world._end_of_day_accounting()
        self.assertGreaterEqual(parent.children_count, 1)
        self.assertGreaterEqual(world.total_births_all_time, 1)

    def test_starvation_when_no_food_and_no_money(self) -> None:
        world = SimulationWorld(
            world_cfg=WorldConfig(initial_blobs=10, ticks_per_day=50, survival_food_need=1.0),
            seed=99,
        )
        victim = world.blobs[0]
        victim.food_today = 0.0
        victim.money = 0.0
        victim.energy = 20.0
        victim.is_at_home = False

        world._end_of_day_accounting()
        self.assertFalse(victim.alive)

    def test_group_formation_and_food_for_money_trade(self) -> None:
        world = SimulationWorld(seed=7)
        a, b = world.blobs[0], world.blobs[1]
        a.x, a.y = 300.0, 300.0
        b.x, b.y = 305.0, 305.0

        # Form group
        world._handle_social_encounter(a, b)
        self.assertIsNotNone(a.group_id)
        self.assertEqual(a.group_id, b.group_id)

        # Give A surplus food and B money + hunger
        a.surplus_food = 1.0
        a_money_before = a.money
        b.food_today = 0.5
        b.money = 20.0

        traded = world._try_peer_trade(a, b)
        self.assertTrue(traded)
        self.assertGreater(a.money, a_money_before)
        self.assertGreater(b.food_today, 0.5)

    def test_blob_can_always_trade_even_below_survival_or_reproduction(self) -> None:
        world = SimulationWorld(seed=11)
        seller, buyer = world.blobs[0], world.blobs[1]
        # Seller only has 0.6 food_today (below 1.0 survival need and 2.0 repro need) and 0.0 surplus_food
        seller.food_today = 0.6
        seller.surplus_food = 0.0
        seller.trade_desire = 0.8  # Wants to sell for gold!
        seller_money_before = seller.money

        buyer.food_today = 0.0
        buyer.money = 15.0
        buyer.trade_desire = 0.5

        traded = world._try_peer_trade(seller, buyer)
        self.assertTrue(traded)
        self.assertAlmostEqual(seller.food_today, 0.1, places=2)
        self.assertAlmostEqual(buyer.food_today, 0.5, places=2)
        self.assertGreater(seller.money, seller_money_before)

    def test_max_residents_per_home_and_multi_home_settlements(self) -> None:
        world = SimulationWorld(
            world_cfg=WorldConfig(initial_blobs=16, max_residents_per_home=4, ticks_per_day=40),
            seed=21,
        )
        # Verify no home exceeds max_residents_per_home initially
        for h in world.homes.values():
            self.assertLessEqual(len(h.resident_ids), 4)

        # Unite two blobs from different homes into a settlement group
        b0 = world.blobs[0]
        b1 = next(b for b in world.blobs if b.home_id != b0.home_id)
        world._handle_social_encounter(b0, b1)
        grp = world.groups[b0.group_id]
        self.assertIn(b0.home_id, grp.home_ids)
        self.assertIn(b1.home_id, grp.home_ids)
        self.assertGreaterEqual(len(grp.home_ids), 2)

        # Run full step loop for 2 days and verify Hall of Fame deduplication & export/import
        for _ in range(80):
            world.step()
        hof_ids = [e["blob_id"] for e in world.hall_of_fame]
        self.assertEqual(len(hof_ids), len(set(hof_ids)))

        exported = world.export_population_json()
        world.import_population_json(exported)
        self.assertEqual(world.day_number, exported["day"])

    def test_batched_brain_pool_and_high_speed_advance_500x(self) -> None:
        world = SimulationWorld(
            world_cfg=WorldConfig(initial_blobs=25, ticks_per_day=360),
            seed=42,
        )
        # Advance 500x (500 ticks in one frame = 1 full day + 140 ticks into day 2)
        world.advance_ticks(500)
        self.assertEqual(world.day_number, 2)
        self.assertEqual(world.tick_in_day, 140)
        self.assertEqual(len(world.history), 1)
        self.assertGreater(len(world.blobs), 0)

    def test_seasons_house_upgrades_and_raid_defense(self) -> None:
        world = SimulationWorld(seed=17)
        # Verify 16-day seasonal cycle (4 days per season)
        world.day_number = 1
        self.assertEqual(world.season_info[0], "spring")
        world.day_number = 5
        self.assertEqual(world.season_info[0], "summer")
        world.day_number = 9
        self.assertEqual(world.season_info[0], "autumn")
        world.day_number = 13
        self.assertEqual(world.season_info[0], "winter")

        # Verify dynamic local price responds to supply & demand
        home = world.homes[world.blobs[0].home_id]
        home.food_depot = 0.2
        scarce_price = home.local_unit_price(5.0)
        home.food_depot = 10.0
        abundant_price = home.local_unit_price(5.0)
        self.assertGreater(scarce_price, abundant_price)

        # Verify rich residents fund house upgrades overnight
        world.blobs[0].money = 80.0
        home.gold_vault = 25.0
        world._process_house_taxes_and_upgrades(world.blobs)
        self.assertGreaterEqual(home.total_upgrades, 1)

        # Verify resource raid: large hungry raider vs small isolated target, then vs fortified watchtower
        raider, defender = world.blobs[0], world.blobs[1]
        raider.group_id = None
        defender.group_id = None
        raider.genes.size = 1.45
        defender.genes.size = 0.75
        raider.food_today = 0.0
        raider.money = 0.0
        raider.energy = 95.0
        defender.food_today = 1.2
        defender.energy = 60.0
        defender.x, defender.y = 600.0, 400.0
        raider.x, raider.y = 602.0, 402.0
        # Move defender far from home so watchtower doesn't protect
        def_home = world.homes[defender.home_id]
        def_home.x, def_home.y = 50.0, 50.0

        self.assertTrue(world._try_peer_raid(raider, defender))
        self.assertGreater(raider.food_today, 0.0)
        self.assertEqual(world.raids_today, 1)

        # Now place defender inside a Lv.3 Watchtower and verify raid is repelled
        def_home.x, def_home.y = defender.x, defender.y
        def_home.defense_level = 3
        raider.trade_cooldown = 0
        defender.trade_cooldown = 0
        raider.food_today = 0.0
        raider.energy = 60.0
        self.assertTrue(world._try_peer_raid(raider, defender))
        self.assertEqual(world.raids_repelled_today, 1)

    def test_oasis_only_food_spawning_and_overgrazing_ecology(self) -> None:
        world = SimulationWorld(seed=33)
        self.assertEqual(world.world_cfg.width, 4800.0)
        self.assertEqual(world.world_cfg.height, 3600.0)
        self.assertEqual(INPUT_DIM, 26)
        self.assertEqual(OUTPUT_DIM, 6)
        self.assertGreaterEqual(len(world.oases), 8)
        self.assertGreater(len(world.foods), 0)

        # Verify 0% random wasteland scatter: every wild food belongs to an oasis and is inside its radius
        for f in world.foods:
            self.assertIsNotNone(f.oasis_id)
            oasis = world.oases[f.oasis_id]
            dist = float(np.hypot(f.x - oasis.x, f.y - oasis.y))
            self.assertLessEqual(dist, oasis.radius * 1.02)

        # Overgraze oasis #1 and verify it stops spawning new food until it regenerates
        target_oasis = next(iter(world.oases.values()))
        target_oasis.fertility = 0.15
        target_oasis.harvest_pressure = 8.0
        self.assertTrue(target_oasis.is_overgrazed)

        world.foods.clear()
        world.spawn_daily_food(120)
        overgrazed_foods = [f for f in world.foods if f.oasis_id == target_oasis.id]
        self.assertEqual(len(overgrazed_foods), 0)

        # Step ecology without harvesting and verify fertility recovers
        fert_before = target_oasis.fertility
        for _ in range(120):
            world._step_oasis_ecology()
        self.assertGreater(target_oasis.fertility, fert_before)

    def test_rock_obstacle_physics_swamp_drag_and_feeler_sensors(self) -> None:
        world = SimulationWorld(seed=44)
        blob = world.blobs[0]

        # Place a rock wall directly in front of the blob (heading = 0.0 -> +X direction)
        world.obstacles.clear()
        world._rebuild_obstacle_arrays()
        world.toggle_rock_at(1050.0, 1000.0)
        blob.x, blob.y = 1000.0, 1000.0
        blob.heading = 0.0
        _, feeler_vecs = world._compute_oasis_and_feeler_sensors(
            [blob],
            np.array([blob.x], dtype=np.float32),
            np.array([blob.y], dtype=np.float32),
            1,
        )
        # Column 1 of feeler_vecs is Front Feeler (Fühler Vorne) -> should strongly detect the rock (> 0.25)
        self.assertGreater(float(feeler_vecs[0, 1]), 0.25)

        # Attempt to walk straight into the rock center and verify collision pushes blob out
        blob.x, blob.y = 1050.0, 1000.0
        blob.vx, blob.vy = 4.0, 0.0
        hit = world._resolve_terrain_physics(blob)
        self.assertTrue(hit)
        rock = world.obstacles[0]
        dist_to_center = float(np.hypot(blob.y - rock.y1, 0.0))
        self.assertGreaterEqual(dist_to_center + 1e-3, rock.radius)

    def test_spatial_memory_and_dynamic_colonization(self) -> None:
        world = SimulationWorld(seed=55)
        pioneer = world.blobs[0]

        # Find an outer oasis far from pioneer's current home
        far_oasis = max(
            world.oases.values(),
            key=lambda o: (o.x - pioneer.home_x) ** 2 + (o.y - pioneer.home_y) ** 2,
        )
        far_oasis.fertility = 0.95
        far_oasis.harvest_pressure = 0.0

        # Place food at far_oasis and let pioneer harvest it -> spatial memory must lock onto far_oasis
        pioneer.x, pioneer.y = far_oasis.x, far_oasis.y
        world.foods[0].x = far_oasis.x + 2.0
        world.foods[0].y = far_oasis.y + 2.0
        world.foods[0].oasis_id = far_oasis.id
        world._process_foraging([pioneer])
        self.assertEqual(pioneer.memory_oasis_id, far_oasis.id)
        self.assertGreater(pioneer.memory_strength, 0.9)

        # Trigger pioneer settlement / migration at the discovered fertile oasis
        old_home_id = pioneer.home_id
        pioneer.colonize_desire = 0.9
        pioneer.colonize_cooldown = 0
        pioneer.energy = 85.0
        pioneer.money = 12.0
        colonized = world._try_colonize_or_relocate(pioneer, tod=0.45)
        self.assertTrue(colonized)
        self.assertNotEqual(pioneer.home_id, old_home_id)
        self.assertGreaterEqual(pioneer.migrations_made, 1)
        new_home = world.homes[pioneer.home_id]
        self.assertGreater(new_home.fertility_bonus, 0.5)

    def test_low_rank_ring_buffer_traces_and_spatial_grid_equivalence(self) -> None:
        world = SimulationWorld(
            world_cfg=WorldConfig(initial_blobs=16, ticks_per_day=120),
            net_cfg=NetworkConfig(depth=6, width=64, activation="gelu"),
            seed=77,
        )
        b0 = world.blobs[0]
        w0_before = b0.brain.weights[0].copy()

        # Run 3 steps so do_trace fires and low-rank ring buffer accumulates pending updates
        for _ in range(3):
            world.step(1)
        b0.brain.apply_online_learning(1.8, 0.05)

        # Cloning must automatically sync/flush b0's pending ring-buffer updates into its weights
        child_brain = b0.brain.clone_and_mutate(mutation_rate=0.0, mutation_strength=0.0)
        self.assertFalse(np.allclose(w0_before, b0.brain.weights[0]))
        self.assertTrue(np.allclose(b0.brain.weights[0], child_brain.weights[0]))

        # Verify O(1) swap-remove dead blob compaction preserves all other blobs in place
        second_blob_id = world.blobs[1].id
        last_blob_id = world.blobs[-1].id
        world._kill_blob(world.blobs[0], reason="test")
        compacted = world._compact_living_blobs()
        self.assertEqual(compacted[0].id, last_blob_id)
        self.assertEqual(compacted[1].id, second_blob_id)

    def test_concentric_curriculum_map_skill_gates_and_start_paradigms(self) -> None:
        world = SimulationWorld(
            world_cfg=WorldConfig(initial_blobs=18, start_paradigm="curriculum"),
            seed=101,
        )
        payload = world.get_state_payload()
        self.assertEqual(len(payload["world"]["zones"]), 3)
        self.assertEqual(len(payload["world"]["skill_gates"]), 6)

        gate_kinds = [g["kind"] for g in payload["world"]["skill_gates"]]
        self.assertEqual(gate_kinds.count("chicane"), 2)
        self.assertEqual(gate_kinds.count("swamp_ford"), 2)
        self.assertEqual(gate_kinds.count("steppe_gap"), 2)

        cradle_oases = [o for o in world.oases.values() if o.biome == "cradle" and o.ring == "inner"]
        gold_oases = [o for o in world.oases.values() if o.biome == "gold_grove" and o.ring == "outer"]
        thermal_oases = [o for o in world.oases.values() if o.biome == "thermal_spring" and o.ring == "outer"]
        self.assertEqual(len(cradle_oases), 3)
        self.assertEqual(len(gold_oases), 5)
        self.assertEqual(len(thermal_oases), 4)

        # Verify all starter homes spawn inside the Cradle zone with ungrouped solo blobs
        for h in world.homes.values():
            self.assertEqual(h.zone, "cradle")
        for b in world.blobs:
            self.assertIsNone(b.group_id)

        # Verify Tabula Rasa Hardcore paradigm sets innate_seed=0.0 (pure random weights)
        world.world_cfg.start_paradigm = "tabula_rasa"
        world.reset_world()
        self.assertEqual(world.get_active_paradigm()["id"], "tabula_rasa")
        self.assertEqual(world.get_active_paradigm()["innate_seed"], 0.0)

        # Verify Rapid Exodus paradigm sets higher Cradle depletion rate
        world.world_cfg.start_paradigm = "rapid_exodus"
        world.reset_world()
        rapid_cradle = [o for o in world.oases.values() if o.biome == "cradle"][0]
        self.assertGreater(rapid_cradle.depletion_rate, 2.0)

    def test_continent_preset_multi_cradle_topography_and_binary_ws(self) -> None:
        import struct

        cfg = WorldConfig.from_preset("continent")
        cfg.initial_blobs = 240
        world = SimulationWorld(world_cfg=cfg, seed=202)

        self.assertEqual(world.world_cfg.width, 40000)
        self.assertEqual(world.world_cfg.height, 30000)
        self.assertEqual(world.world_cfg.num_cradles, 12)

        cradle_oases = [o for o in world.oases.values() if o.biome == "cradle" and o.ring == "inner"]
        frontier_oases = [o for o in world.oases.values() if o.ring == "outer"]
        self.assertEqual(len(cradle_oases), 36)
        self.assertEqual(len(frontier_oases), 72)

        state_json, binary_buf = world.get_ws_state_and_binary(include_topology=True)
        self.assertEqual(len(state_json["world"]["zones"]), 24)
        self.assertEqual(len(state_json["world"]["skill_gates"]), 72)
        self.assertGreater(len(state_json["obstacles"]), 100)

        magic, n_blobs, n_foods, tick = struct.unpack_from("<IIII", binary_buf, 0)
        self.assertEqual(magic, 0x424C4F42)
        self.assertEqual(n_blobs, 240)
        self.assertEqual(n_foods, len(world.foods))
        self.assertEqual(tick, 0)
        expected_len = 16 + n_blobs * 32 + n_foods * 12
        self.assertEqual(len(binary_buf), expected_len)

        blobs_arr = np.frombuffer(binary_buf, dtype=np.float32, count=n_blobs * 8, offset=16).reshape(n_blobs, 8)
        self.assertAlmostEqual(float(blobs_arr[0, 0]), world.blobs[0].x, places=1)
        self.assertAlmostEqual(float(blobs_arr[0, 1]), world.blobs[0].y, places=1)

        state_cached, _ = world.get_ws_state_and_binary(include_topology=False)
        self.assertNotIn("obstacles", state_cached["world"])

    def test_three_cohort_strided_scheduling_and_numba_parallel_scaling(self) -> None:
        cfg = WorldConfig.from_preset("regional")
        cfg.initial_blobs = 360
        world = SimulationWorld(world_cfg=cfg, seed=303)

        # Step 4 ticks so all 3 cohorts (0, 1, 2) execute at least once
        for _ in range(4):
            world.step(1)
        n = len(world.blobs)
        self.assertEqual(n, 360)
        self.assertTrue(np.all(world._brain_pool._out_valid_cap[:n]))

        # Kill blob 0 and verify swap-remove invalidates slot 0 in _out_valid_cap so next step recomputes it
        old_out_359 = world._brain_pool._last_out_cap[359].copy()
        world._kill_blob(world.blobs[0], reason="test")
        world._compact_living_blobs()
        self.assertFalse(bool(world._brain_pool._out_valid_cap[0]))
        world.step(1)
        self.assertTrue(bool(world._brain_pool._out_valid_cap[0]))
        new_out_0 = world._brain_pool._last_out_cap[0].copy()
        self.assertFalse(np.array_equal(new_out_0, old_out_359))

    def test_swap_remove_multi_death_cohort_invalidation_and_neural_update(self) -> None:
        cfg = WorldConfig.from_preset("regional")
        cfg.initial_blobs = 360
        world = SimulationWorld(world_cfg=cfg, seed=303)
        for _ in range(4):
            world.step(1)

        # Kill 3 blobs spanning distinct cohorts: slot 0 (c0), slot 4 (c1), slot 8 (c2)
        dead_slots = [0, 4, 8]
        tail_slots = [359, 358, 357]
        old_tail_outs = [world._brain_pool._last_out_cap[s].copy() for s in tail_slots]

        for s in sorted(dead_slots, reverse=True):
            world._kill_blob(world.blobs[s], reason="test_multi_death")
        world._compact_living_blobs()

        # All 3 relocated slots must be marked invalid prior to forward pass
        for s in dead_slots:
            self.assertFalse(bool(world._brain_pool._out_valid_cap[s]))

        world.step(1)

        # All 3 relocated slots must now be valid and have genuinely recomputed outputs
        for i, s in enumerate(dead_slots):
            self.assertTrue(bool(world._brain_pool._out_valid_cap[s]))
            new_out = world._brain_pool._last_out_cap[s].copy()
            self.assertFalse(np.array_equal(new_out, old_tail_outs[i]))

    def test_procedural_factions_house_conquest_and_colonial_secession(self) -> None:
        from blob_sim.entities import generate_faction_identity

        names = set()
        for gid in range(1, 101):
            name, color, rgb_int = generate_faction_identity(gid)
            names.add(name)
            self.assertTrue(color.startswith("#"))
            self.assertGreater(rgb_int, 0)
        self.assertGreaterEqual(len(names), 85)

        cfg = WorldConfig.from_preset("regional")
        cfg.initial_blobs = 24
        world = SimulationWorld(world_cfg=cfg, seed=404)

        # Create Group 1 and Group 2
        b0, b1 = world.blobs[0], world.blobs[1]
        b2, b3 = world.blobs[8], world.blobs[9]
        world._found_new_settlement_group(b0, b1)
        # Ensure b2 is far from b0 so it founds Group 2
        b2.x = b0.x + 3000.0
        b2.y = b0.y + 3000.0
        world._found_new_settlement_group(b2, b3)
        self.assertNotEqual(b0.group_id, b2.group_id)

        g1 = world.groups[b0.group_id]
        g2 = world.groups[b2.group_id]
        target_home = world.homes[b2.home_id]
        target_home.defense_level = 0
        target_home.terrain_defense_bonus = 0.0
        b2.x, b2.y = target_home.x, target_home.y
        b0.x, b0.y = target_home.x + 5.0, target_home.y
        b0.genes.size = 1.6
        b2.genes.size = 0.65
        b0.social_desire = -0.85
        b0.trade_desire = -0.85
        b0.food_today = 0.0
        b0.money = 0.0
        b0.energy = 100.0
        b2.energy = 12.0
        b2.food_today = 3.0
        b2.money = 30.0

        # Two raids on undefended home should conquer it for Group 1
        world._try_peer_raid(b0, b2)
        b0.trade_cooldown = 0
        b2.trade_cooldown = 0
        b0.energy = 100.0
        b0.food_today = 0.0
        world._try_peer_raid(b0, b2)
        self.assertEqual(target_home.group_id, g1.id)
        self.assertGreaterEqual(g1.conquests, 1)

        # Colonial secession: create a distant colony cluster of 2 homes and 8 members for g1 (plus 5 in capital)
        cap_home = world.homes[g1.home_id]
        for extra_cap_blob in world.blobs[4:6]:
            extra_cap_blob.group_id = g1.id
            if extra_cap_blob.id not in g1.member_ids:
                g1.member_ids.append(extra_cap_blob.id)
        col_h1 = world._construct_new_partner_home(g1, cap_home.x + 5000.0, cap_home.y + 4000.0, 8)
        col_h2 = world._construct_new_partner_home(g1, cap_home.x + 5120.0, cap_home.y + 4050.0, 8)
        for idx, member_blob in enumerate(world.blobs[10:18]):
            member_blob.group_id = g1.id
            if member_blob.id not in g1.member_ids:
                g1.member_ids.append(member_blob.id)
            dest_h = col_h1 if idx < 4 else col_h2
            world._relocate_blob_home(member_blob, dest_h.id)

        groups_before = len(world.groups)
        world._check_empire_secessions()
        self.assertEqual(len(world.groups), groups_before + 1)
        daughter = max(world.groups.values(), key=lambda g: g.id)
        self.assertEqual(daughter.parent_group_id, g1.id)
        self.assertGreaterEqual(g1.secessions, 1)

    def test_telemetry_logger_and_analyse_cockpit_payload(self):
        world = SimulationWorld(
            world_cfg=WorldConfig(initial_blobs=16, ticks_per_day=40, survival_food_need=1.0),
            seed=777,
        )
        tpd = world.world_cfg.ticks_per_day

        # Run 2 full simulated days
        for _ in range(tpd * 2):
            world.step()
        self.assertGreaterEqual(len(world.telemetry.daily_logs), 2)

        rec = world.telemetry.daily_logs[-1]
        self.assertIn("community", rec)
        self.assertIn("actions", rec)
        self.assertIn("genetics", rec)
        self.assertIn("weights", rec)
        self.assertIn("speed", rec["genetics"])
        self.assertIn("basal_cost", rec["genetics"])
        self.assertIn("reputation", rec["genetics"])
        self.assertEqual(len(rec["weights"]["coupling_matrix"]), 26)
        self.assertEqual(len(rec["weights"]["coupling_matrix"][0]), 6)
        self.assertIn("food_nav", rec["weights"]["pathways"])
        self.assertIn("reputation_drive", rec["weights"]["pathways"])

        payload = world.telemetry.get_analytics_payload(world, since_day=0)
        self.assertIn("summary", payload)
        self.assertIn("daily_logs", payload)
        self.assertIn("collapse_logs", payload)
        self.assertIn("tribes", payload)
        self.assertIn("scatter_blobs", payload)
        self.assertEqual(len(payload["input_labels"]), 26)
        self.assertEqual(len(payload["output_labels"]), 6)

    def test_reputation_system_costs_exile_bounties_and_rehabilitation(self) -> None:
        world = SimulationWorld(
            world_cfg=WorldConfig(initial_blobs=16, ticks_per_day=50, survival_food_need=1.0),
            seed=909,
        )
        raider, victim, sheriff, citizen = world.blobs[0], world.blobs[1], world.blobs[2], world.blobs[3]

        # 1. Form a peaceful settlement group between raider, sheriff, and citizen
        world._found_new_settlement_group(raider, sheriff)
        grp_id = raider.group_id
        self.assertIsNotNone(grp_id)
        world._join_blob_to_group(citizen, grp_id)
        sheriff.reputation = 0.35
        citizen.reputation = 0.30

        # 2. Raiding a peaceful solo victim costs net stamina (-12.5 energy) and drops reputation
        victim.group_id = None
        victim.reputation = 0.20
        victim.genes.size = 0.70
        victim.food_today = 2.0
        victim.money = 20.0
        victim.energy = 50.0
        vic_home = world.homes[victim.home_id]
        vic_home.x, vic_home.y = 50.0, 50.0
        victim.x, victim.y = 800.0, 800.0

        raider.x, raider.y = 802.0, 802.0
        raider.genes.size = 1.55
        raider.food_today = 0.0
        raider.money = 0.0
        raider.energy = 90.0
        raider.reputation = 0.0

        energy_before = raider.energy
        self.assertTrue(world._try_peer_raid(raider, victim))
        self.assertLess(raider.energy, energy_before)
        self.assertAlmostEqual(raider.energy, energy_before - 12.5, places=2)
        self.assertLess(raider.reputation, -0.25)

        # 3. Second raid on peaceful victim pushes raider below -0.45 -> triggers immediate group exile
        raider.trade_cooldown = 0
        victim.trade_cooldown = 0
        raider.food_today = 0.0
        raider.energy = 90.0
        self.assertTrue(world._try_peer_raid(raider, victim))
        self.assertTrue(raider.is_outlaw)
        self.assertIsNone(raider.group_id)
        self.assertGreaterEqual(raider.exiles_count, 1)
        self.assertGreaterEqual(world.exiles_today, 1)

        # 4. Economic sanctions: honorable blobs boycott peer trades with outlaws, and market surcharges apply
        raider.surplus_food = 2.0
        citizen.food_today = 0.2
        citizen.money = 30.0
        self.assertFalse(world._try_peer_trade(raider, citizen))
        self.assertGreater(world._get_reputation_buy_multiplier(raider), 1.30)
        self.assertLess(world._get_reputation_sell_multiplier(raider), 0.85)
        self.assertLess(world._get_reputation_buy_multiplier(sheriff), 1.0)

        # 5. Bounty hunter mechanic: attacking an outlaw costs NO reputation and awards a bounty
        sheriff.x, sheriff.y = raider.x + 1.0, raider.y + 1.0
        sheriff.genes.size = 1.65
        sheriff.social_desire = 0.55
        raider.genes.size = 0.75
        raider.energy = 30.0
        raider.food_today = 1.5
        raider.money = 15.0
        raider.trade_cooldown = 0
        sheriff.trade_cooldown = 0
        sheriff.food_today = 0.0
        sheriff.money = 5.0
        sheriff.energy = 95.0
        sheriff_rep_before = sheriff.reputation
        sheriff_money_before = sheriff.money

        self.assertTrue(world._try_peer_raid(sheriff, raider))
        self.assertGreater(sheriff.reputation, sheriff_rep_before)
        self.assertGreaterEqual(sheriff.bounties_won, 1)
        self.assertGreater(sheriff.money, sheriff_money_before + 3.5)

        # 6. Offspring inherits 25% of parent+partner reputation and daily rehabilitation decays reputation by 12%
        child = world._reproduce_blob(sheriff)
        expected_child_rep = round(0.5 * (sheriff.reputation + citizen.reputation) * 0.25, 3)
        self.assertAlmostEqual(child.reputation, expected_child_rep, places=3)

        rep_before_decay = raider.reputation
        raider.surplus_food = 0.0
        raider.money = 5.0
        for b in world.blobs:
            b.food_today = 1.2
            b.energy = 80.0
        world._end_of_day_accounting()
        self.assertAlmostEqual(raider.reputation, rep_before_decay * 0.88, places=2)

    def test_10k_neuron_config_and_hidden_dims_relaxation(self) -> None:
        """Verify NetworkConfig permits hidden layer widths up to 10,000 neurons while preserving defaults."""
        # Default configuration must strictly return [24, 24]
        default_cfg = NetworkConfig()
        self.assertEqual(default_cfg.get_hidden_dims(), [24, 24])

        # Explicit single wide hidden layer (26 -> 10,000 -> 6)
        wide_cfg = NetworkConfig(depth=1, width=10000, activation="tanh")
        self.assertEqual(wide_cfg.get_hidden_dims(), [10000])

        # Explicit custom layer configuration with 10k neurons
        custom_cfg = NetworkConfig(custom_layers=[10000], activation="gelu")
        self.assertEqual(custom_cfg.get_hidden_dims(), [10000])

        # Deep bottleneck topology with 10k neurons
        bottleneck_cfg = NetworkConfig(custom_layers=[32, 10000, 32], activation="silu")
        self.assertEqual(bottleneck_cfg.get_hidden_dims(), [32, 10000, 32])

    def test_10k_neuron_single_brain_forward_and_bounds(self) -> None:
        """Verify BlobBrain with 10,000 neurons initializes, evaluates forward, and preserves output bounds and reflex."""
        cfg = NetworkConfig(depth=1, width=10000, activation="tanh")
        brain = BlobBrain(config=cfg)
        self.assertEqual(brain.layer_dims, [INPUT_DIM, 10000, OUTPUT_DIM])
        self.assertEqual(len(brain.weights), 2)
        self.assertEqual(brain.weights[0].shape, (26, 10000))
        self.assertEqual(brain.weights[1].shape, (10000, 6))

        # Test forward on zero observation
        obs = np.zeros((INPUT_DIM,), dtype=np.float32)
        out = brain.forward(obs, explore_noise=0.0)
        self.assertEqual(out.shape, (OUTPUT_DIM,))
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertTrue(np.all(out >= -1.0) and np.all(out <= 1.0))

        # Test forward on random observation
        rng = np.random.default_rng(42)
        rand_obs = rng.uniform(-1.0, 1.0, size=(INPUT_DIM,)).astype(np.float32)
        out_rand = brain.forward(rand_obs, explore_noise=0.05)
        self.assertEqual(out_rand.shape, (OUTPUT_DIM,))
        self.assertTrue(np.all(np.isfinite(out_rand)))
        self.assertTrue(np.all(out_rand >= -1.0) and np.all(out_rand <= 1.0))

        # Verify innate food reflex forward bias holds at 10,000-neuron scale
        obs_r = np.zeros((INPUT_DIM,), dtype=np.float32); obs_r[0] = 1.0
        obs_l = np.zeros((INPUT_DIM,), dtype=np.float32); obs_l[0] = -1.0
        out_r = brain.forward(obs_r, explore_noise=0.0)
        out_l = brain.forward(obs_l, explore_noise=0.0)
        self.assertGreater(float(out_r[0] - out_l[0]), 0.10)

    def test_10k_neuron_batched_pool_execution(self) -> None:
        """Verify BatchedBrainPool binds and evaluates a batch of 10k-neuron brains with correct shapes and bounds."""
        from blob_sim.neural import BatchedBrainPool
        cfg = NetworkConfig(depth=1, width=10000, activation="tanh")
        n_blobs = 12
        brains = [BlobBrain(config=cfg) for _ in range(n_blobs)]
        pool = BatchedBrainPool()
        obs_batch = np.zeros((n_blobs, INPUT_DIM), dtype=np.float32)
        rng = np.random.default_rng(123)

        out = pool.forward_and_trace(brains, obs_batch, explore_noise=0.0, rng=rng)
        self.assertEqual(out.shape, (n_blobs, OUTPUT_DIM))
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertTrue(np.all(out >= -1.0) and np.all(out <= 1.0))

    def test_10k_neuron_memory_budget_under_4gb(self) -> None:
        """Verify theoretical weight and buffer footprint for 2,000 blobs x 10,000 neurons remains strictly under 4.0 GB."""
        cfg = NetworkConfig(depth=1, width=10000, activation="tanh")
        brain = BlobBrain(config=cfg)
        expected_params = 26 * 10000 + 10000 * 6 + 10000 + 6
        self.assertEqual(brain.parameter_count(), expected_params)

        bytes_per_blob_weights = expected_params * 4
        total_weight_bytes_2000 = bytes_per_blob_weights * 2000
        # 2,000 blobs x 330,006 params x 4 bytes = 2,640,048,000 bytes (~2.46 GB)
        self.assertLess(total_weight_bytes_2000, 3.0 * (1024 ** 3), "Weights alone must fit within 3.0 GB")

    def test_10k_neuron_online_learning_and_trace_stability(self) -> None:
        """Verify online reinforcement learning and mutation operate stably on 10k neurons without divergence."""
        cfg = NetworkConfig(depth=1, width=10000, activation="tanh")
        brain = BlobBrain(config=cfg)
        obs = np.ones((INPUT_DIM,), dtype=np.float32) * 0.5
        _ = brain.forward(obs, explore_noise=0.0)
        w0_before = brain.weights[0].copy()

        brain.apply_online_learning(reward=1.8, learning_rate=0.02)
        self.assertFalse(np.allclose(w0_before, brain.weights[0]))
        self.assertTrue(np.all(np.isfinite(brain.weights[0])))
        self.assertTrue(np.all(brain.weights[0] >= -3.5) and np.all(brain.weights[0] <= 3.5))

        # Offspring cloning and mutation stability
        child = brain.clone_and_mutate(mutation_rate=0.01, mutation_strength=0.1)
        self.assertEqual(child.layer_dims, brain.layer_dims)
        self.assertTrue(np.all(np.isfinite(child.weights[0])))

    def test_10k_neuron_architecture_adaptation(self) -> None:
        """Verify an existing brain can adapt from default [24, 24] to [10000] preserving overlapping sub-matrices."""
        old_brain = BlobBrain(config=NetworkConfig(depth=2, width=24))
        new_cfg = NetworkConfig(depth=1, width=10000, activation="tanh")
        adapted = old_brain.adapt_to_config(new_cfg)
        self.assertEqual(adapted.layer_dims, [INPUT_DIM, 10000, OUTPUT_DIM])
        # Overlapping region [26, 24] from layer 0 must be preserved identically
        self.assertTrue(np.allclose(adapted.weights[0][:INPUT_DIM, :24], old_brain.weights[0][:INPUT_DIM, :24]))

    def test_group_dirty_cache_and_membership_synchronization(self) -> None:
        """Verify Feature 5: Group.member_ids and _get_group_avg_reputation 100% synchronized across conquests, exiles, and direct mutations."""
        world = SimulationWorld(WorldConfig(initial_blobs=20), seed=42)

        # 1. Direct Group Mutation Synchronization
        g1 = Group(id=10, name="Clan1", color="#ff0000", home_id=1, home_ids=[1])
        g2 = Group(id=20, name="Clan2", color="#0000ff", home_id=2, home_ids=[2])
        world.groups[10] = g1
        world.groups[20] = g2

        b1 = world.blobs[0]
        b2 = world.blobs[1]
        b1.group_id = 10
        b2.group_id = 20
        b1.reputation = 0.60
        b2.reputation = -0.40

        self.assertAlmostEqual(world._get_group_avg_reputation(g1), 0.60, places=2)
        self.assertAlmostEqual(world._get_group_avg_reputation(g2), -0.40, places=2)
        self.assertIn(b1.id, g1.member_ids)
        self.assertIn(b2.id, g2.member_ids)

        # Direct mutation: transfer b2 from Clan2 to Clan1
        b2.group_id = 10
        self.assertNotIn(b2.id, g2.member_ids, "Old group must not retain mutated member")
        self.assertIn(b2.id, g1.member_ids, "New group must contain transferred member")
        self.assertEqual(world._get_group_avg_reputation(g2), 0.0, "Empty old group must have 0.0 avg reputation")
        self.assertAlmostEqual(world._get_group_avg_reputation(g1), 0.10, places=2)

        # 2. House Conquest Synchronization
        b_atk = world.blobs[2]
        b_def = world.blobs[3]
        g_atk = Group(id=30, name="AtkClan", color="#00ff00", home_id=3, home_ids=[3])
        g_def = Group(id=40, name="DefClan", color="#ffff00", home_id=4, home_ids=[4])
        world.groups[30] = g_atk
        world.groups[40] = g_def
        b_atk.group_id = 30
        b_def.group_id = 40
        b_def.home_id = 4
        b_atk.reputation = 0.50
        b_def.reputation = -0.30

        target_home = world.homes[4]
        target_home.defense_level = 0
        target_home.terrain_defense_bonus = 0.0
        target_home.raid_damage = 0
        target_home.group_id = 40
        b_def.x, b_def.y = target_home.x, target_home.y
        b_atk.x, b_atk.y = target_home.x + 5.0, target_home.y
        b_atk.genes.size = 1.6
        b_def.genes.size = 0.65
        b_atk.social_desire = -0.85
        b_atk.trade_desire = -0.85
        b_atk.food_today = 0.0
        b_atk.money = 0.0
        b_atk.energy = 100.0
        b_def.energy = 12.0
        b_def.food_today = 3.0
        b_def.money = 30.0

        # Two raids to trigger conquest
        world._try_peer_raid(b_atk, b_def)
        b_atk.trade_cooldown = 0
        b_def.trade_cooldown = 0
        b_atk.energy = 100.0
        b_atk.food_today = 0.0
        world._try_peer_raid(b_atk, b_def)

        self.assertEqual(target_home.group_id, g_atk.id, "Target home must be conquered by attacker")
        self.assertEqual(b_def.group_id, g_atk.id, "Defender must be annexed into attacker group")
        self.assertNotIn(b_def.id, g_def.member_ids, "Conquered defender must be removed from defender group member_ids")
        self.assertIn(b_def.id, g_atk.member_ids, "Conquered defender must be added to attacker group member_ids")
        expected_atk_rep = round((b_atk.reputation + b_def.reputation) / 2, 3)
        self.assertAlmostEqual(world._get_group_avg_reputation(g_atk), expected_atk_rep, places=2)

        # 3. Social Exile Synchronization
        b_exile = world.blobs[4]
        b_peer = world.blobs[5]
        g_peace = Group(id=50, name="PeaceClan", color="#00ffff", home_id=5, home_ids=[5])
        world.groups[50] = g_peace
        b_exile.group_id = 50
        b_peer.group_id = 50
        b_exile.reputation = 0.40
        b_peer.reputation = 0.40
        self.assertAlmostEqual(world._get_group_avg_reputation(g_peace), 0.40, places=2)

        # Push b_exile into outlaw territory
        b_exile.reputation = -0.60
        exiled = world._check_blob_reputation_exile(b_exile)
        self.assertTrue(exiled, "Outlaw blob must be exiled from peaceful group")
        self.assertIsNone(b_exile.group_id, "Exiled blob group_id must be None")
        self.assertNotIn(b_exile.id, g_peace.member_ids, "Exiled blob must be removed from member_ids")
        self.assertAlmostEqual(world._get_group_avg_reputation(g_peace), 0.40, places=2, msg="Group avg rep must only reflect remaining peer")

        # 4. Idempotence & Deduplication Defense
        g_peace.member_ids.append(b_peer.id)
        g_peace.rep_dirty = True
        self.assertAlmostEqual(world._get_group_avg_reputation(g_peace), 0.40, places=2, msg="Duplicate member ID must not inflate rep_sum")
        self.assertEqual(g_peace.cached_rep_count, 1, "Duplicate ID must be counted once")

    def test_zero_blobs_boundary_and_spawn(self) -> None:
        """Verify world with initial_blobs=0 and min_population_floor=0 steps cleanly and supports custom spawn."""
        cfg = WorldConfig.from_preset("classic", initial_blobs=0, min_population_floor=0)
        world = SimulationWorld(world_cfg=cfg, net_cfg=NetworkConfig(width=16), seed=42)
        self.world = world

        self.assertEqual(len(world.blobs), 0)
        # Verify clean step with zero blobs
        world.step(1)
        self.assertEqual(world.tick_count, 1)
        self.assertEqual(len(world.blobs), 0)

        # Spawn a blob and verify stepping without error
        blob = world.spawn_custom_blob(200.0, 200.0)
        self.assertEqual(len(world.blobs), 1)
        self.assertTrue(blob.alive)
        world.step(1)
        self.assertEqual(world.tick_count, 2)
        self.assertTrue(blob.alive)
        self.assertAlmostEqual(world._kin_xs[0], blob.x, delta=1e-4)
        self.assertAlmostEqual(world._kin_ys[0], blob.y, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
