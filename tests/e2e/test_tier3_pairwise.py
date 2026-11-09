"""Tier 3 - Cross-Feature Combinations & Pairwise Interactions E2E Tests.

Tests multi-system interactions:
1. Trading during starvation and sovereign sacrifice (Metabolism + Economics)
2. Reproduction under overgrazed oasis and colonial migration (Ecology + Reproduction + Spatial Memory)
3. Resource raids against watchtowers during seasonal transitions (Combat + Seasons + House Defense)
4. Outlaw stigma, economic boycott, bounty hunting, and rehabilitation (Society + Economics + Combat)
5. Multi-home settlements, colonial secession, and faction proliferation (Settlement + Secession + Factions)
6. Lamarckian online RL traces combined with reproductive genetic mutation (Neural Learning + Genetics)
"""

import unittest
import numpy as np

from blob_sim.config import NetworkConfig, WorldConfig
from blob_sim.entities import generate_faction_identity
from blob_sim.world import SimulationWorld


class TestTier3PairwiseInteractions(unittest.TestCase):
    """Tier 3: Pairwise and multi-feature interaction mechanics."""

    def tearDown(self) -> None:
        if hasattr(self, "world") and self.world is not None:
            if hasattr(self.world, "close"):
                self.world.close()
            self.world = None
        import gc
        gc.collect()

    def test_trading_during_starvation_and_sovereign_sacrifice(self) -> None:
        """Verify economic trade rescues starving blobs, and sovereign sellers can trade below survival need."""
        world = SimulationWorld(world_cfg=WorldConfig(initial_blobs=10, ticks_per_day=40), seed=11)
        self.world = world
        seller, buyer = world.blobs[0], world.blobs[1]
        seller.x, seller.y = 400.0, 400.0
        buyer.x, buyer.y = 405.0, 405.0

        # Scenario A: Desperate starving buyer with money buys food to survive
        seller.food_today = 1.5
        seller.surplus_food = 0.5
        seller.trade_desire = 0.5
        seller_money_before = seller.money

        buyer.food_today = 0.0
        buyer.energy = 15.0
        buyer.money = 20.0
        buyer.trade_desire = 0.8

        traded = world._try_peer_trade(seller, buyer)
        self.assertTrue(traded)
        self.assertAlmostEqual(buyer.food_today, 0.5)
        self.assertGreater(seller.money, seller_money_before)

        # Scenario B: Sovereign seller trades away food even below survival need (food_today = 0.6)
        seller.food_today = 0.6
        seller.surplus_food = 0.0
        seller.trade_desire = 0.85  # Wants gold despite hunger!
        buyer.food_today = 0.1
        buyer.money = 15.0
        seller_money_before = seller.money

        traded_sovereign = world._try_peer_trade(seller, buyer)
        self.assertTrue(traded_sovereign)
        self.assertAlmostEqual(seller.food_today, 0.1, places=2)
        self.assertAlmostEqual(buyer.food_today, 0.6, places=2)
        self.assertGreater(seller.money, seller_money_before)

    def test_reproduction_under_overgrazed_oasis_and_colonial_migration(self) -> None:
        """Verify that overgrazing halts food, prompting spatial memory colonization of distant fertile oases."""
        world = SimulationWorld(world_cfg=WorldConfig(initial_blobs=12, ticks_per_day=50), seed=22)
        self.world = world
        pioneer = world.blobs[0]
        home = world.homes[pioneer.home_id]

        # Overgraze pioneer's local home oasis
        home_oasis = next(o for o in world.oases.values() if o.ring == "inner")
        home_oasis.fertility = 0.10
        home_oasis.harvest_pressure = 10.0
        self.assertTrue(home_oasis.is_overgrazed)

        # Spawning food must yield 0 wild food in this overgrazed oasis
        world.foods = [f for f in world.foods if f.oasis_id != home_oasis.id]
        world.spawn_daily_food(100)
        cradle_foods = [f for f in world.foods if f.oasis_id == home_oasis.id]
        self.assertEqual(len(cradle_foods), 0)

        # Pioneer discovers distant fertile outer oasis
        frontier_oasis = max(
            world.oases.values(),
            key=lambda o: (o.x - home.x) ** 2 + (o.y - home.y) ** 2,
        )
        frontier_oasis.fertility = 0.98
        frontier_oasis.harvest_pressure = 0.0

        pioneer.x, pioneer.y = frontier_oasis.x, frontier_oasis.y
        world.spawn_custom_food_cluster(frontier_oasis.x + 2.0, frontier_oasis.y + 2.0, count=2)
        for f in world.foods:
            if math_hypot := float(np.hypot(f.x - frontier_oasis.x, f.y - frontier_oasis.y)) < 25.0:
                f.oasis_id = frontier_oasis.id

        world._process_foraging([pioneer])
        self.assertEqual(pioneer.memory_oasis_id, frontier_oasis.id)
        self.assertGreater(pioneer.memory_strength, 0.8)

        # Trigger migration and colonization of new home
        old_home_id = pioneer.home_id
        pioneer.colonize_desire = 0.95
        pioneer.colonize_cooldown = 0
        pioneer.energy = 85.0
        pioneer.money = 15.0

        colonized = world._try_colonize_or_relocate(pioneer, tod=0.45)
        self.assertTrue(colonized)
        self.assertNotEqual(pioneer.home_id, old_home_id)
        self.assertGreaterEqual(pioneer.migrations_made, 1)

    def test_raids_against_watchtowers_during_seasonal_transitions(self) -> None:
        """Verify season shifts alter ecology, watchtowers repel raids, and unfortified homes get conquered."""
        world = SimulationWorld(seed=33)
        self.world = world

        # Seasonal progression: Day 1=spring, Day 5=summer, Day 9=autumn, Day 13=winter
        world.day_number = 1
        self.assertEqual(world.season_info[0], "spring")
        world.day_number = 13
        self.assertEqual(world.season_info[0], "winter")

        raider, defender = world.blobs[0], world.blobs[1]
        raider.group_id = None
        defender.group_id = None
        raider.genes.size = 1.50
        defender.genes.size = 0.70
        raider.food_today = 0.0
        raider.money = 0.0
        raider.energy = 95.0
        defender.energy = 45.0
        defender.food_today = 1.5
        defender.x, defender.y = 500.0, 500.0
        raider.x, raider.y = 502.0, 502.0

        # Isolated defender outside watchtower: raid succeeds
        def_home = world.homes[defender.home_id]
        def_home.x, def_home.y = 100.0, 100.0
        raids_before = world.raids_today
        self.assertTrue(world._try_peer_raid(raider, defender))
        self.assertEqual(world.raids_today, raids_before + 1)

        # Defender inside upgraded Lv. 3 Watchtower: raid repelled
        def_home.x, def_home.y = defender.x, defender.y
        def_home.defense_level = 3
        raider.trade_cooldown = 0
        defender.trade_cooldown = 0
        raider.food_today = 0.0
        raider.money = 0.0
        raider.energy = 60.0
        repelled_before = world.raids_repelled_today

        self.assertTrue(world._try_peer_raid(raider, defender))
        self.assertEqual(world.raids_repelled_today, repelled_before + 1)

    def test_outlaw_stigma_boycott_and_bounty_hunter_mechanic(self) -> None:
        """Verify repeated attacks cause exile, economic boycotts block trade, and bounty hunters claim reward."""
        world = SimulationWorld(world_cfg=WorldConfig(initial_blobs=16, ticks_per_day=40), seed=44)
        self.world = world
        raider, victim, sheriff, citizen = (
            world.blobs[0],
            world.blobs[1],
            world.blobs[2],
            world.blobs[3],
        )

        # Found settlement group with raider and sheriff
        world._found_new_settlement_group(raider, sheriff)
        grp_id = raider.group_id
        world._join_blob_to_group(citizen, grp_id)

        # Raider attacks peaceful solo victim -> reputation penalty
        victim.group_id = None
        victim.reputation = 0.2
        victim.x, victim.y = 800.0, 800.0
        raider.x, raider.y = 802.0, 802.0
        raider.genes.size = 1.5
        victim.genes.size = 0.7
        victim.food_today = 2.0
        raider.food_today = 0.0
        raider.money = 0.0
        raider.energy = 90.0
        world.homes[victim.home_id].x = 50.0
        world.homes[victim.home_id].y = 50.0

        world._try_peer_raid(raider, victim)
        self.assertLess(raider.reputation, 0.0)

        # Second attack drops reputation below -0.45 -> triggers exile into outlaw status
        raider.trade_cooldown = 0
        victim.trade_cooldown = 0
        raider.food_today = 0.0
        raider.money = 0.0
        raider.energy = 90.0
        world._try_peer_raid(raider, victim)
        self.assertTrue(raider.is_outlaw)
        self.assertIsNone(raider.group_id)

        # Economic boycott: citizen refuses to trade with outlaw
        raider.surplus_food = 1.0
        citizen.food_today = 0.2
        citizen.money = 20.0
        self.assertFalse(world._try_peer_trade(raider, citizen))

        # Bounty Hunter: Sheriff attacks outlaw -> earns gold bounty + reputation without penalty
        sheriff.x, sheriff.y = raider.x + 1.0, raider.y + 1.0
        sheriff.genes.size = 1.6
        sheriff.social_desire = 0.5
        sheriff.energy = 95.0
        raider.energy = 30.0
        raider.money = 15.0
        sheriff_rep_before = sheriff.reputation
        sheriff_money_before = sheriff.money
        raider.trade_cooldown = 0
        sheriff.trade_cooldown = 0

        self.assertTrue(world._try_peer_raid(sheriff, raider))
        self.assertGreater(sheriff.reputation, sheriff_rep_before)
        self.assertGreaterEqual(sheriff.bounties_won, 1)
        self.assertGreater(sheriff.money, sheriff_money_before + 2.5)

    def test_colonial_secession_and_faction_splintering(self) -> None:
        """Verify distant colony cluster secedes into an autonomous daughter faction."""
        world = SimulationWorld(world_cfg=WorldConfig(initial_blobs=24), seed=55)
        self.world = world
        b0, b1 = world.blobs[0], world.blobs[1]
        world._found_new_settlement_group(b0, b1)
        g1 = world.groups[b0.group_id]

        cap_home = world.homes[g1.home_id]
        # Keep 4 blobs in capital
        for extra_cap in world.blobs[4:7]:
            extra_cap.group_id = g1.id
            if extra_cap.id not in g1.member_ids:
                g1.member_ids.append(extra_cap.id)

        # Build 2 colony homes far from capital
        col_h1 = world._construct_new_partner_home(g1, cap_home.x + 5000.0, cap_home.y + 4000.0, 8)
        col_h2 = world._construct_new_partner_home(g1, cap_home.x + 5120.0, cap_home.y + 4050.0, 8)

        # Relocate 8 members to the distant colony
        for idx, member in enumerate(world.blobs[10:18]):
            member.group_id = g1.id
            if member.id not in g1.member_ids:
                g1.member_ids.append(member.id)
            dest = col_h1 if idx < 4 else col_h2
            world._relocate_blob_home(member, dest.id)

        groups_before = len(world.groups)
        world._check_empire_secessions()
        self.assertEqual(len(world.groups), groups_before + 1)
        daughter = max(world.groups.values(), key=lambda g: g.id)
        self.assertEqual(daughter.parent_group_id, g1.id)
        self.assertGreaterEqual(g1.secessions, 1)

    def test_online_reinforcement_learning_synced_with_reproduction_mutation(self) -> None:
        """Verify Lamarckian online learning updates weights and synchronizes into child upon reproduction."""
        world = SimulationWorld(
            world_cfg=WorldConfig(initial_blobs=16, ticks_per_day=50),
            net_cfg=NetworkConfig(depth=3, width=32, activation="gelu"),
            seed=66,
        )
        self.world = world
        parent = world.blobs[0]
        initial_w0 = parent.brain.weights[0].copy()

        # Step world so traces accumulate in pool ring buffer
        for _ in range(3):
            world.step(1)
        parent.brain.apply_online_learning(reward=2.5, learning_rate=0.08)

        # Reproduce: must flush pending ring-buffer updates into parent's weights and pass to child
        parent.food_today = 3.0
        parent.energy = 90.0
        child = world._reproduce_blob(parent)

        # Parent brain weights must have changed after flush on reproduction
        self.assertFalse(np.allclose(initial_w0, parent.brain.weights[0]))
        self.assertEqual(child.brain.layer_dims, parent.brain.layer_dims)
        # Child weights are correlated with trained parent (>0.45 given default 18% mutation)
        corr = float(np.corrcoef(parent.brain.weights[0].flatten(), child.brain.weights[0].flatten())[0, 1])
        self.assertGreater(corr, 0.45)


if __name__ == "__main__":
    unittest.main()
