"""Unit tests for emergent physical interactions and free bioluminescent/chemical communication channel."""

import math
import numpy as np
import pytest

from blob_sim.config import NetworkConfig, WorldConfig
from blob_sim.entities import Blob, Genes, TerrainObstacle
from blob_sim.neural import BlobBrain
from blob_sim.world import SimulationWorld


def create_test_blob(
    world: SimulationWorld,
    blob_id: int,
    x: float,
    y: float,
    heading: float = 0.0,
    energy: float = 50.0,
    food_today: float = 0.5,
    surplus_food: float = 0.0,
    genes: Genes = None,
) -> Blob:
    """Helper to instantiate and register a blob in a world."""
    if genes is None:
        genes = Genes()
    brain = BlobBrain(world.net_cfg)
    b = Blob(
        blob_id=blob_id,
        x=x,
        y=y,
        home_id=1,
        home_x=x,
        home_y=y,
        brain=brain,
        genes=genes,
    )
    b.heading = heading
    b.energy = energy
    b.food_today = food_today
    b.surplus_food = surplus_food
    b.trade_cooldown = 0
    b.colonize_cooldown = 0
    return b


def test_signal_transmission_and_kinship_perception():
    """
    Blob A transmits signal via actions[4] = 0.85.
    Blob B is next to Blob A and perceives it on sensory channel 24 proportional to proximity.
    Blob B also perceives continuous phenotypic kinship on channel 11.
    """
    cfg = WorldConfig(initial_blobs=0, emergent_interactions=True)
    world = SimulationWorld(world_cfg=cfg, seed=42)

    gA = Genes(hue=100.0, speed=1.0, sense=1.0, size=1.0)
    gB = Genes(hue=105.0, speed=1.05, sense=0.98, size=1.02)  # High genetic similarity

    bA = create_test_blob(world, 1, 1000.0, 1000.0, genes=gA)
    bB = create_test_blob(world, 2, 1015.0, 1000.0, genes=gB)  # Distance 15.0, well within vision
    world.blobs = [bA, bB]
    world._sync_static_blob_kin_arrays(world.blobs)

    bA.signal_value = 0.85
    bB.signal_value = -0.30

    # Step observations
    food_vecs, peer_vecs, nearest_peers, nearest_peer_dists, oasis_vecs, feeler_vecs = world._compute_spatial_perception(
        world.blobs, night_mode=False, dt=1
    )
    obs_batch, *_ = world._prepare_step_observations(
        world.blobs,
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

    # Blob B (index 1) perceives Blob A's signal on channel 24
    # Proximity is obs_batch[1, 8] > 0.0
    prox_B = obs_batch[1, 8]
    assert prox_B > 0.8
    assert math.isclose(obs_batch[1, 24], np.float32(0.85 * prox_B), rel_tol=1e-3, abs_tol=1e-3)

    # Blob B's own signal on channel 25
    assert math.isclose(obs_batch[1, 25], -0.30, abs_tol=1e-3)

    # Blob B's perceived kinship with Blob A on channel 11 should be strongly positive
    kinship_B = obs_batch[1, 11]
    assert kinship_B > 0.6  # Near +1.0 for very similar genes


def test_trophallaxis_energy_sharing():
    """
    Blob A (Output 3 = 0.8, energy = 60.0) feeds hungry neighbor Blob B (energy = 20.0).
    Blob A loses 6.0 energy, Blob B gains 5.4 energy.
    No money is exchanged, trades_today and recent_trade_lines are updated.
    """
    cfg = WorldConfig(initial_blobs=0, emergent_interactions=True)
    world = SimulationWorld(world_cfg=cfg, seed=42)

    bA = create_test_blob(world, 1, 1000.0, 1000.0, energy=60.0)
    bB = create_test_blob(world, 2, 1018.0, 1000.0, energy=20.0)  # Distance 18.0 (< 32.0)
    world.blobs = [bA, bB]
    world._sync_static_blob_kin_arrays(world.blobs)

    actions_batch = np.zeros((2, 6), dtype=np.float32)
    actions_batch[0, 3] = 0.8  # Trophallaxis intent
    actions_batch[1, 3] = 0.0

    nearest_peer_dists = np.array([18.0, 18.0], dtype=np.float32)
    world._last_min_p_idxs = np.array([1, 0], dtype=np.int32)

    trades_before = world.total_trades_all_time
    world._process_emergent_interactions(world.blobs, actions_batch, nearest_peer_dists, dt_scale=1.0)

    assert math.isclose(bA.energy, 54.0, abs_tol=1e-2)
    assert math.isclose(bB.energy, 25.4, abs_tol=1e-2)
    assert bA.trade_cooldown == 15
    assert bB.trade_cooldown == 0  # Receiver cooldown unchanged
    assert world.total_trades_all_time == trades_before + 1
    assert len(world.recent_trade_lines) == 1
    assert world.recent_trade_lines[0]["x1"] == round(bA.x, 1)


def test_predation_bite_and_loot():
    """
    Blob A (Output 3 = -0.8, energy = 50.0) attacks neighbor Blob B (energy = 30.0).
    Attacker pays 3.5 energy, defender loses damage, attacker absorbs 0.65 * damage.
    Raid lines and counters are recorded.
    """
    cfg = WorldConfig(initial_blobs=0, emergent_interactions=True)
    world = SimulationWorld(world_cfg=cfg, seed=42)

    gA = Genes(size=1.4)  # Big predator
    gB = Genes(size=0.8)  # Smaller prey

    bA = create_test_blob(world, 1, 1000.0, 1000.0, energy=50.0, genes=gA)
    bB = create_test_blob(world, 2, 1020.0, 1000.0, energy=30.0, genes=gB)
    world.blobs = [bA, bB]
    world._sync_static_blob_kin_arrays(world.blobs)

    actions_batch = np.zeros((2, 6), dtype=np.float32)
    actions_batch[0, 3] = -0.8  # Bite intent
    actions_batch[1, 3] = 0.0

    nearest_peer_dists = np.array([20.0, 20.0], dtype=np.float32)
    world._last_min_p_idxs = np.array([1, 0], dtype=np.int32)

    raids_before = world.total_raids_all_time
    world._process_emergent_interactions(world.blobs, actions_batch, nearest_peer_dists, dt_scale=1.0)

    # Defender lost energy, attacker paid stamina and gained loot
    assert bB.energy < 30.0
    dmg = 30.0 - bB.energy
    assert dmg > 5.0
    expected_atk_energy = min(bA.max_energy, (50.0 - 3.5) + 0.65 * dmg)
    assert math.isclose(bA.energy, expected_atk_energy, abs_tol=1e-2)
    assert bA.trade_cooldown == 22
    assert bB.trade_cooldown == 0
    assert world.total_raids_all_time == raids_before + 1
    assert len(world.recent_raid_lines) == 1


def test_stash_food_behind_blob():
    """
    Blob stashes food reserve via Output 5 = 0.8.
    Deducts 0.30 food, places FoodItem behind blob outside immediate eat range.
    food_version increments, food_dirty is True.
    """
    cfg = WorldConfig(initial_blobs=0, emergent_interactions=True)
    world = SimulationWorld(world_cfg=cfg, seed=42)

    b = create_test_blob(world, 1, 500.0, 500.0, heading=0.0, surplus_food=0.50, energy=70.0)
    world.blobs = [b]
    world._sync_static_blob_kin_arrays(world.blobs)

    initial_food_count = len(world.foods)
    initial_version = world.food_version

    actions_batch = np.zeros((1, 6), dtype=np.float32)
    actions_batch[0, 5] = 0.8  # Stash intent
    nearest_peer_dists = np.array([9999.0], dtype=np.float32)

    world._process_emergent_interactions(world.blobs, actions_batch, nearest_peer_dists, dt_scale=1.0)

    assert len(world.foods) == initial_food_count + 1
    assert world.food_version == initial_version + 1
    assert world._food_dirty is True
    assert math.isclose(b.surplus_food, 0.20, abs_tol=1e-3)
    assert b.colonize_cooldown == 45

    # Check dropped food position: heading=0.0 -> dropped behind at x < 500.0
    dropped = world.foods[-1]
    assert dropped.nutrition == 0.30
    assert dropped.x < 500.0
    assert math.isclose(dropped.y, 500.0, abs_tol=1e-1)


def test_build_shelter_rock():
    """
    Blob builds a shelter wall via Output 5 = -0.8.
    Costs 14.0 energy, obstacle count increases by 1, cooldown 240.
    """
    cfg = WorldConfig(initial_blobs=0, emergent_interactions=True)
    world = SimulationWorld(world_cfg=cfg, seed=42)

    b = create_test_blob(world, 1, 800.0, 800.0, heading=0.0, energy=80.0)
    world.blobs = [b]
    world._sync_static_blob_kin_arrays(world.blobs)

    initial_obs_count = len(world.obstacles)

    actions_batch = np.zeros((1, 6), dtype=np.float32)
    actions_batch[0, 5] = -0.8  # Build rock wall intent
    nearest_peer_dists = np.array([9999.0], dtype=np.float32)

    world._process_emergent_interactions(world.blobs, actions_batch, nearest_peer_dists, dt_scale=1.0)

    assert len(world.obstacles) == initial_obs_count + 1
    new_obs = world.obstacles[-1]
    assert new_obs.kind == "rock"
    assert new_obs.gate_kind == "nest"
    assert math.isclose(b.energy, 66.0, abs_tol=1e-2)
    assert b.colonize_cooldown == 240


def test_legacy_mode_remains_unaffected():
    """
    Verify that when emergent_interactions=False, the world runs the standard legacy pathways.
    """
    cfg = WorldConfig(initial_blobs=12, emergent_interactions=False)
    world = SimulationWorld(world_cfg=cfg, seed=42)
    ticks_advanced = world.step(5)
    assert ticks_advanced == 5
    assert len(world.blobs) > 0
