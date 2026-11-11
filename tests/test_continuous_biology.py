"""Tests for continuous, asynchronous lifecycle and ecology cycle (`continuous_lifecycle: bool = True`)."""

import math
import numpy as np
import pytest
from blob_sim.config import WorldConfig
from blob_sim.world import SimulationWorld


def test_continuous_lifecycle_config_defaults_and_preset():
    """Verify continuous_lifecycle in WorldConfig and apply_scale_preset."""
    cfg = WorldConfig()
    assert cfg.continuous_lifecycle is False

    cfg.apply_scale_preset("classic")
    assert cfg.continuous_lifecycle is False

    cfg_cont = WorldConfig(continuous_lifecycle=True)
    assert cfg_cont.continuous_lifecycle is True


def test_blob_repro_cooldown_property_and_kinematics():
    """Verify repro_cooldown on Blob synchronizes with _kin_rcooldowns buffer."""
    cfg = WorldConfig(continuous_lifecycle=True, initial_blobs=10)
    world = SimulationWorld(world_cfg=cfg)
    world.step(1)

    blob = world.blobs[0]
    assert blob.repro_cooldown == 0
    blob.repro_cooldown = 150
    assert blob.repro_cooldown == 150
    assert world._kin_rcooldowns[blob._kin_idx] == 150

    world.step(1)
    assert blob.repro_cooldown == 149
    assert world._kin_rcooldowns[blob._kin_idx] == 149


def test_asynchronous_birth_midday_with_exact_energy_conservation():
    """
    Verify midday asynchronous birth occurs without midnight stall,
    and parent.energy + child.energy immediately after birth matches pre-birth energy.
    """
    cfg = WorldConfig(
        continuous_lifecycle=True,
        initial_blobs=16,
        ticks_per_day=400,
        daily_food_spawn=200,
    )
    world = SimulationWorld(world_cfg=cfg)

    # Step into midday: 0 < tick_in_day < ticks_per_day - 5
    for _ in range(50):
        world.step(1)

    assert 0 < world.tick_in_day < (world.world_cfg.ticks_per_day - 5)

    parent = world.blobs[0]
    parent.repro_cooldown = 0
    parent.energy = parent.max_energy * 0.95
    parent.food_today = world.world_cfg.reproduction_food_need
    parent.trade_cooldown = 999  # Prevent selling food before birth evaluation

    pop_before = len(world.blobs)
    birth_recorded = {}

    orig_repro = world._reproduce_blob

    def repro_hook(p, partner=None):
        birth_recorded["parent_e_before"] = p.energy
        return orig_repro(p, partner=partner)

    world._reproduce_blob = repro_hook
    world.step(1)

    assert len(world.blobs) == pop_before + 1
    child = world.blobs[-1]
    assert child.parent_id == parent.id

    e_before = birth_recorded["parent_e_before"]
    assert math.isclose(parent.energy + child.energy, e_before, abs_tol=1e-5)
    assert parent.repro_cooldown >= 120

    # Ensure child was placed within world bounds and not inside a rock
    assert 20.0 <= child.x <= world.world_cfg.width - 20.0
    assert 20.0 <= child.y <= world.world_cfg.height - 20.0
    assert not world._point_hits_rock(child.x, child.y, pad=4.0)


def test_biomass_recycling_on_death():
    """Verify dead blob leaves FoodItem biomass with nutrition scaled to size gene."""
    cfg = WorldConfig(continuous_lifecycle=True, initial_blobs=10)
    world = SimulationWorld(world_cfg=cfg)
    world.step(1)

    blob = world.blobs[0]
    bx, by = blob.x, blob.y
    bsize = blob.genes.size
    expected_nut = round(max(0.35, 0.55 * bsize), 2)
    foods_before = len(world.foods)
    fver_before = world.food_version

    world._kill_blob(blob, reason="starved test")

    assert len(world.foods) == foods_before + 1
    assert world.food_version == fver_before + 1
    new_food = world.foods[-1]
    assert new_food.nutrition == expected_nut
    assert math.isclose(new_food.x, bx, abs_tol=1e-3)
    assert math.isclose(new_food.y, by, abs_tol=1e-3)
    assert new_food.oasis_id is None
    assert new_food.golden is False


def test_three_full_days_simulation_and_telemetry():
    """Verify 3 full days of continuous lifecycle execution with 3 telemetry daily logs."""
    cfg = WorldConfig(
        continuous_lifecycle=True,
        initial_blobs=20,
        ticks_per_day=60,
        daily_food_spawn=120,
    )
    world = SimulationWorld(world_cfg=cfg)

    # 3 full days: 3 * 60 = 180 ticks
    for _ in range(180):
        world.step(1)

    assert world.day_number == 4
    assert len(world.telemetry.daily_logs) == 3
    days = [d["day"] for d in world.telemetry.daily_logs]
    assert days == [1, 2, 3]

    for log in world.telemetry.daily_logs:
        assert "population" in log
        assert log["population"] > 0
