"""Unit and integration tests for ALife emergent long-term simulation fixes.

Verifies:
1. BlobBrain.adapt_to_config preserves innate_weights & innate_biases when baldwin_plasticity=True.
2. Continuous digestion in continuous_lifecycle=True digests food over 200 ticks so blobs don't stay clogged.
3. Emergent interactions disable scripted ghost home creation on births and spawns.
4. Export/import population roundtrip with filled Hall of Fame (after 600 ticks) through real JSON serialization preserves all 5 ALife flags, Baldwin plasticity, and recurrent dimensions.
"""

from __future__ import annotations

import json
import numpy as np
import pytest

from blob_sim.config import NetworkConfig, WorldConfig
from blob_sim.entities import Blob, Genes
from blob_sim.neural import BlobBrain
from blob_sim.world import SimulationWorld


def test_baldwin_innate_genome_preserved_on_adapt():
    """BlobBrain.from_dict(champ).adapt_to_config(net_cfg) must preserve trained/mutated innate weights."""
    orig_cfg = NetworkConfig(depth=2, width=16, baldwin_plasticity=True, recurrent_dim=4)
    orig_brain = BlobBrain(config=orig_cfg)

    # Set distinct patterns in innate weights & biases
    for l_idx, iw in enumerate(orig_brain.innate_weights):
        iw.fill(float(l_idx + 1) * 3.1415)
    for l_idx, ib in enumerate(orig_brain.innate_biases):
        ib.fill(float(l_idx + 1) * 2.7182)

    # Also make sure somatic weights differ from innate weights (simulating lifetime learning)
    for w in orig_brain.weights:
        w.fill(0.1234)

    champ_dict = orig_brain.to_dict()
    champ_brain = BlobBrain.from_dict(champ_dict)

    # Adapt to a larger network configuration
    new_cfg = NetworkConfig(depth=2, width=24, baldwin_plasticity=True, recurrent_dim=8)
    adapted = champ_brain.adapt_to_config(new_cfg)

    assert adapted.config.baldwin_plasticity is True
    assert len(adapted.innate_weights) == len(orig_brain.innate_weights)
    assert len(adapted.innate_biases) == len(orig_brain.innate_biases)

    # Check that overlapping parts match the innate weights, not random or somatic weights
    for l_idx in range(len(adapted.innate_weights)):
        orig_iw = orig_brain.innate_weights[l_idx]
        new_iw = adapted.innate_weights[l_idx]
        orig_ib = orig_brain.innate_biases[l_idx]
        new_ib = adapted.innate_biases[l_idx]

        r = min(orig_iw.shape[0], new_iw.shape[0])
        c = min(orig_iw.shape[1], new_iw.shape[1])
        assert np.allclose(new_iw[:r, :c], orig_iw[:r, :c]), f"Innate weights mismatch at layer {l_idx}"
        cb = min(orig_ib.shape[0], new_ib.shape[0])
        assert np.allclose(new_ib[:cb], orig_ib[:cb]), f"Innate biases mismatch at layer {l_idx}"


def test_continuous_digestion_over_longterm_ticks():
    """A blob with a full stomach digests food over 200 ticks in continuous_lifecycle=True."""
    w_cfg = WorldConfig.from_preset("alife_emergent")
    w_cfg.initial_blobs = 10
    w_cfg.continuous_lifecycle = True
    world = SimulationWorld(world_cfg=w_cfg, net_cfg=NetworkConfig())

    blob = world.blobs[0]
    cap = blob.max_food_capacity
    blob.food_today = cap
    blob.surplus_food = 0.0
    blob.energy = 50.0

    assert np.isclose(blob.total_food, cap)

    # Advance 200 ticks (digestion runs every 10 ticks: 20 digestion cycles)
    for _ in range(200):
        world.step()

    # The blob's stomach must now have digested substantial food (20 cycles * 0.0192 = ~0.38 digested)
    assert blob.total_food < cap, f"Stomach did not digest: {blob.total_food} vs {cap}"
    assert blob.total_food <= cap - 0.35, f"Digestion was too weak: {blob.total_food} remaining of {cap}"

    # Verify that the blob can now acquire fresh food again
    added = blob.add_acquired_food(0.5, world.world_cfg.reproduction_food_need)
    assert added > 0.0, "Blob could not acquire new food after digesting!"
    world.close()


def test_no_ghost_homes_and_roundtrip_after_full_day():
    """Full day run (600 ticks) in alife_emergent: no ghost homes on birth, and json roundtrip preserves all state."""
    w_cfg = WorldConfig.from_preset("alife_emergent")
    w_cfg.initial_blobs = 30
    w_cfg.min_population_floor = 10
    n_cfg = NetworkConfig(
        depth=2,
        width=20,
        baldwin_plasticity=True,
        recurrent_dim=16,
    )
    world = SimulationWorld(world_cfg=w_cfg, net_cfg=n_cfg, seed=123)
    initial_homes_count = len(world.homes)

    # Test that reproducing in emergent mode does not construct new homes
    parent = world.blobs[0]
    for _ in range(5):
        world._reproduce_blob(parent)

    assert len(world.homes) == initial_homes_count, (
        f"Homes increased after births: {len(world.homes)} vs {initial_homes_count}"
    )

    # Ensure Hall of Fame is populated and simulate 1 full day (600 ticks)
    for b in world.blobs[:4]:
        world._consider_for_hall_of_fame(b)

    world.advance_ticks(600)
    assert world.tick_count >= 600
    assert len(world.hall_of_fame) > 0, "Hall of Fame should be populated!"

    # Export population JSON and test strict serialization through json.dumps
    exported = world.export_population_json()
    json_str = json.dumps(exported)
    loaded_data = json.loads(json_str)

    # Import into a new world instance
    target_world = SimulationWorld(world_cfg=WorldConfig(), net_cfg=NetworkConfig())
    target_world.import_population_json(loaded_data)

    # Assert ALife flags and configs preserved
    assert target_world.world_cfg.scale_preset == "alife_emergent"
    assert target_world.world_cfg.every_tick_brain is True
    assert target_world.world_cfg.egocentric_frame is True
    assert target_world.world_cfg.intrinsic_homeostasis is True
    assert target_world.world_cfg.continuous_lifecycle is True
    assert target_world.world_cfg.emergent_interactions is True

    # Assert Network configs preserved
    assert target_world.net_cfg.baldwin_plasticity is True
    assert target_world.net_cfg.recurrent_dim == 16

    # Assert WebSocket & state payload do not render settlement homes in emergent mode
    ws_meta, _ = target_world.get_ws_state_and_binary()
    assert ws_meta["homes"] == []
    state = target_world.get_state_payload()
    assert state["homes"] == []

    world.close()
    target_world.close()
