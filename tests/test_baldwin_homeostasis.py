"""Tests for (1) Baldwin evolution with genome/plastic weight separation & crossover alignment
and (2) body-based intrinsic homeostasis learning signal with evolvable drive genes."""

import numpy as np
import pytest

from blob_sim.config import NetworkConfig, WorldConfig
from blob_sim.entities import Genes, FoodItem
from blob_sim.neural import BlobBrain, BatchedBrainPool
from blob_sim.world import SimulationWorld


def test_baldwin_weight_separation_and_inheritance():
    """
    1. Prüfe Baldwin-Trennung: Online-RL verändert brain.weights[0],
       lässt brain.innate_weights[0] jedoch bitgenau unverändert.
    2. child = brain.clone_and_mutate(0.0, 0.0) besitzt Startgewichte identisch zu parent.innate_weights.
    """
    cfg = NetworkConfig(baldwin_plasticity=True, depth=2, width=16)
    brain = BlobBrain(config=cfg)

    assert brain.config.baldwin_plasticity is True
    assert len(brain.innate_weights) == len(brain.weights)

    # Initial copy check
    w0_innate_before = brain.innate_weights[0].copy()
    b0_innate_before = brain.innate_biases[0].copy()
    assert np.allclose(brain.weights[0], w0_innate_before)

    # Simulate eligibility trace and RL update
    obs = np.ones((26,), dtype=np.float32)
    brain.forward(obs, explore_noise=0.0)
    # Apply online learning
    brain.apply_online_learning(reward=1.5, learning_rate=0.25)

    # Active weights must have changed
    assert not np.array_equal(brain.weights[0], w0_innate_before)
    # Innate weights MUST remain bitwise unchanged
    assert np.array_equal(brain.innate_weights[0], w0_innate_before)
    assert np.array_equal(brain.innate_biases[0], b0_innate_before)

    # Offspring creation with zero mutation
    child = brain.clone_and_mutate(mutation_rate=0.0, mutation_strength=0.0)
    assert child.config.baldwin_plasticity is True
    # Offspring start weights must equal parent's innate weights (not the shifted lifetime weights!)
    assert np.array_equal(child.weights[0], w0_innate_before)
    assert np.array_equal(child.innate_weights[0], w0_innate_before)


def test_baldwin_crossover_neuron_alignment():
    """
    Prüfe Neuronen-Alignment: Erzeuge brain_b durch zufällige Spalten-Permutation des Hidden-Layers
    von brain_a (und korrespondierende Zeilen-Permutation der Folgeschicht).
    Prüfe, dass beide identische Outputs berechnen und ein Crossover-Kind (mutation_rate=0.0)
    funktional identisch bleibt statt zu kollabieren.
    """
    cfg = NetworkConfig(baldwin_plasticity=True, depth=2, width=16)
    brain_a = BlobBrain(config=cfg)

    # Create brain_b by permuting each hidden layer of brain_a
    H = cfg.width
    rng = np.random.default_rng(123)
    perm0 = rng.permutation(H)
    perm1 = rng.permutation(H)

    # Layer 0: shape (In, H) -> columns permuted by perm0
    w_b0 = brain_a.innate_weights[0][:, perm0].copy()
    b_b0 = brain_a.innate_biases[0][perm0].copy()

    # Layer 1: shape (H, H) -> rows permuted by perm0, columns by perm1
    w_b1 = brain_a.innate_weights[1][perm0, :][:, perm1].copy()
    b_b1 = brain_a.innate_biases[1][perm1].copy()

    # Layer 2: shape (H, Out) -> rows permuted by perm1
    w_b2 = brain_a.innate_weights[2][perm1, :].copy()
    b_b2 = brain_a.innate_biases[2].copy()

    brain_b = BlobBrain(
        config=cfg,
        weights=[w_b0.copy(), w_b1.copy(), w_b2.copy()],
        biases=[b_b0.copy(), b_b1.copy(), b_b2.copy()],
        innate_weights=[w_b0, w_b1, w_b2],
        innate_biases=[b_b0, b_b1, b_b2],
    )

    # Test that brain_a and brain_b calculate identical outputs on random inputs
    for _ in range(5):
        test_obs = rng.standard_normal(26, dtype=np.float32)
        out_a = brain_a.forward(test_obs, explore_noise=0.0)
        out_b = brain_b.forward(test_obs, explore_noise=0.0)
        assert np.allclose(out_a, out_b, atol=1e-5), "Permuted network must compute identical function"

    # Now create crossover offspring child between brain_a and brain_b with 0 mutation
    # Because partner neurons align greedily to parent A, crossover columns match 1:1!
    child = brain_a.clone_and_mutate(mutation_rate=0.0, mutation_strength=0.0, partner=brain_b)

    # Child should compute identical output to parent A (and B) without functional collapse
    for _ in range(5):
        test_obs = rng.standard_normal(26, dtype=np.float32)
        out_parent = brain_a.forward(test_obs, explore_noise=0.0)
        out_child = child.forward(test_obs, explore_noise=0.0)
        assert np.allclose(out_child, out_parent, atol=1e-5), "Crossover child must maintain functional equivalence"


def test_intrinsic_homeostasis_learning_signal():
    """
    Prüfe Homöostase: In einer Welt mit intrinsic_homeostasis=True erzeugt bloßes Laufen
    ohne Nahrungsaufnahme ein leicht negatives Signal (Stoffwechselverbrauch),
    während das tatsächliche Fressen eines Nahrungs-Items ein deutlich positives Signal
    skaliert mit genes.drive_energy erzeugt.
    """
    wcfg = WorldConfig(
        initial_blobs=2,
        daily_food_spawn=0,
        intrinsic_homeostasis=True,
    )
    world = SimulationWorld(world_cfg=wcfg, seed=999)
    assert world.world_cfg.intrinsic_homeostasis is True

    # Configure blob 0 with high drive_energy
    b0 = world.blobs[0]
    b0.genes.drive_energy = 2.0
    b0.genes.drive_pain = 1.0
    b0.genes.drive_social = 0.0
    b0.genes.plasticity_lr = 1.0
    world._sync_static_blob_kin_arrays([b0, world.blobs[1]])

    # 1. Step without food: Energy decreases due to basal metabolism
    prev_e = b0.energy
    world.step(1)
    # Blob moved or had basal cost, energy decreased
    assert b0.energy < prev_e
    # Reward should be negative (somatic pain/energy loss)
    assert b0.last_reward < 0.0, f"Expected negative reward from metabolism, got {b0.last_reward}"

    # 2. Place food directly at blob's position and perform a step
    food = FoodItem(id=99999, x=b0.x, y=b0.y, nutrition=1.5)
    world.foods = [food]
    prev_energy_before_feed = b0.energy
    world.step(1)

    # Blob should have eaten the food, increasing energy
    assert b0.energy > prev_energy_before_feed
    # Reward should be distinctly positive, scaled with drive_energy (2.0)
    assert b0.last_reward > 0.05, f"Expected positive reward from foraging, got {b0.last_reward}"
