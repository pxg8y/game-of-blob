"""Tests for recurrent working memory (recurrent_dim, leaky-integrator dynamics, and parity)."""

import numpy as np
import pytest
from blob_sim.config import NetworkConfig, WorldConfig
from blob_sim.neural import BlobBrain, BatchedBrainPool, INPUT_DIM, OUTPUT_DIM, INPUT_LABELS, OUTPUT_LABELS
from blob_sim.world import SimulationWorld


@pytest.mark.parametrize("N,explore_noise", [
    (40, 0.0),
    (40, 0.05),
    (120, 0.0),
    (120, 0.05),
])
def test_dual_engine_parity_over_ticks(N: int, explore_noise: float):
    """Test strict numerical parity between BlobBrain and BatchedBrainPool across 15 ticks."""
    cfg = NetworkConfig(depth=2, width=32, recurrent_dim=16)
    rng_master = np.random.default_rng(12345)

    brains_single = [BlobBrain(cfg, rng=np.random.default_rng(i + 1000)) for i in range(N)]
    brains_pool = [
        BlobBrain(cfg, weights=b.weights, biases=b.biases, rng=np.random.default_rng(i + 1000))
        for i, b in enumerate(brains_single)
    ]

    pool = BatchedBrainPool()

    for tick in range(15):
        # Varying sensory inputs over time
        obs_batch = rng_master.standard_normal((N, INPUT_DIM), dtype=np.float32)

        # Synchronize RNG for explore noise if enabled
        if explore_noise > 0.0:
            seed_tick = 9999 + tick * 100
            rng_single = np.random.default_rng(seed_tick)
            rng_pool = np.random.default_rng(seed_tick)
            for b in brains_single:
                b.rng = rng_single
        else:
            rng_pool = np.random.default_rng(42)

        # Vectorized pool forward
        out_pool = pool.forward_and_trace(brains_pool, obs_batch, explore_noise=explore_noise, rng=rng_pool)

        # Single brain forward
        for i in range(N):
            out_s = brains_single[i].forward(obs_batch[i], explore_noise=explore_noise)
            out_p = out_pool[i]

            diff = np.max(np.abs(out_s - out_p))
            h_diff = np.max(np.abs(brains_single[i].hidden_state - brains_pool[i].hidden_state))
            assert diff < 1e-4, f"Tick {tick}, blob {i}: action diff {diff} exceeds tolerance"
            assert h_diff < 1e-4, f"Tick {tick}, blob {i}: hidden state diff {h_diff} exceeds tolerance"


def test_recurrent_memory_impulse_retention():
    """Test that a sensory impulse at tick 0 affects output at tick 10 even when input returns to zero."""
    cfg = NetworkConfig(depth=2, width=32, recurrent_dim=16)
    brain_stim = BlobBrain(cfg, rng=np.random.default_rng(42))
    brain_null = BlobBrain(cfg, weights=brain_stim.weights, biases=brain_stim.biases, rng=np.random.default_rng(42))

    # Tick 0: Stimulus on brain_stim, zeros on brain_null
    impulse = np.zeros(INPUT_DIM, dtype=np.float32)
    impulse[0] = 5.0  # Strong impulse on food dx
    zero_obs = np.zeros(INPUT_DIM, dtype=np.float32)

    brain_stim.forward(impulse, explore_noise=0.0)
    brain_null.forward(zero_obs, explore_noise=0.0)

    # Ticks 1 to 10: Both receive strictly zero inputs
    for tick in range(1, 11):
        out_stim = brain_stim.forward(zero_obs, explore_noise=0.0)
        out_null = brain_null.forward(zero_obs, explore_noise=0.0)

    # At tick 10, the leaky integrator hidden state and motor output should still differ significantly
    h_diff = np.max(np.abs(brain_stim.hidden_state - brain_null.hidden_state))
    out_diff = np.max(np.abs(out_stim - out_null))

    assert h_diff > 1e-3, f"Hidden state memory decayed too quickly: h_diff={h_diff}"
    assert out_diff > 1e-4, f"Motor action output does not reflect retained memory: out_diff={out_diff}"


def test_simulation_world_swap_remove_and_telemetry_coupling_matrix():
    """Test SimulationWorld with recurrent_dim=16 under blob deaths (swap-remove) and check telemetry coupling matrix."""
    world_cfg = WorldConfig.from_preset("classic", initial_blobs=30)
    world = SimulationWorld(world_cfg=world_cfg)
    world.net_cfg = NetworkConfig(depth=2, width=32, recurrent_dim=16)
    world.reset_world()

    assert world.net_cfg.get_recurrent_dim() == 16
    assert len(world.blobs) == 30
    for b in world.blobs:
        assert b.brain.recurrent_dim == 16
        assert b.brain.hidden_state is not None
        assert b.brain.hidden_state.shape == (16,)

    # Run world for 25 steps to initialize traces and actions
    world.advance_ticks(25)

    # Manually trigger death for 5 blobs to trigger swap-remove in blob list and BatchedBrainPool
    living = [b for b in world.blobs if b.alive]
    for b in living[:5]:
        world._kill_blob(b, "Test-Sterben")

    # Advance more ticks after compaction
    world.advance_ticks(10)

    # Verify that telemetry logger neural weight stats still generate a 26x6 coupling matrix
    living_after = [b for b in world.blobs if b.alive]
    stats = world.telemetry._compute_neural_weight_stats(living_after)
    coupling_matrix = np.asarray(stats["coupling_matrix"])
    assert coupling_matrix.shape == (len(INPUT_LABELS), len(OUTPUT_LABELS))
    assert coupling_matrix.shape == (26, 6)
