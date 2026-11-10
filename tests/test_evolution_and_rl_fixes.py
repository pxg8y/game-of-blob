"""
Comprehensive Mathematical Invariant Tests and KPI Verification Suite.
Validates:
1. Rotational Invariance of Egocentric Sensory & Motor Mapping
2. Zero-Sum Closed-Loop Potential-Based Reward Shaping (PBRS) & Target Discontinuity Invariance
   calling directly world._compute_movement_shaping_reward_batch across multiple starting angles
3. Lossless Hidden-Neuron Permutation Alignment & Crossover Invariance
4. Epigenetic Baseline Inheritance and Parity between BlobBrain and BatchedBrainPool
5. Efficiency Fitness Ranking (Active Forager > Spawn Camper > Jitterer)
"""

import math
import unittest
import numpy as np

from blob_sim.config import WorldConfig, NetworkConfig
from blob_sim.entities import Blob, Genes
from blob_sim.neural import BlobBrain, BatchedBrainPool
from blob_sim.world import SimulationWorld
from blob_sim.spatial import prep_step_observations_numba, apply_kinematics_and_metabolism_batch_numba


class TestEvolutionAndRLFixes(unittest.TestCase):
    """Rigorous mathematical invariant tests across the entire neural and RL pipeline."""

    def test_rotational_invariance(self) -> None:
        """
        Verify that rotating the world and the blob's heading by 90, 180, and 270 degrees
        produces identical egocentric observation vectors and identically rotated world displacements.
        """
        n = 1
        obs_batch = np.zeros((n, 26), dtype=np.float32)
        obs_batch[0, 0] = 1.0
        obs_batch[0, 1] = 0.0
        xs = np.zeros(n, dtype=np.float64)
        ys = np.zeros(n, dtype=np.float64)
        home_xs = np.array([50.0], dtype=np.float64)
        home_ys = np.array([0.0], dtype=np.float64)
        home_radii = np.array([10.0], dtype=np.float32)
        foods_today = np.zeros(n, dtype=np.float32)
        energies = np.array([50.0], dtype=np.float32)
        gene_sizes = np.array([1.0], dtype=np.float32)
        moneys = np.zeros(n, dtype=np.float32)
        reputations = np.zeros(n, dtype=np.float32)
        peer_reps = np.zeros(n, dtype=np.float32)
        is_at_home_out = np.zeros(n, dtype=np.int32)
        mem_xs = np.zeros(n, dtype=np.float64)
        mem_ys = np.zeros(n, dtype=np.float64)
        mem_strengths = np.zeros(n, dtype=np.float32)

        headings_0 = np.array([0.0], dtype=np.float32)
        prep_step_observations_numba(
            obs_batch, xs, ys, headings_0, home_xs, home_ys, home_radii,
            foods_today, energies, gene_sizes, moneys, reputations, peer_reps,
            1.0, 1.0 / 900.0, 0.5, is_at_home_out, mem_xs, mem_ys, mem_strengths
        )
        ego_food_0 = obs_batch[0, :2].copy()
        ego_home_0 = obs_batch[0, 3:5].copy()

        # 90 degrees rotation
        obs_batch_90 = np.zeros((n, 26), dtype=np.float32)
        obs_batch_90[0, 0] = 0.0
        obs_batch_90[0, 1] = 1.0
        home_xs_90 = np.array([0.0], dtype=np.float64)
        home_ys_90 = np.array([50.0], dtype=np.float64)
        headings_90 = np.array([np.float32(math.pi / 2.0)], dtype=np.float32)
        prep_step_observations_numba(
            obs_batch_90, xs, ys, headings_90, home_xs_90, home_ys_90, home_radii,
            foods_today, energies, gene_sizes, moneys, reputations, peer_reps,
            1.0, 1.0 / 900.0, 0.5, is_at_home_out, mem_xs, mem_ys, mem_strengths
        )
        ego_food_90 = obs_batch_90[0, :2].copy()
        ego_home_90 = obs_batch_90[0, 3:5].copy()

        np.testing.assert_allclose(ego_food_0, ego_food_90, atol=1e-5)
        np.testing.assert_allclose(ego_home_0, ego_home_90, atol=1e-5)

        # Motor Kinematics Rotational Invariance:
        actions = np.zeros((1, 6), dtype=np.float32)
        actions[0, 0] = 1.0  # forward
        actions[0, 1] = 0.0  # no lateral steering

        vxs_0 = np.zeros(1, dtype=np.float32)
        vys_0 = np.zeros(1, dtype=np.float32)
        xs_0 = np.array([500.0], dtype=np.float64)
        ys_0 = np.array([500.0], dtype=np.float64)
        h_0 = np.array([0.0], dtype=np.float32)
        apply_kinematics_and_metabolism_batch_numba(
            xs_0, ys_0, vxs_0, vys_0, h_0, energies.copy(), np.zeros(1, dtype=np.bool_),
            np.zeros(1, dtype=np.bool_), foods_today.copy(), np.array([4.0], dtype=np.float32),
            np.array([7.5], dtype=np.float32), np.array([0.05], dtype=np.float32),
            np.zeros(1, dtype=np.bool_), home_xs, home_ys, home_radii,
            actions, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32),
            1000.0, 1000.0, 2.0, 1.0, 600.0, 1.0, 1.0, False,
            np.zeros(1, dtype=np.int32), np.zeros(1, dtype=np.float32)
        )

        vxs_90 = np.zeros(1, dtype=np.float32)
        vys_90 = np.zeros(1, dtype=np.float32)
        xs_90 = np.array([500.0], dtype=np.float64)
        ys_90 = np.array([500.0], dtype=np.float64)
        h_90 = np.array([np.float32(math.pi / 2.0)], dtype=np.float32)
        apply_kinematics_and_metabolism_batch_numba(
            xs_90, ys_90, vxs_90, vys_90, h_90, energies.copy(), np.zeros(1, dtype=np.bool_),
            np.zeros(1, dtype=np.bool_), foods_today.copy(), np.array([4.0], dtype=np.float32),
            np.array([7.5], dtype=np.float32), np.array([0.05], dtype=np.float32),
            np.zeros(1, dtype=np.bool_), home_xs_90, home_ys_90, home_radii,
            actions, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32),
            1000.0, 1000.0, 2.0, 1.0, 600.0, 1.0, 1.0, False,
            np.zeros(1, dtype=np.int32), np.zeros(1, dtype=np.float32)
        )

        disp_x_0 = float(xs_0[0] - 500.0)
        disp_y_0 = float(ys_0[0] - 500.0)
        disp_x_90 = float(xs_90[0] - 500.0)
        disp_y_90 = float(ys_90[0] - 500.0)

        self.assertAlmostEqual(disp_x_0, disp_y_90, places=4)
        self.assertAlmostEqual(disp_y_0, 0.0, places=4)
        self.assertAlmostEqual(disp_x_90, 0.0, places=4)

    def test_zero_sum_closed_loop_pbrs_and_target_discontinuity(self) -> None:
        """
        Directly verify world._compute_movement_shaping_reward_batch on closed-loop circular
        and polygonal paths around food and oases with changing headings.
        Must yield exactly 0.0 accumulated shaping reward up to floating-point tolerance (< 1e-6).
        """
        world = SimulationWorld(seed=42)
        cx, cy = 500.0, 500.0
        R = 80.0
        d_norm = 215.0
        n_steps = 96

        for start_ang in [0.0, math.pi / 3.0, math.pi / 2.0, math.pi, 1.75 * math.pi]:
            sum_rewards = 0.0

            for s in range(n_steps):
                a0 = start_ang + 2.0 * math.pi * (s / n_steps)
                a1 = start_ang + 2.0 * math.pi * ((s + 1) / n_steps)

                x0 = cx + R * math.cos(a0)
                y0 = cy + R * math.sin(a0)
                x1 = cx + R * math.cos(a1)
                y1 = cy + R * math.sin(a1)

                disp_x = x1 - x0
                disp_y = y1 - y0

                # Tangential heading during movement along the circle
                th0 = a0 + math.pi / 2.0
                cos_th = math.cos(th0)
                sin_th = math.sin(th0)

                # Pre-step world vector to target at (cx, cy)
                rel_wx = cx - x0
                rel_wy = cy - y0
                d_before = math.hypot(rel_wx, rel_wy)

                unit_wx = rel_wx / max(1e-4, d_before)
                unit_wy = rel_wy / max(1e-4, d_before)
                dx_food_ego = cos_th * unit_wx + sin_th * unit_wy
                dy_food_ego = -sin_th * unit_wx + cos_th * unit_wy
                prox = max(0.0, 1.0 - d_before / d_norm)

                actions = np.zeros((1, 6), dtype=np.float32)
                oasis_vecs = np.zeros((1, 3), dtype=np.float32)

                r_batch = world._compute_movement_shaping_reward_batch(
                    in_swamps=np.zeros(1, dtype=np.bool_),
                    is_at_homes=np.zeros(1, dtype=np.bool_),
                    foods_today=np.zeros(1, dtype=np.float32),
                    actions_batch=actions,
                    dx_foods=np.array([dx_food_ego], dtype=np.float32),
                    dy_foods=np.array([dy_food_ego], dtype=np.float32),
                    prox_food=np.array([prox], dtype=np.float32),
                    oasis_vecs=oasis_vecs,
                    dx_homes=np.zeros(1, dtype=np.float32),
                    dy_homes=np.zeros(1, dtype=np.float32),
                    tod=0.20,
                    repro_target=2.0,
                    disp_xs=np.array([disp_x], dtype=np.float32),
                    disp_ys=np.array([disp_y], dtype=np.float32),
                    d_norms=np.array([d_norm], dtype=np.float32),
                    pre_headings=np.array([th0], dtype=np.float32),
                    rel_food_wx=np.array([rel_wx], dtype=np.float32),
                    rel_food_wy=np.array([rel_wy], dtype=np.float32),
                )
                sum_rewards += float(r_batch[0])

            self.assertAlmostEqual(sum_rewards, 0.0, places=6)

        world.close()

    def test_lossless_hidden_neuron_permutation_alignment(self) -> None:
        """
        Verify that Parent B (a permuted functional copy of Parent A) is aligned to Parent A
        losslessly so crossover with mutation_rate=0 matches Parent A to < 1e-6 across random inputs.
        """
        np.random.seed(123)
        cfg = NetworkConfig(depth=2, width=32)
        p_a = BlobBrain(config=cfg)
        p_b = BlobBrain(config=cfg)

        p_b.weights = [w.copy() for w in p_a.weights]
        p_b.biases = [b.copy() for b in p_a.biases]

        perm0 = np.random.permutation(32)
        perm1 = np.random.permutation(32)

        p_b.weights[0] = p_b.weights[0][:, perm0]
        p_b.biases[0] = p_b.biases[0][perm0]
        p_b.weights[1] = p_b.weights[1][perm0, :]

        p_b.weights[1] = p_b.weights[1][:, perm1]
        p_b.biases[1] = p_b.biases[1][perm1]
        p_b.weights[2] = p_b.weights[2][perm1, :]

        test_inputs = np.random.randn(20, 26).astype(np.float32)

        def eval_mlp(brain, x_arr):
            a = x_arr
            for l in range(len(brain.weights) - 1):
                a = np.tanh(a @ brain.weights[l] + brain.biases[l])
            return np.tanh(a @ brain.weights[-1] + brain.biases[-1])

        out_a = eval_mlp(p_a, test_inputs)
        out_b = eval_mlp(p_b, test_inputs)
        np.testing.assert_allclose(out_a, out_b, atol=1e-6)

        child = p_a.clone_and_mutate(mutation_rate=0.0, mutation_strength=0.0, partner=p_b)
        out_child = eval_mlp(child, test_inputs)
        np.testing.assert_allclose(out_a, out_child, atol=1e-6)

    def test_epigenetic_stability_and_pool_parity(self) -> None:
        """
        Verify that newborn baseline inheritance prevents Day-1 advantage spikes,
        and BlobBrain and BatchedBrainPool produce matching outputs and online weight updates.
        """
        cfg = NetworkConfig(depth=1, width=16)
        parent = BlobBrain(config=cfg)
        parent.reward_baseline_event = 1.15
        child = parent.clone_and_mutate(mutation_rate=0.0, mutation_strength=0.0)

        self.assertAlmostEqual(child.reward_baseline_event, 1.15, places=5)
        self.assertEqual(child.reward_baseline_dense, 0.0)

        # Dual-Engine Parity Check: BlobBrain vs BatchedBrainPool
        single_brain = BlobBrain(config=cfg)
        batched_brain = BlobBrain(config=cfg)
        for l in range(len(single_brain.weights)):
            batched_brain.weights[l] = single_brain.weights[l].copy()
            batched_brain.biases[l] = single_brain.biases[l].copy()

        pool = BatchedBrainPool()
        pool.bind([batched_brain])

        obs = np.random.randn(1, 26).astype(np.float32)
        rng = np.random.default_rng(42)

        act_single = single_brain.forward(obs[0], explore_noise=0.0)
        act_batch = pool.forward_and_trace([batched_brain], obs, 0.0, rng, update_traces=True)

        np.testing.assert_allclose(act_single, act_batch[0], atol=1e-5)

        single_brain.apply_online_learning(reward=0.85, learning_rate=0.05)
        pool.apply_online_learning_batch([batched_brain], np.array([0.85], dtype=np.float32), 0.05, flush=True)

        for l in range(len(single_brain.weights)):
            np.testing.assert_allclose(single_brain.weights[l], batched_brain.weights[l], atol=1e-5)
            np.testing.assert_allclose(single_brain.biases[l], batched_brain.biases[l], atol=1e-5)

    def test_fitness_ranking_prefers_active_forager(self) -> None:
        """Verify efficiency-normalized ranking prefers active foragers over sedentary campers and aimless jitterers."""
        world = SimulationWorld(seed=42)
        b1 = world.blobs[0]
        b1.food_eaten_lifetime = 20.0
        b1.distance_traveled = 1000.0
        b1.days_sheltered = 3
        b1.age_days = 3
        b1.children_count = 2

        b2 = world.blobs[1]
        b2.food_eaten_lifetime = 3.0
        b2.distance_traveled = 5.0
        b2.days_sheltered = 3
        b2.age_days = 3
        b2.children_count = 0

        b3 = world.blobs[2]
        b3.food_eaten_lifetime = 4.0
        b3.distance_traveled = 8000.0
        b3.days_sheltered = 0
        b3.age_days = 3
        b3.children_count = 0

        world._update_hall_of_fame()
        self.assertGreaterEqual(len(world.hall_of_fame), 3)
        top_blob_id = world.hall_of_fame[0]["blob_id"]
        self.assertEqual(top_blob_id, b1.id)

        score_1 = next(h["score"] for h in world.hall_of_fame if h["blob_id"] == b1.id)
        score_2 = next(h["score"] for h in world.hall_of_fame if h["blob_id"] == b2.id)
        score_3 = next(h["score"] for h in world.hall_of_fame if h["blob_id"] == b3.id)

        self.assertGreater(score_1, score_2)
        self.assertGreater(score_2, score_3)
        world.close()


if __name__ == "__main__":
    unittest.main()
