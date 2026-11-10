"""Configurable Neural Network Brain with Online RL & Neuroevolution."""

from typing import List, Dict, Any, Optional
import numpy as np
from .config import NetworkConfig
from .spatial import parallel_mlp_layer_forward, parallel_mlp_backprop_delta, mutate_neural_layer_numba

_ACT_CODE_MAP = {"tanh": 0, "relu": 1, "gelu": 2, "silu": 3}

INPUT_LABELS: List[str] = [
    "Nahrung ΔX",
    "Nahrung ΔY",
    "Nahrung Nähe",
    "Zuhause ΔX",
    "Zuhause ΔY",
    "Zuhause Nähe",
    "Nachbar ΔX",
    "Nachbar ΔY",
    "Nachbar Nähe",
    "Nachbar hat Essen",
    "Nachbar hat Geld",
    "Gleiche Gruppe",
    "Tages-Essen (0..2x)",
    "Energie / Sättigung",
    "Geld-Vermögen",
    "Tageszeit (Tag/Nacht)",
    "Oase ΔX",
    "Oase ΔY",
    "Oasen-Fruchtbarkeit",
    "Gedächtnis ΔX",
    "Gedächtnis ΔY",
    "Fühler Links (Hindernis)",
    "Fühler Vorne (Hindernis)",
    "Fühler Rechts (Hindernis)",
    "Nachbar Reputation",
    "Eigene Reputation",
]

OUTPUT_LABELS: List[str] = [
    "Bewegung X",
    "Bewegung Y",
    "Sprint / Tempo",
    "Handels-Wunsch",
    "Gruppen-Sozialität",
    "Siedeln / Umziehen",
]

INPUT_DIM = len(INPUT_LABELS)
OUTPUT_DIM = len(OUTPUT_LABELS)


_F32_ZERO = np.float32(0.0)
_F32_HALF = np.float32(0.5)
_F32_ONE = np.float32(1.0)
_F32_SQRT_2_PI = np.float32(0.7978845608028654)
_F32_GELU_COEF = np.float32(0.044715)
_F32_RELU_LEAK = np.float32(0.05)
_F32_TANH_FLOOR = np.float32(0.08)
_F32_SILU_CLIP_LO = np.float32(-15.0)
_F32_SILU_CLIP_HI = np.float32(15.0)
_F32_W_MIN = np.float32(-3.5)
_F32_W_MAX = np.float32(3.5)
_F32_B_MIN = np.float32(-2.5)
_F32_B_MAX = np.float32(2.5)

_TRACE_RANK = 13  # 1 carry-over slot + 12 trace steps per flush window
_FLUSH_CHUNK = 24  # L2-cache-resident batch chunk size (~1.5 MB per chunk at 128x128)
_umath_clip = np._core.umath.clip


def _get_trace_rank_for_dims(dims: List[int]) -> int:
    """Adaptive rank: rank 13 for standard networks (width <= 512); rank 2 for 10k networks."""
    return 2 if (len(dims) > 0 and max(dims) > 512) else _TRACE_RANK


def apply_activation(x: np.ndarray, kind: str, out: Optional[np.ndarray] = None) -> np.ndarray:
    if kind == "relu":
        return np.maximum(_F32_ZERO, x, out=out)
    if kind == "gelu":
        x3 = x * x * x
        res = _F32_HALF * x * (_F32_ONE + np.tanh(_F32_SQRT_2_PI * (x + _F32_GELU_COEF * x3)))
        if out is not None:
            out[:] = res
            return out
        return res
    if kind == "silu":
        sig = _F32_ONE / (_F32_ONE + np.exp(-_umath_clip(x, _F32_SILU_CLIP_LO, _F32_SILU_CLIP_HI)))
        res = x * sig
        if out is not None:
            out[:] = res
            return out
        return res
    # default: tanh
    return np.tanh(x, out=out)


def activation_derivative(a: np.ndarray, kind: str) -> np.ndarray:
    """Approximate derivative w.r.t. pre-activation for credit propagation (strictly float32)."""
    if kind == "relu":
        return np.where(a > _F32_ZERO, _F32_ONE, _F32_RELU_LEAK)
    if kind in ("gelu", "silu"):
        a2 = a * a
        return _F32_ONE - _F32_HALF * (a2 / (_F32_ONE + a2))
    # default: tanh (a is already in [-1, 1], so 1 - a*a <= 1.0)
    return np.maximum(_F32_TANH_FLOOR, _F32_ONE - a * a)


class BlobBrain:
    """
    Feedforward MLP with user-configurable depth and width, backpropagated eligibility
    traces for lifetime reward-modulated learning (Lamarckian RL), and genetic mutation.
    """

    def __init__(
        self,
        config: NetworkConfig,
        weights: Optional[List[np.ndarray]] = None,
        biases: Optional[List[np.ndarray]] = None,
        rng: Optional[np.random.Generator] = None,
        _copy_arrays: bool = True,
        innate_seed: float = 0.72,
    ) -> None:
        self.config = NetworkConfig(
            depth=config.depth,
            width=config.width,
            custom_layers=list(config.custom_layers) if config.custom_layers else None,
            activation=config.activation,
        )
        self.rng = rng or np.random.default_rng()
        hidden_dims = self.config.get_hidden_dims()
        self.layer_dims: List[int] = [INPUT_DIM, *hidden_dims, OUTPUT_DIM]

        shapes_match = (
            weights is not None
            and biases is not None
            and len(weights) == len(self.layer_dims) - 1
            and all(
                w.shape == (self.layer_dims[i], self.layer_dims[i + 1])
                for i, w in enumerate(weights)
            )
            and all(b.shape == (self.layer_dims[i + 1],) for i, b in enumerate(biases))
        )
        if shapes_match and weights is not None and biases is not None:
            if _copy_arrays:
                self.weights = [w.astype(np.float32, copy=True) for w in weights]
                self.biases = [b.astype(np.float32, copy=True) for b in biases]
            else:
                self.weights = weights
                self.biases = biases
        else:
            self.weights = []
            self.biases = []
            for i in range(len(self.layer_dims) - 1):
                fan_in = self.layer_dims[i]
                fan_out = self.layer_dims[i + 1]
                scale = np.float32(0.55 * np.sqrt(2.0 / (fan_in + fan_out)))
                w = self.rng.standard_normal((fan_in, fan_out), dtype=np.float32) * scale
                b = np.zeros((fan_out,), dtype=np.float32)
                self.weights.append(w)
                self.biases.append(b)

            # Minimal initial survival seed: ONLY a gentle food-curiosity reflex on channels 0,1 (when innate_seed > 0).
            # Home return, obstacle avoidance, oasis long-range scent, memory, trading & colonization
            # are NOT pre-wired — blobs must genuinely learn and evolve them through the training map!
            if innate_seed > 0.0 and all(dim >= 2 for dim in hidden_dims):
                s_val = np.float32(innate_seed)
                # Input 0,1 (food dx, dy) -> hidden_0[0,1]
                self.weights[0][0, 0] += np.float32(1.05) * s_val
                self.weights[0][1, 1] += np.float32(1.05) * s_val
                # Residual pass-through on channels 0..1 across intermediate hidden layers
                for mid_idx in range(1, len(self.weights) - 1):
                    for ch in range(2):
                        self.weights[mid_idx][ch, ch] += np.float32(1.08) * max(np.float32(0.75), s_val)
                # Last hidden layer channels 0..1 -> Output 0,1 (move_x, move_y)
                self.weights[-1][0, 0] += np.float32(0.95) * s_val
                self.weights[-1][1, 1] += np.float32(0.95) * s_val

            # If legacy weights were passed with different input/output dims, copy overlapping sub-matrices
            if weights is not None and biases is not None:
                min_l = min(len(weights), len(self.weights))
                for i in range(min_l):
                    ow, nw = weights[i], self.weights[i]
                    r, c = min(ow.shape[0], nw.shape[0]), min(ow.shape[1], nw.shape[1])
                    nw[:r, :c] = ow[:r, :c].astype(np.float32)
                    ob, nb = biases[i], self.biases[i]
                    cb = min(ob.shape[0], nb.shape[0])
                    nb[:cb] = ob[:cb].astype(np.float32)

        self._param_count: int = sum(
            int(w.size + b.size) for w, b in zip(self.weights, self.biases, strict=True)
        )
        # Lazy eligibility traces for single-brain fallback (BatchedBrainPool uses low-rank ring buffers)
        self._traces_w: Optional[List[np.ndarray]] = None
        self._traces_b: Optional[List[np.ndarray]] = None
        self._pool_ref: Optional["BatchedBrainPool"] = None
        self.last_activations: List[np.ndarray] = [
            np.zeros((dim,), dtype=np.float32) for dim in self.layer_dims
        ]
        self.reward_baseline: float = 0.0
        self.reward_baseline_dense: float = 0.0
        self.reward_baseline_event: float = 0.0

    def _sync_from_pool(self) -> None:
        if self._pool_ref is not None:
            self._pool_ref.flush_single_brain(self)

    @property
    def traces_w(self) -> List[np.ndarray]:
        if self._traces_w is None:
            self._traces_w = [np.zeros_like(w) for w in self.weights]
        return self._traces_w

    @traces_w.setter
    def traces_w(self, value: List[np.ndarray]) -> None:
        self._traces_w = value

    @property
    def traces_b(self) -> List[np.ndarray]:
        if self._traces_b is None:
            self._traces_b = [np.zeros_like(b) for b in self.biases]
        return self._traces_b

    @traces_b.setter
    def traces_b(self, value: List[np.ndarray]) -> None:
        self._traces_b = value

    def forward(self, obs: np.ndarray, explore_noise: float = 0.05) -> np.ndarray:
        """Compute action outputs from sensory observation vector and update eligibility traces."""
        a = np.asarray(obs, dtype=np.float32)
        self.last_activations = [a]

        num_layers = len(self.weights)
        for idx in range(num_layers):
            z = a @ self.weights[idx] + self.biases[idx]
            if idx < num_layers - 1:
                a = apply_activation(z, self.config.activation)
            else:
                # Output layer bounded in [-1, 1] via tanh
                if explore_noise > 0.0:
                    z = z + self.rng.standard_normal(z.shape, dtype=np.float32) * np.float32(explore_noise)
                a = np.tanh(z)
            self.last_activations.append(a)

        # Backpropagate output sensitivity delta across layers for structurally valid credit assignment
        decay = np.float32(0.95)
        one_minus_decay = np.float32(0.05)
        out_act = self.last_activations[-1]
        delta = out_act * np.maximum(np.float32(0.15), _F32_ONE - out_act * out_act)
        tw_list = self.traces_w
        tb_list = self.traces_b
        for idx in range(num_layers - 1, -1, -1):
            pre = self.last_activations[idx][:, None]
            post_delta = delta[None, :]
            tw_list[idx] *= decay
            tw_list[idx] += one_minus_decay * (pre @ post_delta)
            tb_list[idx] *= decay
            tb_list[idx] += one_minus_decay * delta
            if idx > 0:
                hidden_act = self.last_activations[idx]
                deriv = activation_derivative(hidden_act, self.config.activation)
                delta = (delta @ self.weights[idx].T) * deriv

        return self.last_activations[-1]

    def apply_online_learning(self, reward: float, learning_rate: float) -> None:
        """
        Update weights online during the blob's lifetime based on reward signal
        (reward-modulated backpropagated eligibility trace update).
        """
        if learning_rate <= 0.0:
            return
        if self._pool_ref is not None and id(self) in self._pool_ref._slot_map:
            self._pool_ref.apply_single_brain_learning(self, reward, learning_rate)
            return

        r = float(reward)
        # Decouple dense movement shaping from sparse discrete events
        is_dense = abs(r) <= 0.35
        if abs(r) < 1e-6:
            # On zero ticks, decay both baselines
            self.reward_baseline_dense *= 0.96
            self.reward_baseline_event *= 0.96
            self.reward_baseline = self.reward_baseline_dense + self.reward_baseline_event
            return
        elif is_dense:
            prev_b = self.reward_baseline_dense
            self.reward_baseline_dense = 0.98 * prev_b + 0.02 * r
            self.reward_baseline_event *= 0.98
            cur_baseline = prev_b
        else:
            prev_b = self.reward_baseline_event
            self.reward_baseline_event = 0.90 * prev_b + 0.10 * r
            self.reward_baseline_dense *= 0.98
            cur_baseline = prev_b
        self.reward_baseline = self.reward_baseline_dense + self.reward_baseline_event

        if abs(r) < 1e-6:
            return

        if r > 0.0:
            raw_adv = max(0.25 * r, r - cur_baseline)
            advantage = min(2.5, max(0.0, raw_adv))
        else:
            raw_adv = min(r, r - cur_baseline)
            advantage = max(-2.5, min(0.0, raw_adv))

        step = np.float32(learning_rate * advantage * 0.18)
        half_step = np.float32(step * 0.5)
        tw_list = self.traces_w
        tb_list = self.traces_b

        for idx in range(len(self.weights)):
            w = self.weights[idx]
            w += step * tw_list[idx]
            _umath_clip(w, _F32_W_MIN, _F32_W_MAX, out=w)

            b = self.biases[idx]
            b += half_step * tb_list[idx]
            _umath_clip(b, _F32_B_MIN, _F32_B_MAX, out=b)

    def _mutate_layer_arrays(
        self, w: np.ndarray, b: np.ndarray, mutation_rate: float, mut_rate_f32: np.float32, mut_str_f32: np.float32
    ) -> None:
        w_size = w.size
        if w_size > 256 and mutation_rate < 0.35:
            k = int(self.rng.binomial(w_size, mutation_rate))
            if k > 0:
                flat_w = w.ravel()
                indices = self.rng.integers(0, w_size, size=k)
                noise_w = self.rng.standard_normal(k, dtype=np.float32) * mut_str_f32
                mut_vals = flat_w[indices] + noise_w
                _umath_clip(mut_vals, _F32_W_MIN, _F32_W_MAX, out=mut_vals)
                flat_w[indices] = mut_vals
        else:
            mut_mask_w = self.rng.random(w.shape, dtype=np.float32) < mut_rate_f32
            noise_w = self.rng.standard_normal(w.shape, dtype=np.float32) * mut_str_f32
            w += mut_mask_w * noise_w
            _umath_clip(w, _F32_W_MIN, _F32_W_MAX, out=w)

        b_size = b.size
        if b_size >= 16 and mutation_rate < 0.35:
            kb = int(self.rng.binomial(b_size, mutation_rate))
            if kb > 0:
                b_idx = self.rng.integers(0, b_size, size=kb)
                noise_b = self.rng.standard_normal(kb, dtype=np.float32) * mut_str_f32
                mut_b = b[b_idx] + noise_b
                _umath_clip(mut_b, _F32_B_MIN, _F32_B_MAX, out=mut_b)
                b[b_idx] = mut_b
        else:
            mut_mask_b = self.rng.random(b.shape, dtype=np.float32) < mut_rate_f32
            if np.any(mut_mask_b):
                noise_b = self.rng.standard_normal(b.shape, dtype=np.float32) * mut_str_f32
                b += mut_mask_b * noise_b
                _umath_clip(b, _F32_B_MIN, _F32_B_MAX, out=b)

    def clone_and_mutate(
        self,
        mutation_rate: float,
        mutation_strength: float,
        partner: Optional["BlobBrain"] = None,
    ) -> "BlobBrain":
        """Create an offspring brain with optional crossover and Gaussian mutation."""
        self._sync_from_pool()
        if partner is not None:
            partner._sync_from_pool()

        new_weights: List[np.ndarray] = []
        new_biases: List[np.ndarray] = []

        can_crossover = (
            partner is not None
            and partner.layer_dims == self.layer_dims
        )
        mut_rate = float(mutation_rate)
        mut_str = float(mutation_strength)

        for idx in range(len(self.weights)):
            w = self.weights[idx]
            b = self.biases[idx]

            if can_crossover and partner is not None:
                pw = partner.weights[idx]
                pb = partner.biases[idx]
                has_partner = True
            else:
                pw = w
                pb = b
                has_partner = False

            new_w, new_b = mutate_neural_layer_numba(
                w, b, pw, pb, has_partner, mut_rate, mut_str
            )
            new_weights.append(new_w)
            new_biases.append(new_b)

        return BlobBrain(
            config=self.config,
            weights=new_weights,
            biases=new_biases,
            rng=self.rng,
            _copy_arrays=False,
        )

    def adapt_to_config(self, new_config: NetworkConfig) -> "BlobBrain":
        """
        Adapt this brain to a new user-defined depth/width configuration,
        preserving overlapping sub-matrices of weights so learned behavior is not lost.
        """
        self._sync_from_pool()
        adapted = BlobBrain(config=new_config, rng=self.rng)
        min_layers = min(len(self.weights), len(adapted.weights))
        for i in range(min_layers):
            old_w = self.weights[i]
            new_w = adapted.weights[i]
            r = min(old_w.shape[0], new_w.shape[0])
            c = min(old_w.shape[1], new_w.shape[1])
            adapted.weights[i][:r, :c] = old_w[:r, :c]

            old_b = self.biases[i]
            new_b = adapted.biases[i]
            cb = min(old_b.shape[0], new_b.shape[0])
            adapted.biases[i][:cb] = old_b[:cb]

        # Also preserve output projection from last layer if layer count changed
        if len(self.weights) != len(adapted.weights):
            old_last_w = self.weights[-1]
            new_last_w = adapted.weights[-1]
            r = min(old_last_w.shape[0], new_last_w.shape[0])
            c = min(old_last_w.shape[1], new_last_w.shape[1])
            adapted.weights[-1][:r, :c] = old_last_w[:r, :c]
            adapted.biases[-1][:c] = self.biases[-1][:c]

        return adapted

    def parameter_count(self) -> int:
        return self._param_count

    def to_inspector_payload(self, max_nodes_per_layer: int = 26) -> Dict[str, Any]:
        """Serialize activations and downsampled weights for live GUI neural network rendering."""
        self._sync_from_pool()
        layers_act = [
            np.round(a[:max_nodes_per_layer].astype(np.float64), 3).tolist()
            for a in self.last_activations
        ]

        weights_viz = []
        for w in self.weights:
            r = min(w.shape[0], max_nodes_per_layer)
            c = min(w.shape[1], max_nodes_per_layer)
            weights_viz.append(np.round(w[:r, :c].astype(np.float64), 2).tolist())

        return {
            "layer_dims": self.layer_dims,
            "activation": self.config.activation,
            "param_count": self._param_count,
            "activations": layers_act,
            "weights": weights_viz,
            "input_labels": INPUT_LABELS,
            "output_labels": OUTPUT_LABELS,
        }

    def to_snapshot_dict(self) -> Dict[str, Any]:
        """Fast in-memory snapshot preserving float32 NumPy arrays without Python float boxing."""
        self._sync_from_pool()
        return {
            "config": self.config.to_dict(),
            "layer_dims": list(self.layer_dims),
            "weights": [w.copy() for w in self.weights],
            "biases": [b.copy() for b in self.biases],
        }

    def to_dict(self) -> Dict[str, Any]:
        self._sync_from_pool()
        return {
            "config": self.config.to_dict(),
            "layer_dims": self.layer_dims,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], rng: Optional[np.random.Generator] = None) -> "BlobBrain":
        cfg_data = data.get("config", {})
        cfg = NetworkConfig(
            depth=cfg_data.get("depth", 2),
            width=cfg_data.get("width", 24),
            custom_layers=cfg_data.get("custom_layers"),
            activation=cfg_data.get("activation", "tanh"),
        )
        weights = [np.asarray(w, dtype=np.float32) for w in data.get("weights", [])]
        biases = [np.asarray(b, dtype=np.float32) for b in data.get("biases", [])]
        return cls(config=cfg, weights=weights, biases=biases, rng=rng, _copy_arrays=True)


class BatchedBrainPool:
    """
    Zero-copy vectorized batch executor for all living BlobBrains sharing a topology.
    Uses capacity-backed tensors for incremental slot compaction on birth/death and
    Low-Rank Ring-Buffer Eligibility Traces so trace updates and RL stay inside CPU L2/L3 cache.
    """

    def __init__(self) -> None:
        self._bound_ids: List[int] = []
        self._slot_map: Dict[int, int] = {}
        self._layer_dims: List[int] = []
        self._capacity: int = 0
        self._ring_ptr: int = 0
        self._current_trace_rank: int = _TRACE_RANK
        self._has_pending_w_updates: bool = False

        self._W_cap: List[np.ndarray] = []
        self._B_cap: List[np.ndarray] = []
        self._TB_cap: List[np.ndarray] = []
        self._A_hist_cap: List[np.ndarray] = []
        self._D_hist_cap: List[np.ndarray] = []
        self._Z_cap: List[np.ndarray] = []
        self._D_back_cap: List[np.ndarray] = []
        self._trace_weights_cap: np.ndarray = np.empty((0, _TRACE_RANK), dtype=np.float32)
        self._accum_coef_cap: np.ndarray = np.empty((0, _TRACE_RANK), dtype=np.float32)
        self._last_out_cap: np.ndarray = np.empty((0, OUTPUT_DIM), dtype=np.float32)
        self._out_valid_cap: np.ndarray = np.empty((0,), dtype=np.bool_)
        self._matmul_buf: List[np.ndarray] = []
        self._a_scaled_buf: List[np.ndarray] = []

        self.W_batch: List[np.ndarray] = []
        self.B_batch: List[np.ndarray] = []
        self.TB_batch: List[np.ndarray] = []

    def invalidate(self) -> None:
        if self._has_pending_w_updates and self._bound_ids:
            self.flush_weights()
        self._bound_ids = []
        self._slot_map.clear()
        self._ring_ptr = 0
        self._has_pending_w_updates = False
        self._last_bound_brains = None

    def close(self) -> None:
        """Release all allocated weight tensors and break references to bound brains."""
        if getattr(self, "_last_bound_brains", None) is not None:
            for b in self._last_bound_brains:
                if getattr(b, "_pool_ref", None) is self:
                    b._pool_ref = None
            if isinstance(self._last_bound_brains, list):
                self._last_bound_brains.clear()
            self._last_bound_brains = None
        self.invalidate()
        self._slot_map.clear()
        self._layer_dims = []
        self._bound_ids = ()
        self._W_cap.clear()
        self._B_cap.clear()
        self._TB_cap.clear()
        self._A_hist_cap.clear()
        self._D_hist_cap.clear()
        self._Z_cap.clear()
        self._D_back_cap.clear()
        self._matmul_buf.clear()
        self._a_scaled_buf.clear()
        self.W_batch.clear()
        self.B_batch.clear()
        self.TB_batch.clear()
        self._trace_weights_cap = np.empty((0, _TRACE_RANK), dtype=np.float32)
        self._accum_coef_cap = np.empty((0, _TRACE_RANK), dtype=np.float32)
        self._last_out_cap = np.empty((0, OUTPUT_DIM), dtype=np.float32)
        self._out_valid_cap = np.empty((0,), dtype=np.bool_)
        self._capacity = 0
        self._last_bound_brains = None

    def flush_single_brain(self, brain: BlobBrain) -> None:
        """Flush pending low-rank weight updates for a single brain (e.g. before cloning or exporting)."""
        if not self._has_pending_w_updates or self._ring_ptr == 0:
            return
        slot = self._slot_map.get(id(brain))
        if slot is None:
            return
        active_k = self._ring_ptr
        coef = self._accum_coef_cap[slot, :active_k]
        if not np.any(coef != _F32_ZERO):
            return
        for idx in range(len(self._W_cap)):
            a_s = self._A_hist_cap[idx][slot, :, :active_k] * coef[None, :]
            d_s = self._D_hist_cap[idx][slot, :active_k, :]
            w = self._W_cap[idx][slot]
            w += a_s @ d_s
            _umath_clip(w, _F32_W_MIN, _F32_W_MAX, out=w)
        self._accum_coef_cap[slot, :active_k] = _F32_ZERO

    def apply_single_brain_learning(self, brain: BlobBrain, reward: float, learning_rate: float) -> None:
        """Route single-brain online RL updates through the pool's low-rank ring buffer."""
        slot = self._slot_map.get(id(brain))
        if slot is None or learning_rate <= 0.0:
            return
        r = float(reward)
        if abs(r) < 1e-6:
            brain.reward_baseline_dense = getattr(brain, "reward_baseline_dense", 0.0) * 0.96
            brain.reward_baseline_event = getattr(brain, "reward_baseline_event", 0.0) * 0.96
            brain.reward_baseline = brain.reward_baseline_dense + brain.reward_baseline_event
            return
        is_dense = abs(r) <= 0.35
        if is_dense:
            prev_b = getattr(brain, "reward_baseline_dense", 0.0)
            brain.reward_baseline_dense = 0.98 * prev_b + 0.02 * r
            brain.reward_baseline_event = getattr(brain, "reward_baseline_event", 0.0) * 0.98
            cur_baseline = prev_b
        else:
            prev_b = getattr(brain, "reward_baseline_event", 0.0)
            brain.reward_baseline_event = 0.90 * prev_b + 0.10 * r
            brain.reward_baseline_dense = getattr(brain, "reward_baseline_dense", 0.0) * 0.98
            cur_baseline = prev_b
        brain.reward_baseline = brain.reward_baseline_dense + brain.reward_baseline_event
        if r > 0.0:
            advantage = min(2.5, max(0.0, 0.25 * r, r - cur_baseline))
        else:
            advantage = max(-2.5, min(0.0, r, r - cur_baseline))
        step = np.float32(learning_rate * advantage * 0.18)
        half_step = np.float32(step * 0.5)
        active_k = self._ring_ptr
        if active_k > 0:
            self._accum_coef_cap[slot, :active_k] += step * self._trace_weights_cap[slot, :active_k]
            self._has_pending_w_updates = True
        for idx in range(len(self._B_cap)):
            b = self._B_cap[idx][slot]
            b += half_step * self._TB_cap[idx][slot]
            _umath_clip(b, _F32_B_MIN, _F32_B_MAX, out=b)

    def _allocate_capacity(self, dims: List[int], new_cap: int, preserve_n: int = 0) -> None:
        num_layers = len(dims) - 1
        is_large = max(dims) > 512
        trace_rank = _get_trace_rank_for_dims(dims)
        self._current_trace_rank = trace_rank
        cohort_cap = min(new_cap, max(64, (new_cap + 2) // 3 + 32)) if is_large else new_cap

        new_W_cap: List[np.ndarray] = []
        new_B_cap: List[np.ndarray] = []
        new_TB_cap: List[np.ndarray] = []
        new_A_cap: List[np.ndarray] = []
        new_D_cap: List[np.ndarray] = []
        new_Z_cap: List[np.ndarray] = []
        new_D_back_cap: List[np.ndarray] = []

        topology_same = (
            (dims == self._layer_dims)
            and len(self._W_cap) == num_layers
            and (self._trace_weights_cap.shape[1] == trace_rank if len(self._trace_weights_cap.shape) > 1 else False)
        )
        if not topology_same:
            self._matmul_buf = [
                np.empty((_FLUSH_CHUNK, dims[lay], dims[lay + 1]), dtype=np.float32)
                for lay in range(num_layers)
            ]
            self._a_scaled_buf = [
                np.empty((_FLUSH_CHUNK, dims[lay], trace_rank), dtype=np.float32)
                for lay in range(num_layers)
            ]

        for lay in range(num_layers):
            fan_in, fan_out = dims[lay], dims[lay + 1]
            w_arr = np.empty((new_cap, fan_in, fan_out), dtype=np.float32)
            b_arr = np.empty((new_cap, fan_out), dtype=np.float32)
            tb_arr = np.zeros((new_cap, fan_out), dtype=np.float32)
            a_arr = np.zeros((new_cap, fan_in, trace_rank), dtype=np.float32)
            d_arr = np.zeros((new_cap, trace_rank, fan_out), dtype=np.float32)
            z_arr = np.empty((cohort_cap, 1, fan_out), dtype=np.float32)
            db_arr = np.empty((cohort_cap, fan_in, 1), dtype=np.float32)
            if topology_same and preserve_n > 0:
                w_arr[:preserve_n] = self._W_cap[lay][:preserve_n]
                b_arr[:preserve_n] = self._B_cap[lay][:preserve_n]
                tb_arr[:preserve_n] = self._TB_cap[lay][:preserve_n]
                a_arr[:preserve_n] = self._A_hist_cap[lay][:preserve_n]
                d_arr[:preserve_n] = self._D_hist_cap[lay][:preserve_n]
            new_W_cap.append(w_arr)
            new_B_cap.append(b_arr)
            new_TB_cap.append(tb_arr)
            new_A_cap.append(a_arr)
            new_D_cap.append(d_arr)
            new_Z_cap.append(z_arr)
            new_D_back_cap.append(db_arr)

        new_tw = np.zeros((new_cap, trace_rank), dtype=np.float32)
        new_ac = np.zeros((new_cap, trace_rank), dtype=np.float32)
        new_last_out = np.zeros((new_cap, dims[-1]), dtype=np.float32)
        new_out_valid = np.zeros((new_cap,), dtype=np.bool_)
        if topology_same and preserve_n > 0:
            new_tw[:preserve_n] = self._trace_weights_cap[:preserve_n]
            new_ac[:preserve_n] = self._accum_coef_cap[:preserve_n]
            new_last_out[:preserve_n] = self._last_out_cap[:preserve_n]
            new_out_valid[:preserve_n] = self._out_valid_cap[:preserve_n]

        self._W_cap = new_W_cap
        self._B_cap = new_B_cap
        self._TB_cap = new_TB_cap
        self._A_hist_cap = new_A_cap
        self._D_hist_cap = new_D_cap
        self._Z_cap = new_Z_cap
        self._D_back_cap = new_D_back_cap
        self._trace_weights_cap = new_tw
        self._accum_coef_cap = new_ac
        self._last_out_cap = new_last_out
        self._out_valid_cap = new_out_valid
        self._capacity = new_cap
        self._layer_dims = list(dims)

    def flush_weights(self) -> None:
        """Flush accumulated low-rank eligibility trace RL updates into W_batch via L2-cached BLAS GEMM."""
        n = len(self._bound_ids)
        active_k = self._ring_ptr
        if not self._has_pending_w_updates or n == 0 or active_k == 0:
            self._has_pending_w_updates = False
            return

        coef = self._accum_coef_cap[:n, :active_k]
        active_mask = np.any(coef != _F32_ZERO, axis=1)
        if not np.any(active_mask):
            self._has_pending_w_updates = False
            return

        active_indices = np.where(active_mask)[0]
        num_layers = len(self.W_batch)
        for idx in range(num_layers):
            w_lay = self.W_batch[idx]
            a_lay = self._A_hist_cap[idx]
            d_lay = self._D_hist_cap[idx]
            m_buf = self._matmul_buf[idx]
            a_buf = self._a_scaled_buf[idx]

            if len(active_indices) == n:
                for s in range(0, n, _FLUSH_CHUNK):
                    e = min(s + _FLUSH_CHUNK, n)
                    c_len = e - s
                    a_sub = a_buf[:c_len, :, :active_k]
                    np.multiply(a_lay[s:e, :, :active_k], coef[s:e, None, :], out=a_sub)
                    dw_sub = m_buf[:c_len]
                    np.matmul(a_sub, d_lay[s:e, :active_k, :], out=dw_sub)
                    w_sub = w_lay[s:e]
                    w_sub += dw_sub
                    _umath_clip(w_sub, _F32_W_MIN, _F32_W_MAX, out=w_sub)
            else:
                for s in range(0, len(active_indices), _FLUSH_CHUNK):
                    chunk_indices = active_indices[s : s + _FLUSH_CHUNK]
                    c_len = len(chunk_indices)
                    a_sub = a_buf[:c_len, :, :active_k]
                    np.multiply(a_lay[chunk_indices, :, :active_k], coef[chunk_indices, None, :], out=a_sub)
                    dw_sub = m_buf[:c_len]
                    np.matmul(a_sub, d_lay[chunk_indices, :active_k, :], out=dw_sub)
                    w_chunk = w_lay[chunk_indices] + dw_sub
                    _umath_clip(w_chunk, _F32_W_MIN, _F32_W_MAX, out=w_chunk)
                    w_lay[chunk_indices] = w_chunk

        self._accum_coef_cap[:n, :active_k] = _F32_ZERO
        self._has_pending_w_updates = False

    def _copy_or_init_slot(
        self, new_idx: int, old_slot: Optional[int], br: BlobBrain, num_layers: int
    ) -> None:
        if old_slot is not None:
            if old_slot != new_idx:
                for lay in range(num_layers):
                    self._W_cap[lay][new_idx] = self._W_cap[lay][old_slot]
                    self._B_cap[lay][new_idx] = self._B_cap[lay][old_slot]
                    self._TB_cap[lay][new_idx] = self._TB_cap[lay][old_slot]
                    self._A_hist_cap[lay][new_idx] = self._A_hist_cap[lay][old_slot]
                    self._D_hist_cap[lay][new_idx] = self._D_hist_cap[lay][old_slot]
                self._trace_weights_cap[new_idx] = self._trace_weights_cap[old_slot]
                self._accum_coef_cap[new_idx] = self._accum_coef_cap[old_slot]
                self._last_out_cap[new_idx] = self._last_out_cap[old_slot]
                self._out_valid_cap[new_idx] = False
        else:
            for lay in range(num_layers):
                self._W_cap[lay][new_idx] = br.weights[lay]
                self._B_cap[lay][new_idx] = br.biases[lay]
                self._TB_cap[lay][new_idx] = _F32_ZERO
                self._A_hist_cap[lay][new_idx] = _F32_ZERO
                self._D_hist_cap[lay][new_idx] = _F32_ZERO
            self._trace_weights_cap[new_idx] = _F32_ZERO
            self._accum_coef_cap[new_idx] = _F32_ZERO
            self._last_out_cap[new_idx] = _F32_ZERO
            self._out_valid_cap[new_idx] = False

        for lay in range(num_layers):
            br.weights[lay] = self._W_cap[lay][new_idx]
            br.biases[lay] = self._B_cap[lay][new_idx]

    def _bind_fallback_copy(self, brains: List[BlobBrain], num_layers: int) -> None:
        if self._has_pending_w_updates and self._bound_ids:
            self.flush_weights()
        needs_cycle = (
            any(
                any(br.weights[lay].base is self._W_cap[lay] for lay in range(num_layers))
                for br in brains
            )
            if len(self._W_cap) == num_layers
            else False
        )

        if not needs_cycle:
            for new_idx, br in enumerate(brains):
                br._pool_ref = self
                for lay in range(num_layers):
                    self._W_cap[lay][new_idx] = br.weights[lay]
                    self._B_cap[lay][new_idx] = br.biases[lay]
                    self._TB_cap[lay][new_idx] = _F32_ZERO
                    self._A_hist_cap[lay][new_idx] = _F32_ZERO
                    self._D_hist_cap[lay][new_idx] = _F32_ZERO
                    br.weights[lay] = self._W_cap[lay][new_idx]
                    br.biases[lay] = self._B_cap[lay][new_idx]
                self._trace_weights_cap[new_idx] = _F32_ZERO
                self._accum_coef_cap[new_idx] = _F32_ZERO
                self._last_out_cap[new_idx] = _F32_ZERO
                self._out_valid_cap[new_idx] = False
        else:
            old_slots = [self._slot_map.get(id(br)) for br in brains]
            saved_w: Dict[int, List[np.ndarray]] = {}
            saved_b: Dict[int, List[np.ndarray]] = {}
            for new_idx, (br, old_slot) in enumerate(zip(brains, old_slots, strict=False)):
                if old_slot is not None and old_slot != new_idx:
                    saved_w[new_idx] = [self._W_cap[lay][old_slot].copy() for lay in range(num_layers)]
                    saved_b[new_idx] = [self._B_cap[lay][old_slot].copy() for lay in range(num_layers)]
            for new_idx, br in enumerate(brains):
                br._pool_ref = self
                for lay in range(num_layers):
                    if new_idx in saved_w:
                        self._W_cap[lay][new_idx] = saved_w[new_idx][lay]
                        self._B_cap[lay][new_idx] = saved_b[new_idx][lay]
                    else:
                        w_src = br.weights[lay]
                        if w_src.base is not self._W_cap[lay]:
                            self._W_cap[lay][new_idx] = w_src
                            self._B_cap[lay][new_idx] = br.biases[lay]
                    self._TB_cap[lay][new_idx] = _F32_ZERO
                    self._A_hist_cap[lay][new_idx] = _F32_ZERO
                    self._D_hist_cap[lay][new_idx] = _F32_ZERO
                    br.weights[lay] = self._W_cap[lay][new_idx]
                    br.biases[lay] = self._B_cap[lay][new_idx]
                self._trace_weights_cap[new_idx] = _F32_ZERO
                self._accum_coef_cap[new_idx] = _F32_ZERO
                self._last_out_cap[new_idx] = _F32_ZERO
                self._out_valid_cap[new_idx] = False
        self._ring_ptr = 0

    def _ensure_pool_capacity(self, dims: List[int], n: int, old_n: int) -> bool:
        if dims != self._layer_dims:
            if self._has_pending_w_updates and self._bound_ids:
                self.flush_weights()
            is_large = max(dims) > 512
            new_cap = max(64, ((n + 127) // 128) * 128) if is_large else max(64, int(n * 1.5))
            self._allocate_capacity(dims, new_cap, preserve_n=0)
            self._slot_map.clear()
            self._ring_ptr = 0
            return True
        if n > self._capacity:
            is_large = max(dims) > 512
            new_cap = max(64, ((n + 127) // 128) * 128) if is_large else max(64, int(self._capacity * 1.5), int(n * 1.25))
            self._allocate_capacity(dims, new_cap, preserve_n=old_n)
            return True
        return False

    def bind(self, brains: List[BlobBrain]) -> None:
        n = len(brains)
        if n == 0:
            self.invalidate()
            return

        dims = brains[0].layer_dims
        old_n = len(self._bound_ids)
        if n == old_n and dims == self._layer_dims:
            if getattr(self, "_last_bound_brains", None) is brains:
                return
            if self._bound_ids and id(brains[0]) == self._bound_ids[0] and id(brains[-1]) == self._bound_ids[-1]:
                if all(id(brains[i]) == self._bound_ids[i] for i in range(n)):
                    self._last_bound_brains = brains
                    return

        self._last_bound_brains = brains
        curr_ids = [id(br) for br in brains]
        num_layers = len(dims) - 1
        cap_reallocated = self._ensure_pool_capacity(dims, n, old_n)

        slot_map = self._slot_map
        is_monotonic = bool(slot_map) and all(
            (slot_map.get(bid) is None or slot_map[bid] >= new_idx)
            for new_idx, bid in enumerate(curr_ids)
        )

        if is_monotonic:
            first_diff = 0
            if not cap_reallocated:
                limit = min(old_n, n)
                while first_diff < limit and curr_ids[first_diff] == self._bound_ids[first_diff]:
                    first_diff += 1

            for new_idx in range(first_diff, n):
                br = brains[new_idx]
                br._pool_ref = self
                old_slot = slot_map.get(curr_ids[new_idx])
                if old_slot == new_idx and not cap_reallocated:
                    continue
                self._copy_or_init_slot(new_idx, old_slot, br, num_layers)
        else:
            self._bind_fallback_copy(brains, num_layers)

        self._bound_ids = curr_ids
        self._slot_map = {bid: idx for idx, bid in enumerate(curr_ids)}
        self.W_batch = [w[:n] for w in self._W_cap]
        self.B_batch = [b[:n] for b in self._B_cap]
        self.TB_batch = [tb[:n] for tb in self._TB_cap]

    def _record_ring_buffer_traces(
        self,
        n: int,
        num_layers: int,
        acts: List[np.ndarray],
        out_act: np.ndarray,
        act_kind: str,
        cohort_slice: Optional[Any] = None,
        start_new_slot: bool = True,
    ) -> None:
        trace_rank = self._current_trace_rank
        if start_new_slot:
            if self._ring_ptr >= trace_rank:
                self.flush_weights()
                last_k = trace_rank - 1
                for idx in range(num_layers):
                    self._A_hist_cap[idx][:n, :, 0] = self._A_hist_cap[idx][:n, :, last_k]
                    self._D_hist_cap[idx][:n, 0, :] = self._D_hist_cap[idx][:n, last_k, :]
                self._trace_weights_cap[:n, 0] = self._trace_weights_cap[:n, last_k]
                self._trace_weights_cap[:n, 1:] = _F32_ZERO
                self._accum_coef_cap[:n, :] = _F32_ZERO
                self._ring_ptr = 1
            k = self._ring_ptr
            self._trace_weights_cap[:n, k] = _F32_ZERO
            self._accum_coef_cap[:n, k] = _F32_ZERO
            self._ring_ptr = k + 1
        else:
            k = max(0, self._ring_ptr - 1)

        decay = np.float32(0.95)
        one_minus_decay = np.float32(0.05)
        sl = cohort_slice if cohort_slice is not None else slice(0, n)

        if k > 0:
            self._trace_weights_cap[sl, :k] *= decay
        self._trace_weights_cap[sl, k] = one_minus_decay
        self._accum_coef_cap[sl, k] = _F32_ZERO

        delta = out_act * np.maximum(np.float32(0.15), _F32_ONE - out_act * out_act)
        act_code = _ACT_CODE_MAP.get(act_kind, 0)
        use_numba = len(out_act) >= 96
        target_len = len(out_act)
        d_back_cap_len = len(self._D_back_cap[0]) if self._D_back_cap else 0

        for idx in range(num_layers - 1, -1, -1):
            self._A_hist_cap[idx][sl, :, k] = acts[idx]
            self._D_hist_cap[idx][sl, k, :] = delta
            if isinstance(sl, slice):
                tb_sub = self.TB_batch[idx][sl]
                tb_sub *= decay
                tb_sub += one_minus_decay * delta
            else:
                self.TB_batch[idx][sl] = self.TB_batch[idx][sl] * decay + one_minus_decay * delta
            if idx > 0:
                if use_numba:
                    if target_len <= d_back_cap_len:
                        d_out = self._D_back_cap[idx][:target_len, :, 0]
                    else:
                        d_out = np.empty((target_len, self._layer_dims[idx]), dtype=np.float32)
                    parallel_mlp_backprop_delta(
                        self.W_batch[idx][sl], delta, acts[idx], d_out, act_code
                    )
                    delta = d_out
                else:
                    deriv = activation_derivative(acts[idx], act_kind)
                    if target_len <= d_back_cap_len:
                        d_3d = self._D_back_cap[idx][:target_len]
                    else:
                        d_3d = np.empty((target_len, self._layer_dims[idx], 1), dtype=np.float32)
                    np.matmul(self.W_batch[idx][sl], delta[:, :, None], out=d_3d)
                    delta = d_3d[:, :, 0]
                    delta *= deriv

    def forward_and_trace(
        self,
        brains: List[BlobBrain],
        obs_batch: np.ndarray,
        explore_noise: float,
        rng: np.random.Generator,
        inspect_idx: int = -1,
        update_traces: bool = True,
        cohort_idx: int = -1,
    ) -> np.ndarray:
        self.bind(brains)
        n = len(brains)
        if n == 0:
            return np.empty((0, OUTPUT_DIM), dtype=np.float32)

        num_layers = len(self.W_batch)
        act_kind = brains[0].config.activation
        act_code = _ACT_CODE_MAP.get(act_kind, 0)
        cohort_cap = len(self._Z_cap[0]) if self._Z_cap else n

        # Check invalid slots (from swap-remove death compaction)
        invalid_mask = ~self._out_valid_cap[:n]
        num_invalid = int(np.sum(invalid_mask))

        # Activate 3-Cohort Strided Scheduling when population >= 180 and invalid slots <= 32
        use_cohort = (0 <= cohort_idx < 3) and (n >= 180) and (num_invalid <= 32)

        if use_cohort:
            cohort_indices = list(range(cohort_idx, n, 3))
            if num_invalid > 0:
                invalid_indices = [int(i) for i in np.flatnonzero(invalid_mask)]
                eval_target: Any = np.asarray(
                    sorted(list(set(cohort_indices).union(invalid_indices))), dtype=np.intp
                )
                sub_n = len(eval_target)
            else:
                eval_target = slice(cohort_idx, n, 3)
                sub_n = len(cohort_indices)
        else:
            eval_target = slice(0, n)
            sub_n = n

        dummy_noise = np.empty((0, 0), dtype=np.float32)

        if sub_n <= cohort_cap:
            a = obs_batch[eval_target]
            acts: List[np.ndarray] = [a]
            use_numba = sub_n >= 96

            for idx in range(num_layers):
                is_out = idx == num_layers - 1
                w_sub = self.W_batch[idx][eval_target]
                b_sub = self.B_batch[idx][eval_target]
                z = self._Z_cap[idx][:sub_n, 0, :]

                if use_numba:
                    has_n = is_out and explore_noise > 0.0
                    n_arr = (
                        rng.standard_normal(z.shape, dtype=np.float32) * np.float32(explore_noise)
                        if has_n
                        else dummy_noise
                    )
                    parallel_mlp_layer_forward(a, w_sub, b_sub, z, act_code, is_out, n_arr, has_n)
                    a = z
                else:
                    z_3d = self._Z_cap[idx][:sub_n]
                    np.matmul(a[:, None, :], w_sub, out=z_3d)
                    z += b_sub
                    if not is_out:
                        a = apply_activation(z, act_kind, out=z)
                    else:
                        if explore_noise > 0.0:
                            z += rng.standard_normal(z.shape, dtype=np.float32) * np.float32(explore_noise)
                        a = np.tanh(z, out=z)
                acts.append(a)

            out_act = acts[-1]
            self._last_out_cap[eval_target] = out_act
            self._out_valid_cap[eval_target] = True

            if update_traces:
                self._record_ring_buffer_traces(
                    n,
                    num_layers,
                    acts,
                    out_act,
                    act_kind,
                    cohort_slice=eval_target if use_cohort else None,
                    start_new_slot=(not use_cohort) or (cohort_idx == 0) or (self._ring_ptr == 0),
                )

            if 0 <= inspect_idx < n:
                if not use_cohort:
                    brains[inspect_idx].last_activations = [layer_act[inspect_idx].copy() for layer_act in acts]
                elif isinstance(eval_target, slice):
                    if (inspect_idx % 3) == cohort_idx:
                        local_i = inspect_idx // 3
                        brains[inspect_idx].last_activations = [layer_act[local_i].copy() for layer_act in acts]
                    else:
                        ins_a = obs_batch[inspect_idx]
                        ins_acts = [ins_a.copy()]
                        for idx in range(num_layers):
                            iz = ins_a @ self.W_batch[idx][inspect_idx] + self.B_batch[idx][inspect_idx]
                            ins_a = apply_activation(iz, act_kind) if idx < num_layers - 1 else np.tanh(iz)
                            ins_acts.append(ins_a)
                        brains[inspect_idx].last_activations = ins_acts
                else:
                    locs = np.where(eval_target == inspect_idx)[0]
                    if len(locs) > 0:
                        local_i = int(locs[0])
                        brains[inspect_idx].last_activations = [layer_act[local_i].copy() for layer_act in acts]
                    else:
                        ins_a = obs_batch[inspect_idx]
                        ins_acts = [ins_a.copy()]
                        for idx in range(num_layers):
                            iz = ins_a @ self.W_batch[idx][inspect_idx] + self.B_batch[idx][inspect_idx]
                            ins_a = apply_activation(iz, act_kind) if idx < num_layers - 1 else np.tanh(iz)
                            ins_acts.append(ins_a)
                        brains[inspect_idx].last_activations = ins_acts
        else:
            # Chunked execution in chunks of cohort_cap to stay strictly within working buffer allocation
            last_acts_full: Optional[List[np.ndarray]] = (
                [np.empty((n, d), dtype=np.float32) for d in self._layer_dims] if update_traces else None
            )
            for c_start in range(0, sub_n, cohort_cap):
                c_end = min(c_start + cohort_cap, sub_n)
                cur_len = c_end - c_start
                c_slice = slice(c_start, c_end)
                a = obs_batch[c_slice]
                c_acts = [a]
                if last_acts_full is not None:
                    last_acts_full[0][c_slice] = a

                use_numba = cur_len >= 96
                for idx in range(num_layers):
                    is_out = idx == num_layers - 1
                    w_sub = self.W_batch[idx][c_slice]
                    b_sub = self.B_batch[idx][c_slice]
                    z = self._Z_cap[idx][:cur_len, 0, :]

                    if use_numba:
                        has_n = is_out and explore_noise > 0.0
                        n_arr = (
                            rng.standard_normal(z.shape, dtype=np.float32) * np.float32(explore_noise)
                            if has_n
                            else dummy_noise
                        )
                        parallel_mlp_layer_forward(a, w_sub, b_sub, z, act_code, is_out, n_arr, has_n)
                        a = z
                    else:
                        z_3d = self._Z_cap[idx][:cur_len]
                        np.matmul(a[:, None, :], w_sub, out=z_3d)
                        z += b_sub
                        if not is_out:
                            a = apply_activation(z, act_kind, out=z)
                        else:
                            if explore_noise > 0.0:
                                z += rng.standard_normal(z.shape, dtype=np.float32) * np.float32(explore_noise)
                            a = np.tanh(z, out=z)
                    c_acts.append(a)
                    if last_acts_full is not None:
                        last_acts_full[idx + 1][c_slice] = a

                c_out = c_acts[-1]
                self._last_out_cap[c_slice] = c_out
                self._out_valid_cap[c_slice] = True

            if update_traces and last_acts_full is not None:
                self._record_ring_buffer_traces(
                    n,
                    num_layers,
                    last_acts_full,
                    self._last_out_cap[:n],
                    act_kind,
                    cohort_slice=None,
                    start_new_slot=True,
                )

            if 0 <= inspect_idx < n:
                ins_a = obs_batch[inspect_idx]
                ins_acts = [ins_a.copy()]
                for idx in range(num_layers):
                    iz = ins_a @ self.W_batch[idx][inspect_idx] + self.B_batch[idx][inspect_idx]
                    ins_a = apply_activation(iz, act_kind) if idx < num_layers - 1 else np.tanh(iz)
                    ins_acts.append(ins_a)
                brains[inspect_idx].last_activations = ins_acts

        return self._last_out_cap[:n]

    def apply_online_learning_batch(
        self,
        brains: List[BlobBrain],
        rewards: np.ndarray,
        learning_rate: float,
        flush: bool = False,
    ) -> None:
        if learning_rate <= 0.0 or not brains:
            return
        self.bind(brains)
        n = len(brains)
        zero_mask = np.abs(rewards) < 1e-6
        dense_mask = np.abs(rewards) <= 0.35
        prev_dense = np.fromiter((getattr(br, "reward_baseline_dense", 0.0) for br in brains), dtype=np.float32, count=n)
        prev_event = np.fromiter((getattr(br, "reward_baseline_event", 0.0) for br in brains), dtype=np.float32, count=n)

        new_dense = np.where(
            zero_mask,
            np.float32(0.96) * prev_dense,
            np.where(dense_mask, np.float32(0.98) * prev_dense + np.float32(0.02) * rewards, np.float32(0.98) * prev_dense),
        )
        new_event = np.where(
            zero_mask,
            np.float32(0.96) * prev_event,
            np.where(~dense_mask, np.float32(0.90) * prev_event + np.float32(0.10) * rewards, np.float32(0.98) * prev_event),
        )

        for i, br in enumerate(brains):
            br.reward_baseline_dense = float(new_dense[i])
            br.reward_baseline_event = float(new_event[i])
            br.reward_baseline = float(new_dense[i] + new_event[i])

        baseline_for_adv = np.where(dense_mask, prev_dense, prev_event)

        pos_mask = rewards > 1e-6
        neg_mask = rewards < -1e-6
        if not np.any(pos_mask | neg_mask):
            if flush and self._has_pending_w_updates:
                self.flush_weights()
            return

        adv = np.zeros_like(rewards, dtype=np.float32)
        if np.any(pos_mask):
            raw_pos = np.maximum(np.float32(0.25) * rewards, rewards - baseline_for_adv)
            adv[pos_mask] = _umath_clip(raw_pos[pos_mask], _F32_ZERO, np.float32(2.5))
        if np.any(neg_mask):
            raw_neg = np.minimum(rewards, rewards - baseline_for_adv)
            adv[neg_mask] = _umath_clip(raw_neg[neg_mask], np.float32(-2.5), _F32_ZERO)

        steps = np.float32(learning_rate * 0.18) * adv
        half_steps = np.float32(0.5) * steps
        steps_2d = half_steps[:, None]

        active_k = self._ring_ptr
        if active_k > 0:
            self._accum_coef_cap[:n, :active_k] += steps[:, None] * self._trace_weights_cap[:n, :active_k]
            self._has_pending_w_updates = True

        for idx in range(len(self.B_batch)):
            b = self.B_batch[idx]
            b += steps_2d * self.TB_batch[idx]
            _umath_clip(b, _F32_B_MIN, _F32_B_MAX, out=b)

        if flush:
            self.flush_weights()
