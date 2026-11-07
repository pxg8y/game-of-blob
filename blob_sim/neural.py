"""Configurable Neural Network Brain with Online RL & Neuroevolution."""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from .config import NetworkConfig

INPUT_LABELS: List[str] = [
    "Nahrung ΔX",
    "Nahrung ΔY",
    "Nahrung Distanz",
    "Zuhause ΔX",
    "Zuhause ΔY",
    "Zuhause Distanz",
    "Nachbar ΔX",
    "Nachbar ΔY",
    "Nachbar Distanz",
    "Nachbar hat Essen",
    "Nachbar hat Geld",
    "Gleiche Gruppe",
    "Tages-Essen (0..2x)",
    "Energie / Sättigung",
    "Geld-Vermögen",
    "Tageszeit (Tag/Nacht)",
]

OUTPUT_LABELS: List[str] = [
    "Bewegung X",
    "Bewegung Y",
    "Sprint / Tempo",
    "Handels-Wunsch",
    "Gruppen-Sozialität",
]

INPUT_DIM = len(INPUT_LABELS)
OUTPUT_DIM = len(OUTPUT_LABELS)


def apply_activation(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "relu":
        return np.maximum(0.0, x)
    if kind == "gelu":
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3))))
    if kind == "silu":
        sig = 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))
        return x * sig
    # default: tanh
    return np.tanh(x)


class BlobBrain:
    """
    Feedforward MLP with user-configurable depth and width, eligibility traces for
    lifetime reward-modulated learning (Lamarckian RL), and genetic mutation.
    """

    def __init__(
        self,
        config: NetworkConfig,
        weights: Optional[List[np.ndarray]] = None,
        biases: Optional[List[np.ndarray]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.config = NetworkConfig(
            depth=config.depth,
            width=config.width,
            custom_layers=list(config.custom_layers) if config.custom_layers else None,
            activation=config.activation,
        )
        self.rng = rng or np.random.default_rng()
        hidden_dims = self.config.get_hidden_dims()
        self.layer_dims: List[int] = [INPUT_DIM] + hidden_dims + [OUTPUT_DIM]

        if weights is not None and biases is not None and len(weights) == len(self.layer_dims) - 1:
            self.weights = [w.astype(np.float32).copy() for w in weights]
            self.biases = [b.astype(np.float32).copy() for b in biases]
        else:
            self.weights = []
            self.biases = []
            for i in range(len(self.layer_dims) - 1):
                fan_in = self.layer_dims[i]
                fan_out = self.layer_dims[i + 1]
                scale = np.sqrt(2.0 / (fan_in + fan_out))
                w = self.rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float32)
                b = np.zeros((fan_out,), dtype=np.float32)
                self.weights.append(w)
                self.biases.append(b)

            # Give new brains a slight innate instinct toward food and home at dusk
            # so initial random populations show lively, understandable behavior right away
            if self.layer_dims[1] >= 4:
                # Input 0,1 (food dx, dy) -> hidden 0,1 -> output 0,1 (move_x, move_y)
                self.weights[0][0, 0] += 0.55
                self.weights[0][1, 1] += 0.55
                # Input 3,4 (home dx, dy) -> hidden 2,3
                self.weights[0][3, 2] += 0.45
                self.weights[0][4, 3] += 0.45
                self.weights[-1][0, 0] += 0.55
                self.weights[-1][1, 1] += 0.55
                self.weights[-1][2, 0] += 0.35
                self.weights[-1][3, 1] += 0.35

        # Eligibility traces for online reward-modulated learning
        self.traces_w: List[np.ndarray] = [np.zeros_like(w) for w in self.weights]
        self.traces_b: List[np.ndarray] = [np.zeros_like(b) for b in self.biases]
        self.last_activations: List[np.ndarray] = [
            np.zeros((dim,), dtype=np.float32) for dim in self.layer_dims
        ]
        self.reward_baseline: float = 0.0

    def forward(self, obs: np.ndarray, explore_noise: float = 0.05) -> np.ndarray:
        """Compute action outputs from sensory observation vector."""
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
                    z = z + self.rng.normal(0.0, explore_noise, size=z.shape).astype(np.float32)
                a = np.tanh(z)
            self.last_activations.append(a)

        # Update eligibility traces (exponentially decaying Hebbian correlation)
        decay = 0.82
        for idx in range(num_layers):
            pre = self.last_activations[idx][:, None]
            post = self.last_activations[idx + 1][None, :]
            self.traces_w[idx] = decay * self.traces_w[idx] + (1.0 - decay) * (pre @ post)
            self.traces_b[idx] = decay * self.traces_b[idx] + (1.0 - decay) * self.last_activations[idx + 1]

        return self.last_activations[-1]

    def apply_online_learning(self, reward: float, learning_rate: float) -> None:
        """
        Update weights online during the blob's lifetime based on reward signal
        (modulated Hebbian / REINFORCE eligibility trace update).
        """
        if learning_rate <= 0.0 or abs(reward) < 1e-6:
            return
        advantage = float(np.clip(reward - self.reward_baseline, -2.5, 2.5))
        self.reward_baseline = 0.95 * self.reward_baseline + 0.05 * reward
        step = float(learning_rate) * advantage * 0.15

        for idx in range(len(self.weights)):
            self.weights[idx] = np.clip(
                self.weights[idx] + step * self.traces_w[idx], -3.5, 3.5
            )
            self.biases[idx] = np.clip(
                self.biases[idx] + step * 0.5 * self.traces_b[idx], -2.5, 2.5
            )

    def clone_and_mutate(
        self,
        mutation_rate: float,
        mutation_strength: float,
        partner: Optional["BlobBrain"] = None,
    ) -> "BlobBrain":
        """Create an offspring brain with optional crossover and Gaussian mutation."""
        new_weights: List[np.ndarray] = []
        new_biases: List[np.ndarray] = []

        can_crossover = (
            partner is not None
            and partner.layer_dims == self.layer_dims
        )

        for idx in range(len(self.weights)):
            w = self.weights[idx].copy()
            b = self.biases[idx].copy()

            if can_crossover and partner is not None:
                mask_w = self.rng.random(w.shape) < 0.35
                w = np.where(mask_w, partner.weights[idx], w)
                mask_b = self.rng.random(b.shape) < 0.35
                b = np.where(mask_b, partner.biases[idx], b)

            if mutation_rate > 0.0 and mutation_strength > 0.0:
                mut_mask_w = self.rng.random(w.shape) < mutation_rate
                noise_w = self.rng.normal(0.0, mutation_strength, size=w.shape).astype(np.float32)
                w = np.clip(w + mut_mask_w * noise_w, -3.5, 3.5)

                mut_mask_b = self.rng.random(b.shape) < mutation_rate
                noise_b = self.rng.normal(0.0, mutation_strength, size=b.shape).astype(np.float32)
                b = np.clip(b + mut_mask_b * noise_b, -2.5, 2.5)

            new_weights.append(w)
            new_biases.append(b)

        return BlobBrain(
            config=self.config,
            weights=new_weights,
            biases=new_biases,
            rng=self.rng,
        )

    def adapt_to_config(self, new_config: NetworkConfig) -> "BlobBrain":
        """
        Adapt this brain to a new user-defined depth/width configuration,
        preserving overlapping sub-matrices of weights so learned behavior is not lost.
        """
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
        return sum(int(w.size + b.size) for w, b in zip(self.weights, self.biases))

    def to_inspector_payload(self, max_nodes_per_layer: int = 24) -> Dict[str, Any]:
        """Serialize activations and downsampled weights for live GUI neural network rendering."""
        layers_act = []
        for a in self.last_activations:
            sliced = a[:max_nodes_per_layer]
            layers_act.append([round(float(v), 3) for v in sliced])

        weights_viz = []
        for w in self.weights:
            r = min(w.shape[0], max_nodes_per_layer)
            c = min(w.shape[1], max_nodes_per_layer)
            sub = w[:r, :c]
            weights_viz.append([[round(float(val), 2) for val in row] for row in sub])

        return {
            "layer_dims": self.layer_dims,
            "activation": self.config.activation,
            "param_count": self.parameter_count(),
            "activations": layers_act,
            "weights": weights_viz,
            "input_labels": INPUT_LABELS,
            "output_labels": OUTPUT_LABELS,
        }

    def to_dict(self) -> Dict[str, Any]:
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
        weights = [np.array(w, dtype=np.float32) for w in data.get("weights", [])]
        biases = [np.array(b, dtype=np.float32) for b in data.get("biases", [])]
        return cls(config=cfg, weights=weights, biases=biases, rng=rng)
