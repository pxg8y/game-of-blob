"""Configuration models for Neural Networks, Training Preferences, and World Rules."""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class NetworkConfig:
    """User-configurable neural network topology."""
    depth: int = 2  # Number of hidden layers (1..6)
    width: int = 24  # Neurons per hidden layer (4..128)
    custom_layers: Optional[List[int]] = None  # Optional explicit layer widths e.g. [32, 16]
    activation: str = "tanh"  # 'tanh', 'relu', 'gelu', 'silu'

    def get_hidden_dims(self) -> List[int]:
        if self.custom_layers and len(self.custom_layers) > 0:
            return [max(4, min(128, int(w))) for w in self.custom_layers[:8]]
        d = max(1, min(6, int(self.depth)))
        w = max(4, min(128, int(self.width)))
        return [w] * d

    def to_dict(self) -> Dict[str, Any]:
        return {
            "depth": self.depth,
            "width": self.width,
            "custom_layers": self.custom_layers,
            "activation": self.activation,
            "hidden_dims": self.get_hidden_dims(),
        }


@dataclass
class TrainingPreferences:
    """User-configurable training & evolutionary preference parameters."""
    pref_food: float = 1.6          # Reward weight for foraging & eating food
    pref_money: float = 1.0         # Reward weight for trading surplus food & accumulating money
    pref_social: float = 0.8        # Reward weight for joining/staying in groups & group trade
    pref_home: float = 1.1          # Reward weight for returning home before nightfall
    pref_exploration: float = 0.35  # Reward weight for active foraging movement when hungry
    learning_rate: float = 0.015    # Lifetime online RL learning rate (Lamarckian plasticity)
    mutation_rate: float = 0.18     # Probability of mutating each weight/gene on reproduction
    mutation_strength: float = 0.14 # Gaussian std dev of weight/gene mutations

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorldConfig:
    """2D world, survival, reproduction, and economy parameters."""
    width: float = 1200.0
    height: float = 800.0
    initial_blobs: int = 45
    daily_food_spawn: int = 135
    ticks_per_day: int = 360
    daylight_ratio: float = 0.72    # 72% day, 28% dusk/night
    survival_food_need: float = 1.0 # 1.0 food per day required to survive
    reproduction_food_Multiplier: float = 2.0  # 2x survival_food_need in a day -> reproduce!
    max_starvation_ticks: int = 480 # Die if no food eaten for this many ticks or energy hits 0
    base_food_price: float = 5.0    # Base money price per 1.0 unit of food
    group_trade_discount: float = 0.30  # 30% better trade deal within same group
    min_population_floor: int = 8   # Respawn from Hall of Fame if population drops below floor

    @property
    def reproduction_food_need(self) -> float:
        return self.survival_food_need * self.reproduction_food_Multiplier

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["reproduction_food_need"] = self.reproduction_food_need
        return d
