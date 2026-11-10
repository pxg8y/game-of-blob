"""Configuration models for Neural Networks, Training Preferences, and World Rules."""

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any


@dataclass
class NetworkConfig:
    """User-configurable neural network topology."""
    depth: int = 2  # Number of hidden layers (1..6)
    width: int = 24  # Neurons per hidden layer (4..10000)
    custom_layers: Optional[List[int]] = None  # Optional explicit layer widths e.g. [32, 16]
    activation: str = "tanh"  # 'tanh', 'relu', 'gelu', 'silu'
    recurrent_dim: int = 0  # Number of recurrent working memory units (0..64)

    def get_hidden_dims(self) -> List[int]:
        if self.custom_layers and len(self.custom_layers) > 0:
            return [max(4, min(10000, int(w))) for w in self.custom_layers[:8]]
        d = max(1, min(6, int(self.depth)))
        w = max(4, min(10000, int(self.width)))
        return [w] * d

    def get_recurrent_dim(self) -> int:
        return max(0, min(64, int(self.recurrent_dim)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "depth": self.depth,
            "width": self.width,
            "custom_layers": self.custom_layers,
            "activation": self.activation,
            "recurrent_dim": self.get_recurrent_dim(),
            "hidden_dims": self.get_hidden_dims(),
        }


@dataclass
class TrainingPreferences:
    """User-configurable training & evolutionary preference parameters."""
    pref_food: float = 1.6          # Reward weight for foraging & eating food
    pref_money: float = 1.0         # Reward weight for trading food & accumulating money
    pref_social: float = 0.8        # Reward weight for joining/forming settlement groups & group trade
    pref_home: float = 1.1          # Reward weight for returning home before nightfall
    pref_exploration: float = 0.35  # Reward weight for active foraging movement when hungry
    learning_rate: float = 0.015    # Lifetime online RL learning rate (Lamarckian plasticity)
    mutation_rate: float = 0.18     # Probability of mutating each weight/gene on reproduction
    mutation_strength: float = 0.14 # Gaussian std dev of weight/gene mutations

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


START_PARADIGMS: Dict[str, Dict[str, Any]] = {
    "curriculum": {
        "id": "curriculum",
        "label": "🎓 Standard Curriculum (Wiege → Exodus)",
        "short_desc": "Minimaler Nahrungs-Keim, 0% Vektor-Tricks. Die Starter-Wiegen nähren Gen 1–2 und versteppen dann durch Überweidung → erzwingt Ausbruch durch die Skill-Gates in die Frontier-Biome.",
        "innate_seed": 0.72,
        "cradle_fertility": 0.86,
        "cradle_depletion": 2.4,
        "cradle_recovery": 0.25,
    },
    "tabula_rasa": {
        "id": "tabula_rasa",
        "label": "🧠 Tabula Rasa Hardcore (100% Zufalls-Gehirne)",
        "short_desc": "0% angeborenes Vorwissen! Ein dichter Kinderstuben-Futterring an Tag 1–2 lässt Zufalls-Blobs erste RL-Erfahrungen sammeln, bevor die Wiegen kollabieren.",
        "innate_seed": 0.0,
        "cradle_fertility": 0.96,
        "cradle_depletion": 2.8,
        "cradle_recovery": 0.20,
    },
    "rapid_exodus": {
        "id": "rapid_exodus",
        "label": "🔥 Schneller Exodus (Dürre-Start in den Wiegen)",
        "short_desc": "Die Starter-Wiegen beginnen bereits halb ausgedörrt (42% 🌱). Sofortiger Selektionsdruck auf Fühler-Navigation, Fern-Witterung und Pionier-Hausbau im Frontier-Raum!",
        "innate_seed": 0.75,
        "cradle_fertility": 0.42,
        "cradle_depletion": 3.2,
        "cradle_recovery": 0.14,
    },
}


WORLD_SCALE_PRESETS: Dict[str, Dict[str, Any]] = {
    "training_kindergarten": {
        "id": "training_kindergarten",
        "label": "🌱 Kindergarten · Perfekte Trainings-Arena (4.800×3.600)",
        "short_desc": "Hindernisfreie Starter-Wiege mit 4 resilienten Oasen, 3.2x Nahrungsüberschuss, 1.350 Ticks Hunger-Puffer und weiten Übergangspässen für optimales RL-Lernen ohne Frühaussterben.",
        "width": 4800.0,
        "height": 3600.0,
        "num_cradles": 1,
        "initial_blobs": 100,
        "daily_food_spawn": 320,
        "max_population_cap": 1500,
        "max_homes_cap": 120,
        "max_groups_cap": 24,
        "num_oases": 12,
        "num_rock_formations": 16,
        "num_swamps": 4,
        "min_population_floor": 24,
        "daylight_ratio": 0.74,
        "max_starvation_ticks": 1350,
    },
    "classic": {
        "id": "classic",
        "label": "🏝️ Klassisch · Trainings-Arena (4.800×3.600)",
        "short_desc": "1 zentrale Starter-Wiege, 12 Oasen, 100 Start-Blobs (bis 1.500 Population & 24 Fraktionen).",
        "width": 4800.0,
        "height": 3600.0,
        "num_cradles": 1,
        "initial_blobs": 100,
        "daily_food_spawn": 240,
        "max_population_cap": 1500,
        "max_homes_cap": 120,
        "max_groups_cap": 24,
        "num_oases": 12,
        "num_rock_formations": 28,
        "num_swamps": 10,
        "min_population_floor": 14,
    },
    "regional": {
        "id": "regional",
        "label": "🗺️ Regional · Große Welt (16.000×12.000)",
        "short_desc": "5 verteilte Starter-Wiegen, 45 Oasen, 1.200 Start-Blobs (bis 6.000 Population & 60 Fraktionen).",
        "width": 16000.0,
        "height": 12000.0,
        "num_cradles": 5,
        "initial_blobs": 1200,
        "daily_food_spawn": 1800,
        "max_population_cap": 6000,
        "max_homes_cap": 450,
        "max_groups_cap": 60,
        "num_oases": 45,
        "num_rock_formations": 90,
        "num_swamps": 28,
        "min_population_floor": 60,
    },
    "continent": {
        "id": "continent",
        "label": "🌍 Kontinent · Mega-Welt (40.000×30.000)",
        "short_desc": "12 regionale Starter-Wiegen, 108 Oasen, 3.600 Start-Blobs (bis 20.000 Blobs & 100+ Fraktionen mit Kolonien, Sezession & Eroberung).",
        "width": 40000.0,
        "height": 30000.0,
        "num_cradles": 12,
        "initial_blobs": 3600,
        "daily_food_spawn": 5400,
        "max_population_cap": 20000,
        "max_homes_cap": 1500,
        "max_groups_cap": 120,
        "num_oases": 108,
        "num_rock_formations": 210,
        "num_swamps": 64,
        "min_population_floor": 160,
    },
}


@dataclass
class WorldConfig:
    """2D world, survival, reproduction, settlement, and economy parameters."""
    width: float = 4800.0
    height: float = 3600.0
    initial_blobs: int = 100
    daily_food_spawn: int = 240
    ticks_per_day: int = 600
    daylight_ratio: float = 0.68    # 68% day, 32% dusk/night for ample return-home & market time
    survival_food_need: float = 1.0 # 1.0 food per day required to survive
    reproduction_food_Multiplier: float = 2.0  # 2x survival_food_need in a day -> reproduce!
    max_starvation_ticks: int = 900 # Die if no food eaten for this many ticks AND energy hits 0
    base_food_price: float = 5.0    # Base money price per 1.0 unit of food
    group_trade_discount: float = 0.30  # 30% better trade deal within same group/settlement
    max_residents_per_home: int = 6     # Max blobs per house before expanding settlement
    min_population_floor: int = 14  # Respawn from Hall of Fame if population drops below floor
    num_oases: int = 12             # 3 inner Cradle oases per cradle + outer Frontier biome oases
    num_rock_formations: int = 28   # Procedural canyon walls & skill-gate baffles
    num_swamps: int = 10            # Swamp shortcut fords & mud basins
    start_paradigm: str = "curriculum"  # 'curriculum', 'tabula_rasa', 'rapid_exodus'
    scale_preset: str = "classic"   # 'classic', 'regional', 'continent'
    num_cradles: int = 1            # Number of regional starter cradles on the map
    max_population_cap: int = 1500  # Carrying capacity ceiling up to 20,000 blobs
    max_homes_cap: int = 120        # Max settlement & pioneer houses
    max_groups_cap: int = 120       # Max concurrent factions/empires (50-100+)
    every_tick_brain: bool = False  # Full 100% population evaluation & online RL trace update every single tick
    egocentric_frame: bool = False  # Local body-frame sensor rotation & differential drive kinematics

    def apply_scale_preset(self, preset_id: str) -> None:
        preset = WORLD_SCALE_PRESETS.get(preset_id, WORLD_SCALE_PRESETS["continent"])
        self.scale_preset = preset["id"]
        self.width = float(preset["width"])
        self.height = float(preset["height"])
        self.num_cradles = int(preset["num_cradles"])
        self.initial_blobs = int(preset["initial_blobs"])
        self.daily_food_spawn = int(preset["daily_food_spawn"])
        self.max_population_cap = int(preset["max_population_cap"])
        self.max_homes_cap = int(preset["max_homes_cap"])
        self.max_groups_cap = int(preset["max_groups_cap"])
        self.num_oases = int(preset["num_oases"])
        self.num_rock_formations = int(preset["num_rock_formations"])
        self.num_swamps = int(preset["num_swamps"])
        self.min_population_floor = int(preset["min_population_floor"])
        self.every_tick_brain = bool(preset.get("every_tick_brain", False))
        self.egocentric_frame = bool(preset.get("egocentric_frame", False))
        if "daylight_ratio" in preset:
            self.daylight_ratio = float(preset["daylight_ratio"])
        if "max_starvation_ticks" in preset:
            self.max_starvation_ticks = int(preset["max_starvation_ticks"])

    @classmethod
    def from_preset(cls, preset_id: str = "continent", **overrides: Any) -> "WorldConfig":
        cfg = cls()
        cfg.apply_scale_preset(preset_id)
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    @property
    def reproduction_food_need(self) -> float:
        return self.survival_food_need * self.reproduction_food_Multiplier

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["reproduction_food_need"] = self.reproduction_food_need
        d["paradigms"] = START_PARADIGMS
        d["scale_presets"] = WORLD_SCALE_PRESETS
        return d
