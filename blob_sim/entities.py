"""Entities for the Blob Life simulation: Genes, FoodItem, Home, Group, and Blob."""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import numpy as np
from .neural import BlobBrain

GROUP_PALETTE = [
    "#0ea5e9",  # Sky Blue
    "#10b981",  # Emerald
    "#f59e0b",  # Amber Gold
    "#f43f5e",  # Coral Rose
    "#14b8a6",  # Teal
    "#eab308",  # Sun Yellow
    "#3b82f6",  # Royal Blue
    "#84cc16",  # Lime Green
]

GROUP_NAMES = [
    "Aura-Clan",
    "Sol-Gilde",
    "Terra-Bund",
    "Nexus-Stamm",
    "Koral-Orden",
    "Helios-Pakt",
    "Vektor-Schar",
    "Zenit-Kollektiv",
]


@dataclass
class Genes:
    """Heritable physical and behavioral traits with energetic trade-offs."""
    speed: float = 1.0       # Base movement speed multiplier (0.55 .. 1.85)
    sense: float = 1.0       # Vision radius multiplier (0.55 .. 1.85)
    size: float = 1.0        # Body size & carrying capacity (0.65 .. 1.60)
    metabolism: float = 1.0  # Efficiency modifier
    hue: float = 195.0       # Individual visual hue when ungrouped

    def basal_energy_cost(self) -> float:
        """
        Higher speed, larger size, and wider vision require more energy per tick.
        Follows biological scaling laws: cost ~ size^1.3 * speed^1.5 + sense * 0.25
        """
        return float(0.065 * (self.size ** 1.25) * (self.speed ** 1.35) + 0.018 * self.sense)

    def mutate(
        self,
        rng: np.random.Generator,
        mutation_rate: float,
        mutation_strength: float,
        partner: Optional["Genes"] = None,
    ) -> "Genes":
        speed = self.speed if partner is None or rng.random() < 0.5 else partner.speed
        sense = self.sense if partner is None or rng.random() < 0.5 else partner.sense
        size = self.size if partner is None or rng.random() < 0.5 else partner.size
        hue = self.hue if partner is None or rng.random() < 0.5 else partner.hue

        if rng.random() < mutation_rate:
            speed = float(np.clip(speed + rng.normal(0.0, mutation_strength * 0.8), 0.55, 1.85))
        if rng.random() < mutation_rate:
            sense = float(np.clip(sense + rng.normal(0.0, mutation_strength * 0.8), 0.55, 1.85))
        if rng.random() < mutation_rate:
            size = float(np.clip(size + rng.normal(0.0, mutation_strength * 0.6), 0.65, 1.60))
        if rng.random() < mutation_rate:
            hue = float((hue + rng.normal(0.0, 14.0)) % 360.0)

        return Genes(
            speed=round(speed, 3),
            sense=round(sense, 3),
            size=round(size, 3),
            metabolism=round(float(0.5 * (speed + size)), 3),
            hue=round(hue, 1),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FoodItem:
    id: int
    x: float
    y: float
    nutrition: float = 0.5  # Typically 0.5 (two bites = 1.0 daily survival need, four bites = 2.0 reproduction!)
    golden: bool = False    # High-value nutrient cluster


@dataclass
class Home:
    """A physical shelter or group village on the 2D map."""
    id: int
    x: float
    y: float
    radius: float = 28.0
    group_id: Optional[int] = None
    food_depot: float = 1.0  # Stored food available for purchase with money
    gold_vault: float = 10.0
    resident_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "radius": round(self.radius, 1),
            "group_id": self.group_id,
            "food_depot": round(self.food_depot, 2),
            "gold_vault": round(self.gold_vault, 1),
            "residents": len(self.resident_ids),
        }


@dataclass
class Group:
    """A social tribe/guild that blobs can form and join."""
    id: int
    name: str
    color: str
    home_id: int
    member_ids: List[int] = field(default_factory=list)
    total_trades: int = 0
    created_day: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "color": self.color,
            "home_id": self.home_id,
            "member_count": len(self.member_ids),
            "total_trades": self.total_trades,
            "created_day": self.created_day,
        }


class Blob:
    """Autonomous neural-network-controlled organism in the 2D world."""

    def __init__(
        self,
        blob_id: int,
        x: float,
        y: float,
        home_id: int,
        home_x: float,
        home_y: float,
        brain: BlobBrain,
        genes: Genes,
        generation: int = 1,
        parent_id: Optional[int] = None,
        group_id: Optional[int] = None,
        money: float = 6.0,
    ) -> None:
        self.id = blob_id
        self.generation = generation
        self.parent_id = parent_id
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.heading = 0.0

        self.home_id = home_id
        self.home_x = float(home_x)
        self.home_y = float(home_y)
        self.group_id = group_id

        self.brain = brain
        self.genes = genes

        # Survival & Reproduction states
        self.energy: float = 85.0              # 0..100; dies if <= 0
        self.food_today: float = 0.0           # 1.0 to survive day, >= 2.0 to reproduce at end of day
        self.surplus_food: float = 0.0         # Extra food available to sell for money
        self.money: float = float(money)       # Persistent currency from selling food
        self.ticks_since_last_meal: int = 0    # Dies if exceeds max_starvation_ticks

        # Action & Social states
        self.is_at_home: bool = True
        self.trade_desire: float = 0.0
        self.social_desire: float = 0.0
        self.sprint_level: float = 0.0
        self.trade_cooldown: int = 0
        self.last_trade_tick: int = -100

        # Lifetime statistics
        self.age_days: int = 0
        self.age_ticks: int = 0
        self.children_count: int = 0
        self.trades_made: int = 0
        self.food_eaten_lifetime: float = 0.0
        self.cumulative_reward: float = 0.0
        self.last_reward: float = 0.0
        self.alive: bool = True
        self.cause_of_death: Optional[str] = None

    @property
    def vision_radius(self) -> float:
        return 145.0 * self.genes.sense

    @property
    def max_speed(self) -> float:
        return 3.2 * self.genes.speed

    @property
    def body_radius(self) -> float:
        return 7.5 * self.genes.size

    def to_summary_dict(self, group_color: Optional[str] = None) -> Dict[str, Any]:
        return {
            "id": self.id,
            "gen": self.generation,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "vx": round(self.vx, 2),
            "vy": round(self.vy, 2),
            "r": round(self.body_radius, 1),
            "energy": round(self.energy, 1),
            "food_today": round(self.food_today, 2),
            "surplus_food": round(self.surplus_food, 2),
            "money": round(self.money, 1),
            "home_id": self.home_id,
            "home_x": round(self.home_x, 1),
            "home_y": round(self.home_y, 1),
            "is_at_home": self.is_at_home,
            "group_id": self.group_id,
            "color": group_color or f"hsl({int(self.genes.hue)}, 75%, 56%)",
            "age_days": self.age_days,
            "children": self.children_count,
            "trades": self.trades_made,
            "reward": round(self.cumulative_reward, 2),
            "recent_trade": (self.age_ticks - self.last_trade_tick) < 25,
            "genes": self.genes.to_dict(),
        }
