"""Entities for the Blob Life simulation: Genes, FoodItem, Home, Group (Settlement), and Blob."""

import colorsys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
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
    "Aura-Siedlung",
    "Sol-Gilde",
    "Terra-Bund",
    "Nexus-Stamm",
    "Koral-Orden",
    "Helios-Pakt",
    "Vektor-Schar",
    "Zenit-Kollektiv",
]

FACTION_PREFIXES = [
    "Aura", "Sol", "Terra", "Nexus", "Koral", "Helios", "Vektor", "Zenit",
    "Aether", "Obsidian", "Boreal", "Zephyr", "Titan", "Nova", "Prisma", "Chronos",
    "Smaragd", "Rubin", "Saphir", "Onyx", "Bernstein", "Astral", "Valkyr", "Hyperion",
    "Orion", "Lyra", "Draco", "Phoenix", "Atlas", "Vortex", "Kosmika", "Sylphen",
]

FACTION_SUFFIXES = [
    "Siedlung", "Gilde", "Bund", "Stamm", "Orden", "Pakt", "Schar", "Kollektiv",
    "Imperium", "Dynastie", "Republik", "Allianz", "Hanse", "Dominion", "Föderation", "Klan",
]

OASIS_NAMES = [
    "Smaragd-Quell",
    "Lotus-Becken",
    "Sonnen-Hain",
    "Jade-Lagune",
    "Bernstein-Oase",
    "Zephyr-Tal",
    "Nebel-Aue",
    "Korallen-Teich",
    "Morgen-Quell",
    "Palmen-Senke",
    "Kristall-Grund",
    "Sternen-Oase",
    "Wurzel-Garten",
    "Silber-Furt",
]


def hex_to_rgb_int(hex_str: str) -> int:
    """Convert '#rrggbb' to packed 24-bit integer 0xRRGGBB."""
    h = hex_str.lstrip("#")
    if len(h) == 6:
        return int(h, 16)
    return 0x10B981


def hsl_to_hex_and_rgb(hue_deg: float, sat: float = 0.78, light: float = 0.56) -> Tuple[str, int]:
    """Convert HSL (hue 0..360, sat 0..1, light 0..1) to ('#rrggbb', 0xRRGGBB)."""
    r, g, b = colorsys.hls_to_rgb((hue_deg % 360.0) / 360.0, light, sat)
    ri = max(0, min(255, int(round(r * 255.0))))
    gi = max(0, min(255, int(round(g * 255.0))))
    bi = max(0, min(255, int(round(b * 255.0))))
    packed = (ri << 16) | (gi << 8) | bi
    return f"#{ri:02x}{gi:02x}{bi:02x}", packed


def generate_faction_identity(gid: int, is_secession: bool = False) -> Tuple[str, str, int]:
    """
    Generate a unique name, CSS hex color, and packed 24-bit RGB integer for 100+ factions.
    Uses GROUP_NAMES/GROUP_PALETTE for the first 8 factions and golden-angle procedural
    identities for all subsequent empires and daughter colonies.
    """
    idx = max(0, gid - 1)
    if idx < len(GROUP_PALETTE) and not is_secession:
        color = GROUP_PALETTE[idx]
        name = GROUP_NAMES[idx]
        return name, color, hex_to_rgb_int(color)

    # Golden-angle hue distribution guarantees high visual contrast even with 120+ factions
    hue = (195.0 + idx * 137.507764) % 360.0
    sat = 0.82 if (idx % 2 == 0) else 0.72
    light = 0.58 if (idx % 3 != 2) else 0.50
    color, rgb_int = hsl_to_hex_and_rgb(hue, sat=sat, light=light)

    p_idx = idx % len(FACTION_PREFIXES)
    s_idx = (idx // len(FACTION_PREFIXES) + idx) % len(FACTION_SUFFIXES)
    prefix = FACTION_PREFIXES[p_idx]
    suffix = FACTION_SUFFIXES[s_idx]
    roman_cycle = idx // (len(FACTION_PREFIXES) * len(FACTION_SUFFIXES))
    tag = f" {roman_cycle + 2}" if roman_cycle > 0 else ""
    neo_prefix = "Neu-" if is_secession and not prefix.startswith("Neu-") else ""
    name = f"{neo_prefix}{prefix}-{suffix}{tag}"
    return name, color, rgb_int


@dataclass
class OasisNode:
    """A localized fertile ecological hotspot where food grows; subject to overgrazing, seasonal specialization, and regeneration."""
    id: int
    name: str
    x: float
    y: float
    radius: float = 230.0
    fertility: float = 1.0          # 0.05 (barren/overgrazed) .. 1.00 (lush)
    max_capacity: int = 26          # Max food items supported at 100% fertility
    harvest_pressure: float = 0.0   # Recent harvesting stress driving overgrazing
    current_food_count: int = 0
    biome: str = "gold_grove"       # 'cradle' (fragile nursery), 'gold_grove' (summer boom), 'thermal_spring' (winter refuge)
    ring: str = "outer"             # 'inner' (starter cradle) or 'outer' (frontier)
    depletion_rate: float = 1.0     # Multiplier on harvest_pressure & fertility drop
    recovery_rate: float = 1.0      # Multiplier on natural fertility recovery
    cradle_id: int = 0              # Regional cradle index (0..num_cradles-1)

    @property
    def is_overgrazed(self) -> bool:
        return self.fertility < 0.38

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "radius": round(self.radius, 1),
            "fertility": round(self.fertility, 2),
            "max_capacity": self.max_capacity,
            "food_count": self.current_food_count,
            "overgrazed": self.is_overgrazed,
            "biome": self.biome,
            "ring": self.ring,
            "cradle_id": self.cradle_id,
        }


@dataclass
class TerrainObstacle:
    """
    A map feature on the 2D arena:
    - kind == 'rock': Impassable capsule wall / canyon ridge from (x1, y1) to (x2, y2) with thickness `radius`; blocks movement & vision.
    - kind == 'swamp': High-drag water/mud zone (x1, y1)-(x2, y2) with `radius`; slows movement & drains extra metabolic energy.
    """
    id: int
    kind: str  # 'rock' or 'swamp'
    x1: float
    y1: float
    x2: float
    y2: float
    radius: float = 48.0
    gate_kind: str = ""  # Optional skill-gate tag ('chicane', 'swamp_ford', 'steppe_gap', 'ring', 'rib')

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "x1": round(self.x1, 1),
            "y1": round(self.y1, 1),
            "x2": round(self.x2, 1),
            "y2": round(self.y2, 1),
            "radius": round(self.radius, 1),
            "gate_kind": self.gate_kind,
        }


@dataclass
class Genes:
    """Heritable physical and behavioral traits with energetic trade-offs."""
    speed: float = 1.0       # Base movement speed multiplier (0.55 .. 1.85)
    sense: float = 1.0       # Vision radius multiplier (0.55 .. 1.85)
    size: float = 1.0        # Body size, energy reserve & food carrying capacity (0.65 .. 1.60)
    metabolism: float = 1.0  # Efficiency modifier
    hue: float = 195.0       # Individual visual hue when ungrouped
    _basal_cost: float = field(init=False, repr=False, default=0.06)
    _rgb_int: int = field(init=False, repr=False, default=0x38BDF8)

    def __post_init__(self) -> None:
        eff = max(0.75, min(1.35, 0.85 + 0.15 * self.metabolism))
        self._basal_cost = float(
            (0.048 * (self.size ** 1.15) * (self.speed ** 1.35) + 0.012 * self.sense) * eff
        )
        _, self._rgb_int = hsl_to_hex_and_rgb(self.hue, sat=0.75, light=0.56)

    def basal_energy_cost(self) -> float:
        """
        Higher speed, larger size, and wider vision require more energy per tick.
        Follows biological scaling laws modulated by metabolic efficiency.
        """
        return self._basal_cost

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
        return {
            "speed": self.speed,
            "sense": self.sense,
            "size": self.size,
            "metabolism": self.metabolism,
            "hue": self.hue,
        }


@dataclass(slots=True)
class FoodItem:
    id: int
    x: float
    y: float
    nutrition: float = 0.6  # Standard apple = 0.6, Golden cluster = 1.2
    golden: bool = False    # High-value nutrient cluster
    oasis_id: Optional[int] = None


@dataclass
class Home:
    """A physical shelter on the 2D map with a maximum resident capacity; can unite with other Homes into a Settlement."""
    id: int
    x: float
    y: float
    radius: float = 48.0
    max_residents: int = 6
    group_id: Optional[int] = None
    food_depot: float = 1.5   # Stored food available for purchase with money
    gold_vault: float = 15.0  # Stored gold reserve in the house market
    farm_level: int = 0       # 0..3: 🌾 Boosts garden yields & winter greenhouse food
    granary_level: int = 0    # 0..3: 🏛️ Expands resident capacity & depot storage
    defense_level: int = 0    # 0..3: 🛡️ Watchtower protection against raiders
    recent_raids: int = 0     # Tracks recent raid pressure to guide upgrade priority
    raid_damage: int = 0      # Tracks consecutive successful raids for house conquest
    fertility_bonus: float = 1.0      # 0.0 (wasteland) .. 1.6 (oasis shore)
    terrain_defense_bonus: float = 0.0 # 0.0 .. 0.75 (canyon / cliff cover)
    zone: str = "cradle"              # 'cradle' (starter valley) or 'frontier' (outer biome)
    resident_ids: List[int] = field(default_factory=list)

    @property
    def is_full(self) -> bool:
        return len(self.resident_ids) >= self.max_residents

    @property
    def max_depot_capacity(self) -> float:
        return 12.0 + self.granary_level * 8.0

    @property
    def total_upgrades(self) -> int:
        return self.farm_level + self.granary_level + self.defense_level

    @property
    def local_price_multiplier(self) -> float:
        """Supply-and-demand price multiplier: scarce depot (<2🍏) raises price up to 1.55x; abundant (>7🍏) lowers to 0.55x."""
        return max(0.55, min(1.55, 1.42 - 0.105 * self.food_depot))

    def local_unit_price(
        self, base_food_price: float, unit_amount: float = 0.5, discount: float = 1.0
    ) -> float:
        return round(
            max(0.4, base_food_price * unit_amount * self.local_price_multiplier * discount), 2
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "radius": round(self.radius, 1),
            "max_residents": self.max_residents,
            "group_id": self.group_id,
            "food_depot": round(self.food_depot, 2),
            "gold_vault": round(self.gold_vault, 1),
            "farm_level": self.farm_level,
            "granary_level": self.granary_level,
            "defense_level": self.defense_level,
            "fertility_bonus": round(self.fertility_bonus, 2),
            "terrain_defense": round(self.terrain_defense_bonus, 2),
            "zone": self.zone,
            "price_mult": round(self.local_price_multiplier, 2),
            "residents": len(self.resident_ids),
        }


@dataclass
class Group:
    """A social settlement / empire (Fraktion) formed by one or more federated Homes and their resident blobs."""
    id: int
    name: str
    color: str
    home_id: int
    home_ids: List[int] = field(default_factory=list)
    member_ids: List[int] = field(default_factory=list)
    total_trades: int = 0
    created_day: int = 1
    parent_group_id: Optional[int] = None
    conquests: int = 0
    secessions: int = 0
    exiles: int = 0
    avg_reputation: float = 0.0
    rgb_int: int = 0x10B981

    # Feature 5: O(1) Group Reputation Cache Fields
    cached_rep_sum: float = 0.0
    cached_rep_count: int = 0
    rep_dirty: bool = True

    def __post_init__(self) -> None:
        if self.home_id not in self.home_ids:
            self.home_ids.append(self.home_id)
        if self.rgb_int == 0x10B981 and self.color:
            self.rgb_int = hex_to_rgb_int(self.color)

    def to_dict(self, cap_x: float = 0.0, cap_y: float = 0.0) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "color": self.color,
            "rgb_int": self.rgb_int,
            "home_id": self.home_id,
            "home_ids": list(self.home_ids),
            "homes_count": len(self.home_ids),
            "member_count": len(self.member_ids),
            "total_trades": self.total_trades,
            "created_day": self.created_day,
            "parent_group_id": self.parent_group_id,
            "conquests": self.conquests,
            "exiles": self.exiles,
            "avg_reputation": round(self.avg_reputation, 2),
            "cap_x": round(cap_x, 1),
            "cap_y": round(cap_y, 1),
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
        reputation: float = 0.0,
    ) -> None:
        self.id = blob_id
        self.generation = generation
        self.parent_id = parent_id
        self._kin_ref: Optional[Any] = None
        self._kin_idx: int = -1
        self._sim_idx: int = -1
        self._x: float = float(x)
        self._y: float = float(y)
        self._vx: float = 0.0
        self._vy: float = 0.0
        self._heading: float = 0.0

        self.home_id = home_id
        self.home_x = float(home_x)
        self.home_y = float(home_y)
        self.home_obj: Optional[Home] = None
        self._group_id: Optional[int] = group_id
        self._on_rep_change: Optional[Callable[["Blob"], None]] = None
        self._on_group_change: Optional[Callable[["Blob", Optional[int], Optional[int]], None]] = None

        self.brain = brain
        self.genes = genes

        # Pre-cached immutable genetic traits
        self.vision_radius: float = 215.0 * self.genes.sense
        self.max_speed: float = 4.2 * self.genes.speed
        self.body_radius: float = 7.5 * self.genes.size
        self.basal_cost: float = float(self.genes._basal_cost)

        # Survival & Reproduction states
        self._energy: float = min(self.max_energy, 85.0)  # 0..max_energy; dies if <= 0
        self._food_today: float = 0.0           # 1.0 to survive day, >= 2.0 to reproduce at end of day
        self._surplus_food: float = 0.0         # Extra carried food beyond reproduction threshold
        self._money: float = float(money)       # Persistent currency from selling food
        self._reputation: float = round(float(np.clip(reputation, -1.0, 1.0)), 3)  # Social reputation [-1.0..+1.0]
        self._ticks_since_last_meal: int = 0    # Dies if exceeds max_starvation_ticks

        # Action, Memory & Colonization states (cached across cohort ticks)
        self._is_at_home: bool = True
        self._in_swamp: bool = False
        self.act_move_x: float = 0.0
        self.act_move_y: float = 0.0
        self.trade_desire: float = 0.0
        self.social_desire: float = 0.0
        self.colonize_desire: float = 0.0
        self.sprint_level: float = 0.0
        self._trade_cooldown: int = 0
        self._colonize_cooldown: int = 0
        self.last_trade_tick: int = -100

        # Spatial Memory (remembers last rich oasis / food grove location & quality)
        self.memory_x: float = float(home_x)
        self.memory_y: float = float(home_y)
        self.memory_strength: float = 0.0
        self.memory_oasis_id: Optional[int] = None

        # Lifetime statistics
        self.age_days: int = 0
        self._age_ticks: int = 0
        self.children_count: int = 0
        self.trades_made: int = 0
        self.raids_won: int = 0
        self.bounties_won: int = 0
        self.exiles_count: int = 0
        self.migrations_made: int = 0
        self.food_eaten_lifetime: float = 0.0
        self.cumulative_reward: float = 0.0
        self.last_reward: float = 0.0
        self.pending_rl_reward: float = 0.0
        self.alive: bool = True
        self.cause_of_death: Optional[str] = None
        self._kin_ref: Optional[Any] = None
        self._kin_idx: int = -1
        self._sim_idx: int = -1

    @property
    def x(self) -> float:
        ref = self._kin_ref
        if ref is not None and self._kin_idx >= 0:
            return float(ref._kin_xs[self._kin_idx])
        return self._x

    @x.setter
    def x(self, val: float) -> None:
        fval = float(val)
        ref = self._kin_ref
        if ref is not None and self._kin_idx >= 0:
            ref._kin_xs[self._kin_idx] = fval
        self._x = fval

    @property
    def y(self) -> float:
        ref = self._kin_ref
        if ref is not None and self._kin_idx >= 0:
            return float(ref._kin_ys[self._kin_idx])
        return self._y

    @y.setter
    def y(self, val: float) -> None:
        fval = float(val)
        ref = self._kin_ref
        if ref is not None and self._kin_idx >= 0:
            ref._kin_ys[self._kin_idx] = fval
        self._y = fval

    @property
    def vx(self) -> float:
        ref = self._kin_ref
        if ref is not None and self._kin_idx >= 0:
            return float(ref._kin_vxs[self._kin_idx])
        return self._vx

    @vx.setter
    def vx(self, val: float) -> None:
        fval = float(val)
        ref = self._kin_ref
        if ref is not None and self._kin_idx >= 0:
            ref._kin_vxs[self._kin_idx] = fval
        self._vx = fval

    @property
    def vy(self) -> float:
        ref = self._kin_ref
        if ref is not None and self._kin_idx >= 0:
            return float(ref._kin_vys[self._kin_idx])
        return self._vy

    @vy.setter
    def vy(self, val: float) -> None:
        fval = float(val)
        ref = self._kin_ref
        if ref is not None and self._kin_idx >= 0:
            ref._kin_vys[self._kin_idx] = fval
        self._vy = fval

    @property
    def heading(self) -> float:
        ref = self._kin_ref
        if ref is not None and self._kin_idx >= 0:
            return float(ref._kin_headings[self._kin_idx])
        return self._heading

    @heading.setter
    def heading(self, val: float) -> None:
        fval = float(val)
        ref = self._kin_ref
        if ref is not None and self._kin_idx >= 0:
            ref._kin_headings[self._kin_idx] = fval
        self._heading = fval

    @property
    def energy(self) -> float:
        ref = self._kin_ref
        if ref is not None and self._kin_idx >= 0:
            return float(ref._kin_energies[self._kin_idx])
        return self._energy

    @energy.setter
    def energy(self, val: float) -> None:
        fval = float(val)
        ref = self._kin_ref
        if ref is not None and self._kin_idx >= 0:
            ref._kin_energies[self._kin_idx] = fval
        self._energy = fval

    @property
    def in_swamp(self) -> bool:
        ref = self._kin_ref
        if ref is not None and self._kin_idx >= 0:
            return bool(ref._kin_in_swamps[self._kin_idx])
        return self._in_swamp

    @in_swamp.setter
    def in_swamp(self, val: bool) -> None:
        bval = bool(val)
        ref = self._kin_ref
        if ref is not None and self._kin_idx >= 0:
            ref._kin_in_swamps[self._kin_idx] = bval
        self._in_swamp = bval

    @property
    def is_at_home(self) -> bool:
        ref = self._kin_ref
        if ref is not None and self._kin_idx >= 0:
            return bool(ref._kin_is_at_homes[self._kin_idx])
        return self._is_at_home

    @is_at_home.setter
    def is_at_home(self, val: bool) -> None:
        bval = bool(val)
        ref = self._kin_ref
        if ref is not None and self._kin_idx >= 0:
            ref._kin_is_at_homes[self._kin_idx] = bval
        self._is_at_home = bval

    @property
    def food_today(self) -> float:
        return self._food_today

    @food_today.setter
    def food_today(self, val: float) -> None:
        fval = float(val)
        ref = self._kin_ref
        if ref is not None and self._kin_idx >= 0:
            ref._kin_foods_today[self._kin_idx] = fval
        self._food_today = fval

    @property
    def surplus_food(self) -> float:
        return self._surplus_food

    @surplus_food.setter
    def surplus_food(self, val: float) -> None:
        fval = float(val)
        ref = self._kin_ref
        if ref is not None and self._kin_idx >= 0:
            ref._kin_surplus_foods[self._kin_idx] = fval
        self._surplus_food = fval

    @property
    def money(self) -> float:
        return self._money

    @money.setter
    def money(self, val: float) -> None:
        fval = float(val)
        ref = self._kin_ref
        if ref is not None and self._kin_idx >= 0:
            ref._kin_moneys[self._kin_idx] = fval
        self._money = fval

    @property
    def trade_cooldown(self) -> int:
        ref = self._kin_ref
        if ref is not None and self._kin_idx >= 0:
            return int(ref._kin_tcooldowns[self._kin_idx])
        return self._trade_cooldown

    @trade_cooldown.setter
    def trade_cooldown(self, val: int) -> None:
        ival = int(val)
        ref = self._kin_ref
        if ref is not None and self._kin_idx >= 0:
            ref._kin_tcooldowns[self._kin_idx] = ival
        self._trade_cooldown = ival

    @property
    def colonize_cooldown(self) -> int:
        ref = self._kin_ref
        if ref is not None and self._kin_idx >= 0:
            return int(ref._kin_ccooldowns[self._kin_idx])
        return self._colonize_cooldown

    @colonize_cooldown.setter
    def colonize_cooldown(self, val: int) -> None:
        ival = int(val)
        ref = self._kin_ref
        if ref is not None and self._kin_idx >= 0:
            ref._kin_ccooldowns[self._kin_idx] = ival
        self._colonize_cooldown = ival

    @property
    def age_ticks(self) -> int:
        ref = self._kin_ref
        if ref is not None and self._kin_idx >= 0:
            return int(ref._kin_age_ticks[self._kin_idx])
        return self._age_ticks

    @age_ticks.setter
    def age_ticks(self, val: int) -> None:
        ival = int(val)
        ref = self._kin_ref
        if ref is not None and self._kin_idx >= 0:
            ref._kin_age_ticks[self._kin_idx] = ival
        self._age_ticks = ival

    @property
    def ticks_since_last_meal(self) -> int:
        ref = self._kin_ref
        if ref is not None and self._kin_idx >= 0:
            return int(ref._kin_ticks_since_meal[self._kin_idx])
        return self._ticks_since_last_meal

    @ticks_since_last_meal.setter
    def ticks_since_last_meal(self, val: int) -> None:
        ival = int(val)
        ref = self._kin_ref
        if ref is not None and self._kin_idx >= 0:
            ref._kin_ticks_since_meal[self._kin_idx] = ival
        self._ticks_since_last_meal = ival

    @property
    def is_outlaw(self) -> bool:
        """Whether this blob is socially stigmatized as a bandit/outlaw."""
        return self.reputation <= -0.35

    @property
    def is_honorable(self) -> bool:
        """Whether this blob has built strong social trust through trade and taxes."""
        return self.reputation >= 0.35

    @property
    def reputation(self) -> float:
        """Social reputation score in [-1.0, +1.0]."""
        return self._reputation

    @reputation.setter
    def reputation(self, val: float) -> None:
        old_val = getattr(self, "_reputation", 0.0)
        c = -1.0 if val < -1.0 else (1.0 if val > 1.0 else val)
        self._reputation = round(c, 3)
        ref = getattr(self, "_kin_ref", None)
        kidx = getattr(self, "_kin_idx", -1)
        if ref is not None and 0 <= kidx < getattr(ref, "_kin_cap", 0):
            ref._kin_reputations[kidx] = self._reputation
        cb = getattr(self, "_on_rep_change", None)
        if cb is not None and old_val != self._reputation:
            try:
                cb(self, old_val, self._reputation)
            except TypeError:
                cb(self)

    @property
    def group_id(self) -> Optional[int]:
        """Settlement Group ID or None if solo/outlaw."""
        return self._group_id

    @group_id.setter
    def group_id(self, val: Optional[int]) -> None:
        old_val = getattr(self, "_group_id", None)
        self._group_id = val
        if old_val != val:
            cb = getattr(self, "_on_group_change", None)
            if cb is not None:
                cb(self, old_val, val)

    def adjust_reputation(self, delta: float) -> float:
        """Adjust social reputation score within [-1.0, +1.0] and return the updated value."""
        self.reputation = self._reputation + float(delta)
        return self._reputation


    @property
    def max_energy(self) -> float:
        """Larger blobs have a larger metabolic energy reserve."""
        return 85.0 + 15.0 * self.genes.size

    @property
    def max_food_capacity(self) -> float:
        """Larger blobs can carry more total food (stomach + tradable surplus)."""
        return 3.4 * self.genes.size

    @property
    def total_food(self) -> float:
        """Total food currently held by the blob (all of which can be traded if desired)."""
        return self.food_today + self.surplus_food

    @property
    def total_food_eaten(self) -> float:
        """Total food eaten over lifetime."""
        return self.food_eaten_lifetime

    @property
    def combat_power(self) -> float:
        """Physical strength governed by the size gene and remaining energy."""
        e_ratio = max(0.15, self.energy / max(1.0, self.max_energy))
        return (self.genes.size ** 1.35) * (0.55 + 0.45 * e_ratio)

    def deduct_traded_food(self, amount: float) -> float:
        """
        Deduct `amount` of food for trading.
        Draws from surplus_food first, then from food_today (even if that drops food_today
        below reproduction or survival thresholds!). Returns how much came from food_today.
        """
        from_surplus = min(self.surplus_food, amount)
        self.surplus_food = round(self.surplus_food - from_surplus, 3)
        rem = round(amount - from_surplus, 3)
        from_stomach = 0.0
        if rem > 0.0:
            from_stomach = min(self.food_today, rem)
            self.food_today = round(self.food_today - from_stomach, 3)
        return from_stomach

    def add_acquired_food(self, amount: float, repro_need: float) -> float:
        """
        Add acquired food (from foraging or buying) with 1:1 conservation up to max_food_capacity.
        Fills food_today up to repro_need first, then stores remainder in surplus_food.
        Returns actual amount stored.
        """
        free_cap = max(0.0, self.max_food_capacity - (self.food_today + self.surplus_food))
        added = round(min(float(amount), free_cap), 3)
        if added <= 0.0:
            return 0.0
        to_stomach = max(0.0, min(added, repro_need - self.food_today))
        self.food_today = round(self.food_today + to_stomach, 3)
        to_surplus = round(added - to_stomach, 3)
        if to_surplus > 0.0:
            self.surplus_food = round(self.surplus_food + to_surplus, 3)
        return added

    def to_summary_dict(self, group_color: Optional[str] = None) -> Dict[str, Any]:
        return {
            "id": int(self.id),
            "gen": int(self.generation),
            "x": round(float(self.x), 1),
            "y": round(float(self.y), 1),
            "vx": round(float(self.vx), 2),
            "vy": round(float(self.vy), 2),
            "heading": round(float(self.heading), 2),
            "r": round(float(self.body_radius), 1),
            "energy": round(float(self.energy), 1),
            "max_energy": float(self.max_energy),
            "food_today": round(float(self.food_today), 2),
            "surplus_food": round(float(self.surplus_food), 2),
            "max_food": float(self.max_food_capacity),
            "money": round(float(self.money), 1),
            "reputation": round(float(self.reputation), 2),
            "home_id": self.home_id,
            "home_x": round(float(self.home_x), 1) if self.home_x is not None else 0.0,
            "home_y": round(float(self.home_y), 1) if self.home_y is not None else 0.0,
            "is_at_home": bool(self.is_at_home),
            "in_swamp": bool(self.in_swamp),
            "mem_x": round(float(self.memory_x), 1),
            "mem_y": round(float(self.memory_y), 1),
            "mem_s": round(float(self.memory_strength), 2),
            "group_id": self.group_id,
            "color": group_color or f"hsl({int(self.genes.hue)}, 75%, 56%)",
            "age_days": int(self.age_days),
            "children": int(self.children_count),
            "trades": int(self.trades_made),
            "raids": int(self.raids_won),
            "bounties": int(self.bounties_won),
            "exiles": int(self.exiles_count),
            "migrations": int(self.migrations_made),
            "reward": round(float(self.cumulative_reward), 2),
            "recent_trade": (self.age_ticks - self.last_trade_tick) < 25,
            "genes": self.genes.to_dict(),
        }
