"""Core 2D Artificial Life, Settlement & Economy Simulation World."""

from typing import List, Dict, Any, Optional, Tuple, Set
import math
import struct
import weakref
import heapq
import numpy as np
from .config import NetworkConfig, TrainingPreferences, WorldConfig, START_PARADIGMS, WORLD_SCALE_PRESETS
from .neural import BlobBrain, BatchedBrainPool, INPUT_DIM, OUTPUT_DIM
from .logger import SimulationTelemetryLogger
from .entities import (
    Genes,
    FoodItem,
    Home,
    Group,
    Blob,
    OasisNode,
    TerrainObstacle,
    GROUP_PALETTE,
    GROUP_NAMES,
    OASIS_NAMES,
    generate_faction_identity,
)
from .spatial import (
    SPATIAL_CELL_SIZE,
    INV_SPATIAL_CELL,
    build_spatial_linked_grid,
    build_obstacle_csr_grid,
    query_nearest_foods_grid,
    query_nearest_peers_grid,
    compute_oasis_and_feelers_numba,
    find_foraging_candidates_grid,
    apply_kinematics_and_metabolism_batch_numba,
    prep_step_observations_numba,
    batch_point_hits_rock_csr,
    filter_active_peers_numba,
    filter_active_market_numba,
    pack_binary_blobs_kernel,
    warmup_simulation_engine,
)

CRADLE_OASIS_NAMES = [
    "Wiegen-Quell Süd",
    "Wiegen-Quell Nordwest",
    "Wiegen-Quell Nordost",
]

GOLD_GROVE_NAMES = [
    "Sonnen-Hain",
    "Bernstein-Oase",
    "Gold-Lagune",
    "Sternen-Hain",
    "Lotus-Garten",
]

THERMAL_SPRING_NAMES = [
    "Smaragd-Therme",
    "Kristall-Geysir",
    "Jade-Quell",
    "Nebel-Therme",
]

SEASONS: List[Tuple[str, str, float, float, float]] = [
    ("spring", "🌸 Frühling", 1.15, 0.18, 1.00),
    ("summer", "☀️ Sommer", 1.35, 0.25, 0.95),
    ("autumn", "🍂 Herbst", 0.85, 0.16, 1.02),
    ("winter", "❄️ Winter", 0.45, 0.12, 1.14),
]

class SimulationWorld:
    """Manages the 2D arena, blobs, homes, multi-home settlements, trading economy, and neuroevolution."""

    def __init__(
        self,
        world_cfg: Optional[WorldConfig] = None,
        net_cfg: Optional[NetworkConfig] = None,
        prefs: Optional[TrainingPreferences] = None,
        seed: Optional[int] = 42,
    ) -> None:
        self.world_cfg = world_cfg or WorldConfig()
        self.net_cfg = net_cfg or NetworkConfig()
        self.prefs = prefs or TrainingPreferences()
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            np.random.seed(seed)

        self.tick_count: int = 0
        self.day_number: int = 1
        self.tick_in_day: int = 0

        self._next_blob_id: int = 1
        self._next_food_id: int = 1
        self._next_home_id: int = 1
        self._next_group_id: int = 1
        self._next_oasis_id: int = 1
        self._next_obstacle_id: int = 1

        self.blobs: List[Blob] = []
        self._blobs_by_id: Dict[int, Blob] = {}
        self.foods: List[FoodItem] = []
        self.homes: Dict[int, Home] = {}
        self.groups: Dict[int, Group] = {}
        self.oases: Dict[int, OasisNode] = {}
        self.obstacles: List[TerrainObstacle] = []
        self._brain_pool = BatchedBrainPool()

        # Vectorized obstacle caches for fast NumPy collision & raycasting
        self._rock_x1 = np.empty(0, dtype=np.float32)
        self._rock_y1 = np.empty(0, dtype=np.float32)
        self._rock_dx = np.empty(0, dtype=np.float32)
        self._rock_dy = np.empty(0, dtype=np.float32)
        self._rock_len_sq = np.empty(0, dtype=np.float32)
        self._rock_r = np.empty(0, dtype=np.float32)

        self._swamp_x1 = np.empty(0, dtype=np.float32)
        self._swamp_y1 = np.empty(0, dtype=np.float32)
        self._swamp_dx = np.empty(0, dtype=np.float32)
        self._swamp_dy = np.empty(0, dtype=np.float32)
        self._swamp_len_sq = np.empty(0, dtype=np.float32)
        self._swamp_r = np.empty(0, dtype=np.float32)

        # Feature 6: Vectorized kinematics buffers
        self._kin_cap = 0
        self._kin_xs = np.empty(0, dtype=np.float64)
        self._kin_ys = np.empty(0, dtype=np.float64)
        self._kin_vxs = np.empty(0, dtype=np.float64)
        self._kin_vys = np.empty(0, dtype=np.float64)
        self._kin_headings = np.empty(0, dtype=np.float32)
        self._kin_energies = np.empty(0, dtype=np.float32)
        self._kin_in_swamps = np.empty(0, dtype=np.bool_)
        self._kin_is_at_homes = np.empty(0, dtype=np.bool_)
        self._kin_foods_today = np.empty(0, dtype=np.float32)
        self._kin_max_speeds = np.empty(0, dtype=np.float32)
        self._kin_body_radii = np.empty(0, dtype=np.float32)
        self._kin_basal_costs = np.empty(0, dtype=np.float32)
        self._kin_has_groups = np.empty(0, dtype=np.bool_)
        self._kin_home_xs = np.empty(0, dtype=np.float64)
        self._kin_home_ys = np.empty(0, dtype=np.float64)
        self._kin_home_radii = np.empty(0, dtype=np.float32)
        self._kin_hit_rocks = np.empty(0, dtype=np.uint8)
        self._kin_speed_norms = np.empty(0, dtype=np.float32)
        self._kin_gene_sizes = np.empty(0, dtype=np.float32)
        self._kin_moneys = np.empty(0, dtype=np.float32)
        self._kin_reputations = np.empty(0, dtype=np.float32)
        self._kin_peer_reps = np.empty(0, dtype=np.float32)
        self._kin_is_at_home_int = np.empty(0, dtype=np.int32)
        self._kin_sprints = np.empty(0, dtype=np.float32)
        self._kin_mem_xs = np.empty(0, dtype=np.float64)
        self._kin_mem_ys = np.empty(0, dtype=np.float64)
        self._kin_mem_strengths = np.empty(0, dtype=np.float64)
        self._kin_surplus_foods = np.empty(0, dtype=np.float32)
        self._kin_gids = np.empty(0, dtype=np.int32)
        self._kin_hids = np.empty(0, dtype=np.int32)
        self._kin_tcooldowns = np.empty(0, dtype=np.int32)
        self._kin_ccooldowns = np.empty(0, dtype=np.int32)
        self._kin_rcooldowns = np.empty(0, dtype=np.int32)
        self._kin_age_ticks = np.empty(0, dtype=np.int32)
        self._kin_ticks_since_meal = np.empty(0, dtype=np.int32)
        self._kin_vision_radii = np.empty(0, dtype=np.float32)
        self._kin_ids = np.empty(0, dtype=np.float32)
        self._kin_gene_rgbs = np.empty(0, dtype=np.uint32)
        self._kin_drive_energy = np.empty(0, dtype=np.float32)
        self._kin_drive_pain = np.empty(0, dtype=np.float32)
        self._kin_drive_social = np.empty(0, dtype=np.float32)
        self._kin_plasticity_lr = np.empty(0, dtype=np.float32)
        self._group_rgb_table = np.zeros(256, dtype=np.uint32)
        self.food_version: int = 1
        self.recent_eaten_indices: List[int] = []
        self._kin_populated: bool = False
        self._kin_static_dirty: bool = True
        self._in_step_once: bool = False

        # Obstacle CSR Spatial Index
        self._rock_offsets = np.empty(0, dtype=np.int32)
        self._rock_indices = np.empty(0, dtype=np.int32)
        self._rock_inv_cell: float = 0.0
        self._rock_gw: int = 0
        self._rock_gh: int = 0
        self._swamp_offsets = np.empty(0, dtype=np.int32)
        self._swamp_indices = np.empty(0, dtype=np.int32)
        self._swamp_inv_cell: float = 0.0
        self._swamp_gw: int = 0
        self._swamp_gh: int = 0
        self._step_eff_visions = np.empty(0, dtype=np.float32)
        self._step_blob_foods_ok = np.empty(0, dtype=np.float32)
        self._step_blob_moneys_ok = np.empty(0, dtype=np.float32)
        self._step_blob_groups = np.empty(0, dtype=np.int32)
        self._step_home_objs: List[Optional[Home]] = []
        self._step_home_radii_list: List[float] = []
        self._last_min_p_idxs = np.empty(0, dtype=np.int32)

        # Persistent food coordinate cache
        self._food_xs: Optional[np.ndarray] = None
        self._food_ys: Optional[np.ndarray] = None
        self._food_base_reaches: Optional[np.ndarray] = None
        self._food_dirty: bool = True

        # Feature 7: Cached static oasis arrays
        self._cached_oasis_count: int = -1
        self._cached_oasis_ox = np.empty(0, dtype=np.float32)
        self._cached_oasis_oy = np.empty(0, dtype=np.float32)
        self._cached_oasis_orad = np.empty(0, dtype=np.float32)
        self._cached_oasis_ofert = np.empty(0, dtype=np.float32)
        self._cached_oasis_oring_bonus = np.empty(0, dtype=np.float32)

        # Daily counters
        self.births_today: int = 0
        self.deaths_today: int = 0
        self.trades_today: int = 0
        self.raids_today: int = 0
        self.raids_repelled_today: int = 0
        self.exiles_today: int = 0
        self.bounties_today: int = 0
        self.total_trades_all_time: int = 0
        self.total_births_all_time: int = 0
        self.total_deaths_all_time: int = 0
        self.total_raids_all_time: int = 0
        self.total_raids_repelled: int = 0
        self.total_exiles_all_time: int = 0
        self.total_bounties_all_time: int = 0
        self.total_upgrades_built: int = 0
        self.total_migrations_all_time: int = 0

        # Hall of Fame (best unique brains by cumulative reward / longevity)
        self.hall_of_fame: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self.event_log: List[Dict[str, Any]] = []
        self.recent_trade_lines: List[Dict[str, Any]] = []
        self.recent_raid_lines: List[Dict[str, Any]] = []

        self.selected_blob_id: Optional[int] = None
        self.topology_version: int = 1
        self.cradle_centers: List[Tuple[float, float, float, float]] = []
        self.telemetry = SimulationTelemetryLogger()
        self._pending_food_spawn: int = 0
        self._eod_maintenance_done: bool = False
        self._eod_settlements_done: bool = False
        self._eod_blob_homes: Optional[List[Optional[Home]]] = None

        self.reset_world()

    @property
    def time_of_day(self) -> float:
        """0.0 = dawn, 0.72 = dusk/nightfall, 1.0 = end of night."""
        return float(self.tick_in_day) / max(1, self.world_cfg.ticks_per_day)

    @property
    def is_night(self) -> bool:
        return self.time_of_day >= self.world_cfg.daylight_ratio

    @property
    def season_info(self) -> Tuple[str, str, float, float, float]:
        """Return (season_id, label, wild_food_mult, golden_chance, cold_drain_mult) for a 16-day year (4 days/season)."""
        idx = ((self.day_number - 1) // 4) % 4
        return SEASONS[idx]

    def log_event(self, kind: str, message: str) -> None:
        self.event_log.append(
            {
                "tick": self.tick_count,
                "day": self.day_number,
                "kind": kind,
                "msg": message,
            }
        )
        if len(self.event_log) > 45:
            self.event_log = self.event_log[-45:]
        if hasattr(self, "telemetry"):
            self.telemetry.log_event(self.day_number, self.tick_count, kind, message)

    def get_active_paradigm(self) -> Dict[str, Any]:
        pid = getattr(self.world_cfg, "start_paradigm", "curriculum")
        return START_PARADIGMS.get(pid, START_PARADIGMS["curriculum"])

    def _register_blob(self, blob: Blob) -> None:
        """Register a blob in the ID index and bind reputation/group change listeners."""
        self._blobs_by_id[blob.id] = blob
        if blob.home_obj is None and blob.home_id in self.homes:
            blob.home_obj = self.homes[blob.home_id]
        wref = weakref.ref(self)
        blob._on_rep_change = lambda b, o=None, n=None, w=wref: w()._on_blob_reputation_change(b, o, n) if w() is not None else None
        blob._on_group_change = lambda b, o, n, w=wref: w()._on_blob_group_change(b, o, n) if w() is not None else None
        if blob.group_id is not None and blob.group_id in self.groups:
            grp = self.groups[blob.group_id]
            grp.rep_dirty = True
            living_ids = getattr(grp, "_living_member_ids", None)
            if living_ids is not None and blob.alive and blob.id not in living_ids:
                living_ids.add(blob.id)
                grp.cached_rep_sum = round(grp.cached_rep_sum + blob._reputation, 5)
                grp.cached_rep_count += 1
                c = grp.cached_rep_count
                grp.avg_reputation = round(grp.cached_rep_sum / c, 3) if c > 0 else 0.0

    def _unregister_blob(self, blob: Blob) -> None:
        """Unregister dead blob from ID index, snapshot terminal values, detach kin ref, and mark its group dirty."""
        if getattr(blob, "_kin_ref", None) is not None:
            idx = blob._kin_idx
            if 0 <= idx < self._kin_cap:
                blob._x = float(self._kin_xs[idx])
                blob._y = float(self._kin_ys[idx])
                blob._vx = float(self._kin_vxs[idx])
                blob._vy = float(self._kin_vys[idx])
                blob._heading = float(self._kin_headings[idx])
                blob._energy = float(self._kin_energies[idx])
                blob._in_swamp = bool(self._kin_in_swamps[idx])
                blob._is_at_home = bool(self._kin_is_at_homes[idx])
                blob._food_today = float(self._kin_foods_today[idx])
                blob._surplus_food = float(self._kin_surplus_foods[idx])
                blob._money = float(self._kin_moneys[idx])
                blob._trade_cooldown = int(self._kin_tcooldowns[idx])
                blob._colonize_cooldown = int(self._kin_ccooldowns[idx])
                blob._repro_cooldown = int(self._kin_rcooldowns[idx])
                blob._age_ticks = int(self._kin_age_ticks[idx])
                blob._ticks_since_last_meal = int(self._kin_ticks_since_meal[idx])
            blob._kin_ref = None
            blob._kin_idx = -1
        blob._sim_idx = -1
        self._blobs_by_id.pop(blob.id, None)
        blob._on_rep_change = None
        blob._on_group_change = None
        gid = blob.group_id
        if gid is not None and gid in self.groups:
            grp = self.groups[gid]
            grp.rep_dirty = True
            living_ids = getattr(grp, "_living_member_ids", None)
            if living_ids is not None and blob.id in living_ids:
                living_ids.discard(blob.id)
                grp.cached_rep_sum = round(grp.cached_rep_sum - blob._reputation, 5)
                grp.cached_rep_count = max(0, grp.cached_rep_count - 1)
                c = grp.cached_rep_count
                grp.avg_reputation = round(grp.cached_rep_sum / c, 3) if c > 0 else 0.0

    def _on_blob_reputation_change(self, blob: Blob, old_rep: Optional[float] = None, new_rep: Optional[float] = None) -> None:
        new_r = new_rep if new_rep is not None else blob._reputation
        idx = getattr(blob, "_sim_idx", -1)
        if 0 <= idx < self._kin_cap:
            self._kin_reputations[idx] = new_r
        kidx = getattr(blob, "_kin_idx", -1)
        if 0 <= kidx < self._kin_cap:
            self._kin_reputations[kidx] = new_r
        gid = blob._group_id
        if gid is not None and gid in self.groups:
            grp = self.groups[gid]
            grp.rep_dirty = True
            living_ids = getattr(grp, "_living_member_ids", None)
            if living_ids is not None and blob.id in living_ids:
                old_r = old_rep if old_rep is not None else 0.0
                grp.cached_rep_sum = round(grp.cached_rep_sum + (new_r - old_r), 5)
                c = grp.cached_rep_count
                grp.avg_reputation = round(grp.cached_rep_sum / c, 3) if c > 0 else 0.0

    def _on_blob_group_change(self, blob: Blob, old_gid: Optional[int], new_gid: Optional[int]) -> None:
        if old_gid is not None and old_gid in self.groups:
            old_grp = self.groups[old_gid]
            old_grp.rep_dirty = True
            if blob.id in old_grp.member_ids:
                old_grp.member_ids = [mid for mid in old_grp.member_ids if mid != blob.id]
            old_living = getattr(old_grp, "_living_member_ids", None)
            if old_living is not None and blob.id in old_living:
                old_living.discard(blob.id)
                old_grp.cached_rep_sum = round(old_grp.cached_rep_sum - blob._reputation, 5)
                old_grp.cached_rep_count = max(0, old_grp.cached_rep_count - 1)
                c = old_grp.cached_rep_count
                old_grp.avg_reputation = round(old_grp.cached_rep_sum / c, 3) if c > 0 else 0.0
        if new_gid is not None and new_gid in self.groups:
            new_grp = self.groups[new_gid]
            new_grp.rep_dirty = True
            if blob.id not in new_grp.member_ids:
                new_grp.member_ids.append(blob.id)
            new_living = getattr(new_grp, "_living_member_ids", None)
            if new_living is not None and blob.alive and blob.id not in new_living:
                new_living.add(blob.id)
                new_grp.cached_rep_sum = round(new_grp.cached_rep_sum + blob._reputation, 5)
                new_grp.cached_rep_count += 1
                c = new_grp.cached_rep_count
                new_grp.avg_reputation = round(new_grp.cached_rep_sum / c, 3) if c > 0 else 0.0
        idx = getattr(blob, "_sim_idx", -1)
        if 0 <= idx < self._kin_cap:
            self._kin_gids[idx] = new_gid if new_gid is not None else -1
            self._kin_has_groups[idx] = (new_gid is not None)

    def _update_group_rgb_table(self) -> None:
        """Maintain lookup table mapping group_id to 24-bit 0xRRGGBB integer."""
        max_gid = max(self.groups.keys()) if self.groups else 0
        needed = max(256, max_gid + 1)
        if getattr(self, "_group_rgb_table", None) is None or len(self._group_rgb_table) < needed:
            self._group_rgb_table = np.zeros(needed, dtype=np.uint32)
        for gid, g in self.groups.items():
            if 0 <= gid < len(self._group_rgb_table):
                self._group_rgb_table[gid] = np.uint32(g.rgb_int)

    def _sync_food_arrays(self) -> None:
        """Sync food coordinates and golden reaches into contiguous float32 arrays."""
        n_foods = len(self.foods)
        if n_foods > 0:
            self._food_xs = np.fromiter((f.x for f in self.foods), dtype=np.float32, count=n_foods)
            self._food_ys = np.fromiter((f.y for f in self.foods), dtype=np.float32, count=n_foods)
            self._food_base_reaches = np.fromiter((10.0 if f.golden else 8.0 for f in self.foods), dtype=np.float32, count=n_foods)
            self._food_id_to_slot = {f.id: idx for idx, f in enumerate(self.foods)}
        else:
            self._food_xs = np.empty(0, dtype=np.float32)
            self._food_ys = np.empty(0, dtype=np.float32)
            self._food_base_reaches = np.empty(0, dtype=np.float32)
            self._food_id_to_slot = {}
        self._food_dirty = False

    def _sync_static_blob_kin_arrays(self, living_blobs: List[Blob]) -> None:
        """Sync immutable genetic and group attributes to persistent _kin_* buffers."""
        n = len(living_blobs)
        self._ensure_kin_capacity(n)
        self._last_synced_blob_count = n
        self._update_group_rgb_table()
        for i, b in enumerate(living_blobs):
            b._sim_idx = i
            self._kin_max_speeds[i] = b.max_speed
            self._kin_body_radii[i] = b.body_radius
            self._kin_basal_costs[i] = b.basal_cost
            self._kin_gene_sizes[i] = b.genes.size
            self._kin_has_groups[i] = (b._group_id is not None)
            h_obj = b.home_obj
            if h_obj is None or h_obj.id != b.home_id:
                h_obj = self.homes.get(b.home_id)
                b.home_obj = h_obj
            self._step_home_objs[i] = h_obj
            hr = h_obj.radius if h_obj is not None else 48.0
            self._step_home_radii_list[i] = hr
            self._kin_home_xs[i] = b.home_x
            self._kin_home_ys[i] = b.home_y
            self._kin_home_radii[i] = hr
            self._kin_mem_xs[i] = b.memory_x
            self._kin_mem_ys[i] = b.memory_y
            self._kin_mem_strengths[i] = b.memory_strength
            self._kin_reputations[i] = b._reputation
            self._kin_moneys[i] = b.money
            self._kin_foods_today[i] = b.food_today
            bgid = b._group_id
            self._kin_gids[i] = bgid if bgid is not None else -1
            self._kin_hids[i] = b.home_id
            self._kin_tcooldowns[i] = b.trade_cooldown
            self._kin_surplus_foods[i] = b.surplus_food
            self._kin_ccooldowns[i] = b.colonize_cooldown
            self._kin_rcooldowns[i] = b.repro_cooldown
            self._kin_age_ticks[i] = b.age_ticks
            self._kin_ticks_since_meal[i] = b.ticks_since_last_meal
            self._kin_vision_radii[i] = b.vision_radius
            self._kin_ids[i] = float(b.id)
            self._kin_gene_rgbs[i] = np.uint32(b.genes._rgb_int)
            self._kin_drive_energy[i] = float(b.genes.drive_energy)
            self._kin_drive_pain[i] = float(b.genes.drive_pain)
            self._kin_drive_social[i] = float(b.genes.drive_social)
            self._kin_plasticity_lr[i] = float(b.genes.plasticity_lr)
            if getattr(b, "_kin_ref", None) is None:
                self._kin_xs[i] = b._x
                self._kin_ys[i] = b._y
                self._kin_vxs[i] = b._vx
                self._kin_vys[i] = b._vy
                self._kin_headings[i] = b._heading
                self._kin_energies[i] = b._energy
                self._kin_in_swamps[i] = b._in_swamp
                self._kin_is_at_homes[i] = b._is_at_home
                self._kin_foods_today[i] = b._food_today
                self._kin_surplus_foods[i] = b._surplus_food
                self._kin_moneys[i] = b._money
                self._kin_tcooldowns[i] = b._trade_cooldown
                self._kin_ccooldowns[i] = b._colonize_cooldown
                self._kin_rcooldowns[i] = b._repro_cooldown
                self._kin_age_ticks[i] = b._age_ticks
                self._kin_ticks_since_meal[i] = b._ticks_since_last_meal
                b._kin_ref = self
                b._kin_idx = i
            else:
                b._kin_idx = i

    def reset_world(self) -> None:
        """Initialize the Concentric Evolutionary Training Map ('Cradle-to-Frontier') with active Start-Paradigm."""
        self.tick_count = 0
        self._substep_count = 0
        self.day_number = 1
        self.tick_in_day = 0
        self._next_blob_id = 1
        self._next_food_id = 1
        self._next_home_id = 1
        self._next_group_id = 1
        self._next_oasis_id = 1
        self._next_obstacle_id = 1
        self.topology_version = getattr(self, "topology_version", 0) + 1

        self.blobs.clear()
        self._blobs_by_id.clear()
        self.foods.clear()
        self.homes.clear()
        self._oasis_to_homes: Dict[int, List[Home]] = {}
        self.groups.clear()
        self.oases.clear()
        self._cached_oasis_count = -1
        self.obstacles.clear()
        self.history.clear()
        self.event_log.clear()
        self.recent_trade_lines.clear()
        self.recent_raid_lines.clear()
        self.map_zones: List[Dict[str, Any]] = []
        self.skill_gates: List[Dict[str, Any]] = []
        self.cradle_centers = []

        self.births_today = 0
        self.deaths_today = 0
        self.trades_today = 0
        self.raids_today = 0
        self.raids_repelled_today = 0
        self.exiles_today = 0
        self.bounties_today = 0
        self.total_trades_all_time = 0
        self.total_births_all_time = 0
        self.total_deaths_all_time = 0
        self.total_raids_all_time = 0
        self.total_raids_repelled = 0
        self.total_exiles_all_time = 0
        self.total_bounties_all_time = 0
        self.total_upgrades_built = 0
        self.total_migrations_all_time = 0

        paradigm = self.get_active_paradigm()
        self._generate_world_topography()

        max_res = max(2, self.world_cfg.max_residents_per_home)
        min_homes_for_pop = math.ceil(self.world_cfg.initial_blobs / max(1, max_res - 1))
        num_cradles = max(1, int(getattr(self.world_cfg, "num_cradles", 1)))
        num_initial_homes = max(6 * num_cradles, min_homes_for_pop)

        # Place modest starter huts ('Survive Solo -> Thrive Together') exclusively around the inner Cradle oases!
        cx, cy = self.world_cfg.width * 0.5, self.world_cfg.height * 0.5
        starter_oases = [o for o in self.oases.values() if o.ring == "inner"]
        if not starter_oases:
            starter_oases = sorted(self.oases.values(), key=lambda o: math.hypot(o.x - cx, o.y - cy))[:3]
        starter_centers = [(o.x, o.y, o.radius) for o in starter_oases] if starter_oases else [(cx, cy, 180.0)]

        is_rapid_exodus = paradigm["id"] == "rapid_exodus"
        starter_depot = 0.45 if is_rapid_exodus else 0.85
        starter_vault = 8.0 if is_rapid_exodus else 12.0

        placed_by_center: List[List[Tuple[float, float]]] = [[] for _ in range(len(starter_centers))]
        for i in range(num_initial_homes):
            c_idx = i % len(starter_centers)
            scx, scy, srad = starter_centers[c_idx]
            local_placed = placed_by_center[c_idx]
            hx, hy = scx, scy
            for attempt in range(25):
                angle = float(self.rng.uniform(0.0, 2.0 * math.pi))
                max_ring = srad * (0.92 if attempt < 15 else 1.28)
                dist = float(self.rng.uniform(srad * 0.40, max_ring))
                cand_x = float(np.clip(scx + math.cos(angle) * dist, 120.0, self.world_cfg.width - 120.0))
                cand_y = float(np.clip(scy + math.sin(angle) * dist, 120.0, self.world_cfg.height - 120.0))
                if not self._point_hits_rock(cand_x, cand_y, pad=58.0) and all(
                    (px - cand_x) ** 2 + (py - cand_y) ** 2 >= 62.0 * 62.0 for px, py in local_placed
                ):
                    hx, hy = cand_x, cand_y
                    break
            local_placed.append((hx, hy))
            hid = self._next_home_id
            self._next_home_id += 1
            home = Home(
                id=hid,
                x=hx,
                y=hy,
                radius=46.0,
                max_residents=max_res,
                food_depot=starter_depot,
                gold_vault=starter_vault,
                zone="cradle",
            )
            self._evaluate_home_location(home)
            self.homes[hid] = home

        self._oasis_to_homes = {oid: [] for oid in self.oases}
        for home in self.homes.values():
            for oid, oasis in self.oases.items():
                max_reach = oasis.radius * 1.55
                if math.hypot(home.x - oasis.x, home.y - oasis.y) <= max_reach:
                    self._oasis_to_homes[oid].append(home)

        innate_seed = float(paradigm.get("innate_seed", 0.72))
        home_list = list(self.homes.values())
        initial_n = self.world_cfg.initial_blobs
        use_pool_prealloc = (
            initial_n >= 500
            and hasattr(self, "_brain_pool")
            and self._brain_pool is not None
            and any(w > 256 for w in self.net_cfg.get_hidden_dims())
        )
        if use_pool_prealloc:
            rec_dim = self.net_cfg.get_recurrent_dim()
            in_dim = INPUT_DIM + rec_dim
            out_dim = OUTPUT_DIM + rec_dim
            layer_dims = [in_dim] + self.net_cfg.get_hidden_dims() + [out_dim]
            self._brain_pool._allocate_capacity(layer_dims, initial_n)
            num_layers = len(layer_dims) - 1

        for i in range(initial_n):
            home = home_list[i % len(home_list)]
            if home.is_full:
                home = self._find_or_create_settlement_home(None, home.x, home.y)
            if self.world_cfg.intrinsic_homeostasis:
                genes = Genes(
                    speed=round(float(self.rng.uniform(0.85, 1.25)), 3),
                    sense=round(float(self.rng.uniform(0.85, 1.30)), 3),
                    size=round(float(self.rng.uniform(0.85, 1.15)), 3),
                    metabolism=1.0,
                    hue=round(float(self.rng.uniform(0, 360)), 1),
                    drive_energy=round(float(self.rng.uniform(0.75, 1.25)), 3),
                    drive_pain=round(float(self.rng.uniform(0.75, 1.25)), 3),
                    drive_social=round(float(self.rng.uniform(-0.2, 0.8)), 3),
                    plasticity_lr=round(float(self.rng.uniform(0.75, 1.25)), 3),
                )
            else:
                genes = Genes(
                    speed=round(float(self.rng.uniform(0.85, 1.25)), 3),
                    sense=round(float(self.rng.uniform(0.85, 1.30)), 3),
                    size=round(float(self.rng.uniform(0.85, 1.15)), 3),
                    metabolism=1.0,
                    hue=round(float(self.rng.uniform(0, 360)), 1),
                )
            if use_pool_prealloc:
                w_slices = [self._brain_pool._W_cap[lay][i] for lay in range(num_layers)]
                b_slices = [self._brain_pool._B_cap[lay][i] for lay in range(num_layers)]
                for lay in range(num_layers):
                    fan_in = layer_dims[lay]
                    fan_out = layer_dims[lay + 1]
                    scale = np.float32(0.55 * np.sqrt(2.0 / (fan_in + fan_out)))
                    w_slices[lay][:] = self.rng.standard_normal((fan_in, fan_out), dtype=np.float32) * scale
                    b_slices[lay].fill(0.0)
                if innate_seed > 0.0 and all(dim >= 2 for dim in self.net_cfg.get_hidden_dims()):
                    s_val = np.float32(innate_seed)
                    w_slices[0][0, 0] += np.float32(1.05) * s_val
                    w_slices[0][1, 1] += np.float32(1.05) * s_val
                    for mid_idx in range(1, len(w_slices) - 1):
                        for ch in range(2):
                            w_slices[mid_idx][ch, ch] += np.float32(1.08) * max(np.float32(0.75), s_val)
                    w_slices[-1][0, 0] += np.float32(0.95) * s_val
                    w_slices[-1][1, 1] += np.float32(0.95) * s_val
                brain = BlobBrain(
                    config=self.net_cfg,
                    weights=w_slices,
                    biases=b_slices,
                    rng=self.rng,
                    _copy_arrays=False,
                )
            else:
                brain = BlobBrain(config=self.net_cfg, rng=self.rng, innate_seed=innate_seed)
            bx = float(np.clip(home.x + self.rng.normal(0.0, 14.0), 30.0, self.world_cfg.width - 30.0))
            by = float(np.clip(home.y + self.rng.normal(0.0, 14.0), 30.0, self.world_cfg.height - 30.0))
            blob = Blob(
                blob_id=self._next_blob_id,
                x=bx,
                y=by,
                home_id=home.id,
                home_x=home.x,
                home_y=home.y,
                brain=brain,
                genes=genes,
                generation=1,
                group_id=None,  # Start solo ('Survive Solo -> Thrive Together')
                money=round(float(self.rng.uniform(4.5, 7.5)), 1),
            )
            self._next_blob_id += 1
            home.resident_ids.append(blob.id)
            self._register_blob(blob)
            self.blobs.append(blob)
            blob._sim_idx = len(self.blobs) - 1

        self.spawn_daily_food(self.world_cfg.daily_food_spawn)
        # In Tabula Rasa mode, seed an initial Day-1 nursery ring inside the Cradle oases so random movers discover food
        if paradigm["id"] == "tabula_rasa" and starter_oases:
            for co in starter_oases:
                for _ in range(10):
                    ang = float(self.rng.uniform(0.0, 2.0 * math.pi))
                    r_d = float(self.rng.uniform(18.0, co.radius * 0.82))
                    fx = float(np.clip(co.x + math.cos(ang) * r_d, 35.0, self.world_cfg.width - 35.0))
                    fy = float(np.clip(co.y + math.sin(ang) * r_d, 35.0, self.world_cfg.height - 35.0))
                    if not self._point_hits_rock(fx, fy, pad=8.0):
                        self.foods.append(
                            FoodItem(
                                id=self._next_food_id,
                                x=fx,
                                y=fy,
                                nutrition=0.6,
                                golden=False,
                                oasis_id=co.id,
                            )
                        )
                        self._next_food_id += 1
                        co.current_food_count += 1
        self._sync_food_arrays()

        if self.blobs:
            self.selected_blob_id = self.blobs[0].id
            self._sync_static_blob_kin_arrays(self.blobs)
            self._kin_static_dirty = False
            if hasattr(self, "_brain_pool") and self._brain_pool is not None:
                self._brain_pool.bind([b.brain for b in self.blobs])
                import gc
                gc.collect()
        self.telemetry.reset(self)
        self.record_daily_snapshot()
        n_inner = len(starter_oases)
        n_outer = len(self.oases) - n_inner
        self.log_event(
            "world",
            f"Welt ({paradigm['label']}) gestartet: {num_cradles} Starter-Wiege(n), {n_inner} Wiegen-Quellen & {n_outer} Frontier-Oasen.",
        )

    def _add_obstacle(
        self, kind: str, x1: float, y1: float, x2: float, y2: float, radius: float, gate_kind: str = ""
    ) -> None:
        obs_id = self._next_obstacle_id
        self._next_obstacle_id += 1
        self.obstacles.append(
            TerrainObstacle(
                id=obs_id,
                kind=kind,
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                radius=float(radius),
                gate_kind=gate_kind,
            )
        )

    def _build_canyon_ring_and_gates(
        self,
        cx: float,
        cy: float,
        rx_canyon: float,
        ry_canyon: float,
        scale_f: float,
        angle_offset: float = 0.0,
    ) -> None:
        """Build a 6-Gate Canyon Ring around (cx, cy) with Chicane, Swamp-Ford, and Steppe-Gap passes."""
        gate_specs = [
            (0.0 + angle_offset, "steppe_gap", "🧭 Ost-Steppe (Fernsinn)"),
            (math.pi / 3.0 + angle_offset, "chicane", "🪨 Südost-Schikane (Fühler)"),
            (2.0 * math.pi / 3.0 + angle_offset, "swamp_ford", "🐊 Südwest-Furt (Moor)"),
            (math.pi + angle_offset, "steppe_gap", "🧭 West-Steppe (Fernsinn)"),
            (4.0 * math.pi / 3.0 + angle_offset, "chicane", "🪨 Nordwest-Schikane (Fühler)"),
            (5.0 * math.pi / 3.0 + angle_offset, "swamp_ford", "🐊 Nordost-Furt (Moor)"),
        ]
        rock_thick = 36.0 * scale_f

        for g_idx, (g_ang, g_kind, g_label) in enumerate(gate_specs):
            next_ang = gate_specs[(g_idx + 1) % len(gate_specs)][0]
            if next_ang <= g_ang:
                next_ang += 2.0 * math.pi

            open_half = 0.145 if g_kind == "chicane" else (0.175 if g_kind == "swamp_ford" else 0.19)
            wall_start = g_ang + open_half
            wall_end = next_ang - open_half
            wall_mid = 0.5 * (wall_start + wall_end)

            for a1, a2 in ((wall_start, wall_mid), (wall_mid, wall_end)):
                wx1 = cx + math.cos(a1) * rx_canyon
                wy1 = cy + math.sin(a1) * ry_canyon
                wx2 = cx + math.cos(a2) * rx_canyon
                wy2 = cy + math.sin(a2) * ry_canyon
                self._add_obstacle("rock", wx1, wy1, wx2, wy2, rock_thick, gate_kind="ring")

            gx = cx + math.cos(g_ang) * rx_canyon
            gy = cy + math.sin(g_ang) * ry_canyon
            nx = math.cos(g_ang)
            ny = math.sin(g_ang)
            tx = -ny
            ty = nx

            self.skill_gates.append(
                {
                    "kind": g_kind,
                    "label": g_label,
                    "x": round(gx, 1),
                    "y": round(gy, 1),
                }
            )

            if g_kind == "chicane":
                bx_center = gx + nx * (155.0 * scale_f)
                by_center = gy + ny * (155.0 * scale_f)
                b_half = 135.0 * scale_f
                self._add_obstacle(
                    "rock",
                    bx_center - tx * b_half,
                    by_center - ty * b_half,
                    bx_center + tx * b_half,
                    by_center + ty * b_half,
                    30.0 * scale_f,
                    gate_kind="chicane",
                )
                ix_center = gx - nx * (135.0 * scale_f) + tx * (85.0 * scale_f)
                iy_center = gy - ny * (135.0 * scale_f) + ty * (85.0 * scale_f)
                self._add_obstacle(
                    "rock",
                    ix_center - tx * (65.0 * scale_f),
                    iy_center - ty * (65.0 * scale_f),
                    ix_center + tx * (65.0 * scale_f),
                    iy_center + ty * (65.0 * scale_f),
                    26.0 * scale_f,
                    gate_kind="chicane",
                )
            elif g_kind == "swamp_ford":
                self._add_obstacle(
                    "swamp",
                    gx - tx * (70.0 * scale_f),
                    gy - ty * (70.0 * scale_f),
                    gx + tx * (70.0 * scale_f),
                    gy + ty * (70.0 * scale_f),
                    118.0 * scale_f,
                    gate_kind="swamp_ford",
                )
                ox_swamp = gx + nx * (260.0 * scale_f)
                oy_swamp = gy + ny * (260.0 * scale_f)
                self._add_obstacle(
                    "swamp",
                    ox_swamp - tx * (55.0 * scale_f),
                    oy_swamp - ty * (55.0 * scale_f),
                    ox_swamp + tx * (55.0 * scale_f),
                    oy_swamp + ty * (55.0 * scale_f),
                    102.0 * scale_f,
                    gate_kind="swamp_ford",
                )
            elif g_kind == "steppe_gap":
                for side in (-1.0, 1.0):
                    c_ang = g_ang + side * 0.165
                    cx1 = cx + math.cos(c_ang) * (rx_canyon * 1.02)
                    cy1 = cy + math.sin(c_ang) * (ry_canyon * 1.02)
                    cx2 = cx + math.cos(c_ang) * (rx_canyon * 1.32)
                    cy2 = cy + math.sin(c_ang) * (ry_canyon * 1.32)
                    self._add_obstacle("rock", cx1, cy1, cx2, cy2, 28.0 * scale_f, gate_kind="steppe_gap")

    def _generate_multi_cradle_topography(self, num_cradles: int) -> None:
        """
        Generate a Continental Multi-Cradle Region Network:
        - `num_cradles` distributed regional starter cradles, each with 3 fragile inner nursery oases
          and a 6-Gate Canyon Ring.
        - Wide Frontier Corridors between cradles containing dozens of Summer Gold Groves ('gold_grove'),
          Winter Thermal Springs ('thermal_spring'), protective rock ribs, and swamp hazards.
        """
        W, H = self.world_cfg.width, self.world_cfg.height
        paradigm = self.get_active_paradigm()
        cradle_fert = float(paradigm.get("cradle_fertility", 0.86))
        cradle_dep = float(paradigm.get("cradle_depletion", 2.4))
        cradle_rec = float(paradigm.get("cradle_recovery", 0.25))
        init_pressure = 2.1 if paradigm["id"] == "rapid_exodus" else 0.0

        cols = max(2, round(math.sqrt(num_cradles * (W / max(1.0, H)))))
        rows = max(1, math.ceil(num_cradles / cols))
        cell_w = W / cols
        cell_h = H / rows

        rx_cradle = min(840.0, cell_w * 0.16)
        ry_cradle = min(680.0, cell_h * 0.16)
        rx_canyon = rx_cradle * 1.32
        ry_canyon = ry_cradle * 1.32
        scale_f = min(1.0, rx_cradle / 740.0)

        self.map_zones = []
        self.skill_gates = []
        self.cradle_centers = []

        cradle_angles = [math.pi * 0.5, math.pi * 1.17, math.pi * 1.83]
        for c_idx in range(num_cradles):
            col = c_idx % cols
            row = c_idx // cols
            base_cx = (col + 0.5) * cell_w
            base_cy = (row + 0.5) * cell_h
            ccx = float(np.clip(base_cx + self.rng.uniform(-cell_w * 0.08, cell_w * 0.08), rx_canyon + 350.0, W - rx_canyon - 350.0))
            ccy = float(np.clip(base_cy + self.rng.uniform(-cell_h * 0.08, cell_h * 0.08), ry_canyon + 350.0, H - ry_canyon - 350.0))
            self.cradle_centers.append((ccx, ccy, rx_cradle * 1.08, ry_cradle * 1.08))

            self.map_zones.append(
                {
                    "id": "cradle",
                    "name": f"WIEGE #{c_idx + 1} · STARTER-TAL",
                    "cx": round(ccx, 1),
                    "cy": round(ccy, 1),
                    "rx": round(rx_cradle, 1),
                    "ry": round(ry_cradle, 1),
                }
            )
            self.map_zones.append(
                {
                    "id": "canyon",
                    "name": f"CANYON-RING #{c_idx + 1}",
                    "cx": round(ccx, 1),
                    "cy": round(ccy, 1),
                    "rx": round(rx_canyon, 1),
                    "ry": round(ry_canyon, 1),
                }
            )

            # 3 Inner Cradle Oases per regional cradle
            for idx, base_ang in enumerate(cradle_angles):
                ang = base_ang + float(self.rng.uniform(-0.08, 0.08))
                dist_x = rx_cradle * 0.43 + float(self.rng.uniform(-16.0, 16.0))
                dist_y = ry_cradle * 0.45 + float(self.rng.uniform(-14.0, 14.0))
                ox = ccx + math.cos(ang) * dist_x
                oy = ccy + math.sin(ang) * dist_y
                rad = round(float(self.rng.uniform(190.0, 212.0)) * scale_f, 1)
                oid = self._next_oasis_id
                self._next_oasis_id += 1
                self.oases[oid] = OasisNode(
                    id=oid,
                    name=f"{CRADLE_OASIS_NAMES[idx % len(CRADLE_OASIS_NAMES)]} {c_idx + 1}",
                    x=float(np.clip(ox, rad + 40.0, W - rad - 40.0)),
                    y=float(np.clip(oy, rad + 40.0, H - rad - 40.0)),
                    radius=rad,
                    fertility=cradle_fert,
                    max_capacity=95,
                    harvest_pressure=init_pressure,
                    biome="cradle",
                    ring="inner",
                    depletion_rate=cradle_dep,
                    recovery_rate=cradle_rec,
                    cradle_id=c_idx + 1,
                )

            # Build the 6-Gate Canyon Ring around this cradle
            ang_off = float((c_idx % 3) * (math.pi / 9.0))
            self._build_canyon_ring_and_gates(ccx, ccy, rx_canyon, ry_canyon, scale_f, angle_offset=ang_off)

        # Outer Frontier Oases distributed across the corridors between cradles
        num_outer = max(num_cradles * 4, self.world_cfg.num_oases - num_cradles * 3)
        gold_idx = 0
        therm_idx = 0
        placed_outer: List[Tuple[float, float, float]] = [(o.x, o.y, o.radius) for o in self.oases.values()]

        for idx in range(num_outer):
            rad = round(float(self.rng.uniform(230.0, 275.0)), 1)
            ox, oy = W * 0.5, H * 0.5
            found_spot = False
            for _ in range(45):
                cand_x = float(self.rng.uniform(rad + 180.0, W - rad - 180.0))
                cand_y = float(self.rng.uniform(rad + 180.0, H - rad - 180.0))
                # Must lie outside all Cradle Canyon rings
                in_any_canyon = any(
                    ((cand_x - ccx) / (rx_canyon * 1.26)) ** 2 + ((cand_y - ccy) / (ry_canyon * 1.26)) ** 2 < 1.0
                    for ccx, ccy, _, _ in self.cradle_centers
                )
                if in_any_canyon:
                    continue
                if all((cand_x - px) ** 2 + (cand_y - py) ** 2 >= (rad + prad + 320.0) ** 2 for px, py, prad in placed_outer):
                    ox, oy = cand_x, cand_y
                    found_spot = True
                    break
            if not found_spot:
                continue

            placed_outer.append((ox, oy, rad))
            is_gold_grove = (idx % 2) == 0
            if is_gold_grove:
                b_kind = "gold_grove"
                o_name = f"{GOLD_GROVE_NAMES[gold_idx % len(GOLD_GROVE_NAMES)]} #{gold_idx + 1}"
                gold_idx += 1
                cap = 140
            else:
                b_kind = "thermal_spring"
                o_name = f"{THERMAL_SPRING_NAMES[therm_idx % len(THERMAL_SPRING_NAMES)]} #{therm_idx + 1}"
                therm_idx += 1
                cap = 125

            oid = self._next_oasis_id
            self._next_oasis_id += 1
            self.oases[oid] = OasisNode(
                id=oid,
                name=o_name,
                x=ox,
                y=oy,
                radius=rad,
                fertility=round(float(self.rng.uniform(0.92, 1.0)), 2),
                max_capacity=cap,
                biome=b_kind,
                ring="outer",
                depletion_rate=0.72,
                recovery_rate=1.22,
                cradle_id=0,
            )

            # Add protective rock ridge or swamp near alternating frontier oases
            rib_ang = float(self.rng.uniform(0.0, 2.0 * math.pi))
            rib_dist = rad + 185.0
            rcx = ox + math.cos(rib_ang) * rib_dist
            rcy = oy + math.sin(rib_ang) * rib_dist
            tx = -math.sin(rib_ang)
            ty = math.cos(rib_ang)
            if 160.0 < rcx < W - 160.0 and 160.0 < rcy < H - 160.0:
                if (idx % 3) != 2:
                    self._add_obstacle(
                        "rock",
                        rcx - tx * 160.0,
                        rcy - ty * 160.0,
                        rcx + tx * 160.0,
                        rcy + ty * 160.0,
                        32.0,
                        gate_kind="rib",
                    )
                else:
                    self._add_obstacle(
                        "swamp",
                        rcx - tx * 65.0,
                        rcy - ty * 65.0,
                        rcx + tx * 65.0,
                        rcy + ty * 65.0,
                        105.0,
                        gate_kind="outer_swamp",
                    )

        self._rebuild_obstacle_arrays()

    def _generate_kindergarten_topography(self) -> None:
        """
        Generate the Perfect Basic Training Map ('Kindergarten Arena'):
        - Zone 0: The Central Nursery Sanctuary (0 <= r < 850):
          Completely flat, smooth ground. Zero rocks, zero swamps.
          4 balanced nursery oases in a cross/diamond pattern:
          North: (cx, cy - 320), South: (cx, cy + 320),
          East: (cx + 380, cy), West: (cx - 380, cy).
          depletion_rate=0.85, recovery_rate=0.65, max_capacity=80.
        - Zone 1: Transition Ring (850 <= r < 1600):
          8 radial convex boulders (rounded pillars, length 80, width 40, radius 20).
          Minimum clearance between any two obstacles > 320 units.
          4 open radial food spokes along cardinal axes.
        - Zone 2: Frontier Biomes (1600 <= r < 2400):
          4 Golden Groves in diagonal quadrants (NE, NW, SE, SW; rich golden cluster chance).
          4 Thermal Springs on outer perimeter (cold resilience).
          4 small mud swamps (r=85) off-axis.
        """
        W, H = self.world_cfg.width, self.world_cfg.height
        cx, cy = W * 0.5, H * 0.5
        paradigm = self.get_active_paradigm()
        cradle_fert = float(paradigm.get("cradle_fertility", 1.0))
        if paradigm["id"] == "curriculum":
            cradle_fert = 1.0  # Lush kindergarten baseline
        init_pressure = 2.1 if paradigm["id"] == "rapid_exodus" else 0.0

        self.map_zones = [
            {
                "id": "cradle",
                "name": "ZONE 0 · SANCTUARY (Kindergarten Basin)",
                "cx": round(cx, 1),
                "cy": round(cy, 1),
                "rx": 850.0,
                "ry": 850.0,
            },
            {
                "id": "canyon",
                "name": "ZONE 1 · TRANSITION RING (Convex Guideposts)",
                "cx": round(cx, 1),
                "cy": round(cy, 1),
                "rx": 1600.0,
                "ry": 1600.0,
            },
            {
                "id": "frontier",
                "name": "ZONE 2 · FRONTIER BIOMES (Golden Groves & Thermal Springs)",
                "cx": round(cx, 1),
                "cy": round(cy, 1),
                "rx": 2400.0,
                "ry": 1800.0,
            },
        ]
        self.skill_gates = []
        self.cradle_centers = [(cx, cy, 850.0, 850.0)]

        # 1. ZONE 0: 4 Inner Sanctuary Oases in Cross/Diamond Pattern
        inner_oasis_specs = [
            ("Wiegen-Quell Nord", cx, cy - 320.0),
            ("Wiegen-Quell Süd", cx, cy + 320.0),
            ("Wiegen-Quell Ost", cx + 380.0, cy),
            ("Wiegen-Quell West", cx - 380.0, cy),
        ]
        for name, ox, oy in inner_oasis_specs:
            oid = self._next_oasis_id
            self._next_oasis_id += 1
            rad = 220.0
            self.oases[oid] = OasisNode(
                id=oid,
                name=name,
                x=float(ox),
                y=float(oy),
                radius=rad,
                fertility=cradle_fert,
                max_capacity=80,
                harvest_pressure=init_pressure,
                biome="cradle",
                ring="inner",
                depletion_rate=0.85,
                recovery_rate=0.65,
                cradle_id=1,
            )

        # 2. ZONE 1: 8 Radial Convex Boulder Guideposts along r = 1250 with >320 unit clearance
        # Placed at angles offset by pi/8 from cardinal directions to keep cardinal spokes wide open!
        r_boulder = 1250.0
        for i in range(8):
            ang = (math.pi / 8.0) + (i * 2.0 * math.pi / 8.0)
            bx = cx + math.cos(ang) * r_boulder
            by = cy + math.sin(ang) * r_boulder
            # Tangential pillar (length 80, thickness 40 -> radius 20)
            tx = -math.sin(ang)
            ty = math.cos(ang)
            half_len = 40.0
            self._add_obstacle(
                "rock",
                bx - tx * half_len,
                by - ty * half_len,
                bx + tx * half_len,
                by + ty * half_len,
                20.0,
                gate_kind="guidepost",
            )
            self.skill_gates.append(
                {
                    "kind": "guidepost",
                    "label": f"Fels-Leitpfosten #{i + 1}",
                    "x": round(bx, 1),
                    "y": round(by, 1),
                }
            )

        # 3. ZONE 2: 4 Golden Groves (diagonal quadrants) + 4 Thermal Springs (outer perimeter)
        frontier_specs = [
            # 4 Golden Groves in diagonal quadrants
            ("gold_grove", "Sonnen-Hain NE", cx + 1400.0, cy - 1000.0, 250.0),
            ("gold_grove", "Bernstein-Hain NW", cx - 1400.0, cy - 1000.0, 250.0),
            ("gold_grove", "Gold-Lagune SE", cx + 1400.0, cy + 1000.0, 250.0),
            ("gold_grove", "Sternen-Hain SW", cx - 1400.0, cy + 1000.0, 250.0),
            # 4 Thermal Springs on cardinal perimeters
            ("thermal_spring", "Smaragd-Therme Nord", cx, cy - 1380.0, 240.0),
            ("thermal_spring", "Kristall-Geysir Süd", cx, cy + 1380.0, 240.0),
            ("thermal_spring", "Jade-Quell Ost", cx + 1850.0, cy, 240.0),
            ("thermal_spring", "Nebel-Therme West", cx - 1850.0, cy, 240.0),
        ]

        for b_kind, o_name, ox, oy, rad in frontier_specs:
            oid = self._next_oasis_id
            self._next_oasis_id += 1
            cap = 110 if b_kind == "gold_grove" else 95
            self.oases[oid] = OasisNode(
                id=oid,
                name=o_name,
                x=float(np.clip(ox, rad + 40.0, W - rad - 40.0)),
                y=float(np.clip(oy, rad + 40.0, H - rad - 40.0)),
                radius=rad,
                fertility=1.0,
                max_capacity=cap,
                biome=b_kind,
                ring="outer",
                depletion_rate=0.72,
                recovery_rate=1.22,
                cradle_id=0,
            )

        # 4. ZONE 2 Hazards: 4 small mud swamps (radius 85) positioned off-axis in outer frontier
        swamp_offsets = [
            (cx + 950.0, cy - 1200.0),
            (cx - 950.0, cy - 1200.0),
            (cx + 950.0, cy + 1200.0),
            (cx - 950.0, cy + 1200.0),
        ]
        for sx_mid, sy_mid in swamp_offsets:
            self._add_obstacle(
                "swamp",
                sx_mid - 25.0,
                sy_mid - 20.0,
                sx_mid + 25.0,
                sy_mid + 20.0,
                65.0,
                gate_kind="outer_swamp",
            )

        self._rebuild_obstacle_arrays()

    def _generate_world_topography(self) -> None:
        """
        Generate the Concentric Evolutionary Training Map ('Cradle-to-Frontier') for single-cradle maps,
        or the Multi-Cradle Continental Region Network when `num_cradles > 1`.
        """
        if getattr(self.world_cfg, "scale_preset", "") == "training_kindergarten":
            self._generate_kindergarten_topography()
            return

        num_cradles = max(1, int(getattr(self.world_cfg, "num_cradles", 1)))
        if num_cradles > 1:
            self._generate_multi_cradle_topography(num_cradles)
            return

        W, H = self.world_cfg.width, self.world_cfg.height
        cx, cy = W * 0.5, H * 0.5
        sx, sy = W / 4800.0, H / 3600.0
        scale_f = min(1.0, math.sqrt((W * H) / (4800.0 * 3600.0)))
        paradigm = self.get_active_paradigm()

        rx_cradle, ry_cradle = 740.0 * sx, 580.0 * sy
        rx_canyon, ry_canyon = 980.0 * sx, 760.0 * sy
        rx_frontier, ry_frontier = 1760.0 * sx, 1300.0 * sy
        self.cradle_centers = [(cx, cy, max(1.0, 820.0 * sx), max(1.0, 640.0 * sy))]

        self.map_zones = [
            {
                "id": "cradle",
                "name": "RING I · STARTER-WIEGE (Überweidungs-Zone)",
                "cx": round(cx, 1),
                "cy": round(cy, 1),
                "rx": round(rx_cradle, 1),
                "ry": round(ry_cradle, 1),
            },
            {
                "id": "canyon",
                "name": "RING II · CANYON-GÜRTEL & SKILL-GATES",
                "cx": round(cx, 1),
                "cy": round(cy, 1),
                "rx": round(rx_canyon, 1),
                "ry": round(ry_canyon, 1),
            },
            {
                "id": "frontier",
                "name": "RING III · FRONTIER-BIOME (Gold-Haine & Thermal-Oasen)",
                "cx": round(cx, 1),
                "cy": round(cy, 1),
                "rx": round(rx_frontier, 1),
                "ry": round(ry_frontier, 1),
            },
        ]
        self.skill_gates = []

        # 1. RING I: 3 Fragile Starter-Cradle Oases
        cradle_angles = [math.pi * 0.5, math.pi * 1.17, math.pi * 1.83]  # South, NW, NE
        cradle_fert = float(paradigm.get("cradle_fertility", 0.86))
        cradle_dep = float(paradigm.get("cradle_depletion", 2.4))
        cradle_rec = float(paradigm.get("cradle_recovery", 0.25))
        init_pressure = 2.1 if paradigm["id"] == "rapid_exodus" else 0.0

        for idx, base_ang in enumerate(cradle_angles):
            ang = base_ang + float(self.rng.uniform(-0.07, 0.07))
            dist_x = 320.0 * sx + float(self.rng.uniform(-18.0, 18.0)) * sx
            dist_y = 265.0 * sy + float(self.rng.uniform(-15.0, 15.0)) * sy
            ox = cx + math.cos(ang) * dist_x
            oy = cy + math.sin(ang) * dist_y
            rad = round(float(self.rng.uniform(188.0, 208.0)) * scale_f, 1)
            oid = self._next_oasis_id
            self._next_oasis_id += 1
            self.oases[oid] = OasisNode(
                id=oid,
                name=CRADLE_OASIS_NAMES[idx % len(CRADLE_OASIS_NAMES)],
                x=float(np.clip(ox, rad + 40.0, W - rad - 40.0)),
                y=float(np.clip(oy, rad + 40.0, H - rad - 40.0)),
                radius=rad,
                fertility=cradle_fert,
                max_capacity=18,
                harvest_pressure=init_pressure,
                biome="cradle",
                ring="inner",
                depletion_rate=cradle_dep,
                recovery_rate=cradle_rec,
            )

        # 2. RING III: 9 Specialized Frontier Oases (Alternating Summer Gold Groves & Winter Thermal Springs)
        num_outer = max(6, self.world_cfg.num_oases - 3)
        gold_idx = 0
        therm_idx = 0
        for idx in range(num_outer):
            base_ang = (2.0 * math.pi * idx) / num_outer + (math.pi / num_outer) * 0.5
            ang = base_ang + float(self.rng.uniform(-0.05, 0.05))
            # Push East/West Steppe-Gap target oases slightly further out to test long-range sensory navigation
            is_steppe_axis = abs(math.sin(ang)) < 0.38
            r_mult = 1.07 if is_steppe_axis else float(self.rng.uniform(0.94, 1.03))
            ox = cx + math.cos(ang) * rx_frontier * r_mult
            oy = cy + math.sin(ang) * ry_frontier * r_mult
            rad = round(float(self.rng.uniform(225.0, 260.0)) * scale_f, 1)

            is_gold_grove = (idx % 2) == 0
            if is_gold_grove:
                b_kind = "gold_grove"
                o_name = GOLD_GROVE_NAMES[gold_idx % len(GOLD_GROVE_NAMES)]
                gold_idx += 1
                cap = 32
            else:
                b_kind = "thermal_spring"
                o_name = THERMAL_SPRING_NAMES[therm_idx % len(THERMAL_SPRING_NAMES)]
                therm_idx += 1
                cap = 30

            oid = self._next_oasis_id
            self._next_oasis_id += 1
            self.oases[oid] = OasisNode(
                id=oid,
                name=o_name,
                x=float(np.clip(ox, rad + 55.0, W - rad - 55.0)),
                y=float(np.clip(oy, rad + 55.0, H - rad - 55.0)),
                radius=rad,
                fertility=round(float(self.rng.uniform(0.92, 1.0)), 2),
                max_capacity=cap,
                biome=b_kind,
                ring="outer",
                depletion_rate=0.72,
                recovery_rate=1.22,
            )

        # 3. RING II: Canyon Belt & 6 Asymmetric Skill-Gates at 0°, 60°, 120°, 180°, 240°, 300°
        self._build_canyon_ring_and_gates(cx, cy, rx_canyon, ry_canyon, scale_f, angle_offset=0.0)

        # 4. Outer Frontier Radial Canyon Ribs & Defensive Alcoves (separating outer biomes & granting Watchtower cover)
        outer_oases = [o for o in self.oases.values() if o.ring == "outer"]
        for idx, o in enumerate(outer_oases):
            o_ang = math.atan2((o.y - cy) / max(1.0, sy), (o.x - cx) / max(1.0, sx))
            rib_ang = o_ang + (math.pi / len(outer_oases))
            rx1 = cx + math.cos(rib_ang) * (rx_canyon * 1.24)
            ry1 = cy + math.sin(rib_ang) * (ry_canyon * 1.24)
            rx2 = cx + math.cos(rib_ang) * (rx_frontier * 0.86)
            ry2 = cy + math.sin(rib_ang) * (ry_frontier * 0.86)
            if all(
                self._point_to_seg_dist(oo.x, oo.y, rx1, ry1, rx2, ry2) >= (oo.radius * 0.72 + 45.0)
                for oo in self.oases.values()
            ):
                self._add_obstacle("rock", rx1, ry1, rx2, ry2, 30.0 * scale_f, gate_kind="rib")

            # Add 4 outer inter-biome swamp hazards on alternating ribs
            if (idx % 2) == 1:
                sx_mid = cx + math.cos(rib_ang) * (rx_frontier * 0.96)
                sy_mid = cy + math.sin(rib_ang) * (ry_frontier * 0.96)
                if all(math.hypot(oo.x - sx_mid, oo.y - sy_mid) >= (oo.radius * 0.8 + 95.0) for oo in self.oases.values()):
                    self._add_obstacle(
                        "swamp",
                        sx_mid - 35.0 * scale_f,
                        sy_mid - 25.0 * scale_f,
                        sx_mid + 35.0 * scale_f,
                        sy_mid + 25.0 * scale_f,
                        92.0 * scale_f,
                        gate_kind="outer_swamp",
                    )

        self._rebuild_obstacle_arrays()

    @staticmethod
    def _point_to_seg_dist(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
        dx = x2 - x1
        dy = y2 - y1
        len_sq = dx * dx + dy * dy
        if len_sq <= 1e-6:
            return math.hypot(px - x1, py - y1)
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
        return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))

    @staticmethod
    def _build_spatial_grid(
        tuples_list: Tuple[Tuple[float, float, float, float, float, float], ...],
        gw: int,
        gh: int,
        extra_pad: float,
    ) -> Tuple[Tuple[Tuple[Tuple[float, float, float, float, float, float], ...], ...], ...]:
        grid: List[List[List[Tuple[float, float, float, float, float, float]]]] = [
            [[] for _ in range(gh)] for _ in range(gw)
        ]
        for seg in tuples_list:
            sx1, sy1, sdx, sdy, s_inv_l2, sr = seg
            sx2 = sx1 + sdx
            sy2 = sy1 + sdy
            pad = sr + extra_pad
            gx0 = max(0, int((min(sx1, sx2) - pad) * 0.01))
            gx1 = min(gw - 1, int((max(sx1, sx2) + pad) * 0.01))
            gy0 = max(0, int((min(sy1, sy2) - pad) * 0.01))
            gy1 = min(gh - 1, int((max(sy1, sy2) + pad) * 0.01))
            max_cell_d2 = (pad + 72.0) ** 2
            for gx in range(gx0, gx1 + 1):
                cx_cell = gx * 100.0 + 50.0
                for gy in range(gy0, gy1 + 1):
                    cy_cell = gy * 100.0 + 50.0
                    t = ((cx_cell - sx1) * sdx + (cy_cell - sy1) * sdy) * s_inv_l2
                    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
                    dx = cx_cell - (sx1 + t * sdx)
                    dy = cy_cell - (sy1 + t * sdy)
                    if dx * dx + dy * dy <= max_cell_d2:
                        grid[gx][gy].append(seg)
        return tuple(tuple(tuple(cell) for cell in col) for col in grid)

    def _rebuild_obstacle_arrays(self) -> None:
        """Cache vectorized NumPy arrays and a 100x100 spatial lookup grid for rock walls and swamps."""
        self.topology_version = getattr(self, "topology_version", 0) + 1
        rocks = [o for o in self.obstacles if o.kind == "rock"]
        if rocks:
            self._rock_x1 = np.array([o.x1 for o in rocks], dtype=np.float32)
            self._rock_y1 = np.array([o.y1 for o in rocks], dtype=np.float32)
            self._rock_dx = np.array([o.x2 - o.x1 for o in rocks], dtype=np.float32)
            self._rock_dy = np.array([o.y2 - o.y1 for o in rocks], dtype=np.float32)
            self._rock_len_sq = np.maximum(np.float32(1e-5), self._rock_dx ** 2 + self._rock_dy ** 2)
            self._rock_inv_len_sq = np.float32(1.0) / self._rock_len_sq
            self._rock_r = np.array([o.radius for o in rocks], dtype=np.float32)
            self._rock_feeler_r2 = (self._rock_r + np.float32(56.0)) ** 2
            self._rock_tuples = tuple(
                (
                    float(o.x1),
                    float(o.y1),
                    float(o.x2 - o.x1),
                    float(o.y2 - o.y1),
                    1.0 / max(1e-5, (o.x2 - o.x1) ** 2 + (o.y2 - o.y1) ** 2),
                    float(o.radius),
                )
                for o in rocks
            )
        else:
            self._rock_x1 = np.empty(0, dtype=np.float32)
            self._rock_y1 = np.empty(0, dtype=np.float32)
            self._rock_dx = np.empty(0, dtype=np.float32)
            self._rock_dy = np.empty(0, dtype=np.float32)
            self._rock_len_sq = np.empty(0, dtype=np.float32)
            self._rock_inv_len_sq = np.empty(0, dtype=np.float32)
            self._rock_r = np.empty(0, dtype=np.float32)
            self._rock_feeler_r2 = np.empty(0, dtype=np.float32)
            self._rock_tuples = ()

        swamps = [o for o in self.obstacles if o.kind == "swamp"]
        if swamps:
            self._swamp_x1 = np.array([o.x1 for o in swamps], dtype=np.float32)
            self._swamp_y1 = np.array([o.y1 for o in swamps], dtype=np.float32)
            self._swamp_dx = np.array([o.x2 - o.x1 for o in swamps], dtype=np.float32)
            self._swamp_dy = np.array([o.y2 - o.y1 for o in swamps], dtype=np.float32)
            self._swamp_len_sq = np.maximum(np.float32(1e-5), self._swamp_dx ** 2 + self._swamp_dy ** 2)
            self._swamp_inv_len_sq = np.float32(1.0) / self._swamp_len_sq
            self._swamp_r = np.array([o.radius for o in swamps], dtype=np.float32)
            self._swamp_feeler_r2 = (self._swamp_r + np.float32(48.0)) ** 2
            self._swamp_tuples = tuple(
                (
                    float(o.x1),
                    float(o.y1),
                    float(o.x2 - o.x1),
                    float(o.y2 - o.y1),
                    1.0 / max(1e-5, (o.x2 - o.x1) ** 2 + (o.y2 - o.y1) ** 2),
                    float(o.radius),
                )
                for o in swamps
            )
        else:
            self._swamp_x1 = np.empty(0, dtype=np.float32)
            self._swamp_y1 = np.empty(0, dtype=np.float32)
            self._swamp_dx = np.empty(0, dtype=np.float32)
            self._swamp_dy = np.empty(0, dtype=np.float32)
            self._swamp_len_sq = np.empty(0, dtype=np.float32)
            self._swamp_inv_len_sq = np.empty(0, dtype=np.float32)
            self._swamp_r = np.empty(0, dtype=np.float32)
            self._swamp_feeler_r2 = np.empty(0, dtype=np.float32)
            self._swamp_tuples = ()

        gw = max(1, math.ceil(self.world_cfg.width * 0.01) + 1)
        gh = max(1, math.ceil(self.world_cfg.height * 0.01) + 1)
        self._grid_w = gw
        self._grid_h = gh
        self._rock_grid = self._build_spatial_grid(self._rock_tuples, gw, gh, 64.0)
        self._swamp_grid = self._build_spatial_grid(self._swamp_tuples, gw, gh, 32.0)
        self._rock_offsets, self._rock_indices, self._rock_inv_cell, self._rock_gw, self._rock_gh = build_obstacle_csr_grid(
            self._rock_x1, self._rock_y1, self._rock_dx, self._rock_dy, self._rock_r, 56.0, self.world_cfg.width, self.world_cfg.height
        )
        self._swamp_offsets, self._swamp_indices, self._swamp_inv_cell, self._swamp_gw, self._swamp_gh = build_obstacle_csr_grid(
            self._swamp_x1, self._swamp_y1, self._swamp_dx, self._swamp_dy, self._swamp_r, 48.0, self.world_cfg.width, self.world_cfg.height
        )
        self._obstacles_dict_cache = [ob.to_dict() for ob in self.obstacles]
        for h in getattr(self, "homes", {}).values():
            self._evaluate_home_location(h, update_defense=True)

    def _point_hits_rock(self, px: float, py: float, pad: float = 12.0) -> bool:
        if not self._rock_tuples:
            return False
        gx = int(px * 0.01)
        gy = int(py * 0.01)
        if pad <= 64.0 and 0 <= gx < self._grid_w and 0 <= gy < self._grid_h:
            candidates = self._rock_grid[gx][gy]
            if not candidates:
                return False
        else:
            candidates = self._rock_tuples

        for rx1, ry1, rdx, rdy, r_inv_l2, rr in candidates:
            t = ((px - rx1) * rdx + (py - ry1) * rdy) * r_inv_l2
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            dx = px - (rx1 + t * rdx)
            dy = py - (ry1 + t * rdy)
            thresh = rr + pad
            if dx * dx + dy * dy < thresh * thresh:
                return True
        return False

    def _evaluate_home_location(self, home: Home, update_defense: bool = True) -> None:
        """
        Compute location-dependent bonuses and zone ('cradle' vs 'frontier') for `home`:
        - `fertility_bonus`: Modest & fragile in the Cradle; high at Outer Frontier Oases (with Winter Thermal bonus)!
        - `terrain_defense_bonus`: High when nestled near protective Rock canyon ridges!
        """
        hx, hy = home.x, home.y
        if getattr(home, "zone", None) is None:
            if self.cradle_centers:
                in_cradle = any(
                    (((hx - ccx) / rx_c) ** 2 + ((hy - ccy) / ry_c) ** 2) <= 1.0
                    for ccx, ccy, rx_c, ry_c in self.cradle_centers
                )
            else:
                W, H = self.world_cfg.width, self.world_cfg.height
                cx, cy = W * 0.5, H * 0.5
                rx_c = max(1.0, 820.0 * (W / 4800.0))
                ry_c = max(1.0, 640.0 * (H / 3600.0))
                in_cradle = (((hx - cx) / rx_c) ** 2 + ((hy - cy) / ry_c) ** 2) <= 1.0
            home.zone = "cradle" if in_cradle else "frontier"

        best_fert = 0.0
        season_id = self.season_info[0]
        nearby = getattr(home, "_nearby_oases", None)
        if nearby is None:
            nearby = []
            for o in self.oases.values():
                max_reach = o.radius * 2.16
                if abs(hx - o.x) <= max_reach and abs(hy - o.y) <= max_reach:
                    nearby.append(o)
            home._nearby_oases = nearby

        for o in nearby:
            d = math.hypot(hx - o.x, hy - o.y)
            if o.ring == "inner":
                # Starter Cradle oases cannot support high-yield industrial farms once grazed
                if d <= o.radius * 1.45:
                    best_fert = max(best_fert, (0.22 + 0.58 * o.fertility) * (0.65 if o.is_overgrazed else 1.0))
            else:
                # Outer Frontier oases unlock full agricultural & seasonal bonuses
                biome_mult = 1.0
                if o.biome == "thermal_spring" and season_id in ("autumn", "winter"):
                    biome_mult = 1.32
                elif o.biome == "gold_grove" and season_id in ("spring", "summer"):
                    biome_mult = 1.18
                if d <= o.radius * 1.40:
                    best_fert = max(best_fert, (0.70 + 0.92 * o.fertility) * biome_mult)
                elif d <= o.radius * 2.15:
                    falloff = 1.0 - (d - o.radius * 1.40) / max(1.0, o.radius * 0.75)
                    best_fert = max(best_fert, (0.30 + 0.68 * o.fertility) * max(0.0, falloff) * biome_mult)
        home.fertility_bonus = round(best_fert, 2)

        if update_defense:
            def_bonus = 0.0
            if self._rock_tuples:
                gx = int(hx * 0.01)
                gy = int(hy * 0.01)
                seen_segs = set()
                near_rocks = 0
                for dgx in (-1, 0, 1):
                    ngx = gx + dgx
                    if 0 <= ngx < self._grid_w:
                        col = self._rock_grid[ngx]
                        for dgy in (-1, 0, 1):
                            ngy = gy + dgy
                            if 0 <= ngy < self._grid_h:
                                for seg in col[ngy]:
                                    if seg in seen_segs:
                                        continue
                                    seen_segs.add(seg)
                                    rx1, ry1, rdx, rdy, r_inv_l2, rr = seg
                                    t = ((hx - rx1) * rdx + (hy - ry1) * rdy) * r_inv_l2
                                    if t < 0.0:
                                        t = 0.0
                                    elif t > 1.0:
                                        t = 1.0
                                    dx = hx - (rx1 + t * rdx)
                                    dy = hy - (ry1 + t * rdy)
                                    thresh = rr + 130.0
                                    if dx * dx + dy * dy <= thresh * thresh:
                                        near_rocks += 1
                if near_rocks > 0:
                    def_bonus = min(0.75, 0.30 + 0.20 * near_rocks)
            home.terrain_defense_bonus = round(def_bonus, 2)

    def _refresh_all_home_locations(self, update_defense: bool = False) -> None:
        for h in self.homes.values():
            self._evaluate_home_location(h, update_defense=update_defense)

    def spawn_daily_food(self, count: int) -> None:
        """
        Spawn food strictly inside fertile Oasis nodes (0% random wasteland scatter!),
        modulated by each oasis's biome specialization ('cradle', 'gold_grove', 'thermal_spring'),
        current fertility (Overgrazing vs. Lush state), and season.
        """
        if not self.oases:
            return
        season_id, _slabel, wild_mult, golden_chance, _cold = self.season_info
        scaled_count = max(14, round(count * max(0.62, wild_mult)))
        max_food_on_map = max(360, int(count * 2.2))
        if len(self.foods) >= max_food_on_map:
            return
        to_spawn = min(scaled_count, max_food_on_map - len(self.foods))
        if to_spawn <= 0:
            return

        # Count current food per oasis
        oasis_counts: Dict[int, int] = {oid: 0 for oid in self.oases}
        for f in self.foods:
            if f.oasis_id is not None and f.oasis_id in oasis_counts:
                oasis_counts[f.oasis_id] += 1
        for oid, o in self.oases.items():
            o.current_food_count = oasis_counts[oid]

        oasis_list = list(self.oases.values())
        n_oases = len(oasis_list)
        weights_list: List[float] = []
        eff_caps: List[int] = []
        eff_goldens: List[float] = []

        for o in oasis_list:
            if o.is_overgrazed and o.harvest_pressure >= 3.2:
                w_val = 0.0
            else:
                w_val = max(0.04, o.fertility ** 1.45)

            # Biome-specific seasonal multipliers
            if o.biome == "cradle":
                b_mult = 0.85 * wild_mult
                g_mult = 0.45
                if o.is_overgrazed:
                    w_val *= 0.25
            elif o.biome == "thermal_spring":
                # Geothermal refuge: resists Autumn/Winter freeze!
                if season_id == "winter":
                    b_mult = 1.25
                    w_val *= 1.65
                elif season_id == "autumn":
                    b_mult = 1.15
                    w_val *= 1.30
                else:
                    b_mult = 0.92 * wild_mult
                g_mult = 0.90
            else:
                # 'gold_grove': Booms in Spring/Summer with rich Golden Clusters, freezes in Winter
                if season_id in ("spring", "summer"):
                    b_mult = 1.28 * wild_mult
                    g_mult = 2.05
                    w_val *= 1.35
                elif season_id == "winter":
                    b_mult = 0.32
                    g_mult = 0.55
                    w_val *= 0.45
                else:
                    b_mult = 0.95 * wild_mult
                    g_mult = 1.25

            weights_list.append(w_val)
            eff_caps.append(max(2, round(o.max_capacity * (0.18 + 0.85 * o.fertility) * b_mult)))
            fert_gold = 1.35 if o.fertility >= 0.78 else (0.30 if o.is_overgrazed else 1.0)
            eff_goldens.append(min(0.65, golden_chance * g_mult * fert_gold))

        weights = np.array(weights_list, dtype=np.float64)
        w_sum = float(np.sum(weights))
        if w_sum <= 0.0:
            weights[:] = 1.0 / n_oases
        else:
            weights /= w_sum

        # In kindergarten training arena: ensure at least 65% of daily food spawns inside Zone 0 during initial days / when pop < 150
        is_kindergarten = getattr(self.world_cfg, "scale_preset", "") == "training_kindergarten"
        living_pop = len([b for b in self.blobs if b.alive])
        if is_kindergarten and (self.day_number <= 3 or living_pop < 150):
            inner_indices = [idx for idx, o in enumerate(oasis_list) if o.ring == "inner"]
            outer_indices = [idx for idx, o in enumerate(oasis_list) if o.ring != "inner"]
            if inner_indices and outer_indices:
                w_inner_sum = float(np.sum(weights[inner_indices]))
                w_outer_sum = float(np.sum(weights[outer_indices]))
                if w_inner_sum > 0.0 and w_outer_sum > 0.0:
                    weights[inner_indices] = (weights[inner_indices] / w_inner_sum) * 0.68
                    weights[outer_indices] = (weights[outer_indices] / w_outer_sum) * 0.32
                elif w_inner_sum > 0.0:
                    weights[inner_indices] = 1.0 / len(inner_indices)
                    weights[outer_indices] = 0.0
                weights /= float(np.sum(weights))

        chosen_indices = self.rng.choice(n_oases, size=to_spawn, p=weights)
        angles = self.rng.uniform(0.0, 2.0 * math.pi, size=to_spawn).astype(np.float32)
        r_fracs = np.sqrt(self.rng.uniform(0.02, 0.88, size=to_spawn)).astype(np.float32)
        rand_golds = self.rng.random(to_spawn).astype(np.float32)
        if len(self._cached_oasis_ox) == n_oases:
            ox = self._cached_oasis_ox
            oy = self._cached_oasis_oy
            orad = self._cached_oasis_orad
        else:
            ox = np.fromiter((o.x for o in oasis_list), dtype=np.float32, count=n_oases)
            oy = np.fromiter((o.y for o in oasis_list), dtype=np.float32, count=n_oases)
            orad = np.fromiter((o.radius for o in oasis_list), dtype=np.float32, count=n_oases)
            self._cached_oasis_ox = ox
            self._cached_oasis_oy = oy
            self._cached_oasis_orad = orad
        r_dists = orad[chosen_indices] * r_fracs
        raw_xs = ox[chosen_indices] + np.cos(angles) * r_dists
        raw_ys = oy[chosen_indices] + np.sin(angles) * r_dists
        max_x = self.world_cfg.width - 25.0
        max_y = self.world_cfg.height - 25.0
        fxs = np.clip(raw_xs, 25.0, max_x)
        fys = np.clip(raw_ys, 25.0, max_y)

        rock_hits = batch_point_hits_rock_csr(
            fxs, fys, 10.0,
            self._rock_x1, self._rock_y1, self._rock_dx, self._rock_dy,
            self._rock_inv_len_sq, self._rock_r,
            self._rock_offsets, self._rock_indices,
            self._rock_inv_cell, self._rock_gw, self._rock_gh
        )

        cur_counts = [o.current_food_count for o in oasis_list]
        foods = self.foods
        fid = self._next_food_id
        for i in range(to_spawn):
            chosen_idx = chosen_indices[i]
            if cur_counts[chosen_idx] >= eff_caps[chosen_idx] or rock_hits[i]:
                continue
            is_gold = bool(rand_golds[i] < eff_goldens[chosen_idx])
            nutr = 1.2 if is_gold else 0.6
            o = oasis_list[chosen_idx]
            foods.append(FoodItem(id=fid, x=float(fxs[i]), y=float(fys[i]), nutrition=nutr, golden=is_gold, oasis_id=o.id))
            fid += 1
            cur_counts[chosen_idx] += 1
        for idx, o in enumerate(oasis_list):
            o.current_food_count = cur_counts[idx]
        self._next_food_id = fid
        self.food_version += 1
        self.recent_eaten_indices.clear()
        self._sync_food_arrays()

    def _step_oasis_ecology(self) -> None:
        """
        Update oasis overgrazing recovery and continuous ecological regrowth:
        - Inner Cradle oases have high `depletion_rate` and low `recovery_rate`, collapsing under overcrowding.
        - Outer Frontier oases recover briskly and reward sustainable colonization.
        """
        if not self.oases:
            return
        oasis_counts: Dict[int, int] = {oid: 0 for oid in self.oases}
        for f in self.foods:
            if f.oasis_id is not None and f.oasis_id in oasis_counts:
                oasis_counts[f.oasis_id] += 1

        season_id, _slabel, wild_mult, _gold, _cold = self.season_info
        is_kindergarten = getattr(self.world_cfg, "scale_preset", "") == "training_kindergarten"
        hp_threshold = 6.0 if is_kindergarten else 2.2
        for oid, o in self.oases.items():
            o.current_food_count = oasis_counts[oid]
            # Decay harvest pressure
            decay = (0.42 if is_kindergarten else 0.72) if o.ring == "inner" else 0.58
            o.harvest_pressure *= decay
            eff_season_rec = 1.15 if (o.biome == "thermal_spring" and season_id == "winter") else wild_mult
            if is_kindergarten and o.ring == "inner" and self.day_number <= 5:
                # Epoch 1 (Nursery Phase): Cradle oases are highly resilient, maintaining lush conditions
                recovery = 0.12 * o.recovery_rate
                o.fertility = round(min(1.0, max(0.68, o.fertility + recovery)), 3)
            elif o.harvest_pressure < hp_threshold:
                base_rec = 0.075 * eff_season_rec if o.harvest_pressure < (hp_threshold * 0.4) else 0.038 * eff_season_rec
                recovery = base_rec * o.recovery_rate
                o.fertility = round(min(1.0, o.fertility + recovery), 3)
            else:
                # Continuous harvesting depresses fertility scaled by depletion_rate
                o.fertility = round(max(0.05, o.fertility - 0.012 * o.depletion_rate), 3)

        # Sprout a batch of new food in fertile oases
        pulse_budget = max(8, self.world_cfg.daily_food_spawn // 6)
        self.spawn_daily_food(pulse_budget)
        self._cached_overgrazed_oases = None
        self._refresh_all_home_locations()

    def _spawn_settlement_gardens(self) -> None:
        """
        Spawn local garden fruits around inhabited settlement houses, scaled by `fertility_bonus`
        (Oasis proximity), 🌾 Farm upgrades, and Autumn harvest. Wasteland homes yield 0 crops!
        """
        max_food_on_map = max(380, int(self.world_cfg.daily_food_spawn * 2.2))
        if len(self.foods) >= max_food_on_map:
            return
        self._refresh_all_home_locations()
        season_id, _, _, _, _ = self.season_info
        max_gardens = max(75, getattr(self.world_cfg, "num_cradles", 1) * 45)
        spawned_gardens = 0
        for h in self.homes.values():
            if len(self.foods) >= max_food_on_map or spawned_gardens >= max_gardens:
                break
            if h.fertility_bonus < 0.15:
                # Barren wasteland: no garden crops can grow without oasis moisture!
                continue
            if len(h.resident_ids) < 2 and h.farm_level == 0:
                continue
            base_crops = 2 if (h.group_id is not None and len(h.resident_ids) >= 4) else 1
            raw_crops = (base_crops + h.farm_level) * h.fertility_bonus
            crops = max(0, round(raw_crops))
            if season_id == "autumn":
                crops = math.ceil(crops * 1.6)
                if h.farm_level > 0:
                    h.food_depot = min(
                        h.max_depot_capacity,
                        round(h.food_depot + 0.45 * h.farm_level * h.fertility_bonus, 2),
                    )
            elif season_id == "winter":
                crops = max(1 if (h.farm_level > 0 and h.fertility_bonus >= 0.5) else 0, int(h.farm_level * h.fertility_bonus))
                if h.farm_level > 0:
                    h.food_depot = min(
                        h.max_depot_capacity,
                        round(h.food_depot + 0.35 * h.farm_level * h.fertility_bonus, 2),
                    )

            for _ in range(crops):
                angle = float(self.rng.uniform(0.0, 2.0 * math.pi))
                dist = float(self.rng.uniform(28.0, 78.0))
                fx = float(np.clip(h.x + math.cos(angle) * dist, 25.0, self.world_cfg.width - 25.0))
                fy = float(np.clip(h.y + math.sin(angle) * dist, 25.0, self.world_cfg.height - 25.0))
                if self._point_hits_rock(fx, fy, pad=8.0):
                    continue
                self.foods.append(
                    FoodItem(
                        id=self._next_food_id,
                        x=fx,
                        y=fy,
                        nutrition=0.6,
                        golden=False,
                    )
                )
                self._next_food_id += 1
                spawned_gardens += 1

    def update_network_architecture(self, new_cfg: NetworkConfig, reset_weights: bool = False) -> None:
        """Update neural network depth/width/activation for all living blobs."""
        self.net_cfg = new_cfg
        for blob in self.blobs:
            if reset_weights:
                blob.brain = BlobBrain(config=new_cfg, rng=self.rng)
            else:
                blob.brain = blob.brain.adapt_to_config(new_cfg)
        if reset_weights:
            self.hall_of_fame.clear()
        self.log_event(
            "brain",
            f"Netz-Architektur aktualisiert: Tiefe={len(new_cfg.get_hidden_dims())}, "
            f"Hidden={new_cfg.get_hidden_dims()}, Aktivierung={new_cfg.activation.upper()}",
        )

    def _compute_spatial_perception(
        self,
        living_blobs: List[Blob],
        night_mode: bool,
        dt: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, List[Optional[Blob]], np.ndarray, np.ndarray, np.ndarray]:
        """O(1) Spatial-Grid & 16-Thread Numba computation of nearest food, nearest peer, oasis scent, and raycast feelers."""
        n_blobs = len(living_blobs)
        if n_blobs == 0:
            self._last_min_p_idxs = np.empty(0, dtype=np.int32)
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 6), dtype=np.float32),
                [],
                np.empty(0, dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
            )

        self._ensure_kin_capacity(n_blobs)
        if (
            getattr(self, "_kin_static_dirty", True)
            or getattr(self, "_last_synced_blob_count", -1) != n_blobs
            or getattr(living_blobs[0], "_kin_ref", None) is None
        ):
            self._sync_static_blob_kin_arrays(living_blobs)
            self._kin_static_dirty = False

        blob_xs = self._kin_xs[:n_blobs]
        blob_ys = self._kin_ys[:n_blobs]
        headings = self._kin_headings[:n_blobs]
        eff_visions = self._step_eff_visions[:n_blobs]
        has_grp = self._kin_has_groups[:n_blobs]
        if self.world_cfg.continuous_lifecycle:
            tod = self.time_of_day
            light = 0.5 + 0.5 * math.sin(2.0 * math.pi * (tod - 0.15))
            night_factor = np.float32(0.62 + 0.38 * light)
        else:
            night_factor = np.float32(0.65 if night_mode else 1.0)
        half_price = np.float32(self.world_cfg.base_food_price * 0.5)

        # Vectorized NumPy sensory traits extraction (0.04 ms vs 3.45 ms)
        eff_visions[:] = self._kin_vision_radii[:n_blobs] * (night_factor * np.where(has_grp, np.float32(1.28), np.float32(1.0)))
        total_foods = self._kin_foods_today[:n_blobs] + self._kin_surplus_foods[:n_blobs]
        blob_foods_ok = self._step_blob_foods_ok[:n_blobs]
        blob_foods_ok[:] = np.where(total_foods >= np.float32(0.5), np.float32(1.0), np.float32(-1.0))
        blob_moneys_ok = self._step_blob_moneys_ok[:n_blobs]
        blob_moneys_ok[:] = np.where(self._kin_moneys[:n_blobs] >= half_price, np.float32(1.0), np.float32(-1.0))
        blob_groups = self._kin_gids[:n_blobs]

        self._kin_populated = False

        grid_w = max(1, int(self.world_cfg.width * INV_SPATIAL_CELL) + 2)
        grid_h = max(1, int(self.world_cfg.height * INV_SPATIAL_CELL) + 2)
        num_cells = grid_w * grid_h
        self._step_grid_w = grid_w
        self._step_grid_h = grid_h
        self._step_num_cells = num_cells

        food_vecs = self._query_nearest_foods(blob_xs, blob_ys, eff_visions, n_blobs)
        peer_vecs, nearest_peers, nearest_peer_dists = self._query_nearest_peers(
            living_blobs,
            blob_xs,
            blob_ys,
            eff_visions,
            n_blobs,
            blob_foods_ok=blob_foods_ok,
            blob_moneys_ok=blob_moneys_ok,
            blob_groups=blob_groups,
        )
        oasis_vecs, feeler_vecs = self._compute_oasis_and_feeler_sensors(
            living_blobs, blob_xs, blob_ys, n_blobs, headings=headings
        )
        return food_vecs, peer_vecs, nearest_peers, nearest_peer_dists, oasis_vecs, feeler_vecs

    def _compute_oasis_and_feeler_sensors(
        self,
        living_blobs: List[Blob],
        blob_xs: np.ndarray,
        blob_ys: np.ndarray,
        n_blobs: int,
        headings: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute via 16-Thread Numba kernel (zero N*M heap allocations):
        - `oasis_vecs` (N, 3): [dx_best_oasis, dy_best_oasis, local_fertility (-1..+1)]
        - `feeler_vecs` (N, 3): [left_feeler, front_feeler, right_feeler] (+1=rock, -0.8=swamp, 0=clear)
        """
        if n_blobs == 0:
            oasis_vecs = np.zeros((0, 3), dtype=np.float32)
            feeler_vecs = np.zeros((0, 3), dtype=np.float32)
            return oasis_vecs, feeler_vecs

        oasis_list = list(self.oases.values())
        n_oases = len(oasis_list)
        if n_oases > 0:
            if getattr(self, "_cached_oasis_count", -1) != n_oases:
                self._cached_oasis_ox = np.fromiter((o.x for o in oasis_list), dtype=np.float32, count=n_oases)
                self._cached_oasis_oy = np.fromiter((o.y for o in oasis_list), dtype=np.float32, count=n_oases)
                self._cached_oasis_orad = np.fromiter((o.radius for o in oasis_list), dtype=np.float32, count=n_oases)
                self._cached_oasis_ofert = np.empty(n_oases, dtype=np.float32)
                self._cached_oasis_oring_bonus = np.fromiter(
                    (1.22 if getattr(o, "ring", "outer") == "outer" else 0.85 for o in oasis_list),
                    dtype=np.float32,
                    count=n_oases,
                )
                self._cached_oasis_count = n_oases

            ofert = self._cached_oasis_ofert
            for i, o in enumerate(oasis_list):
                ofert[i] = o.fertility

            ox = self._cached_oasis_ox
            oy = self._cached_oasis_oy
            orad = self._cached_oasis_orad
            oring_bonus = self._cached_oasis_oring_bonus
        else:
            ox = np.empty(0, dtype=np.float32)
            oy = np.empty(0, dtype=np.float32)
            orad = np.empty(0, dtype=np.float32)
            ofert = np.empty(0, dtype=np.float32)
            oring_bonus = np.empty(0, dtype=np.float32)

        if headings is None:
            headings = np.fromiter((b.heading for b in living_blobs), dtype=np.float32, count=n_blobs)

        return compute_oasis_and_feelers_numba(
            blob_xs,
            blob_ys,
            headings,
            ox,
            oy,
            orad,
            ofert,
            oring_bonus,
            self._rock_x1,
            self._rock_y1,
            self._rock_dx,
            self._rock_dy,
            self._rock_inv_len_sq,
            self._rock_r,
            self._rock_feeler_r2,
            self._swamp_x1,
            self._swamp_y1,
            self._swamp_dx,
            self._swamp_dy,
            self._swamp_inv_len_sq,
            self._swamp_r,
            self._swamp_feeler_r2,
            self._rock_offsets,
            self._rock_indices,
            self._rock_inv_cell,
            self._rock_gw,
            self._rock_gh,
            self._swamp_offsets,
            self._swamp_indices,
            self._swamp_inv_cell,
            self._swamp_gw,
            self._swamp_gh,
        )

    def _query_nearest_foods(
        self,
        blob_xs: np.ndarray,
        blob_ys: np.ndarray,
        eff_visions: np.ndarray,
        n_blobs: int,
    ) -> np.ndarray:
        n_foods = len(self.foods)
        if n_foods == 0 or n_blobs == 0:
            self._step_food_xs = np.empty(0, dtype=np.float32)
            self._step_food_ys = np.empty(0, dtype=np.float32)
            return np.zeros((n_blobs, 3), dtype=np.float32)

        if getattr(self, "_food_dirty", True) or getattr(self, "_food_xs", None) is None or len(self._food_xs) != n_foods:
            self._food_xs = np.fromiter((f.x for f in self.foods), dtype=np.float32, count=n_foods)
            self._food_ys = np.fromiter((f.y for f in self.foods), dtype=np.float32, count=n_foods)
            self._food_base_reaches = np.fromiter((10.0 if f.golden else 8.0 for f in self.foods), dtype=np.float32, count=n_foods)
            self._food_dirty = False
        food_xs = self._food_xs
        food_ys = self._food_ys
        self._step_food_xs = food_xs
        self._step_food_ys = food_ys

        grid_w = getattr(self, "_step_grid_w", max(1, int(self.world_cfg.width * INV_SPATIAL_CELL) + 2))
        grid_h = getattr(self, "_step_grid_h", max(1, int(self.world_cfg.height * INV_SPATIAL_CELL) + 2))
        num_cells = grid_w * grid_h

        food_cell_head, food_next = build_spatial_linked_grid(
            food_xs, food_ys, INV_SPATIAL_CELL, grid_w, grid_h
        )
        return query_nearest_foods_grid(
            blob_xs,
            blob_ys,
            eff_visions,
            food_xs,
            food_ys,
            food_cell_head,
            food_next,
            INV_SPATIAL_CELL,
            grid_w,
            grid_h,
            self._rock_x1,
            self._rock_y1,
            self._rock_dx,
            self._rock_dy,
            self._rock_inv_len_sq,
            self._rock_r,
            self._rock_offsets,
            self._rock_indices,
            self._rock_inv_cell,
            self._rock_gw,
            self._rock_gh,
        )

    def _query_nearest_peers(
        self,
        living_blobs: List[Blob],
        blob_xs: np.ndarray,
        blob_ys: np.ndarray,
        eff_visions: np.ndarray,
        n_blobs: int,
        blob_foods_ok: Optional[np.ndarray] = None,
        blob_moneys_ok: Optional[np.ndarray] = None,
        blob_groups: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[Optional[Blob]], np.ndarray]:
        if n_blobs <= 1:
            self._last_min_p_idxs = np.full((n_blobs,), -1, dtype=np.int32)
            return (
                np.zeros((n_blobs, 6), dtype=np.float32),
                [None] * n_blobs,
                np.full((n_blobs,), 999999.0, dtype=np.float32),
            )

        grid_w = getattr(self, "_step_grid_w", max(1, int(self.world_cfg.width * INV_SPATIAL_CELL) + 2))
        grid_h = getattr(self, "_step_grid_h", max(1, int(self.world_cfg.height * INV_SPATIAL_CELL) + 2))

        half_price = self.world_cfg.base_food_price * 0.5
        if blob_foods_ok is None:
            blob_foods_ok = np.fromiter(
                ((1.0 if (b.food_today + b.surplus_food) >= 0.5 else -1.0) for b in living_blobs),
                dtype=np.float32,
                count=n_blobs,
            )
        if blob_moneys_ok is None:
            blob_moneys_ok = np.fromiter(
                ((1.0 if b.money >= half_price else -1.0) for b in living_blobs),
                dtype=np.float32,
                count=n_blobs,
            )
        if blob_groups is None:
            blob_groups = np.fromiter(
                ((b.group_id if b.group_id is not None else -1) for b in living_blobs),
                dtype=np.int32,
                count=n_blobs,
            )

        blob_cell_head, blob_next = build_spatial_linked_grid(
            blob_xs, blob_ys, INV_SPATIAL_CELL, grid_w, grid_h
        )
        peer_vecs, min_p_idxs, min_p_dists = query_nearest_peers_grid(
            blob_xs,
            blob_ys,
            eff_visions,
            blob_foods_ok,
            blob_moneys_ok,
            blob_groups,
            blob_cell_head,
            blob_next,
            INV_SPATIAL_CELL,
            grid_w,
            grid_h,
            0.0,
        )
        self._last_min_p_idxs = min_p_idxs
        nearest_peers: List[Optional[Blob]] = [
            living_blobs[idx] if idx >= 0 else None for idx in min_p_idxs.tolist()
        ]
        return peer_vecs, nearest_peers, min_p_dists

    @staticmethod
    def _clamp_home_arrival_step(
        blob: Blob,
        step_dx: float,
        step_dy: float,
        hx: float,
        hy: float,
        home_radius: float,
    ) -> Tuple[float, float]:
        to_hx = hx - blob.x
        to_hy = hy - blob.y
        step_len_sq = step_dx * step_dx + step_dy * step_dy
        if step_len_sq > 1e-6 and (step_dx * to_hx + step_dy * to_hy) > 0.0:
            t_proj = min(1.0, max(0.0, (to_hx * step_dx + to_hy * step_dy) / step_len_sq))
            cx = blob.x + t_proj * step_dx - hx
            cy = blob.y + t_proj * step_dy - hy
            if (cx * cx + cy * cy) <= (home_radius * home_radius):
                blob.vx *= 0.25
                blob.vy *= 0.25
                return step_dx * t_proj, step_dy * t_proj
        return step_dx, step_dy

    def _reflect_arena_bounds(self, blob: Blob) -> bool:
        pad = blob.body_radius + 4.0
        max_x = self.world_cfg.width - pad
        hit_wall = False
        if blob.x < pad:
            blob.x = pad
            blob.vx *= -0.6
            hit_wall = True
        elif blob.x > max_x:
            blob.x = max_x
            blob.vx *= -0.6
            hit_wall = True

        max_y = self.world_cfg.height - pad
        if blob.y < pad:
            blob.y = pad
            blob.vy *= -0.6
            hit_wall = True
        elif blob.y > max_y:
            blob.y = max_y
            blob.vy *= -0.6
            hit_wall = True
        return hit_wall

    def _resolve_terrain_physics(self, blob: Blob) -> bool:
        """
        Check swamp immersion and resolve physical capsule collisions with rock labyrinth walls
        using the O(1) 100x100 spatial acceleration grid.
        Returns True if the blob collided with a rock obstacle.
        """
        bx = blob.x
        by = blob.y
        br = blob.body_radius
        gx = int(bx * 0.01)
        gy = int(by * 0.01)
        in_grid = 0 <= gx < self._grid_w and 0 <= gy < self._grid_h

        # 1. Check Swamp immersion
        swamp_cands = self._swamp_grid[gx][gy] if in_grid else self._swamp_tuples
        if swamp_cands:
            in_swamp = False
            for sx1, sy1, sdx, sdy, s_inv_l2, sr in swamp_cands:
                t = ((bx - sx1) * sdx + (by - sy1) * sdy) * s_inv_l2
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                dx = bx - (sx1 + t * sdx)
                dy = by - (sy1 + t * sdy)
                thresh = sr + br
                if dx * dx + dy * dy <= thresh * thresh:
                    in_swamp = True
                    break
            blob.in_swamp = in_swamp
        else:
            blob.in_swamp = False

        # 2. Resolve Rock Wall collisions with smooth tangential sliding
        rock_cands = self._rock_grid[gx][gy] if in_grid else self._rock_tuples
        if not rock_cands:
            return False

        hit_rock = False
        pad_r = br + 2.0
        for rx1, ry1, rdx, rdy, r_inv_l2, rr in rock_cands:
            t = ((bx - rx1) * rdx + (by - ry1) * rdy) * r_inv_l2
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            cx = rx1 + t * rdx
            cy = ry1 + t * rdy
            dx = bx - cx
            dy = by - cy
            d2 = dx * dx + dy * dy
            target_r = rr + pad_r
            if d2 < target_r * target_r:
                hit_rock = True
                dist = math.sqrt(d2)
                if dist > 1e-4:
                    inv_d = 1.0 / dist
                    nx = dx * inv_d
                    ny = dy * inv_d
                else:
                    nx, ny = 1.0, 0.0
                bx = cx + nx * target_r
                by = cy + ny * target_r
                blob.x = bx
                blob.y = by
                vn = blob.vx * nx + blob.vy * ny
                if vn < 0.0:
                    blob.vx -= 1.15 * vn * nx
                    blob.vy -= 1.15 * vn * ny
        return hit_rock

    def _apply_kinematics_and_metabolism(
        self,
        blob: Blob,
        move_x: float,
        move_y: float,
        sprint: float,
        home_obj: Optional[Home],
        home_radius: float,
        night_mode: bool,
        dt_scale: float = 1.0,
        egocentric_mode: bool = False,
    ) -> float:
        """Update blob position, terrain physics (swamps & rock walls), home status, and energy drain."""
        speed_mult = 1.0 + 0.55 * sprint
        if blob.in_swamp:
            speed_mult *= 0.44
        if blob.is_at_home and (
            blob.food_today >= self.world_cfg.reproduction_food_need
            or (night_mode and blob.food_today >= self.world_cfg.survival_food_need)
        ):
            speed_mult *= 0.18

        blend = 0.28 if dt_scale <= 1.0 else (1.0 - (0.72 ** min(dt_scale, 10.0)))
        if egocentric_mode:
            raw_thrust = max(-1.0, min(1.0, float(move_x)))
            eff_thrust = raw_thrust if raw_thrust >= 0.0 else 0.35 * raw_thrust
            turn = max(-1.0, min(1.0, float(move_y)))
            hd = blob.heading + turn * 0.28 * dt_scale
            hd = math.atan2(math.sin(hd), math.cos(hd))
            blob.heading = hd
            target_vx = math.cos(hd) * eff_thrust * blob.max_speed * speed_mult
            target_vy = math.sin(hd) * eff_thrust * blob.max_speed * speed_mult
            blob.vx = (1.0 - blend) * blob.vx + blend * target_vx
            blob.vy = (1.0 - blend) * blob.vy + blend * target_vy
        else:
            mag = math.hypot(move_x, move_y)
            if mag > 1.0:
                move_x /= mag
                move_y /= mag

            blob.vx = (1.0 - blend) * blob.vx + blend * (move_x * blob.max_speed * speed_mult)
            blob.vy = (1.0 - blend) * blob.vy + blend * (move_y * blob.max_speed * speed_mult)

        hx = home_obj.x if home_obj is not None else blob.home_x
        hy = home_obj.y if home_obj is not None else blob.home_y
        step_dx = blob.vx * dt_scale
        step_dy = blob.vy * dt_scale

        if dt_scale > 1.0 and (
            night_mode or blob.food_today >= self.world_cfg.survival_food_need
        ):
            step_dx, step_dy = self._clamp_home_arrival_step(
                blob, step_dx, step_dy, hx, hy, home_radius
            )

        blob.x += step_dx
        blob.y += step_dy
        hit_rock = self._resolve_terrain_physics(blob)
        if hit_rock:
            blob.pending_rl_reward -= 0.015
            self.telemetry.on_rock_collision(blob)

        hit_wall = self._reflect_arena_bounds(blob)

        speed_mag = math.hypot(blob.vx, blob.vy)
        if egocentric_mode:
            if (hit_rock or hit_wall) and speed_mag > 0.08:
                blob.heading = math.atan2(blob.vy, blob.vx)
        else:
            if speed_mag > 0.08:
                blob.heading = math.atan2(blob.vy, blob.vx)

        dhx = hx - blob.x
        dhy = hy - blob.y
        blob.is_at_home = (dhx * dhx + dhy * dhy) <= (home_radius * home_radius)

        move_speed_norm = speed_mag / max(0.1, blob.max_speed)
        movement_cost = blob.genes._basal_cost * (
            0.35 + 0.60 * (move_speed_norm * move_speed_norm) + 0.42 * (sprint * sprint)
        )
        if blob.in_swamp:
            movement_cost *= 1.75
        if blob.is_at_home:
            movement_cost *= 0.25 if blob.group_id is not None else 0.35
        else:
            movement_cost *= self.season_info[4]
            if night_mode:
                movement_cost *= 1.18

        day_len_scale = 360.0 / max(180.0, float(self.world_cfg.ticks_per_day))
        e_new = blob.energy - movement_cost * dt_scale * day_len_scale
        blob.energy = e_new if e_new > 0.0 else 0.0
        return move_speed_norm

    def _apply_kinematics_and_metabolism_batch(
        self,
        living_blobs: List[Blob],
        actions_batch: np.ndarray,
        home_objs: List[Optional[Home]],
        home_radii: List[float],
        night_mode: bool,
        dt_scale: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(living_blobs)
        if n == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.uint8)

        self._ensure_kin_capacity(n)
        xs = self._kin_xs[:n]
        ys = self._kin_ys[:n]
        vxs = self._kin_vxs[:n]
        vys = self._kin_vys[:n]
        headings = self._kin_headings[:n]
        energies = self._kin_energies[:n]
        in_swamps = self._kin_in_swamps[:n]
        is_at_homes = self._kin_is_at_homes[:n]
        foods_today = self._kin_foods_today[:n]
        max_speeds = self._kin_max_speeds[:n]
        body_radii = self._kin_body_radii[:n]
        basal_costs = self._kin_basal_costs[:n]
        has_groups = self._kin_has_groups[:n]
        home_xs = self._kin_home_xs[:n]
        home_ys = self._kin_home_ys[:n]
        home_radii_arr = self._kin_home_radii[:n]
        hit_rocks = self._kin_hit_rocks[:n]
        speed_norms = self._kin_speed_norms[:n]

        if not getattr(self, "_kin_populated", False):
            for i, b in enumerate(living_blobs):
                h_obj = home_objs[i]
                xs[i] = b.x
                ys[i] = b.y
                vxs[i] = b.vx
                vys[i] = b.vy
                headings[i] = b.heading
                energies[i] = b.energy
                in_swamps[i] = b.in_swamp
                is_at_homes[i] = b.is_at_home
                foods_today[i] = b.food_today
                max_speeds[i] = b.max_speed
                body_radii[i] = b.body_radius
                basal_costs[i] = b.basal_cost
                has_groups[i] = (b._group_id is not None)
                home_xs[i] = h_obj.x if h_obj is not None else b.home_x
                home_ys[i] = h_obj.y if h_obj is not None else b.home_y
                home_radii_arr[i] = home_radii[i]
                if getattr(b, "_kin_ref", None) is None:
                    b._kin_ref = self
                    b._kin_idx = i
                    b._sim_idx = i
                else:
                    b._kin_idx = i
        else:
            is_at_homes[:] = (self._kin_is_at_home_int[:n] != 0)
            for i, b in enumerate(living_blobs):
                if getattr(b, "_kin_ref", None) is None:
                    b._kin_ref = self
                b._kin_idx = i
        self._kin_populated = False

        apply_kinematics_and_metabolism_batch_numba(
            xs, ys, vxs, vys, headings, energies,
            in_swamps, is_at_homes, foods_today,
            max_speeds, body_radii, basal_costs, has_groups,
            home_xs, home_ys, home_radii_arr,
            actions_batch,
            self._rock_x1, self._rock_y1, self._rock_dx, self._rock_dy, self._rock_inv_len_sq, self._rock_r,
            self._swamp_x1, self._swamp_y1, self._swamp_dx, self._swamp_dy, self._swamp_inv_len_sq, self._swamp_r,
            self.world_cfg.width, self.world_cfg.height,
            self.world_cfg.reproduction_food_need, self.world_cfg.survival_food_need,
            float(self.world_cfg.ticks_per_day), self.season_info[4], dt_scale, night_mode,
            hit_rocks, speed_norms,
            egocentric_mode=self.world_cfg.egocentric_frame,
        )

        # Elimination of 8 .tolist() array allocations and 16,800 pure-Python attribute assignments!
        # Blob entities access kinematics/metabolism properties lazily via persistent _kin_* buffers.

        hit_indices = np.flatnonzero(hit_rocks)
        coll_count = len(hit_indices)
        if coll_count > 0:
            for idx in hit_indices:
                living_blobs[idx].pending_rl_reward -= 0.015
            self.telemetry.daily_counters["rock_collisions"] += coll_count

        return speed_norms, hit_rocks

    def _ensure_kin_capacity(self, n: int) -> None:
        if n > self._kin_cap:
            old_cap = self._kin_cap
            new_cap = max(n, int(self._kin_cap * 1.5), 1024)
            self._kin_cap = new_cap
            self._kin_static_dirty = True
            def resize_arr(attr_name, dtype):
                arr = getattr(self, attr_name, None)
                new_arr = np.empty(new_cap, dtype=dtype)
                if arr is not None and old_cap > 0 and len(arr) > 0:
                    copy_len = min(old_cap, len(arr), new_cap)
                    new_arr[:copy_len] = arr[:copy_len]
                return new_arr
            self._kin_xs = resize_arr("_kin_xs", np.float64)
            self._kin_ys = resize_arr("_kin_ys", np.float64)
            self._kin_vxs = resize_arr("_kin_vxs", np.float64)
            self._kin_vys = resize_arr("_kin_vys", np.float64)
            self._kin_headings = resize_arr("_kin_headings", np.float32)
            self._kin_energies = resize_arr("_kin_energies", np.float32)
            self._kin_in_swamps = resize_arr("_kin_in_swamps", np.bool_)
            self._kin_is_at_homes = resize_arr("_kin_is_at_homes", np.bool_)
            self._kin_foods_today = resize_arr("_kin_foods_today", np.float32)
            self._kin_max_speeds = resize_arr("_kin_max_speeds", np.float32)
            self._kin_body_radii = resize_arr("_kin_body_radii", np.float32)
            self._kin_basal_costs = resize_arr("_kin_basal_costs", np.float32)
            self._kin_has_groups = resize_arr("_kin_has_groups", np.bool_)
            self._kin_home_xs = resize_arr("_kin_home_xs", np.float64)
            self._kin_home_ys = resize_arr("_kin_home_ys", np.float64)
            self._kin_home_radii = resize_arr("_kin_home_radii", np.float32)
            self._kin_hit_rocks = resize_arr("_kin_hit_rocks", np.uint8)
            self._kin_speed_norms = resize_arr("_kin_speed_norms", np.float32)
            self._kin_gene_sizes = resize_arr("_kin_gene_sizes", np.float32)
            self._kin_moneys = resize_arr("_kin_moneys", np.float32)
            self._kin_reputations = resize_arr("_kin_reputations", np.float32)
            self._kin_peer_reps = resize_arr("_kin_peer_reps", np.float32)
            self._kin_is_at_home_int = resize_arr("_kin_is_at_home_int", np.int32)
            self._kin_sprints = resize_arr("_kin_sprints", np.float32)
            self._kin_mem_xs = resize_arr("_kin_mem_xs", np.float64)
            self._kin_mem_ys = resize_arr("_kin_mem_ys", np.float64)
            self._kin_mem_strengths = resize_arr("_kin_mem_strengths", np.float64)
            self._kin_gids = resize_arr("_kin_gids", np.int32)
            self._kin_hids = resize_arr("_kin_hids", np.int32)
            self._kin_tcooldowns = resize_arr("_kin_tcooldowns", np.int32)
            self._kin_surplus_foods = resize_arr("_kin_surplus_foods", np.float32)
            self._kin_ccooldowns = resize_arr("_kin_ccooldowns", np.int32)
            self._kin_rcooldowns = resize_arr("_kin_rcooldowns", np.int32)
            self._kin_age_ticks = resize_arr("_kin_age_ticks", np.int32)
            self._kin_ticks_since_meal = resize_arr("_kin_ticks_since_meal", np.int32)
            self._kin_vision_radii = resize_arr("_kin_vision_radii", np.float32)
            self._kin_ids = resize_arr("_kin_ids", np.float32)
            self._kin_gene_rgbs = resize_arr("_kin_gene_rgbs", np.uint32)
            self._kin_drive_energy = resize_arr("_kin_drive_energy", np.float32)
            self._kin_drive_pain = resize_arr("_kin_drive_pain", np.float32)
            self._kin_drive_social = resize_arr("_kin_drive_social", np.float32)
            self._kin_plasticity_lr = resize_arr("_kin_plasticity_lr", np.float32)
            self._step_eff_visions = resize_arr("_step_eff_visions", np.float32)
            self._step_blob_foods_ok = resize_arr("_step_blob_foods_ok", np.float32)
            self._step_blob_moneys_ok = resize_arr("_step_blob_moneys_ok", np.float32)
            self._step_blob_groups = resize_arr("_step_blob_groups", np.int32)
            self._step_home_objs = [None] * new_cap
            self._step_home_radii_list = [48.0] * new_cap

    def _compute_movement_shaping_reward(
        self,
        blob: Blob,
        move_x: float,
        move_y: float,
        dx_food: float,
        dy_food: float,
        prox_food: float,
        dx_oasis: float,
        dy_oasis: float,
        local_fertility: float,
        dx_home: float,
        dy_home: float,
        tod: float,
        repro_target: float,
        vx: float = 0.0,
        vy: float = 0.0,
        max_speed: float = 1.0,
    ) -> float:
        """
        Reward intelligent foraging & navigation (NO blind straight-line speed reward!):
        - Reward approaching visible food items based on kinematic velocity & intended action.
        - When no food is immediately visible, reward following Oasis Scent / Memory toward fertile oases.
        - Penalize lingering in overgrazed barren nurseries when hungry.
        - Penalize wading through energy-draining swamps when avoidable.
        - Reward returning home in dusk/evening or once sated.
        """
        reward = 0.0
        if blob.in_swamp:
            reward -= 0.018

        # Blend kinematic velocity with action direction for physical progress & coasting reward
        inv_spd = 1.0 / max(1e-4, max_speed)
        eff_vx = 0.5 * move_x + 0.5 * (vx * inv_spd)
        eff_vy = 0.5 * move_y + 0.5 * (vy * inv_spd)

        if blob.food_today < repro_target:
            if prox_food > 0.0:
                approach = eff_vx * dx_food + eff_vy * dy_food
                if approach > 0.0:
                    reward += self.prefs.pref_food * 0.068 * approach
            else:
                if local_fertility < -0.35:
                    reward -= 0.008 * self.prefs.pref_exploration
                if local_fertility < 0.25:
                    oasis_align = eff_vx * dx_oasis + eff_vy * dy_oasis
                    if oasis_align > 0.0:
                        reward += self.prefs.pref_exploration * 0.052 * oasis_align

        if tod >= 0.60 or blob.food_today >= repro_target:
            if blob.is_at_home:
                reward += self.prefs.pref_home * 0.038
            else:
                home_align = eff_vx * dx_home + eff_vy * dy_home
                if home_align > 0.0:
                    reward += self.prefs.pref_home * 0.058 * home_align
        return reward

    def _compute_movement_shaping_reward_batch(
        self,
        in_swamps: np.ndarray,
        is_at_homes: np.ndarray,
        foods_today: np.ndarray,
        actions_batch: np.ndarray,
        dx_foods: np.ndarray,
        dy_foods: np.ndarray,
        prox_food: np.ndarray,
        oasis_vecs: np.ndarray,
        dx_homes: np.ndarray,
        dy_homes: np.ndarray,
        tod: float,
        repro_target: float,
        vxs: Optional[np.ndarray] = None,
        vys: Optional[np.ndarray] = None,
        max_speeds: Optional[np.ndarray] = None,
        disp_xs: Optional[np.ndarray] = None,
        disp_ys: Optional[np.ndarray] = None,
        d_norms: Optional[np.ndarray] = None,
        dist_homes: Optional[np.ndarray] = None,
        pre_headings: Optional[np.ndarray] = None,
        rel_food_wx: Optional[np.ndarray] = None,
        rel_food_wy: Optional[np.ndarray] = None,
        rel_home_wx: Optional[np.ndarray] = None,
        rel_home_wy: Optional[np.ndarray] = None,
        headings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Exact Canonical Potential-Based Reward Shaping (PBRS, gamma=1.0) & smooth diurnal arbitration.
        - F(s, s') = Phi(s') - Phi(s) = (min(d_before, D_norm) - min(d_after, D_norm)) / D_norm.
        - Uses exact Euclidean distance difference in world coordinates (independent of heading).
        - Telescopes on any closed path to EXACTLY 0.0 (no circular Taylor-approximation leak).
        - Smooth, continuous ETA- and hunger-based homing arbitration without artificial cliffs.
        """
        n = len(in_swamps)
        pref_food = self.prefs.pref_food
        pref_home = self.prefs.pref_home
        pref_exp = self.prefs.pref_exploration

        reward = np.where(in_swamps, -0.018, 0.0)

        speeds = max_speeds if max_speeds is not None else np.full(n, 4.2, dtype=np.float32)

        if self.world_cfg.egocentric_frame and headings is not None:
            c = np.cos(headings)
            s = np.sin(headings)
            inv_speed = 1.0 / np.maximum(1e-4, speeds)
            vx_arr = vxs if vxs is not None else np.zeros(n, dtype=np.float32)
            vy_arr = vys if vys is not None else np.zeros(n, dtype=np.float32)
            eff_vx = (vx_arr * c + vy_arr * s) * inv_speed
            eff_vy = (-vx_arr * s + vy_arr * c) * inv_speed
            dx_step = eff_vx
            dy_step = eff_vy
        elif disp_xs is not None and disp_ys is not None:
            dx_step = disp_xs
            dy_step = disp_ys
        elif vxs is not None and vys is not None:
            dx_step = vxs
            dy_step = vys
        else:
            dx_step = np.zeros(n, dtype=np.float32)
            dy_step = np.zeros(n, dtype=np.float32)

        if d_norms is None:
            d_norm_f = np.full(n, 215.0, dtype=np.float32)
        else:
            d_norm_f = np.maximum(50.0, d_norms)

        tpd = float(self.world_cfg.ticks_per_day)
        ticks_to_night = max(0.0, (0.74 - tod) * tpd)
        speeds = max_speeds if max_speeds is not None else np.full(n, 4.2, dtype=np.float32)
        cruise_speeds = np.maximum(0.5, speeds * 0.75)

        # -------------------------------------------------------------
        # 1. Exact Home Distance & Diurnal Weights
        # -------------------------------------------------------------
        if rel_home_wx is not None and rel_home_wy is not None:
            d_homes_before = np.sqrt(rel_home_wx * rel_home_wx + rel_home_wy * rel_home_wy)
            d_homes_after = np.sqrt((rel_home_wx - dx_step) ** 2 + (rel_home_wy - dy_step) ** 2)
        elif dist_homes is not None:
            d_homes_before = dist_homes
            proj_h = dx_step * dx_homes + dy_step * dy_homes
            d_homes_after = np.maximum(0.0, d_homes_before - proj_h)
        else:
            d_homes_before = np.full(n, 300.0, dtype=np.float32)
            d_homes_after = d_homes_before

        eta_homes = (d_homes_before / cruise_speeds) * 1.3
        urgency_home = np.where(
            ticks_to_night <= 0.0,
            1.0,
            np.clip(eta_homes / max(1.0, ticks_to_night), 0.0, 1.0)
        )

        hunger_ratio = np.clip((repro_target - foods_today) / max(0.1, repro_target), 0.0, 1.0)
        w_home = np.clip((1.0 - hunger_ratio * 0.75) * urgency_home, 0.0, 1.0)
        w_food = np.clip(hunger_ratio * (1.0 - 0.70 * urgency_home), 0.0, 1.0)

        # -------------------------------------------------------------
        # 2. Exact Canonical Food PBRS (World Euclidean Distance)
        # -------------------------------------------------------------
        has_food = prox_food > 0.0
        if rel_food_wx is not None and rel_food_wy is not None:
            d_before_food = np.sqrt(rel_food_wx * rel_food_wx + rel_food_wy * rel_food_wy)
            d_after_food = np.sqrt((rel_food_wx - dx_step) ** 2 + (rel_food_wy - dy_step) ** 2)
        else:
            d_before_food = (np.float32(1.0) - prox_food) * d_norm_f
            if pre_headings is not None:
                cos_th = np.cos(pre_headings)
                sin_th = np.sin(pre_headings)
                rf_wx = cos_th * dx_foods - sin_th * dy_foods
                rf_wy = sin_th * dx_foods + cos_th * dy_foods
                vec_fx = d_before_food * rf_wx
                vec_fy = d_before_food * rf_wy
                d_after_food = np.sqrt((vec_fx - dx_step) ** 2 + (vec_fy - dy_step) ** 2)
            else:
                proj_f = dx_step * dx_foods + dy_step * dy_foods
                d_after_food = np.maximum(0.0, d_before_food - proj_f)

        delta_phi_food = np.where(
            has_food,
            (np.minimum(d_before_food, d_norm_f) - np.minimum(d_after_food, d_norm_f)) / d_norm_f,
            0.0
        )
        reward += w_food * (pref_food * 0.045) * delta_phi_food

        # -------------------------------------------------------------
        # 3. Oasis PBRS & Exploration Signal (when no food is visible)
        # -------------------------------------------------------------
        local_fertility = oasis_vecs[:, 2]
        d_norm_oasis = np.float32(650.0)
        if pre_headings is not None:
            cos_th = np.cos(pre_headings)
            sin_th = np.sin(pre_headings)
            ro_wx = cos_th * oasis_vecs[:, 0] - sin_th * oasis_vecs[:, 1]
            ro_wy = sin_th * oasis_vecs[:, 0] + cos_th * oasis_vecs[:, 1]
            proj_oasis = dx_step * ro_wx + dy_step * ro_wy
        else:
            proj_oasis = dx_step * oasis_vecs[:, 0] + dy_step * oasis_vecs[:, 1]

        delta_phi_oasis = np.clip(proj_oasis / d_norm_oasis, -0.05, 0.05)
        barren_penalty = np.where(local_fertility < -0.35, -0.002 * pref_exp, 0.0)
        oasis_reward = np.where(
            local_fertility < 0.35,
            pref_exp * 0.025 * delta_phi_oasis + barren_penalty,
            0.0
        )
        reward += np.where(~has_food, w_food * oasis_reward, 0.0)

        # -------------------------------------------------------------
        # 4. Exact Canonical Home PBRS
        # -------------------------------------------------------------
        d_norm_h = np.maximum(100.0, d_norm_f * 2.0)
        delta_phi_home = np.where(
            is_at_homes,
            0.0,
            (np.minimum(d_homes_before, d_norm_h) - np.minimum(d_homes_after, d_norm_h)) / d_norm_h
        )
        reward += w_home * (pref_home * 0.035) * delta_phi_home

        return reward
    def _try_colonize_or_relocate(self, blob: Blob, tod: float) -> bool:
        """
        Allow a blob to actively migrate into a better-situated oasis home or pioneer a new home
        when it discovers a fertile oasis far from its current (or barren/overgrazed) home.
        """
        if blob.colonize_cooldown > 0 or not self.oases:
            return False

        dhx = blob.x - blob.home_x
        dhy = blob.y - blob.home_y
        dist_to_home_sq = dhx * dhx + dhy * dhy
        if dist_to_home_sq <= 340.0 * 340.0:
            return False

        dist_to_home = math.sqrt(dist_to_home_sq)
        wants_to_settle = blob.colonize_desire > 0.06 or (tod >= 0.54 and dist_to_home > 720.0)
        if not wants_to_settle:
            return False

        # Find if the blob is currently at a fertile oasis
        bx, by = blob.x, blob.y
        nearest_oasis: Optional[OasisNode] = None
        best_o_d2 = float("inf")
        for o in self.oases.values():
            odx = bx - o.x
            ody = by - o.y
            od2 = odx * odx + ody * ody
            if od2 < best_o_d2:
                best_o_d2 = od2
                nearest_oasis = o
        if nearest_oasis is None:
            return False
        dist_to_oasis = math.sqrt(best_o_d2)
        if dist_to_oasis > nearest_oasis.radius * 1.38 or nearest_oasis.fertility < 0.42:
            return False

        cur_home = self.homes.get(blob.home_id)
        cur_fert = cur_home.fertility_bonus if cur_home is not None else 0.0
        cur_zone = getattr(cur_home, "zone", "cradle") if cur_home is not None else "cradle"

        # Worth relocating if far from current home, moving out of the Cradle, or current home is overgrazed
        worth_relocating = (
            dist_to_home > 580.0
            or (cur_zone == "cradle" and getattr(nearest_oasis, "ring", "outer") == "outer")
            or (dist_to_home > 340.0 and cur_fert + 0.30 < nearest_oasis.fertility)
        )
        if not worth_relocating:
            return False

        # 1. Check if there is already an open home near this oasis
        max_o_reach = nearest_oasis.radius * 1.55
        cand_homes = getattr(self, "_oasis_to_homes", {}).get(nearest_oasis.id, self.homes.values())
        open_oasis_homes = [
            h
            for h in cand_homes
            if not h.is_full
            and (h.group_id is None or h.group_id == blob.group_id)
            and abs(h.x - nearest_oasis.x) <= max_o_reach
            and abs(h.y - nearest_oasis.y) <= max_o_reach
            and math.hypot(h.x - nearest_oasis.x, h.y - nearest_oasis.y) <= max_o_reach
        ]
        if open_oasis_homes:
            target_home = min(open_oasis_homes, key=lambda h: math.hypot(h.x - blob.x, h.y - blob.y))
            self._relocate_blob_home(blob, target_home.id)
            if blob.group_id is not None and blob.group_id in self.groups:
                self._federate_home_into_group(target_home, self.groups[blob.group_id])
            blob.colonize_cooldown = 160
            blob.migrations_made += 1
            self.total_migrations_all_time += 1
            self.telemetry.on_migration(blob, target_home, nearest_oasis, pioneer=False)
            mig_rew = 0.50 * (self.prefs.pref_home + self.prefs.pref_exploration)
            blob.cumulative_reward += mig_rew
            blob.pending_rl_reward += mig_rew
            return True

        # 2. Otherwise, found a new Pioneer Home on the shore of this fertile oasis!
        max_homes_cap = getattr(self.world_cfg, "max_homes_cap", 1500)
        if len(self.homes) >= max_homes_cap or blob.energy < 34.0:
            return False
        can_pay_gold = blob.money >= 4.5
        can_pay_food = blob.total_food >= 0.85
        if not (can_pay_gold or can_pay_food):
            return False

        if can_pay_gold:
            blob.money = round(blob.money - 4.0, 2)
        else:
            blob.deduct_traded_food(0.35)
        blob.energy = max(18.0, blob.energy - 8.0)

        grp = self.groups.get(blob.group_id) if blob.group_id is not None else None
        max_res = max(2, self.world_cfg.max_residents_per_home)
        new_home = self._construct_new_partner_home(grp, blob.x, blob.y, max_res)
        self._relocate_blob_home(blob, new_home.id)
        blob.colonize_cooldown = 210
        blob.migrations_made += 1
        self.total_migrations_all_time += 1
        self.telemetry.on_migration(blob, new_home, nearest_oasis, pioneer=True)

        frontier_bonus = 1.25 if getattr(new_home, "zone", "frontier") == "frontier" else 1.0
        col_rew = 0.80 * frontier_bonus * (self.prefs.pref_home + self.prefs.pref_exploration)
        blob.cumulative_reward += col_rew
        blob.pending_rl_reward += col_rew
        self.log_event(
            "world",
            f"🏕️ Blob #{blob.id} gründet Pionier-Haus #{new_home.id} an {nearest_oasis.name} (Fruchtbarkeit {int(nearest_oasis.fertility * 100)}%)!",
        )
        return True

    def _evaluate_step_interactions(
        self,
        blob: Blob,
        move_x: float,
        move_y: float,
        dx_food: float,
        dy_food: float,
        prox_food: float,
        dx_oasis: float,
        dy_oasis: float,
        local_fertility: float,
        dx_home: float,
        dy_home: float,
        nearest_peer: Optional[Blob],
        nearest_peer_dist: float,
        home_obj: Optional[Home],
        tod: float,
        repro_target: float,
    ) -> float:
        """Compute step shaping reward and execute social, trading, and colonization interactions."""
        step_reward = self._compute_movement_shaping_reward(
            blob,
            move_x,
            move_y,
            dx_food,
            dy_food,
            prox_food,
            dx_oasis,
            dy_oasis,
            local_fertility,
            dx_home,
            dy_home,
            tod,
            repro_target,
            vx=blob.vx,
            vy=blob.vy,
            max_speed=blob.max_speed,
        )

        if blob.colonize_cooldown == 0 and (blob.colonize_desire > 0.06 or tod >= 0.54):
            self._try_colonize_or_relocate(blob, tod)

        if nearest_peer is not None and nearest_peer_dist < 44.0:
            if blob.social_desire > 0.05 and nearest_peer.social_desire > -0.05:
                if self._handle_social_encounter(blob, nearest_peer):
                    step_reward += self.prefs.pref_social * 0.45

            if blob.trade_cooldown == 0 and nearest_peer.trade_cooldown == 0:
                traded = False
                if blob.trade_desire > -0.15 and nearest_peer.trade_desire > -0.25:
                    traded = self._try_peer_trade(blob, nearest_peer)
                if not traded and (
                    blob.group_id is None or blob.group_id != nearest_peer.group_id
                ):
                    self._try_peer_raid(blob, nearest_peer)

        if blob.is_at_home and home_obj is not None and blob.trade_cooldown == 0:
            self._try_village_market_trade(blob, home_obj)

        return step_reward

    def _check_midday_starvation(self, living_blobs: List[Blob]) -> None:
        """Check for mid-day starvation and allow emergency ration purchase at home."""
        n = len(living_blobs)
        if n == 0:
            return
        energies = self._kin_energies[:n] if len(self._kin_energies) >= n else None
        max_starve = self.world_cfg.max_starvation_ticks
        for idx in range(n):
            blob = living_blobs[idx]
            if not blob.alive:
                continue
            e = energies[idx] if energies is not None else blob.energy
            if e > 0.0 and blob.ticks_since_last_meal < max_starve:
                continue

            if blob.money >= self.world_cfg.base_food_price * 0.8 and blob.is_at_home:
                depot_home = self._find_settlement_home_with_food(blob, min_food=0.4)
                if depot_home is not None:
                    rep_buy_mult = self._get_reputation_buy_multiplier(blob)
                    cost = round(self.world_cfg.base_food_price * 0.4 * rep_buy_mult, 2)
                    if blob.money >= cost:
                        depot_home.food_depot = round(depot_home.food_depot - 0.4, 2)
                        depot_home.gold_vault = round(depot_home.gold_vault + cost, 2)
                        blob.money = round(blob.money - cost, 2)
                        blob.food_today = round(blob.food_today + 0.4, 3)
                        blob.energy = min(blob.max_energy, blob.energy + 38.0)
                        blob.ticks_since_last_meal = 0
                        blob.trades_made += 1
                        self.trades_today += 1
                        self.total_trades_all_time += 1
                        self.telemetry.on_emergency_ration_buy(blob, depot_home, 0.4, cost)
                        continue

            self._kill_blob(blob, reason="verhungert (Energie leer)")

    def _modulate_context_vectors(
        self,
        blob: Blob,
        dx_food: float,
        dy_food: float,
        dx_home: float,
        dy_home: float,
        tod: float,
        surv_target: float,
        repro_target: float,
    ) -> Tuple[float, float, float, float]:
        """
        Provide honest, unflipped directional vectors scaled only by user preference weights,
        so the neural network must learn context-dependent behavior via `time_of_day` and `satiety`.
        """
        home_gain = 0.85 + 0.35 * self.prefs.pref_home
        food_gain = 0.85 + 0.25 * self.prefs.pref_food
        if blob.is_at_home:
            return dx_food * food_gain, dy_food * food_gain, 0.0, 0.0
        return dx_food * food_gain, dy_food * food_gain, dx_home * home_gain, dy_home * home_gain

    def advance_ticks(self, total_ticks: int) -> None:
        """
        Advance the simulation by `total_ticks` (1 .. 500+).
        For 1..15 ticks, executes exact single-tick steps.
        For high turbo speeds (45x..500x), uses event-aligned adaptive multi-tick sub-stepping
        so midday food spawns and end-of-day accounting always fire on exact boundaries while
        maintaining a smooth 30 FPS frame time.
        """
        if total_ticks <= 10:
            for _ in range(max(1, total_ticks)):
                self.step(1)
            return

        base_dt = max(2, min(45, (total_ticks + 11) // 12))
        remaining = total_ticks
        while remaining > 0:
            done = self.step(min(remaining, base_dt))
            remaining -= done

    def _compute_safe_dt(self, requested_dt: int) -> int:
        if requested_dt <= 1:
            return 1
        tpd = self.world_cfg.ticks_per_day
        midday = tpd // 2
        if self.tick_in_day < midday:
            return max(1, min(requested_dt, midday - self.tick_in_day))
        return max(1, min(requested_dt, tpd - self.tick_in_day))

    def _prepare_step_observations(
        self,
        living_blobs: List[Blob],
        food_vecs: np.ndarray,
        peer_vecs: np.ndarray,
        oasis_vecs: np.ndarray,
        feeler_vecs: np.ndarray,
        tod: float,
        surv_target: float,
        repro_target: float,
        dt: int,
        nearest_peers: Optional[List[Optional[Blob]]] = None,
    ) -> Tuple[
        np.ndarray,
        List[Optional[Home]],
        List[float],
        List[float],
        List[float],
        List[float],
        List[float],
        int,
    ]:
        n_living = len(living_blobs)
        if getattr(self, "_obs_batch_cache", None) is None or len(self._obs_batch_cache) < n_living:
            self._obs_batch_cache = np.empty((max(n_living + 256, 1024), 26), dtype=np.float32)
        obs_batch = self._obs_batch_cache[:n_living]
        obs_batch[:, 2] = food_vecs[:, 2]
        obs_batch[:, 6:12] = peer_vecs
        obs_batch[:, 15] = np.float32(tod * 2.0 - 1.0)
        obs_batch[:, 16:19] = oasis_vecs
        obs_batch[:, 21:24] = feeler_vecs

        food_gain = 0.85 + 0.25 * self.prefs.pref_food
        obs_batch[:, 0] = food_vecs[:, 0] * food_gain
        obs_batch[:, 1] = food_vecs[:, 1] * food_gain

        sel_id = self.selected_blob_id
        inspect_idx = -1
        homes_dict = self.homes
        oases_dict = self.oases

        home_gain = 0.85 + 0.35 * self.prefs.pref_home
        inv_900 = 1.0 / 900.0
        inv_repro = 1.0 / repro_target

        self._ensure_kin_capacity(n_living)
        if getattr(self, "_kin_static_dirty", True):
            self._sync_static_blob_kin_arrays(living_blobs)
            self._kin_static_dirty = False

        xs = self._kin_xs[:n_living]
        ys = self._kin_ys[:n_living]
        home_xs = self._kin_home_xs[:n_living]
        home_ys = self._kin_home_ys[:n_living]
        home_radii_arr = self._kin_home_radii[:n_living]
        gene_sizes = self._kin_gene_sizes[:n_living]
        moneys = self._kin_moneys[:n_living]
        reputations = self._kin_reputations[:n_living]
        foods_today = self._kin_foods_today[:n_living]
        energies = self._kin_energies[:n_living]
        peer_reps = self._kin_peer_reps[:n_living]
        is_at_home_out = self._kin_is_at_home_int[:n_living]
        home_objs = self._step_home_objs[:n_living]
        home_radii_list = self._step_home_radii_list[:n_living]
        peer_reps.fill(0.0)

        mem_xs = self._kin_mem_xs[:n_living]
        mem_ys = self._kin_mem_ys[:n_living]
        mem_strengths = self._kin_mem_strengths[:n_living]
        mem_strengths.fill(0.0)

        gids = self._kin_gids[:n_living]
        hids = self._kin_hids[:n_living]
        tcds = self._kin_tcooldowns[:n_living]
        sufs = self._kin_surplus_foods[:n_living]

        substep = self._substep_count
        groups = self.groups
        overgrazed_oases = getattr(self, "_cached_overgrazed_oases", None)
        if overgrazed_oases is None:
            overgrazed_oases = {oid: o for oid, o in oases_dict.items() if o.is_overgrazed}
            self._cached_overgrazed_oases = overgrazed_oases

        if not getattr(self, "_in_step_once", False):
            for idx, blob in enumerate(living_blobs):
                xs[idx] = blob.x
                ys[idx] = blob.y
                self._kin_headings[idx] = blob.heading
                h_obj = blob.home_obj
                if h_obj is None or h_obj.id != blob.home_id:
                    h_obj = homes_dict.get(blob.home_id)
                    blob.home_obj = h_obj
                home_objs[idx] = h_obj
                hr = h_obj.radius if h_obj is not None else 48.0
                home_radii_list[idx] = hr
                home_radii_arr[idx] = hr
                home_xs[idx] = blob.home_x
                home_ys[idx] = blob.home_y
                foods_today[idx] = blob.food_today
                energies[idx] = blob.energy
                moneys[idx] = blob.money
                reputations[idx] = blob._reputation
                gene_sizes[idx] = blob.genes.size
                mem_xs[idx] = blob.memory_x
                mem_ys[idx] = blob.memory_y
                mem_strengths[idx] = blob.memory_strength
                bgid = blob._group_id
                gids[idx] = bgid if bgid is not None else -1
                hids[idx] = blob.home_id
                tcds[idx] = blob.trade_cooldown
                sufs[idx] = blob.surplus_food
        else:
            # Contiguous in-place updates (0.01 ms vs 2.70 ms)
            self._kin_age_ticks[:n_living] += dt
            self._kin_ticks_since_meal[:n_living] += dt
            np.maximum(0, tcds - dt, out=tcds)
            ccds = self._kin_ccooldowns[:n_living]
            np.maximum(0, ccds - dt, out=ccds)
            rcds = self._kin_rcooldowns[:n_living]
            np.maximum(0, rcds - dt, out=rcds)

            # Overgrazed oasis spatial memory decay (only if overgrazed oases exist)
            if overgrazed_oases:
                for idx in range(n_living):
                    ms = mem_strengths[idx]
                    if ms > 0.02:
                        blob = living_blobs[idx]
                        oid = blob.memory_oasis_id
                        if oid is not None and oid in overgrazed_oases:
                            rem_oasis = overgrazed_oases[oid]
                            dmx = blob.memory_x - xs[idx]
                            dmy = blob.memory_y - ys[idx]
                            if (dmx * dmx + dmy * dmy) <= (rem_oasis.radius * rem_oasis.radius):
                                ms = round(ms * 0.65, 3)
                                blob.memory_strength = ms
                                mem_strengths[idx] = ms

            stride_offset = (-substep) & 15
            for idx in range(stride_offset, n_living, 16):
                blob = living_blobs[idx]
                bgid = blob._group_id
                if bgid is not None and not self._kin_is_at_homes[idx]:
                    grp = groups.get(bgid)
                    if grp is not None and len(grp.home_ids) > 1:
                        near_home = self._get_nearest_settlement_home(blob)
                        if near_home is not None:
                            blob.home_obj = near_home
                            blob.home_x = near_home.x
                            blob.home_y = near_home.y
                            home_xs[idx] = near_home.x
                            home_ys[idx] = near_home.y
                            home_radii_arr[idx] = near_home.radius
                            home_objs[idx] = near_home
                            home_radii_list[idx] = near_home.radius

        if sel_id is not None:
            for idx in range(n_living):
                if living_blobs[idx].id == sel_id:
                    inspect_idx = idx
                    break

        p_idxs = getattr(self, "_last_min_p_idxs", None)
        has_valid_p = False
        if p_idxs is not None and len(p_idxs) >= n_living:
            p_sub = p_idxs[:n_living]
            valid_mask = (p_sub >= 0) & (p_sub < n_living)
            if np.any(valid_mask):
                peer_reps[valid_mask] = reputations[p_sub[valid_mask]]
                has_valid_p = True
        if not has_valid_p and nearest_peers is not None:
            for idx in range(n_living):
                np_peer = nearest_peers[idx]
                if np_peer is not None:
                    peer_reps[idx] = np.float32(np_peer._reputation)

        self._kin_populated = True

        headings = self._kin_headings[:n_living]
        prep_step_observations_numba(
            obs_batch,
            xs,
            ys,
            headings,
            home_xs,
            home_ys,
            home_radii_arr,
            foods_today,
            energies,
            gene_sizes,
            moneys,
            reputations,
            peer_reps,
            float(home_gain),
            float(inv_900),
            float(inv_repro),
            is_at_home_out,
            mem_xs,
            mem_ys,
            mem_strengths,
        )

        return (
            obs_batch,
            home_objs,
            home_radii_list,
            obs_batch[:, 0],
            obs_batch[:, 1],
            obs_batch[:, 3],
            obs_batch[:, 4],
            inspect_idx,
        )

    def _snapshot_blob_kin_state(self, b: Blob) -> None:
        if getattr(b, "_kin_ref", None) is not None:
            idx = b._kin_idx
            if 0 <= idx < self._kin_cap:
                b._x = float(self._kin_xs[idx])
                b._y = float(self._kin_ys[idx])
                b._vx = float(self._kin_vxs[idx])
                b._vy = float(self._kin_vys[idx])
                b._heading = float(self._kin_headings[idx])
                b._energy = float(self._kin_energies[idx])
                b._in_swamp = bool(self._kin_in_swamps[idx])
                b._is_at_home = bool(self._kin_is_at_homes[idx])
                b._food_today = float(self._kin_foods_today[idx])
                b._surplus_food = float(self._kin_surplus_foods[idx])
                b._money = float(self._kin_moneys[idx])
                b._trade_cooldown = int(self._kin_tcooldowns[idx])
                b._colonize_cooldown = int(self._kin_ccooldowns[idx])
                b._repro_cooldown = int(self._kin_rcooldowns[idx])
                b._age_ticks = int(self._kin_age_ticks[idx])
                b._ticks_since_last_meal = int(self._kin_ticks_since_meal[idx])
                b._reputation = round(float(self._kin_reputations[idx]), 3)
            b._kin_ref = None
            b._kin_idx = -1
        b._sim_idx = -1

    def _compact_living_blobs(self) -> List[Blob]:
        """Compact self.blobs in-place using O(1) swap-remove so surviving blobs do not shift left in BatchedBrainPool."""
        blobs = self.blobs
        n = len(blobs)
        write_idx = 0
        dead_blobs: List[Blob] = []
        valid_cap = self._brain_pool._out_valid_cap
        cap_len = len(valid_cap)
        while write_idx < n:
            if blobs[write_idx].alive:
                blobs[write_idx]._sim_idx = write_idx
                write_idx += 1
            else:
                self._snapshot_blob_kin_state(blobs[write_idx])
                dead_blobs.append(blobs[write_idx])
                n -= 1
                while n > write_idx and not blobs[n].alive:
                    self._snapshot_blob_kin_state(blobs[n])
                    dead_blobs.append(blobs[n])
                    n -= 1
                if write_idx < n:
                    blobs[write_idx] = blobs[n]
                    blobs[write_idx]._sim_idx = write_idx
                    if getattr(blobs[write_idx], "_kin_ref", None) is not None:
                        blobs[write_idx]._kin_idx = write_idx
                    if write_idx < cap_len:
                        valid_cap[write_idx] = False
                    if write_idx < self._kin_cap and n < self._kin_cap:
                        self._kin_max_speeds[write_idx] = self._kin_max_speeds[n]
                        self._kin_body_radii[write_idx] = self._kin_body_radii[n]
                        self._kin_basal_costs[write_idx] = self._kin_basal_costs[n]
                        self._kin_gene_sizes[write_idx] = self._kin_gene_sizes[n]
                        self._kin_has_groups[write_idx] = self._kin_has_groups[n]
                        self._kin_vxs[write_idx] = self._kin_vxs[n]
                        self._kin_vys[write_idx] = self._kin_vys[n]
                        self._kin_headings[write_idx] = self._kin_headings[n]
                        self._kin_in_swamps[write_idx] = self._kin_in_swamps[n]
                        self._step_home_objs[write_idx] = self._step_home_objs[n]
                        self._step_home_radii_list[write_idx] = self._step_home_radii_list[n]
                        self._kin_home_xs[write_idx] = self._kin_home_xs[n]
                        self._kin_home_ys[write_idx] = self._kin_home_ys[n]
                        self._kin_home_radii[write_idx] = self._kin_home_radii[n]
                        self._kin_mem_xs[write_idx] = self._kin_mem_xs[n]
                        self._kin_mem_ys[write_idx] = self._kin_mem_ys[n]
                        self._kin_mem_strengths[write_idx] = self._kin_mem_strengths[n]
                        self._kin_xs[write_idx] = self._kin_xs[n]
                        self._kin_ys[write_idx] = self._kin_ys[n]
                        self._kin_energies[write_idx] = self._kin_energies[n]
                        self._kin_foods_today[write_idx] = self._kin_foods_today[n]
                        self._kin_moneys[write_idx] = self._kin_moneys[n]
                        self._kin_reputations[write_idx] = self._kin_reputations[n]
                        self._kin_gids[write_idx] = self._kin_gids[n]
                        self._kin_hids[write_idx] = self._kin_hids[n]
                        self._kin_tcooldowns[write_idx] = self._kin_tcooldowns[n]
                        self._kin_surplus_foods[write_idx] = self._kin_surplus_foods[n]
                        self._kin_is_at_homes[write_idx] = self._kin_is_at_homes[n]
                        self._kin_is_at_home_int[write_idx] = self._kin_is_at_home_int[n]
                        self._kin_sprints[write_idx] = self._kin_sprints[n]
                        self._kin_peer_reps[write_idx] = self._kin_peer_reps[n]
                        self._kin_hit_rocks[write_idx] = self._kin_hit_rocks[n]
                        self._kin_speed_norms[write_idx] = self._kin_speed_norms[n]
                        self._kin_ccooldowns[write_idx] = self._kin_ccooldowns[n]
                        self._kin_rcooldowns[write_idx] = self._kin_rcooldowns[n]
                        self._kin_age_ticks[write_idx] = self._kin_age_ticks[n]
                        self._kin_ticks_since_meal[write_idx] = self._kin_ticks_since_meal[n]
                        self._kin_vision_radii[write_idx] = self._kin_vision_radii[n]
                        self._kin_ids[write_idx] = self._kin_ids[n]
                        self._kin_gene_rgbs[write_idx] = self._kin_gene_rgbs[n]
                        self._kin_drive_energy[write_idx] = self._kin_drive_energy[n]
                        self._kin_drive_pain[write_idx] = self._kin_drive_pain[n]
                        self._kin_drive_social[write_idx] = self._kin_drive_social[n]
                        self._kin_plasticity_lr[write_idx] = self._kin_plasticity_lr[n]
                    write_idx += 1
        for dead_b in dead_blobs:
            self._unregister_blob(dead_b)
        del blobs[n:]
        if dead_blobs:
            self._last_min_p_idxs = np.full(n, -1, dtype=np.int32)
        return blobs

    def step(self, dt_ticks: int = 1) -> int:
        """Advance the simulation by `dt_ticks` (default 1 tick). Returns actual ticks advanced."""
        if dt_ticks <= 1:
            return self._step_once(1)
        safe_dt = self._compute_safe_dt(dt_ticks)
        if dt_ticks <= safe_dt:
            return self._step_once(dt_ticks)
        remaining = dt_ticks
        total_done = 0
        while remaining > 0:
            sub_dt = self._compute_safe_dt(remaining)
            done = self._step_once(sub_dt)
            total_done += done
            remaining -= done
            if done == 0:
                break
        return total_done

    def _step_once(self, dt: int = 1) -> int:
        self._in_step_once = True
        try:
            return self._do_step_once(dt)
        finally:
            self._in_step_once = False

    def _do_step_once(self, dt: int = 1) -> int:
        dt_scale = float(dt)
        prev_tick_in_day = self.tick_in_day
        self.tick_count += dt
        self.tick_in_day += dt
        self._substep_count += 1

        self.recent_trade_lines = [
            t for t in self.recent_trade_lines if (self.tick_count - t["tick"]) <= 18
        ]
        self.recent_raid_lines = [
            r for r in self.recent_raid_lines if (self.tick_count - r["tick"]) <= 22
        ]

        if self.world_cfg.continuous_lifecycle:
            # Kontinuierlicher Oasen-Wuchs: alle 20 Ticks Nahrung spawnen
            if (prev_tick_in_day // 20) < (self.tick_in_day // 20):
                continuous_spawn = max(2, round(self.world_cfg.daily_food_spawn / (self.world_cfg.ticks_per_day / 20.0)))
                self.spawn_daily_food(continuous_spawn)
            # Alle 60 Ticks sanft die Oasen-Regeneration aktualisieren
            if (prev_tick_in_day // 60) < (self.tick_in_day // 60):
                self._step_oasis_ecology()
        else:
            # Periodic ecological oasis pulse (overgrazing recovery & ripe fruit growth)
            if (prev_tick_in_day // 75) < (self.tick_in_day // 75):
                self._step_oasis_ecology()

            # Amortized dawn food spawning
            if getattr(self, "_pending_food_spawn", 0) > 0:
                chunk = min(1200, self._pending_food_spawn)
                self._pending_food_spawn -= chunk
                self.spawn_daily_food(chunk)

            if self.tick_in_day == self.world_cfg.ticks_per_day // 2:
                self.spawn_daily_food(max(25, self.world_cfg.daily_food_spawn // 3))

        living_blobs = self._compact_living_blobs()
        if not living_blobs:
            self._respawn_from_hall_of_fame()
            living_blobs = self._compact_living_blobs()

        tod = self.time_of_day
        night_mode = self.is_night
        surv_target = self.world_cfg.survival_food_need
        repro_target = max(0.5, self.world_cfg.reproduction_food_need)

        (
            food_vecs,
            peer_vecs,
            nearest_peers,
            nearest_peer_dists,
            oasis_vecs,
            feeler_vecs,
        ) = self._compute_spatial_perception(living_blobs, night_mode, dt=dt)
        (
            obs_batch,
            home_objs,
            home_radii,
            dx_foods,
            dy_foods,
            dx_homes,
            dy_homes,
            inspect_idx,
        ) = self._prepare_step_observations(
            living_blobs,
            food_vecs,
            peer_vecs,
            oasis_vecs,
            feeler_vecs,
            tod,
            surv_target,
            repro_target,
            dt,
            nearest_peers=nearest_peers,
        )
        n_living = len(living_blobs)

        brains = [b.brain for b in living_blobs]
        if self.world_cfg.every_tick_brain:
            cohort_idx = -1
            do_trace = True
        else:
            cohort_idx = self._substep_count % 3
            hidden_dims = self.net_cfg.get_hidden_dims()
            max_dim = max(hidden_dims) if hidden_dims else 0
            do_trace = (max_dim <= 512) or (cohort_idx == 0)

        actions_batch = self._brain_pool.forward_and_trace(
            brains,
            obs_batch,
            0.04,
            self.rng,
            inspect_idx,
            update_traces=do_trace,
            cohort_idx=cohort_idx,
        )
        self.telemetry.record_step_actions(living_blobs, actions_batch, dt)
        pre_xs = self._kin_xs[:n_living].copy()
        pre_ys = self._kin_ys[:n_living].copy()
        pre_headings = self._kin_headings[:n_living].copy()
        prev_energies = self._kin_energies[:n_living].copy() if self.world_cfg.intrinsic_homeostasis else None
        rel_home_wx = self._kin_home_xs[:n_living] - pre_xs
        rel_home_wy = self._kin_home_ys[:n_living] - pre_ys

        speed_norms, hit_rocks = self._apply_kinematics_and_metabolism_batch(
            living_blobs, actions_batch, home_objs, home_radii, night_mode, dt_scale
        )

        disp_xs = self._kin_xs[:n_living] - pre_xs
        disp_ys = self._kin_ys[:n_living] - pre_ys

        if not self.world_cfg.intrinsic_homeostasis:
            eff_oasis_vecs = obs_batch[:, 16:19] if self.world_cfg.egocentric_frame else oasis_vecs
            base_rewards = self._compute_movement_shaping_reward_batch(
                self._kin_in_swamps[:n_living],
                self._kin_is_at_homes[:n_living],
                self._kin_foods_today[:n_living],
                actions_batch,
                dx_foods,
                dy_foods,
                food_vecs[:, 2],
                eff_oasis_vecs,
                dx_homes,
                dy_homes,
                tod,
                repro_target,
                vxs=self._kin_vxs[:n_living],
                vys=self._kin_vys[:n_living],
                max_speeds=self._kin_max_speeds[:n_living],
                disp_xs=disp_xs,
                disp_ys=disp_ys,
                d_norms=self._step_eff_visions[:n_living],
                pre_headings=pre_headings,
                rel_home_wx=rel_home_wx,
                rel_home_wy=rel_home_wy,
                headings=self._kin_headings[:n_living] if self.world_cfg.egocentric_frame else None,
            )
            scaled_reward_vec = base_rewards * dt_scale
            step_rewards = (scaled_reward_vec * 2.5).astype(np.float32)
            srs = scaled_reward_vec.tolist()
        else:
            scaled_reward_vec = None
            step_rewards = None
            srs = None

        sprints_arr = (actions_batch[:, 2] + 1.0) * 0.5
        self._kin_sprints[:n_living] = sprints_arr

        sps = sprints_arr.tolist()
        step_dists = np.hypot(disp_xs, disp_ys).tolist()
        for idx in range(n_living):
            b = living_blobs[idx]
            if srs is not None:
                sr = srs[idx]
                b.last_reward = sr
                b.cumulative_reward += sr
            b.sprint_level = sps[idx]
            b.distance_traveled += step_dists[idx]

        # Vectorized colonize candidate detection using true coordinates
        tod_56 = tod >= 0.56
        dhx = self._kin_xs[:n_living] - self._kin_home_xs[:n_living]
        dhy = self._kin_ys[:n_living] - self._kin_home_ys[:n_living]
        dist_c_sq = dhx * dhx + dhy * dhy
        col_cand_mask = (actions_batch[:, 5] > 0.08) & (dist_c_sq > 122500.0)
        if tod_56:
            col_cand_mask |= (dist_c_sq > 608400.0) & (actions_batch[:, 5] > 0.15)
        col_candidate_indices = np.flatnonzero(col_cand_mask)
        if len(col_candidate_indices) > 12:
            desires = actions_batch[col_candidate_indices, 5]
            col_candidate_indices = col_candidate_indices[np.argsort(-desires)[:12]]
        for c_idx in col_candidate_indices:
            c_blob = living_blobs[c_idx]
            if c_blob.colonize_cooldown == 0:
                c_blob.colonize_desire = float(actions_batch[c_idx, 5])
                success = self._try_colonize_or_relocate(c_blob, tod)
                if not success:
                    c_blob.colonize_cooldown = 15

        # -------------------------------------------------------------
        # 2. Vectorized Numba Peer Filter & Active Interactions
        # -------------------------------------------------------------
        peer_reach = 44.0 + (dt_scale - 1.0) * 2.2
        if n_living > 1 and getattr(self, "_last_min_p_idxs", None) is not None and len(self._last_min_p_idxs) >= n_living:
            min_p_idxs = self._last_min_p_idxs[:n_living]
            active_i, active_j, active_kind = filter_active_peers_numba(
                n_living,
                min_p_idxs,
                nearest_peer_dists[:n_living],
                peer_reach,
                actions_batch[:, 4],
                actions_batch[:, 3],
                self._kin_foods_today[:n_living],
                self._kin_surplus_foods[:n_living],
                self._kin_moneys[:n_living],
                self._kin_energies[:n_living],
                self._kin_reputations[:n_living],
                self._kin_gids[:n_living],
                self._kin_hids[:n_living],
                self._kin_tcooldowns[:n_living],
                surv_target,
                repro_target,
                self.world_cfg.base_food_price * 0.55,
            )
        else:
            active_i = np.empty(0, dtype=np.int32)
            active_j = np.empty(0, dtype=np.int32)
            active_kind = np.empty(0, dtype=np.int32)

        for k in range(len(active_i)):
            i = int(active_i[k])
            j = int(active_j[k])
            kind = int(active_kind[k])

            blob = living_blobs[i]
            p = living_blobs[j]
            if not p.alive:
                continue

            if (kind & 1):
                blob.social_desire = float(actions_batch[i, 4])
                p.social_desire = float(actions_batch[j, 4])
                if self._handle_social_encounter(blob, p):
                    if not self.world_cfg.intrinsic_homeostasis:
                        bonus = self.prefs.pref_social * 0.45
                        blob.last_reward += bonus
                        blob.cumulative_reward += bonus
                        step_rewards[i] += bonus * 2.5

            traded = False
            if (kind & 2):
                blob.trade_desire = float(actions_batch[i, 3])
                p.trade_desire = float(actions_batch[j, 3])
                traded = self._try_peer_trade(blob, p)

            if not traded and (kind & 4):
                blob.social_desire = float(actions_batch[i, 4])
                blob.trade_desire = float(actions_batch[i, 3])
                self._try_peer_raid(blob, p)

        # -------------------------------------------------------------
        # 3. Vectorized Numba Market Filter & Village Trades
        # -------------------------------------------------------------
        groups_with_depot_food = {
            h.group_id for h in self.homes.values() if h.group_id is not None and h.food_depot >= 0.5
        }
        if getattr(self, "_market_grp_has_depot", None) is None:
            self._market_grp_has_depot = np.zeros(512, dtype=np.bool_)
            self._homes_gold_arr = np.zeros(1024, dtype=np.float32)
            self._homes_depot_arr = np.zeros(1024, dtype=np.float32)

        self._market_grp_has_depot.fill(False)
        max_gid = 0
        for gid in groups_with_depot_food:
            if 0 <= gid < 512:
                self._market_grp_has_depot[gid] = True
                if gid > max_gid:
                    max_gid = gid

        homes_gold = self._homes_gold_arr
        homes_depot = self._homes_depot_arr
        for hid, h in self.homes.items():
            if hid < 1024:
                homes_gold[hid] = h.gold_vault
                homes_depot[hid] = h.food_depot

        active_market_indices = filter_active_market_numba(
            n_living,
            self._kin_is_at_home_int[:n_living],
            self._kin_tcooldowns[:n_living],
            self._kin_foods_today[:n_living],
            self._kin_surplus_foods[:n_living],
            self._kin_moneys[:n_living],
            self._kin_energies[:n_living],
            actions_batch[:, 3],
            self._kin_gids[:n_living],
            self._kin_hids[:n_living],
            homes_gold,
            homes_depot,
            self._market_grp_has_depot,
            max_gid,
            tod,
            surv_target,
            repro_target,
        )

        for idx in active_market_indices:
            idx_int = int(idx)
            blob = living_blobs[idx_int]
            home_obj = home_objs[idx_int]
            if home_obj is not None:
                blob.trade_desire = float(actions_batch[idx_int, 3])
                if self._try_village_market_trade(blob, home_obj, groups_with_depot_food=groups_with_depot_food, tod=tod):
                    hid = home_obj.id
                    if hid < 1024:
                        homes_gold[hid] = home_obj.gold_vault
                        homes_depot[hid] = home_obj.food_depot

        self._process_foraging(living_blobs, dt_scale, use_kin_cache=True)

        if self.world_cfg.intrinsic_homeostasis:
            delta_e = (self._kin_energies[:n_living] - prev_energies) * np.float32(1.0 / 25.0)
            acute_pain = np.maximum(np.float32(0.0), (-delta_e - np.float32(0.15)) * np.float32(1.5))
            pain = (
                np.float32(0.08) * hit_rocks.astype(np.float32)
                + np.float32(0.04) * self._kin_in_swamps[:n_living].astype(np.float32)
                + acute_pain
            )
            peer_prox = peer_vecs[:, 2]
            base_intrinsic = (
                self._kin_drive_energy[:n_living] * delta_e
                - self._kin_drive_pain[:n_living] * pain
                + np.float32(0.015) * self._kin_drive_social[:n_living] * peer_prox
            )
            bi_list = base_intrinsic.tolist()
            for idx in range(n_living):
                bi_val = bi_list[idx]
                living_blobs[idx].last_reward = bi_val
                living_blobs[idx].cumulative_reward += bi_val
                living_blobs[idx].pending_rl_reward = 0.0
            step_rewards = (base_intrinsic * self._kin_plasticity_lr[:n_living] * np.float32(2.5)).astype(np.float32)
        else:
            for idx in range(n_living):
                if living_blobs[idx].pending_rl_reward != 0.0:
                    # Scale discrete event reward uniformly by 2.5x to match movement shaping reward learning scale
                    step_rewards[idx] += living_blobs[idx].pending_rl_reward * 2.5
                    living_blobs[idx].pending_rl_reward = 0.0

        if self.world_cfg.every_tick_brain or do_trace or np.any(np.abs(step_rewards) > 0.25):
            self._brain_pool.apply_online_learning_batch(
                brains, step_rewards, self.prefs.learning_rate
            )

        # Asynchrone Fortpflanzung mit Energieerhaltung (continuous_lifecycle)
        midtick_babies: List[Blob] = []
        if self.world_cfg.continuous_lifecycle and n_living > 0 and len(self.blobs) < self.world_cfg.max_population_cap:
            # 1. repro_cooldown == 0
            # 2. energy >= 0.78 * max_energy
            # 3. food_today >= repro_need * 0.85
            # 4. Pop-Cap
            rcooldowns = self._kin_rcooldowns[:n_living]
            energies = self._kin_energies[:n_living]
            foods_today = self._kin_foods_today[:n_living]
            max_energies = 85.0 + 15.0 * self._kin_gene_sizes[:n_living]
            repro_mask = (
                (rcooldowns == 0)
                & (energies >= (0.78 * max_energies))
                & (foods_today >= (repro_target * 0.85))
            )
            cand_indices = np.flatnonzero(repro_mask)
            if len(cand_indices) > 0:
                if len(cand_indices) > 24:
                    cand_energies = energies[cand_indices]
                    cand_indices = cand_indices[np.argsort(-cand_energies)[:24]]

                remaining_slots = self.world_cfg.max_population_cap - len(self.blobs)
                if len(cand_indices) > remaining_slots:
                    cand_indices = cand_indices[:remaining_slots]

                min_p_idxs = getattr(self, "_last_min_p_idxs", None)
                for c_i in cand_indices:
                    parent = living_blobs[c_i]
                    if not parent.alive:
                        continue

                    # Partner finding
                    partner: Optional[Blob] = None
                    if (
                        min_p_idxs is not None
                        and len(min_p_idxs) > c_i
                        and len(nearest_peer_dists) > c_i
                        and nearest_peer_dists[c_i] <= 95.0
                    ):
                        p_idx = int(min_p_idxs[c_i])
                        if 0 <= p_idx < n_living and p_idx != c_i:
                            cand_partner = living_blobs[p_idx]
                            if cand_partner.alive:
                                partner = cand_partner

                    # Reproduction
                    child = self._reproduce_blob(parent, partner=partner)

                    # Energy conservation with float32 buffer precision
                    e_before = parent.energy
                    target_child_e = min(round(0.38 * e_before, 2), child.max_energy)
                    parent.energy = max(1.0, e_before - target_child_e)
                    actual_parent_e = parent.energy
                    child.energy = e_before - actual_parent_e

                    # Food cost and cooldown
                    food_cost = min(parent.food_today, self.world_cfg.reproduction_food_need * 0.85)
                    parent.food_today = round(max(0.0, parent.food_today - food_cost), 3)
                    if parent.surplus_food > 0.0 and parent.food_today < repro_target:
                        shift = min(parent.surplus_food, repro_target - parent.food_today)
                        parent.surplus_food = round(parent.surplus_food - shift, 3)
                        parent.food_today = round(parent.food_today + shift, 3)

                    parent.repro_cooldown = max(120, self.world_cfg.ticks_per_day // 3)

                    # Local placement near parent avoiding rocks
                    w_w = self.world_cfg.width
                    w_h = self.world_cfg.height
                    px, py = parent.x, parent.y
                    placed = False
                    for _ in range(6):
                        cx = float(np.clip(px + self.rng.normal(0.0, 10.0), 25.0, w_w - 25.0))
                        cy = float(np.clip(py + self.rng.normal(0.0, 10.0), 25.0, w_h - 25.0))
                        if not self._point_hits_rock(cx, cy, pad=child.body_radius + 4.0):
                            child.x = cx
                            child.y = cy
                            placed = True
                            break
                    if not placed:
                        child.x = px
                        child.y = py

                    midtick_babies.append(child)

        self._check_midday_starvation(living_blobs)
        self._compact_living_blobs()

        if midtick_babies:
            base_idx = len(self.blobs)
            self.blobs.extend(midtick_babies)
            for b_i, b in enumerate(midtick_babies):
                b._sim_idx = base_idx + b_i
            self._kin_static_dirty = True

        tpd = self.world_cfg.ticks_per_day
        if self.world_cfg.continuous_lifecycle:
            if self.tick_in_day >= tpd:
                # Leichtgewichtiger Tageswechsel ohne Mitternachts-Stall
                for b in self.blobs:
                    if b.alive:
                        b.age_days += 1
                self.telemetry.on_eod_pre_accounting(
                    self.blobs, self.world_cfg.survival_food_need, self.world_cfg.reproduction_food_need
                )
                self._update_hall_of_fame()
                self._cleanup_groups_and_homes()
                if self.world_cfg.min_population_floor > 0 and len(self.blobs) < self.world_cfg.min_population_floor:
                    self._respawn_from_hall_of_fame()
                self.record_daily_snapshot()
                self.day_number += 1
                self.tick_in_day = 0
                self.births_today = 0
                self.deaths_today = 0
                self.trades_today = 0
                self.raids_today = 0
                self.raids_repelled_today = 0
                self.exiles_today = 0
                self.bounties_today = 0
        else:
            if tpd >= 20 and dt == 1:
                if self.tick_in_day == tpd - 2:
                    self._stage_eod_home_maintenance(living_blobs)
                elif self.tick_in_day == tpd - 1:
                    self._stage_eod_settlements_and_learning(living_blobs)
                elif self.tick_in_day >= tpd:
                    self._stage_eod_reproduction_and_rollover(living_blobs, amortize_food=True)
            elif self.tick_in_day >= tpd:
                self._end_of_day_accounting(amortize_food=True)
        return dt

    def _process_foraging(
        self, living_blobs: List[Blob], dt_scale: float = 1.0, use_kin_cache: bool = False
    ) -> None:
        """
        Check collisions between living blobs and food items using the O(1) Numba spatial grid,
        awarding each food to the closest blob. Applies Overgrazing harvest pressure to the source Oasis
        and updates the blob's Spatial Memory!
        """
        if not self.foods or not living_blobs:
            return

        repro_need = self.world_cfg.reproduction_food_need
        alive_blobs = [b for b in living_blobs if b.alive]
        if not alive_blobs:
            return

        n_alive = len(alive_blobs)
        n_foods = len(self.foods)
        if use_kin_cache and len(self._kin_xs) >= n_alive:
            blob_xs = self._kin_xs[:n_alive]
            blob_ys = self._kin_ys[:n_alive]
            blob_reaches = self._kin_body_radii[:n_alive] * (
                np.float32(1.08) - np.float32(0.35) * self._kin_sprints[:n_alive]
            )
        else:
            blob_xs = np.fromiter((b.x for b in alive_blobs), dtype=np.float32, count=n_alive)
            blob_ys = np.fromiter((b.y for b in alive_blobs), dtype=np.float32, count=n_alive)
            blob_reaches = np.fromiter(
                (b.body_radius * (1.08 - 0.35 * b.sprint_level) for b in alive_blobs),
                dtype=np.float32,
                count=n_alive,
            )

        food_xs = self._food_xs
        food_ys = self._food_ys
        if getattr(self, "_food_dirty", False) or food_xs is None or len(food_xs) != n_foods or not use_kin_cache:
            food_xs = np.fromiter((f.x for f in self.foods), dtype=np.float32, count=n_foods)
            food_ys = np.fromiter((f.y for f in self.foods), dtype=np.float32, count=n_foods)
            self._food_base_reaches = np.fromiter(
                ((10.0 if f.golden else 8.0) for f in self.foods),
                dtype=np.float32,
                count=n_foods,
            )
            self._food_xs = food_xs
            self._food_ys = food_ys
            self._food_dirty = False
        sweep_bonus = min(38.0, (dt_scale - 1.0) * 1.8)
        extra_reaches = self._food_base_reaches + sweep_bonus

        grid_w = getattr(self, "_step_grid_w", max(1, int(self.world_cfg.width * INV_SPATIAL_CELL) + 2))
        grid_h = getattr(self, "_step_grid_h", max(1, int(self.world_cfg.height * INV_SPATIAL_CELL) + 2))

        blob_cell_head, blob_next = build_spatial_linked_grid(
            blob_xs, blob_ys, INV_SPATIAL_CELL, grid_w, grid_h
        )
        cand1, cand2 = find_foraging_candidates_grid(
            food_xs,
            food_ys,
            extra_reaches,
            blob_xs,
            blob_ys,
            blob_reaches,
            blob_cell_head,
            blob_next,
            INV_SPATIAL_CELL,
            grid_w,
            grid_h,
        )

        hit_food_idxs = np.flatnonzero(cand1 >= 0)
        if len(hit_food_idxs) == 0:
            return

        eaten_set = set()
        for f_idx_int in hit_food_idxs:
            f_idx = int(f_idx_int)
            food = self.foods[f_idx]
            for c_idx in (int(cand1[f_idx]), int(cand2[f_idx])):
                if c_idx < 0:
                    break
                blob = alive_blobs[c_idx]
                if blob.total_food >= blob.max_food_capacity - 1e-4 and blob.energy >= blob.max_energy - 1.0:
                    continue
                eaten_set.add(f_idx)
                nutr = food.nutrition
                stored = blob.add_acquired_food(nutr, repro_need)
                effective_nutr = stored if stored > 0.0 else nutr * 0.25
                blob.food_eaten_lifetime += effective_nutr
                blob.ticks_since_last_meal = 0
                blob.energy = min(blob.max_energy, blob.energy + effective_nutr * 44.0)

                if food.oasis_id is not None and food.oasis_id in self.oases:
                    oasis = self.oases[food.oasis_id]
                    dep_mult = float(getattr(oasis, "depletion_rate", 1.0))
                    is_kg = getattr(self.world_cfg, "scale_preset", "") == "training_kindergarten"
                    hp_add = (0.20 if (is_kg and self.day_number <= 5) else 0.65) * dep_mult
                    oasis.harvest_pressure += hp_add
                    fert_drop = (0.003 if (is_kg and self.day_number <= 5) else 0.012) * dep_mult
                    min_floor = 0.68 if (is_kg and oasis.ring == "inner" and self.day_number <= 5) else 0.06
                    oasis.fertility = round(max(min_floor, oasis.fertility - fert_drop), 3)
                    oasis.current_food_count = max(0, oasis.current_food_count - 1)
                    blob.memory_x = oasis.x
                    blob.memory_y = oasis.y
                    blob.memory_oasis_id = oasis.id
                    blob.memory_strength = round(max(0.25, oasis.fertility), 2)
                elif blob.memory_strength < 0.4:
                    blob.memory_x = food.x
                    blob.memory_y = food.y
                    blob.memory_strength = 0.55

                eat_reward = self.prefs.pref_food * (1.1 if food.golden else 0.75)
                blob.last_reward += eat_reward
                blob.cumulative_reward += eat_reward
                blob.pending_rl_reward += eat_reward
                self.telemetry.on_food_foraged(
                    blob,
                    food,
                    effective_nutr,
                    self.oases.get(food.oasis_id) if food.oasis_id is not None else None,
                )
                break

        if eaten_set:
            if hasattr(self, "recent_eaten_indices"):
                slot_map = getattr(self, "_food_id_to_slot", {})
                for ei in eaten_set:
                    fid = self.foods[ei].id
                    slot = slot_map.get(fid, ei)
                    self.recent_eaten_indices.append(slot)
            keep_mask = np.ones(n_foods, dtype=np.bool_)
            for ei in eaten_set:
                keep_mask[ei] = False
            self.foods = [f for i, f in enumerate(self.foods) if i not in eaten_set]
            if food_xs is not None and len(food_xs) == n_foods:
                self._food_xs = food_xs[keep_mask]
                self._food_ys = food_ys[keep_mask]
                self._food_base_reaches = self._food_base_reaches[keep_mask]
                self._food_dirty = False
            else:
                self._food_dirty = True

    # --- Settlement & Multi-Home Management ---

    def _get_nearest_settlement_home(self, blob: Blob) -> Optional[Home]:
        """Return the blob's assigned home, or if in a multi-home settlement, the closest settlement home."""
        primary = self.homes.get(blob.home_id)
        if blob.is_at_home and primary is not None:
            return primary
        bgid = blob._group_id
        if bgid is None:
            return primary
        grp = self.groups.get(bgid)
        if grp is None or len(grp.home_ids) <= 1:
            return primary
        grp_homes = getattr(grp, "_cached_homes", None)
        if grp_homes is None or len(grp_homes) != len(grp.home_ids):
            grp_homes = [self.homes[hid] for hid in grp.home_ids if hid in self.homes]
            grp._cached_homes = grp_homes
        bx, by = blob.x, blob.y
        best_home = primary
        best_d2 = ((primary.x - bx) ** 2 + (primary.y - by) ** 2) if primary is not None else 1e12
        for h in grp_homes:
            d2 = (h.x - bx) ** 2 + (h.y - by) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_home = h
        return best_home

    def _find_settlement_home_with_food(self, blob: Blob, min_food: float = 0.4) -> Optional[Home]:
        """Find a home in the blob's settlement (or its own home) that has at least `min_food` in its depot."""
        home = self.homes.get(blob.home_id)
        if home is not None and home.food_depot >= min_food:
            return home
        if blob.group_id is not None and blob.group_id in self.groups:
            grp = self.groups[blob.group_id]
            for hid in grp.home_ids:
                h = self.homes.get(hid)
                if h is not None and h.food_depot >= min_food:
                    return h
        return None

    def _federate_home_into_group(self, home: Home, grp: Group) -> None:
        """Unite an ungrouped (or same-group) Home into a Settlement Group."""
        if home.group_id is None or home.group_id == grp.id:
            if home.group_id != grp.id:
                self.telemetry.on_home_federated(grp, home)
                home.group_id = grp.id
            home.radius = 52.0
            if home.id not in grp.home_ids:
                grp.home_ids.append(home.id)
            for rid in home.resident_ids:
                b = self._blobs_by_id.get(rid)
                if b is not None:
                    b.home_obj = home
                    idx = getattr(b, "_kin_idx", -1)
                    if not (0 <= idx < len(self.blobs) and self.blobs[idx] is b):
                        idx = getattr(b, "_sim_idx", -1)
                    if not (0 <= idx < len(self.blobs) and self.blobs[idx] is b):
                        try:
                            idx = self.blobs.index(b)
                            b._sim_idx = idx
                            b._kin_idx = idx
                        except ValueError:
                            idx = -1
                    if 0 <= idx < self._kin_cap:
                        if idx < len(self._kin_home_radii):
                            self._kin_home_radii[idx] = 52.0
                        if idx < len(self._step_home_radii_list):
                            self._step_home_radii_list[idx] = 52.0
                        if idx < len(self._step_home_objs):
                            self._step_home_objs[idx] = home

    def _find_or_create_settlement_home(
        self,
        group_id: Optional[int],
        ref_x: float,
        ref_y: float,
    ) -> Home:
        """
        Find a home with free resident capacity (< max_residents) in the settlement or nearby,
        or construct a new partner house that unites with the settlement.
        """
        max_res = max(2, self.world_cfg.max_residents_per_home)
        grp = self.groups.get(group_id) if group_id is not None else None

        if grp is not None:
            open_settlement_homes = [
                self.homes[hid]
                for hid in grp.home_ids
                if hid in self.homes
                and not self.homes[hid].is_full
                and abs(self.homes[hid].x - ref_x) <= 950.0
                and abs(self.homes[hid].y - ref_y) <= 950.0
            ]
            if open_settlement_homes:
                return min(open_settlement_homes, key=lambda h: math.hypot(h.x - ref_x, h.y - ref_y))

        open_ungrouped = [
            h
            for h in self.homes.values()
            if (h.group_id is None or h.group_id == group_id)
            and not h.is_full
            and abs(h.x - ref_x) <= 280.0
            and abs(h.y - ref_y) <= 280.0
            and math.hypot(h.x - ref_x, h.y - ref_y) <= 280.0
        ]
        if open_ungrouped:
            chosen = min(open_ungrouped, key=lambda h: math.hypot(h.x - ref_x, h.y - ref_y))
            if grp is not None:
                self._federate_home_into_group(chosen, grp)
            return chosen

        max_homes_cap = getattr(self.world_cfg, "max_homes_cap", 1500)
        if len(self.homes) >= max_homes_cap and self.homes:
            return min(self.homes.values(), key=lambda h: math.hypot(h.x - ref_x, h.y - ref_y))

        return self._construct_new_partner_home(grp, ref_x, ref_y, max_res)

    def _construct_new_partner_home(
        self,
        grp: Optional[Group],
        ref_x: float,
        ref_y: float,
        max_res: int,
    ) -> Home:
        hx = float(np.clip(ref_x + self.rng.normal(0.0, 80.0), 65.0, self.world_cfg.width - 65.0))
        hy = float(np.clip(ref_y + self.rng.normal(0.0, 80.0), 65.0, self.world_cfg.height - 65.0))
        for _ in range(18):
            angle = float(self.rng.uniform(0.0, 2.0 * math.pi))
            dist = float(self.rng.uniform(75.0, 145.0))
            cand_x = float(np.clip(ref_x + math.cos(angle) * dist, 65.0, self.world_cfg.width - 65.0))
            cand_y = float(np.clip(ref_y + math.sin(angle) * dist, 65.0, self.world_cfg.height - 65.0))
            if not self._point_hits_rock(cand_x, cand_y, pad=58.0) and all(
                abs(h.x - cand_x) >= 62.0 or abs(h.y - cand_y) >= 62.0 or math.hypot(h.x - cand_x, h.y - cand_y) >= 62.0
                for h in self.homes.values()
            ):
                hx, hy = cand_x, cand_y
                break

        hid = self._next_home_id
        self._next_home_id += 1
        new_home = Home(
            id=hid,
            x=hx,
            y=hy,
            radius=52.0 if grp is not None else 48.0,
            max_residents=max_res,
            group_id=grp.id if grp is not None else None,
            food_depot=1.8,
            gold_vault=16.0,
        )
        self._evaluate_home_location(new_home)
        self.homes[hid] = new_home
        if hasattr(self, "_oasis_to_homes"):
            for oid, oasis in self.oases.items():
                if math.hypot(new_home.x - oasis.x, new_home.y - oasis.y) <= oasis.radius * 1.55:
                    if oid not in self._oasis_to_homes:
                        self._oasis_to_homes[oid] = []
                    self._oasis_to_homes[oid].append(new_home)
        self.telemetry.on_home_constructed(new_home, grp)
        if grp is not None:
            self._federate_home_into_group(new_home, grp)
            self.log_event("group", f"{grp.name} hat ein neues Siedlungs-Haus #{hid} errichtet!")
        return new_home

    def _recompute_group_reputation(self, grp: Group) -> None:
        """Exact, zero-drift group reputation recomputation over group members in O(M)."""
        blobs_by_id = self._blobs_by_id
        living_ids = set()
        seen = set()
        stale_ids = []
        living_reps = []
        for mid in grp.member_ids:
            if mid in seen:
                continue
            seen.add(mid)
            b = blobs_by_id.get(mid)
            if b is None or b._group_id != grp.id:
                stale_ids.append(mid)
                continue
            if b.alive:
                living_ids.add(mid)
                living_reps.append(b.reputation)
        if stale_ids:
            stale_set = set(stale_ids)
            grp.member_ids = [mid for mid in grp.member_ids if mid not in stale_set]
        count = len(living_ids)
        rep_sum = 0.0
        for r in living_reps:
            rep_sum += r
        grp.cached_rep_sum = rep_sum
        grp.cached_rep_count = count
        if count > 0:
            grp.avg_reputation = round(rep_sum / count, 3)
        else:
            grp.avg_reputation = 0.0
        grp._living_member_ids = living_ids
        grp.rep_dirty = False
        grp._last_recompute_tick = getattr(self, "tick_count", 0)

    def _get_group_avg_reputation(self, grp: Group, exclude_blob_id: Optional[int] = None) -> float:
        """Exact, zero-drift O(1) cached group reputation index query with dirty-flag invalidation."""
        is_dirty = getattr(grp, "rep_dirty", True)
        if is_dirty:
            in_step = getattr(self, "_in_step_once", False)
            last_tick = getattr(grp, "_last_recompute_tick", -1)
            if not in_step or last_tick != self.tick_count:
                self._recompute_group_reputation(grp)

        if exclude_blob_id is None:
            return grp.avg_reputation

        living_ids = getattr(grp, "_living_member_ids", None)
        if living_ids is None:
            self._recompute_group_reputation(grp)
            living_ids = grp._living_member_ids

        if exclude_blob_id in living_ids:
            ex_blob = self._blobs_by_id.get(exclude_blob_id)
            if ex_blob is not None:
                rem_count = grp.cached_rep_count - 1
                if rem_count > 0:
                    rem_sum = round(grp.cached_rep_sum - ex_blob.reputation, 5)
                    return round(rem_sum / rem_count, 3)
                return 0.0

        return grp.avg_reputation

    @staticmethod
    def _get_reputation_buy_multiplier(blob: Blob) -> float:
        """Outlaws pay up to +75% risk surcharge when buying food; honorable blobs receive up to -15% trust discount."""
        if blob.reputation < 0.0:
            return 1.0 + 0.75 * abs(blob.reputation)
        if blob.reputation > 0.0:
            return max(0.85, 1.0 - 0.15 * blob.reputation)
        return 1.0

    @staticmethod
    def _get_reputation_sell_multiplier(blob: Blob) -> float:
        """Outlaws suffer up to -35% fence markdown when selling goods; honorable blobs earn up to +10% trust bonus."""
        if blob.reputation < 0.0:
            return max(0.65, 1.0 - 0.35 * abs(blob.reputation))
        if blob.reputation > 0.0:
            return 1.0 + 0.10 * blob.reputation
        return 1.0

    def _check_blob_reputation_exile(self, blob: Blob) -> bool:
        """
        If a group member's reputation falls into outlaw territory (<= -0.45) inside a peaceful settlement
        (group avg_reputation >= -0.25), banish the blob from the settlement and relocate it out of group homes.
        """
        if blob.group_id is None or blob.reputation > -0.45:
            return False
        grp = self.groups.get(blob.group_id)
        if grp is None:
            blob.group_id = None
            return False

        peer_avg_rep = self._get_group_avg_reputation(grp, exclude_blob_id=blob.id)
        # Bandit clans (where other members also have negative reputation < -0.25) tolerate outlaws
        if peer_avg_rep < -0.25 and len(grp.member_ids) >= 2:
            return False

        old_grp_name = grp.name
        blob.group_id = None
        if blob.id in grp.member_ids:
            grp.member_ids.remove(blob.id)
        grp.exiles += 1
        blob.exiles_count += 1
        self.exiles_today += 1
        self.total_exiles_all_time += 1

        cur_home = self.homes.get(blob.home_id)
        if cur_home is not None and cur_home.group_id == grp.id:
            if len(cur_home.resident_ids) <= 1 and len(grp.home_ids) > 1:
                cur_home.group_id = None
                cur_home.radius = 38.0
                if cur_home.id in grp.home_ids:
                    grp.home_ids.remove(cur_home.id)
                    if grp.home_id == cur_home.id and grp.home_ids:
                        grp.home_id = grp.home_ids[0]
            else:
                outlaw_shelter = self._find_or_create_settlement_home(None, blob.x, blob.y)
                self._relocate_blob_home(blob, outlaw_shelter.id)

        exile_pen = -0.45 * self.prefs.pref_social
        blob.cumulative_reward += exile_pen
        blob.pending_rl_reward += exile_pen
        if hasattr(self.telemetry, "on_blob_exiled"):
            self.telemetry.on_blob_exiled(blob, grp)
        self.log_event(
            "raid",
            f"⚖️ Verbannung: {old_grp_name} verstößt Geächteten #{blob.id} (Ruf {blob.reputation:+.2f}) aus der Siedlung!",
        )
        return True

    def _found_new_settlement_group(self, a: Blob, b: Blob) -> bool:
        # Peaceful blobs refuse to co-found a settlement with an outlaw (unless both are low-rep bandits forming a Bandit Clan)
        if (a.is_outlaw and b.reputation > -0.15) or (b.is_outlaw and a.reputation > -0.15):
            return False

        max_groups_cap = getattr(self.world_cfg, "max_groups_cap", 120)
        if self.groups:
            nearest_grp: Optional[Group] = None
            nearest_d_sq = float("inf")
            local_groups_count = 0
            ax, ay = a.x, a.y
            max_local_dist_sq = 1250.0 * 1250.0
            close_threshold_sq = 650.0 * 650.0
            for g in self.groups.values():
                gh = self.homes.get(g.home_id)
                if gh is None:
                    continue
                dx = gh.x - ax
                dy = gh.y - ay
                d_sq = dx * dx + dy * dy
                if d_sq < nearest_d_sq:
                    nearest_d_sq = d_sq
                    nearest_grp = g
                if d_sq <= max_local_dist_sq:
                    local_groups_count += 1

            if nearest_grp is not None and (
                len(self.groups) >= max_groups_cap
                or (local_groups_count >= 5 and nearest_d_sq <= close_threshold_sq)
            ):
                ca = self._join_blob_to_group(a, nearest_grp.id)
                cb = self._join_blob_to_group(b, nearest_grp.id)
                if ca or cb:
                    return True
                if len(self.groups) >= max_groups_cap:
                    return False

        gid = self._next_group_id
        self._next_group_id += 1
        name, color, rgb_int = generate_faction_identity(gid)
        both_bandits = a.reputation <= -0.25 and b.reputation <= -0.25
        if both_bandits and not name.startswith("Banditen-"):
            name = f"Banditen-{name.split('-')[0]}"
        grp = Group(
            id=gid,
            name=name,
            color=color,
            home_id=a.home_id,
            home_ids=[a.home_id],
            member_ids=[a.id, b.id],
            created_day=self.day_number,
            avg_reputation=round(0.5 * (a.reputation + b.reputation), 3),
            rgb_int=rgb_int,
        )
        self.groups[gid] = grp
        a.group_id = gid
        b.group_id = gid
        self.telemetry.on_group_founded(grp, a, b)

        home_a = self.homes.get(a.home_id)
        if home_a is not None:
            self._federate_home_into_group(home_a, grp)

        home_b = self.homes.get(b.home_id)
        if (
            home_b is not None
            and (home_b.group_id is None or home_b.group_id == gid)
            and len(home_b.resident_ids) <= home_b.max_residents
        ):
            self._federate_home_into_group(home_b, grp)
        else:
            target_home = self._find_or_create_settlement_home(gid, a.home_x, a.home_y)
            self._relocate_blob_home(b, target_home.id)

        reward = self.prefs.pref_social * 0.5
        a.pending_rl_reward += reward
        b.pending_rl_reward += reward
        self.log_event("group", f"Neue Siedlung gegründet: {name} (Blob #{a.id} & #{b.id})")
        return True

    def _handle_social_encounter(self, a: Blob, b: Blob) -> bool:
        """
        Allow two nearby friendly blobs to form or join a Settlement Group,
        or unite their Homes into a multi-home Settlement. Returns True if a change occurred.
        """
        if a.group_id is not None and b.group_id == a.group_id:
            grp = self.groups.get(a.group_id)
            if grp is not None:
                for member in (a, b):
                    mh = self.homes.get(member.home_id)
                    if mh is not None and mh.group_id is None:
                        self._federate_home_into_group(mh, grp)
                        return True
            return False

        if a.group_id is not None and b.group_id is None:
            return self._join_blob_to_group(b, a.group_id)

        if b.group_id is not None and a.group_id is None:
            return self._join_blob_to_group(a, b.group_id)

        if a.group_id is None and b.group_id is None:
            return self._found_new_settlement_group(a, b)

        return False

    def _join_blob_to_group(self, blob: Blob, group_id: int) -> bool:
        grp = self.groups.get(group_id)
        if grp is None:
            return False

        # Peaceful settlements reject outlaws; honorable blobs refuse bandit clans
        if blob.is_outlaw or blob.is_honorable:
            grp_rep = self._get_group_avg_reputation(grp)
            if blob.is_outlaw and grp_rep >= -0.25:
                return False
            if blob.is_honorable and grp_rep <= -0.35:
                return False

        if blob.group_id != grp.id:
            self.telemetry.on_group_join(grp, blob)
        blob.group_id = grp.id
        if blob.id not in grp.member_ids:
            grp.member_ids.append(blob.id)

        cur_home = self.homes.get(blob.home_id)
        if cur_home is not None and (cur_home.group_id is None or cur_home.group_id == grp.id) and len(cur_home.resident_ids) <= cur_home.max_residents:
            self._federate_home_into_group(cur_home, grp)
        else:
            bx, by = blob.x, blob.y
            grp_homes = [self.homes[hid] for hid in grp.home_ids if hid in self.homes]
            ref_home = (
                min(grp_homes, key=lambda h: math.hypot(h.x - bx, h.y - by))
                if grp_homes
                else self.homes.get(grp.home_id)
            )
            rx = ref_home.x if ref_home else bx
            ry = ref_home.y if ref_home else by
            target_home = self._find_or_create_settlement_home(grp.id, rx, ry)
            self._relocate_blob_home(blob, target_home.id)

        reward = self.prefs.pref_social * 0.5
        blob.pending_rl_reward += reward
        return True

    def _relocate_blob_home(self, blob: Blob, new_home_id: int) -> None:
        if blob.home_id in self.homes:
            old_h = self.homes[blob.home_id]
            if blob.id in old_h.resident_ids:
                old_h.resident_ids.remove(blob.id)
        if new_home_id in self.homes:
            new_h = self.homes[new_home_id]
            blob.home_id = new_h.id
            blob.home_obj = new_h
            blob.home_x = new_h.x
            blob.home_y = new_h.y
            if blob.id not in new_h.resident_ids:
                new_h.resident_ids.append(blob.id)
            idx = getattr(blob, "_sim_idx", -1)
            if not (0 <= idx < len(self.blobs) and self.blobs[idx] is blob):
                try:
                    idx = self.blobs.index(blob)
                    blob._sim_idx = idx
                except ValueError:
                    idx = -1
            if 0 <= idx < self._kin_cap:
                self._kin_home_xs[idx] = new_h.x
                self._kin_home_ys[idx] = new_h.y
                self._kin_home_radii[idx] = new_h.radius
                self._kin_hids[idx] = new_h.id
                if idx < len(self._step_home_objs):
                    self._step_home_objs[idx] = new_h
                if idx < len(self._step_home_radii_list):
                    self._step_home_radii_list[idx] = new_h.radius

    def _is_seller_willing(
        self,
        seller: Blob,
        unit_amount: float,
        surv_need: float,
        repro_need: float,
    ) -> bool:
        """
        A blob can ALWAYS trade food if it holds >= unit_amount, even below 2x or 1x!
        However, selling below reproduction or survival thresholds must be an active neural
        network decision (trade_desire), never a forced rule against the brain's will.
        """
        if seller.total_food < unit_amount:
            return False
        remaining = seller.total_food - unit_amount
        if seller.surplus_food >= unit_amount or remaining >= repro_need - 1e-4:
            return seller.trade_desire > -0.10
        if remaining >= surv_need - 1e-4:
            return seller.trade_desire > 0.12
        return seller.trade_desire > 0.25

    def _compute_seller_trade_reward(
        self,
        seller: Blob,
        same_group: bool,
        surv_need: float,
        repro_need: float,
    ) -> float:
        if seller.food_today < surv_need - 1e-4:
            # Selling food when hungry/starving threatens survival - heavily penalize
            return -1.0 * self.prefs.pref_food
        social_bonus = self.prefs.pref_social * 0.25 if same_group else 0.0
        if seller.food_today >= repro_need - 1e-4:
            return self.prefs.pref_money * 0.60 + social_bonus
        return self.prefs.pref_money * 0.22 + social_bonus * 0.5

    def _try_peer_trade(self, a: Blob, b: Blob) -> bool:
        """
        Execute peer-to-peer trade (Food <-> Money).
        A blob can ALWAYS trade food if it holds >= unit_amount, even if doing so leaves it
        below reproduction (2x) or daily survival (1x) thresholds!
        Honorable/normal peers boycott peer-to-peer trades with known outlaws (reputation <= -0.35).
        """
        unit_amount = 0.5
        if (a.total_food < unit_amount and b.total_food < unit_amount) or (a.money < 0.4 and b.money < 0.4):
            return False
        if (a.is_outlaw and b.reputation > -0.05) or (b.is_outlaw and a.reputation > -0.05):
            return False

        same_group = a.group_id is not None and a.group_id == b.group_id
        surv_need = self.world_cfg.survival_food_need
        repro_need = self.world_cfg.reproduction_food_need

        for seller, buyer in ((a, b), (b, a)):
            if seller.total_food < unit_amount or buyer.money < 0.4:
                continue
            if not self._is_seller_willing(seller, unit_amount, surv_need, repro_need):
                continue
            buyer_can_carry = (buyer.total_food + unit_amount) <= buyer.max_food_capacity + 1e-4
            if not buyer_can_carry:
                continue
            wants_to_buy = (
                buyer.food_today < repro_need
                or buyer.energy < 75.0
                or buyer.trade_desire > 0.15
            )
            if not wants_to_buy:
                continue

            discount = (1.0 - self.world_cfg.group_trade_discount) if same_group else 1.0
            home_a = self.homes.get(a.home_id)
            local_mult = home_a.local_price_multiplier if home_a is not None else 1.0
            base_price = max(0.4, self.world_cfg.base_food_price * unit_amount * local_mult * discount)
            rep_buy_mult = self._get_reputation_buy_multiplier(buyer)
            price = round(max(0.4, base_price * rep_buy_mult), 2)
            if buyer.money < price:
                continue

            pre_buy_food = buyer.food_today
            from_stomach = seller.deduct_traded_food(unit_amount)
            if from_stomach > 0.0:
                seller.energy = max(15.0, seller.energy - from_stomach * 18.0)

            buyer.add_acquired_food(unit_amount, repro_need)
            buyer.energy = min(buyer.max_energy, buyer.energy + unit_amount * 38.0)
            buyer.ticks_since_last_meal = 0

            buyer.money = round(buyer.money - price, 2)
            seller.money = round(seller.money + price, 2)

            seller.trades_made += 1
            buyer.trades_made += 1
            seller.adjust_reputation(0.05)
            buyer.adjust_reputation(0.05)
            seller.trade_cooldown = 30
            buyer.trade_cooldown = 30
            seller.last_trade_tick = seller.age_ticks
            buyer.last_trade_tick = buyer.age_ticks

            self.trades_today += 1
            self.total_trades_all_time += 1
            if same_group and seller.group_id in self.groups:
                self.groups[seller.group_id].total_trades += 1
            self.telemetry.on_peer_trade(
                seller,
                buyer,
                unit_amount,
                price,
                same_group,
                from_stomach,
                pre_buy_food < surv_need,
            )

            self.recent_trade_lines.append(
                {
                    "x1": round(seller.x, 1),
                    "y1": round(seller.y, 1),
                    "x2": round(buyer.x, 1),
                    "y2": round(buyer.y, 1),
                    "same_group": same_group,
                    "tick": self.tick_count,
                }
            )

            seller_reward = self._compute_seller_trade_reward(
                seller, same_group, surv_need, repro_need
            )
            if seller_reward > 0.0:
                seller_reward += 0.08 * self.prefs.pref_social
            buyer_reward = self.prefs.pref_money * 0.45 + (
                self.prefs.pref_food * 0.45 if pre_buy_food < surv_need else self.prefs.pref_food * 0.20
            ) + 0.08 * self.prefs.pref_social
            seller.cumulative_reward += seller_reward
            buyer.cumulative_reward += buyer_reward
            seller.pending_rl_reward += seller_reward
            buyer.pending_rl_reward += buyer_reward
            return True

        return False

    def _try_peer_raid(self, aggressor: Blob, defender: Blob) -> bool:
        """
        Attempt a resource raid against an outsider blob (or its nearby depot).
        Strength is governed by the `size` gene, energy, clan support, and 🛡️ Watchtower defense.
        - Raiding peaceful blobs (`defender.reputation > -0.40`) severely damages reputation, costs real stamina,
          and triggers social exile if the aggressor drops below -0.45 in a peaceful settlement.
        - Engaging known outlaws (`defender.reputation <= -0.40`) costs NO reputation and grants a bounty reward!
        """
        surv_need = self.world_cfg.survival_food_need
        repro_need = self.world_cfg.reproduction_food_need
        victim_is_outlaw = defender.reputation <= -0.40
        is_desperate = (
            aggressor.food_today < surv_need
            and aggressor.money < self.world_cfg.base_food_price * 0.55
        )
        is_hostile = (
            aggressor.social_desire < -0.16
            and aggressor.trade_desire < -0.12
            and aggressor.food_today < repro_need
        )
        is_bounty_intervention = (
            victim_is_outlaw
            and aggressor.reputation >= 0.0
            and aggressor.social_desire > 0.15
            and aggressor.energy >= 38.0
        )
        if not (is_desperate or is_hostile or is_bounty_intervention) or aggressor.energy < 28.0:
            return False

        def_home = self.homes.get(defender.home_id)
        near_def_home = False
        if def_home is not None:
            dhx = defender.x - def_home.x
            dhy = defender.y - def_home.y
            near_def_home = (dhx * dhx + dhy * dhy) <= (def_home.radius * 1.65) ** 2

        target_has_loot = (
            defender.total_food >= 0.40
            or defender.money >= 3.5
            or (near_def_home and def_home is not None and def_home.food_depot >= 0.6)
        )
        if not target_has_loot:
            return False

        tower_bonus = (
            (0.32 + def_home.defense_level * 0.52 + def_home.terrain_defense_bonus)
            if (near_def_home and def_home is not None)
            else 0.0
        )
        clan_bonus = 0.20 if defender.group_id is not None else 0.0
        # Known outlaws trigger heightened Watchtower & Militia alertness when attacking!
        outlaw_alert_bonus = (
            abs(aggressor.reputation) * (0.48 if near_def_home else 0.26)
            if (aggressor.reputation < -0.20 and not victim_is_outlaw)
            else 0.0
        )
        atk_score = aggressor.combat_power
        def_score = defender.combat_power + tower_bonus + clan_bonus + outlaw_alert_bonus

        if near_def_home and def_home is not None:
            def_home.recent_raids += 1

        aggressor.trade_cooldown = 34
        defender.trade_cooldown = 34

        if atk_score > def_score * 1.04:
            if defender.total_food >= 0.35:
                loot_kind = "food"
                stolen_food = round(min(0.45, defender.total_food), 2)
                defender.deduct_traded_food(stolen_food)
                aggressor.add_acquired_food(stolen_food, repro_need)
            elif near_def_home and def_home is not None and def_home.defense_level == 0 and def_home.food_depot >= 0.45:
                loot_kind = "depot"
                stolen_food = 0.40
                def_home.food_depot = round(max(0.0, def_home.food_depot - stolen_food), 2)
                aggressor.add_acquired_food(stolen_food, repro_need)
            else:
                loot_kind = "gold"
                stolen_gold = round(min(3.5, defender.money * 0.35), 2)
                defender.money = round(max(0.0, defender.money - stolen_gold), 2)
                aggressor.money = round(aggressor.money + stolen_gold, 2)

            # Pillar 3: Real combat stamina cost (no free +10 energy heal from raiding!)
            aggressor.energy = max(8.0, aggressor.energy - 12.5)
            defender.energy = max(8.0, defender.energy - 11.0)
            aggressor.raids_won += 1
            self.raids_today += 1
            self.total_raids_all_time += 1
            self.telemetry.on_raid_outcome(
                aggressor,
                defender,
                repelled=False,
                loot_kind=loot_kind,
                bounty=victim_is_outlaw,
            )

            # House Siege & Conquest: repeated successful raids near an enemy home can conquer it!
            if near_def_home and def_home is not None:
                def_home.raid_damage += 1
                if (
                    aggressor.group_id is not None
                    and aggressor.group_id in self.groups
                    and def_home.group_id != aggressor.group_id
                    and def_home.raid_damage >= (2 + def_home.defense_level)
                ):
                    atk_grp = self.groups[aggressor.group_id]
                    old_gid = def_home.group_id
                    if old_gid is not None and old_gid in self.groups:
                        old_grp = self.groups[old_gid]
                        if def_home.id in old_grp.home_ids:
                            old_grp.home_ids.remove(def_home.id)
                        if old_grp.home_id == def_home.id and old_grp.home_ids:
                            old_grp.home_id = old_grp.home_ids[0]
                        if defender.id in old_grp.member_ids:
                            old_grp.member_ids = [mid for mid in old_grp.member_ids if mid != defender.id]
                        old_grp.rep_dirty = True
                    def_home.group_id = atk_grp.id
                    def_home.raid_damage = 0
                    def_home.defense_level = max(0, def_home.defense_level - 1)
                    if def_home.id not in atk_grp.home_ids:
                        atk_grp.home_ids.append(def_home.id)
                    atk_grp.conquests += 1
                    defender.group_id = atk_grp.id
                    if defender.id not in atk_grp.member_ids:
                        atk_grp.member_ids.append(defender.id)
                    atk_grp.rep_dirty = True
                    self.telemetry.on_house_conquered(atk_grp, old_gid, def_home)
                    self.log_event(
                        "raid",
                        f"⚔️ Eroberung: {atk_grp.name} erobert Haus #{def_home.id} im Sturm!",
                    )

            if victim_is_outlaw:
                # Bounty / Self-defense against an outlaw: no reputation loss, gold bounty + positive social reward
                bounty_gold = round(2.5 + min(3.0, defender.money * 0.25), 2)
                defender.money = round(max(0.0, defender.money - min(defender.money, 1.5)), 2)
                aggressor.money = round(aggressor.money + bounty_gold, 2)
                aggressor.adjust_reputation(0.10)
                aggressor.bounties_won += 1
                self.bounties_today += 1
                self.total_bounties_all_time += 1
                raid_rew = 0.35 * self.prefs.pref_food + 0.28 * self.prefs.pref_social
            else:
                rep_loss = 0.28 + 0.12 * max(0.0, defender.reputation)
                aggressor.adjust_reputation(-rep_loss)
                social_pen = (0.34 + 0.18 * abs(min(0.0, aggressor.reputation))) * self.prefs.pref_social
                raid_rew = 0.28 * self.prefs.pref_food - social_pen
                self._check_blob_reputation_exile(aggressor)

            aggressor.cumulative_reward += raid_rew
            aggressor.pending_rl_reward += raid_rew
            defender.energy = max(5.0, defender.energy - 10.0)

            self.recent_raid_lines.append(
                {
                    "x1": round(aggressor.x, 1),
                    "y1": round(aggressor.y, 1),
                    "x2": round(defender.x, 1),
                    "y2": round(defender.y, 1),
                    "repelled": False,
                    "tick": self.tick_count,
                }
            )
            return True

        outlaw_recoil = 6.0 * abs(aggressor.reputation) if aggressor.reputation < -0.25 else 0.0
        recoil = (
            14.0
            + (def_home.defense_level * 8.5 if (near_def_home and def_home is not None) else 0.0)
            + outlaw_recoil
        )
        aggressor.energy = max(5.0, aggressor.energy - recoil)
        aggressor.vx = -aggressor.vx * 0.75
        aggressor.vy = -aggressor.vy * 0.75
        self.raids_repelled_today += 1
        self.total_raids_repelled += 1
        defender.adjust_reputation(0.08)
        if not victim_is_outlaw:
            aggressor.adjust_reputation(-0.16)
            aggressor.pending_rl_reward -= (0.38 * self.prefs.pref_food + 0.25 * self.prefs.pref_social)
            self._check_blob_reputation_exile(aggressor)
        else:
            aggressor.pending_rl_reward -= 0.22 * self.prefs.pref_food

        self.telemetry.on_raid_outcome(
            aggressor,
            defender,
            repelled=True,
            loot_kind="none",
            bounty=victim_is_outlaw,
        )
        defender.cumulative_reward += 0.28 * self.prefs.pref_social
        defender.pending_rl_reward += 0.28 * self.prefs.pref_social

        self.recent_raid_lines.append(
            {
                "x1": round(aggressor.x, 1),
                "y1": round(aggressor.y, 1),
                "x2": round(defender.x, 1),
                "y2": round(defender.y, 1),
                "repelled": True,
                "tick": self.tick_count,
            }
        )
        return True

    def _try_village_market_trade(
        self,
        blob: Blob,
        home: Home,
        groups_with_depot_food: Optional[Set[int]] = None,
        tod: Optional[float] = None,
    ) -> bool:
        """
        At home or in a settlement house, blobs can sell food for money into the depot
        (even dipping below reproduction or survival food if trade_desire is high!),
        or buy food from the settlement's depots.
        Reputation modulates prices: outlaws pay a risk surcharge when buying and receive a fence markdown when selling.
        """
        unit = 0.5
        surv_need = self.world_cfg.survival_food_need
        repro_need = self.world_cfg.reproduction_food_need

        can_sell = (
            (blob.surplus_food >= unit and blob.trade_desire >= -0.15)
            or (blob.total_food >= unit and blob.trade_desire > 0.25)
        ) and home.gold_vault >= 0.3

        can_buy = False
        if blob.money >= 0.4:
            has_depot_food = home.food_depot >= unit
            if not has_depot_food and blob._group_id is not None:
                if groups_with_depot_food is not None:
                    has_depot_food = blob._group_id in groups_with_depot_food
                elif blob._group_id in self.groups:
                    grp = self.groups[blob._group_id]
                    if len(grp.home_ids) > 1:
                        has_depot_food = any(
                            self.homes[hid].food_depot >= unit
                            for hid in grp.home_ids
                            if hid in self.homes
                        )
            if has_depot_food:
                ft = blob.food_today
                tod_val = tod if tod is not None else self.time_of_day
                needs_survival = ft < surv_need and (
                    tod_val > 0.55 or blob.energy < 55.0
                )
                close_to_repro = (
                    ft >= surv_need
                    and ft < repro_need
                    and (repro_need - ft) <= unit
                )
                wants_market_buy = blob.trade_desire > 0.25 and ft < repro_need
                can_buy = needs_survival or close_to_repro or wants_market_buy

        if not can_sell and not can_buy:
            return False

        discount = (
            (1.0 - self.world_cfg.group_trade_discount * 0.5)
            if (blob.group_id is not None and home.group_id == blob.group_id)
            else 1.0
        )
        base_price = home.local_unit_price(
            self.world_cfg.base_food_price, unit_amount=unit, discount=discount
        )

        can_sell_surplus = blob.surplus_food >= unit and blob.trade_desire >= -0.15
        can_sell_stomach = blob.total_food >= unit and blob.trade_desire > 0.25
        if can_sell and (can_sell_surplus or can_sell_stomach):
            sell_price = round(max(0.3, base_price * self._get_reputation_sell_multiplier(blob)), 2)
            if home.gold_vault >= sell_price:
                from_stomach = blob.deduct_traded_food(unit)
                if from_stomach > 0.0:
                    blob.energy = max(15.0, blob.energy - from_stomach * 18.0)
                blob.money = round(blob.money + sell_price, 2)
                home.gold_vault = max(0.0, round(home.gold_vault - sell_price, 2))
                home.food_depot = min(home.max_depot_capacity, round(home.food_depot + unit, 2))
                blob.trades_made += 1
                blob.adjust_reputation(0.035)
                blob.trade_cooldown = 25
                blob.last_trade_tick = blob.age_ticks
                self.trades_today += 1
                self.total_trades_all_time += 1
                if blob.group_id and blob.group_id in self.groups:
                    self.groups[blob.group_id].total_trades += 1
                self.telemetry.on_market_sell(blob, home, unit, sell_price, from_stomach)
                if blob.food_today < surv_need - 1e-4:
                    sell_pen = -1.0 * self.prefs.pref_food
                    blob.cumulative_reward += sell_pen
                    blob.pending_rl_reward += sell_pen
                else:
                    sell_rew = self.prefs.pref_money * 0.35
                    blob.cumulative_reward += sell_rew
                    blob.pending_rl_reward += sell_rew
                return True

        if can_buy:
            buy_price = round(max(0.4, base_price * self._get_reputation_buy_multiplier(blob)), 2)
            if blob.money >= buy_price:
                depot_home = self._find_settlement_home_with_food(blob, min_food=unit)
                if depot_home is not None and (blob.total_food + unit) <= blob.max_food_capacity + 1e-4:
                    depot_home.food_depot = round(depot_home.food_depot - unit, 2)
                    depot_home.gold_vault = round(depot_home.gold_vault + buy_price, 2)
                    blob.money = round(blob.money - buy_price, 2)
                    blob.add_acquired_food(unit, repro_need)
                    blob.energy = min(blob.max_energy, blob.energy + unit * 38.0)
                    blob.ticks_since_last_meal = 0
                    blob.trades_made += 1
                    blob.adjust_reputation(0.035)
                    blob.trade_cooldown = 25
                    blob.last_trade_tick = blob.age_ticks
                    self.trades_today += 1
                    self.total_trades_all_time += 1
                    if blob.group_id and blob.group_id in self.groups:
                        self.groups[blob.group_id].total_trades += 1
                    self.telemetry.on_market_buy(blob, depot_home, unit, buy_price)
                    buy_rew = self.prefs.pref_money * 0.35
                    blob.cumulative_reward += buy_rew
                    blob.pending_rl_reward += buy_rew
                    return True

        return False

    def _settle_blob_surplus_food(self, blob: Blob, home: Optional[Home], repro_need: float) -> None:
        if blob.surplus_food <= 0.0:
            return
        if blob.food_today < repro_need:
            transfer = min(blob.surplus_food, repro_need - blob.food_today)
            blob.food_today = round(blob.food_today + transfer, 3)
            blob.surplus_food = round(blob.surplus_food - transfer, 3)
        if blob.surplus_food > 0.0 and blob.is_at_home and home is not None:
            sell_mult = self._get_reputation_sell_multiplier(blob)
            earned = round(blob.surplus_food * self.world_cfg.base_food_price * 0.8 * sell_mult, 2)
            blob.money = round(blob.money + earned, 2)
            home.food_depot = round(home.food_depot + blob.surplus_food, 2)
            home.gold_vault = max(0.0, round(home.gold_vault - earned, 2))
            blob.trades_made += 1
            blob.adjust_reputation(0.03)
            self.trades_today += 1
            self.total_trades_all_time += 1
            self.telemetry.on_eod_surplus_sold(blob, home, blob.surplus_food, earned)
        blob.surplus_food = 0.0

    def _settle_blob_survival_deficit(
        self,
        blob: Blob,
        home: Optional[Home],
        survival_need: float,
        grp_depot_cache: Optional[Dict[int, Optional[Home]]] = None,
    ) -> None:
        if blob.food_today < survival_need:
            if blob.money >= 0.04:
                deficit = round(survival_need - blob.food_today, 3)
                buy_mult = self._get_reputation_buy_multiplier(blob)
                unit_price = round(max(0.5, self.world_cfg.base_food_price * buy_mult), 2)
                affordable = round(blob.money / unit_price, 3)
                if affordable >= 0.1:
                    target_depot = home if (home and home.food_depot >= 0.1) else (
                        grp_depot_cache.get(blob.group_id) if (grp_depot_cache is not None and blob.group_id is not None) else (
                            self._find_settlement_home_with_food(blob, min_food=0.1) if home is not None else None
                        )
                    )
                    if target_depot is not None and target_depot.food_depot >= 0.1:
                        actual_food = round(min(deficit, affordable, target_depot.food_depot), 3)
                        cost = round(actual_food * unit_price, 2)
                        blob.money = max(0.0, round(blob.money - cost, 2))
                        blob.food_today = round(blob.food_today + actual_food, 3)
                        blob.energy = min(blob.max_energy, blob.energy + actual_food * 35.0)
                        blob.trades_made += 1
                        self.trades_today += 1
                        self.total_trades_all_time += 1
                        target_depot.food_depot = max(0.0, round(target_depot.food_depot - actual_food, 2))
                        target_depot.gold_vault = round(target_depot.gold_vault + cost, 2)
                        self.telemetry.on_eod_deficit_bought(blob, target_depot, actual_food, cost)

        if blob.food_today >= survival_need - 1e-4:
            recovery = 45.0 if blob.is_at_home else 20.0
            blob.energy = min(blob.max_energy, blob.energy + recovery)
            blob.ticks_since_last_meal = 0
        else:
            remaining_deficit = survival_need - blob.food_today
            is_kindergarten = getattr(self.world_cfg, "scale_preset", "") == "training_kindergarten"
            base_outdoor = 2.5 if (is_kindergarten or self.day_number <= 3) else 8.0
            outdoor_penalty = 0.0 if blob.is_at_home else base_outdoor
            blob.energy -= 32.0 * (remaining_deficit / max(0.1, survival_need)) + outdoor_penalty

    def _apply_overnight_carryover_and_repro(
        self,
        blob: Blob,
        home: Optional[Home],
        survival_need: float,
        repro_need: float,
        new_babies: List[Blob],
    ) -> None:
        if blob.food_today >= repro_need - 1e-5:
            max_pop_cap = getattr(self.world_cfg, "max_population_cap", 20000)
            max_carrying_pop = min(max_pop_cap, max(250, int(self.world_cfg.daily_food_spawn * 1.15)))
            cur_total = len(self.blobs) + len(new_babies)
            if cur_total < max_pop_cap and (cur_total < max_carrying_pop or blob.is_at_home):
                if cur_total < min(max_pop_cap, int(max_carrying_pop * 1.2)):
                    blob.energy = max(25.0, blob.energy - 18.0)
                    new_babies.append(self._reproduce_blob(blob))
            blob.food_today = min(0.15, round(max(0.0, blob.food_today - repro_need), 2))
            return

        if blob.food_today > survival_need:
            leftover = round(blob.food_today - survival_need, 3)
            if blob.is_at_home and home is not None:
                to_depot = round(leftover * 0.5, 2)
                home.food_depot = min(18.0, round(home.food_depot + to_depot, 2))
                self.telemetry.on_eod_leftover_stored(blob, home, to_depot)
                if home.gold_vault >= 0.5:
                    stipend = min(home.gold_vault, round(to_depot * self.world_cfg.base_food_price * 0.6, 2))
                    home.gold_vault = round(home.gold_vault - stipend, 2)
                    blob.money = round(blob.money + stipend, 2)
                blob.food_today = min(0.35, round(leftover * 0.5, 2))
            else:
                blob.food_today = min(0.25, round(leftover * 0.4, 2))
            return

        blob.food_today = 0.0

    def _check_empire_secessions(self) -> None:
        """
        Check if any distant frontier colony cluster of a large Empire secedes (Schisma)
        into a new independent daughter faction.
        """
        max_groups_cap = getattr(self.world_cfg, "max_groups_cap", 120)
        if len(self.groups) >= max_groups_cap:
            return

        secession_dist = min(3200.0, max(1250.0, self.world_cfg.width * 0.22))
        blob_by_id = self._blobs_by_id

        for grp in list(self.groups.values()):
            if len(self.groups) >= max_groups_cap:
                break
            if len(grp.home_ids) < 3 or len(grp.member_ids) < 12:
                continue
            cap_home = self.homes.get(grp.home_id)
            if cap_home is None:
                continue

            distant_homes = [
                self.homes[hid]
                for hid in grp.home_ids
                if hid in self.homes
                and hid != cap_home.id
                and math.hypot(self.homes[hid].x - cap_home.x, self.homes[hid].y - cap_home.y) >= secession_dist
            ]
            if len(distant_homes) < 2:
                continue

            for anchor in distant_homes:
                cluster_homes = [
                    h for h in distant_homes if math.hypot(h.x - anchor.x, h.y - anchor.y) <= 1100.0
                ]
                if len(cluster_homes) < 2:
                    continue

                cluster_hids = {h.id for h in cluster_homes}
                cluster_blobs = [
                    blob_by_id[mid]
                    for mid in grp.member_ids
                    if mid in blob_by_id and blob_by_id[mid].home_id in cluster_hids
                ]
                if len(cluster_blobs) < 6 or (len(grp.member_ids) - len(cluster_blobs)) < 4:
                    continue

                new_gid = self._next_group_id
                self._next_group_id += 1
                name, color, rgb_int = generate_faction_identity(new_gid)
                daughter = Group(
                    id=new_gid,
                    name=name,
                    color=color,
                    home_id=anchor.id,
                    home_ids=[h.id for h in cluster_homes],
                    member_ids=[b.id for b in cluster_blobs],
                    created_day=self.day_number,
                    parent_group_id=grp.id,
                    rgb_int=rgb_int,
                )
                self.groups[new_gid] = daughter
                grp.secessions += 1
                self.telemetry.on_secession(grp, daughter, cluster_homes, cluster_blobs)

                for h in cluster_homes:
                    h.group_id = new_gid
                    if h.id in grp.home_ids:
                        grp.home_ids.remove(h.id)
                for b in cluster_blobs:
                    b.group_id = new_gid
                    if b.id in grp.member_ids:
                        grp.member_ids.remove(b.id)

                self.log_event(
                    "group",
                    f"🚩 Schisma: Ferne Kolonie um Haus #{anchor.id} spaltet sich von {grp.name} ab → {name}!",
                )
                break

    def _stage_eod_home_maintenance(self, living_blobs: Optional[List[Blob]] = None) -> None:
        """Phase 1: Settlement & Home Maintenance (executed at dusk/night, tick tpd - 2)."""
        if getattr(self, "_eod_maintenance_done", False):
            return
        if living_blobs is None:
            living_blobs = list(self._compact_living_blobs())

        # Overnight oasis recovery & location-weighted house depot regeneration
        self._cached_overgrazed_oases = None
        for o in self.oases.values():
            rec_mult = float(getattr(o, "recovery_rate", 1.0))
            o.harvest_pressure *= max(0.35, 0.55 - 0.10 * rec_mult)
            if o.harvest_pressure < 2.0:
                o.fertility = round(min(1.0, o.fertility + 0.075 * rec_mult), 3)

        self._refresh_all_home_locations()
        for h in self.homes.values():
            if h.raid_damage > 0:
                h.raid_damage = max(0, h.raid_damage - 1)
            zone_mult = 0.50 if getattr(h, "zone", "cradle") == "cradle" else 1.0
            loc_mult = max(0.10, h.fertility_bonus) * zone_mult
            regen = ((0.32 if len(h.resident_ids) > 0 else 0.08) + 0.25 * h.granary_level) * loc_mult
            h.food_depot = min(h.max_depot_capacity, round(h.food_depot + regen, 2))
            h.gold_vault = min(90.0, round(h.gold_vault + 2.0, 2))

        self._process_house_taxes_and_upgrades(living_blobs)
        self._eod_maintenance_done = True

    def _stage_eod_settlements_and_learning(self, living_blobs: Optional[List[Blob]] = None) -> None:
        """Phase 2: Biological Markets & Learning (executed at late night, tick tpd - 1)."""
        if getattr(self, "_eod_settlements_done", False):
            return
        if living_blobs is None:
            living_blobs = list(self._compact_living_blobs())
        n_living = len(living_blobs)
        survival_need = self.world_cfg.survival_food_need
        repro_need = self.world_cfg.reproduction_food_need
        self.telemetry.on_eod_pre_accounting(living_blobs, survival_need, repro_need)

        blob_homes: List[Optional[Home]] = [None] * n_living
        eod_rewards = np.empty(n_living, dtype=np.float32)
        home_rew_pos = 0.35 * self.prefs.pref_home
        home_rew_neg = -0.25 * self.prefs.pref_home
        starve_pen = -1.0 * self.prefs.pref_food
        repro_rew = self.prefs.pref_food * 1.4

        # Precompute top-2 mates per group in O(N) so reproduction is O(1) per parent
        self._eod_group_mates: Dict[int, List[Blob]] = {}
        for b in living_blobs:
            if b.group_id is not None:
                lst = self._eod_group_mates.setdefault(b.group_id, [])
                lst.append(b)
        for gid, lst in self._eod_group_mates.items():
            grp_obj = self.groups.get(gid)
            grp_avg_rep = self._get_group_avg_reputation(grp_obj) if grp_obj is not None else 0.0
            rep_weight = 4.5 if grp_avg_rep >= -0.25 else 0.5
            lst.sort(
                key=lambda m, rw=rep_weight: (
                    m.food_eaten_lifetime * 1.5
                    + m.children_count * 3.0
                    + m.cumulative_reward * 0.3
                    + m.reputation * rw
                ),
                reverse=True,
            )
            if len(lst) > 2:
                del lst[2:]

        # Precompute settlement depot homes with food >= 0.1 for O(1) deficit buying
        grp_depot_cache = {}
        for gid, grp in self.groups.items():
            depot_h = None
            for hid in grp.home_ids:
                h = self.homes.get(hid)
                if h is not None and h.food_depot >= 0.1:
                    depot_h = h
                    break
            grp_depot_cache[gid] = depot_h

        for idx, blob in enumerate(living_blobs):
            blob.age_days += 1
            # Daily reputation decay (~12%/day toward 0.0) allows desperate one-time thieves to rehabilitate
            if blob.reputation != 0.0:
                decayed_rep = round(blob.reputation * 0.88, 3)
                blob.reputation = 0.0 if abs(decayed_rep) < 0.01 else decayed_rep
            if blob.reputation <= -0.45:
                self._check_blob_reputation_exile(blob)

            home = blob.home_obj or self.homes.get(blob.home_id)
            blob_homes[idx] = home
            self._settle_blob_surplus_food(blob, home, repro_need)
            self._settle_blob_survival_deficit(blob, home, survival_need, grp_depot_cache=grp_depot_cache)

            if not self.world_cfg.intrinsic_homeostasis:
                rew = home_rew_pos if blob.is_at_home else home_rew_neg
                if blob.energy <= 0.0:
                    rew += starve_pen
                elif blob.food_today >= repro_need - 1e-5:
                    rew += repro_rew
                eod_rewards[idx] = rew
                blob.cumulative_reward += rew
            else:
                eod_rewards[idx] = 0.0

        for grp in self.groups.values():
            grp.rep_dirty = True

        if n_living > 0 and not self.world_cfg.intrinsic_homeostasis:
            self._brain_pool.apply_online_learning_batch(
                [b.brain for b in living_blobs], eod_rewards, self.prefs.learning_rate
            )

        self._eod_blob_homes = blob_homes
        self._eod_settlements_done = True

    def _stage_eod_reproduction_and_rollover(
        self, living_blobs: Optional[List[Blob]] = None, amortize_food: bool = True
    ) -> None:
        """Phase 3: Reproduction, Compaction & Midnight Rollover (executed at tick tpd)."""
        if not getattr(self, "_eod_maintenance_done", False):
            self._stage_eod_home_maintenance(living_blobs)
        if not getattr(self, "_eod_settlements_done", False):
            self._stage_eod_settlements_and_learning(living_blobs)

        if living_blobs is None:
            living_blobs = list(self._compact_living_blobs())

        survival_need = self.world_cfg.survival_food_need
        repro_need = self.world_cfg.reproduction_food_need
        new_babies: List[Blob] = []
        blob_homes = getattr(self, "_eod_blob_homes", None)
        if blob_homes is None or len(blob_homes) != len(living_blobs):
            blob_homes = [b.home_obj or self.homes.get(b.home_id) for b in living_blobs]

        for idx, blob in enumerate(living_blobs):
            if blob.energy <= 0.0:
                self._kill_blob(blob, reason="verhungert am Tagesende")
                continue

            self._apply_overnight_carryover_and_repro(
                blob, blob_homes[idx], survival_need, repro_need, new_babies
            )

        if hasattr(self, "_eod_group_mates"):
            self._eod_group_mates.clear()
        self._eod_blob_homes = None
        self._eod_maintenance_done = False
        self._eod_settlements_done = False

        self._update_hall_of_fame()
        self._compact_living_blobs()
        self.blobs.extend(new_babies)
        if new_babies:
            base_idx = len(self.blobs) - len(new_babies)
            for b_i, b in enumerate(new_babies):
                b._sim_idx = base_idx + b_i
            self._kin_static_dirty = True
        self._cleanup_groups_and_homes()
        self._check_empire_secessions()

        if len(self.blobs) < self.world_cfg.min_population_floor:
            self._respawn_from_hall_of_fame()

        self.record_daily_snapshot()
        self.day_number += 1
        self.tick_in_day = 0
        self.births_today = 0
        self.deaths_today = 0
        self.trades_today = 0
        self.raids_today = 0
        self.raids_repelled_today = 0
        self.exiles_today = 0
        self.bounties_today = 0

        if ((self.day_number - 1) % 4) == 0:
            _sid, slabel, w_mult, _g, _c = self.season_info
            self.log_event(
                "season",
                f"Jahreszeitenwechsel: {slabel} beginnt (Wildnahrung {int(w_mult * 100)}%)!",
            )

        daily_food = self.world_cfg.daily_food_spawn
        if amortize_food:
            spawn_now = min(1200, daily_food)
            self.spawn_daily_food(spawn_now)
            self._pending_food_spawn = max(0, daily_food - spawn_now)
        else:
            self.spawn_daily_food(daily_food)
            self._pending_food_spawn = 0
        self._spawn_settlement_gardens()

    def _end_of_day_accounting(self, amortize_food: bool = False) -> None:
        """
        Execute end-of-day rules (synchronous full fallback):
        - Sell any remaining surplus_food at home or lose unstored surplus
        - Blobs with >= 1x survival_food_need regenerate energy overnight (+45 at home, +20 outdoors)
        - Blobs with < 1x survival_food_need try to buy missing food; else lose moderate energy
        - Blobs with 1x..2x food store half in home depot and carry up to 0.35 into next morning
        - Blobs with >= 2x survival_food_need reproduce a new blob with their genes!
        """
        living = list(self._compact_living_blobs())
        if not getattr(self, "_eod_maintenance_done", False):
            self._stage_eod_home_maintenance(living)
        if not getattr(self, "_eod_settlements_done", False):
            self._stage_eod_settlements_and_learning(living)
        self._stage_eod_reproduction_and_rollover(living, amortize_food=amortize_food)

    @staticmethod
    def _select_next_home_upgrade(h: Home) -> str:
        if (h.recent_raids > 0 or (h.food_depot >= 4.5 and h.defense_level < h.farm_level)) and h.defense_level < 3:
            return "defense"
        if (h.is_full or h.food_depot >= h.max_depot_capacity * 0.75) and h.granary_level < 3:
            return "granary"
        if h.farm_level < 3:
            return "farm"
        if h.granary_level < 3:
            return "granary"
        return "defense" if h.defense_level < 3 else ""

    def _apply_home_upgrade(self, h: Home, choice: str) -> None:
        self.telemetry.on_home_upgrade(h, choice)
        grp_tag = (
            f" ({self.groups[h.group_id].name})"
            if (h.group_id is not None and h.group_id in self.groups)
            else ""
        )
        if choice == "farm":
            h.farm_level += 1
            self.log_event("upgrade", f"Haus #{h.id}{grp_tag} baut 🌾 Farm Lv.{h.farm_level} aus!")
        elif choice == "granary":
            h.granary_level += 1
            h.max_residents += 2
            self.log_event(
                "upgrade",
                f"Haus #{h.id}{grp_tag} baut 🏛️ Speicher Lv.{h.granary_level} (+2 Wohnplätze) aus!",
            )
        else:
            h.defense_level += 1
            h.recent_raids = 0
            self.log_event(
                "upgrade",
                f"Haus #{h.id}{grp_tag} errichtet 🛡️ Wachturm Lv.{h.defense_level}!",
            )

    def _process_house_taxes_and_upgrades(self, living_blobs: List[Blob]) -> None:
        """Collect overnight wealth contributions (>12🪙) and build needed house upgrades (🌾, 🏛️, 🛡️)."""
        for blob in living_blobs:
            if not blob.alive or blob.money <= 12.0:
                continue
            home = self.homes.get(blob.home_id)
            if home is None:
                continue
            contribution = round((blob.money - 12.0) * 0.20, 2)
            if contribution >= 0.4:
                blob.money = round(blob.money - contribution, 2)
                home.gold_vault = min(120.0, round(home.gold_vault + contribution, 2))
                blob.adjust_reputation(0.07)
                blob.cumulative_reward += 0.12 * self.prefs.pref_home + 0.08 * self.prefs.pref_social
                self.telemetry.on_tax_paid(blob, home, contribution)

        for h in self.homes.values():
            if len(h.resident_ids) == 0 or h.total_upgrades >= 9:
                continue
            cost = 18.0 + h.total_upgrades * 9.0
            if h.gold_vault < cost:
                continue

            choice = self._select_next_home_upgrade(h)
            if choice:
                h.gold_vault = round(max(0.0, h.gold_vault - cost), 2)
                self.total_upgrades_built += 1
                self._apply_home_upgrade(h, choice)

    def _reproduce_blob(self, parent: Blob, partner: Optional[Blob] = None) -> Blob:
        """Produce a new child blob inheriting parent's trained neural network & physical genes."""
        if partner is None and parent.group_id is not None and parent.group_id in self.groups:
            eod_mates = getattr(self, "_eod_group_mates", None)
            if eod_mates and parent.group_id in eod_mates:
                for cand in eod_mates[parent.group_id]:
                    if cand.id != parent.id and cand.alive:
                        partner = cand
                        break
            else:
                grp = self.groups[parent.group_id]
                mates = [
                    self._blobs_by_id[mid]
                    for mid in grp.member_ids
                    if mid != parent.id and mid in self._blobs_by_id and self._blobs_by_id[mid].alive
                ]
                if mates:
                    grp_avg_rep = self._get_group_avg_reputation(grp)
                    rep_weight = 4.5 if grp_avg_rep >= -0.25 else 0.5
                    partner = max(
                        mates,
                        key=lambda m: (
                            m.food_eaten_lifetime * 1.5
                            + m.children_count * 3.0
                            + m.cumulative_reward * 0.3
                            + m.reputation * rep_weight
                        ),
                    )

        child_brain = parent.brain.clone_and_mutate(
            mutation_rate=self.prefs.mutation_rate,
            mutation_strength=self.prefs.mutation_strength,
            partner=partner.brain if partner else None,
        )
        child_genes = parent.genes.mutate(
            rng=self.rng,
            mutation_rate=self.prefs.mutation_rate,
            mutation_strength=self.prefs.mutation_strength,
            partner=partner.genes if partner else None,
        )

        starter_money = 4.0
        if parent.money > 10.0:
            gift = round(parent.money * 0.2, 1)
            parent.money = round(parent.money - gift, 2)
            starter_money += gift

        # Offspring inherit 25% of their parents' social reputation (honor or stigma)
        parent_rep = (
            0.5 * (parent.reputation + partner.reputation)
            if partner is not None
            else parent.reputation
        )
        child_reputation = round(0.25 * parent_rep, 3)

        parent_home = self.homes.get(parent.home_id)
        if parent_home is not None and not parent_home.is_full:
            child_home = parent_home
        else:
            child_home = self._find_or_create_settlement_home(
                parent.group_id, parent.home_x, parent.home_y
            )

        bx = float(np.clip(child_home.x + self.rng.normal(0.0, 14.0), 20.0, self.world_cfg.width - 20.0))
        by = float(np.clip(child_home.y + self.rng.normal(0.0, 14.0), 20.0, self.world_cfg.height - 20.0))

        child = Blob(
            blob_id=self._next_blob_id,
            x=bx,
            y=by,
            home_id=child_home.id,
            home_x=child_home.x,
            home_y=child_home.y,
            brain=child_brain,
            genes=child_genes,
            generation=parent.generation + 1,
            parent_id=parent.id,
            group_id=parent.group_id,
            money=starter_money,
            reputation=child_reputation,
        )
        self._next_blob_id += 1
        parent.children_count += 1
        self.births_today += 1
        self.total_births_all_time += 1
        self.telemetry.on_birth(parent, partner, child, child_home)

        child_home.resident_ids.append(child.id)
        if parent.group_id is not None and parent.group_id in self.groups:
            self.groups[parent.group_id].member_ids.append(child.id)
            self._federate_home_into_group(child_home, self.groups[parent.group_id])

        if self.births_today <= 8:
            self.log_event(
                "birth",
                f"Blob #{parent.id} hat 2x Tagesnahrung erreicht → Kind #{child.id} (Gen {child.generation}) in Haus #{child_home.id} geboren!",
            )
        self._register_blob(child)
        return child

    def _kill_blob(self, blob: Blob, reason: str) -> None:
        if not blob.alive:
            return
        self.telemetry.on_blob_death(self, blob, reason)
        blob.alive = False
        blob.cause_of_death = reason
        self._unregister_blob(blob)
        self.deaths_today += 1
        self.total_deaths_all_time += 1
        self._consider_for_hall_of_fame(blob)
        if self.selected_blob_id == blob.id:
            living = [b for b in self.blobs if b.alive]
            self.selected_blob_id = living[0].id if living else None

        if self.world_cfg.continuous_lifecycle and len(self.foods) < (self.world_cfg.daily_food_spawn * 3 + 200):
            bx, by = blob.x, blob.y
            if not self._point_hits_rock(bx, by, pad=8.0):
                nutrition = round(max(0.35, 0.55 * blob.genes.size), 2)
                self.foods.append(
                    FoodItem(
                        id=self._next_food_id,
                        x=bx,
                        y=by,
                        nutrition=nutrition,
                        golden=False,
                        oasis_id=None,
                    )
                )
                self._next_food_id += 1
                self.food_version += 1
                self.recent_eaten_indices.clear()
                self._food_dirty = True

    def _consider_for_hall_of_fame(self, blob: Blob) -> None:
        score = float(
            blob.food_eaten_lifetime * 1.6
            + blob.age_days * 3.0
            + blob.children_count * 5.0
            + blob.trades_made * 0.8
            + blob.money * 0.15
            + blob.reputation * 3.5
            + blob.bounties_won * 1.5
            + blob.cumulative_reward * 0.4
        )
        existing_idx = next(
            (i for i, e in enumerate(self.hall_of_fame) if e.get("blob_id") == blob.id),
            None,
        )
        if existing_idx is not None:
            if score < self.hall_of_fame[existing_idx]["score"]:
                return
        elif len(self.hall_of_fame) >= 12 and score <= self.hall_of_fame[-1]["score"]:
            return

        entry = {
            "blob_id": blob.id,
            "generation": blob.generation,
            "score": round(score, 2),
            "age_days": blob.age_days,
            "children": blob.children_count,
            "money": round(blob.money, 1),
            "reputation": round(blob.reputation, 2),
            "genes": blob.genes.to_dict(),
            "brain": blob.brain.to_snapshot_dict(),
        }
        if existing_idx is not None:
            self.hall_of_fame[existing_idx] = entry
        else:
            self.hall_of_fame.append(entry)

        self.hall_of_fame.sort(key=lambda e: e["score"], reverse=True)
        self.hall_of_fame = self.hall_of_fame[:12]

    def _update_hall_of_fame(self) -> None:
        best_by_id: Dict[int, Tuple[float, Any]] = {}
        for e in self.hall_of_fame:
            bid = e.get("blob_id")
            if bid is not None:
                best_by_id[bid] = (float(e["score"]), e)

        for b in self.blobs:
            if b.alive and (b.age_days >= 1 or b.children_count >= 1):
                # Efficiency-normalized scoring (Section 4.5):
                # 1. Foraging efficiency with physical scale D_0 = 200.0 (prevents zero-movement exploit)
                foraging_eff = b.food_eaten_lifetime / (b.distance_traveled + 200.0)
                # 2. Homing success with Laplace smoothing (prevents Day-1 camper inflation)
                homing_eff = b.days_sheltered / (b.age_days + 2.0)

                s = float(
                    foraging_eff * 850.0
                    + homing_eff * 18.0
                    + b.children_count * 6.5
                    + b.age_days * 2.5
                    + b.trades_made * 0.8
                    + b.money * 0.12
                    + b._reputation * 3.5
                    + b.bounties_won * 1.5
                )
                prev = best_by_id.get(b.id)
                if prev is None or s > prev[0]:
                    best_by_id[b.id] = (s, b)

        top_items = sorted(best_by_id.values(), key=lambda x: x[0], reverse=True)[:12]
        new_hof = []
        for s, item in top_items:
            if isinstance(item, dict):
                item["score"] = round(s, 2)
                new_hof.append(item)
            else:
                new_hof.append({
                    "blob_id": item.id,
                    "generation": item.generation,
                    "score": round(s, 2),
                    "age_days": item.age_days,
                    "children": item.children_count,
                    "money": round(item.money, 1),
                    "reputation": round(item.reputation, 2),
                    "genes": item.genes.to_dict(),
                    "brain": item.brain.to_snapshot_dict(),
                })
        self.hall_of_fame = new_hof

    def _respawn_from_hall_of_fame(self) -> None:
        """Seed mutated offspring from the Hall of Fame if population drops below floor."""
        needed = max(0, self.world_cfg.min_population_floor - len([b for b in self.blobs if b.alive]))
        if needed <= 0:
            return
        home_list = list(self.homes.values())
        if not home_list:
            return
        self.telemetry.on_hof_rescue(needed, self.day_number, self.tick_count)

        for i in range(needed):
            home = home_list[i % len(home_list)]
            if home.is_full:
                home = self._find_or_create_settlement_home(home.group_id, home.x, home.y)
            if self.hall_of_fame:
                champ = self.hall_of_fame[i % len(self.hall_of_fame)]
                base_brain = BlobBrain.from_dict(champ["brain"], rng=self.rng).adapt_to_config(self.net_cfg)
                brain = base_brain.clone_and_mutate(
                    mutation_rate=self.prefs.mutation_rate,
                    mutation_strength=self.prefs.mutation_strength,
                )
                g_dict = champ["genes"]
                base_genes = Genes(
                    speed=g_dict.get("speed", 1.0),
                    sense=g_dict.get("sense", 1.0),
                    size=g_dict.get("size", 1.0),
                    metabolism=g_dict.get("metabolism", 1.0),
                    hue=g_dict.get("hue", 195.0),
                    drive_energy=g_dict.get("drive_energy", 1.0),
                    drive_pain=g_dict.get("drive_pain", 1.0),
                    drive_social=g_dict.get("drive_social", 0.3),
                    plasticity_lr=g_dict.get("plasticity_lr", 1.0),
                )
                genes = base_genes.mutate(self.rng, self.prefs.mutation_rate, self.prefs.mutation_strength)
                gen = champ.get("generation", 1) + 1
            else:
                innate_seed = float(self.get_active_paradigm()["innate_seed"])
                brain = BlobBrain(config=self.net_cfg, rng=self.rng, innate_seed=innate_seed)
                genes = Genes(hue=float(self.rng.uniform(0, 360)))
                gen = 1

            blob = Blob(
                blob_id=self._next_blob_id,
                x=float(home.x + self.rng.normal(0.0, 12.0)),
                y=float(home.y + self.rng.normal(0.0, 12.0)),
                home_id=home.id,
                home_x=home.x,
                home_y=home.y,
                brain=brain,
                genes=genes,
                generation=gen,
                group_id=home.group_id,
                money=6.0,
            )
            self._next_blob_id += 1
            home.resident_ids.append(blob.id)
            if home.group_id is not None and home.group_id in self.groups:
                self.groups[home.group_id].member_ids.append(blob.id)
            self._register_blob(blob)
            self.blobs.append(blob)
            blob._sim_idx = len(self.blobs) - 1
        if needed > 0:
            self._kin_static_dirty = True

    def _prune_group_homes(self, grp: Group) -> None:
        active_homes = [
            hid
            for hid in grp.home_ids
            if hid in self.homes and (len(self.homes[hid].resident_ids) > 0 or hid == grp.home_id)
        ]
        for hid in list(grp.home_ids):
            if hid not in active_homes and hid in self.homes:
                self.homes[hid].group_id = None
                self.homes[hid].radius = 28.0
        grp.home_ids = active_homes or ([grp.home_id] if grp.home_id in self.homes else [])
        if grp.home_ids and grp.home_id not in grp.home_ids:
            grp.home_id = grp.home_ids[0]

    def _cleanup_groups_and_homes(self) -> None:
        alive_ids = set(self._blobs_by_id.keys())
        for home in self.homes.values():
            home.resident_ids = [rid for rid in home.resident_ids if rid in alive_ids]

        empty_groups = []
        for gid, grp in self.groups.items():
            prev_len = len(grp.member_ids)
            grp.member_ids = [mid for mid in grp.member_ids if mid in alive_ids]
            if len(grp.member_ids) != prev_len:
                grp.rep_dirty = True
            self._prune_group_homes(grp)
            if len(grp.member_ids) == 0:
                empty_groups.append(gid)

        for gid in empty_groups:
            grp = self.groups.pop(gid)
            self.telemetry.on_group_extinct(grp, self.day_number, self.tick_count)
            for hid in grp.home_ids:
                if hid in self.homes:
                    self.homes[hid].group_id = None
                    self.homes[hid].radius = 28.0

        min_keep_homes = max(24, getattr(self.world_cfg, "num_cradles", 1) * 9)
        if len(self.homes) > min_keep_homes:
            empty_ungrouped = [
                hid for hid, h in self.homes.items() if len(h.resident_ids) == 0 and h.group_id is None
            ]
            pruned_ids = set()
            for hid in empty_ungrouped[: len(self.homes) - min_keep_homes]:
                self.homes.pop(hid, None)
                pruned_ids.add(hid)
            if pruned_ids and hasattr(self, "_oasis_to_homes"):
                for oid, homes_list in self._oasis_to_homes.items():
                    self._oasis_to_homes[oid] = [h for h in homes_list if h.id not in pruned_ids]

    def record_daily_snapshot(self) -> None:
        living = [b for b in self.blobs if b.alive]
        pop = len(living)
        for grp in self.groups.values():
            self._get_group_avg_reputation(grp)
        if pop > 0:
            if pop <= self._kin_cap and self._kin_populated and all(getattr(b, "_kin_ref", None) is self for b in living[:min(pop, 8)]):
                moneys_sub = self._kin_moneys[:pop]
                reps_sub = self._kin_reputations[:pop]
                avg_money = float(np.mean(moneys_sub))
                avg_reputation = float(np.mean(reps_sub))
                outlaws_count = int(np.count_nonzero(reps_sub <= -0.4))
                honorable_count = int(np.count_nonzero(reps_sub >= 0.4))
                grouped_count = int(np.count_nonzero(self._kin_has_groups[:pop]))
                g_speeds = self._kin_max_speeds[:pop] * (1.0 / 4.2)
                g_sizes = self._kin_gene_sizes[:pop]
                avg_speed = float(np.mean(g_speeds))
                avg_size = float(np.mean(g_sizes))
                sum_sns = sum_rew = 0.0
                max_gen = 0
                for b in living:
                    sum_sns += b.genes.sense
                    sum_rew += b.cumulative_reward
                    if b.generation > max_gen:
                        max_gen = b.generation
                inv_pop = 1.0 / pop
                avg_sense = sum_sns * inv_pop
                avg_reward = sum_rew * inv_pop
            else:
                sum_m = sum_rep = sum_spd = sum_sns = sum_sz = sum_rew = 0.0
                outlaws_count = honorable_count = grouped_count = max_gen = 0
                for b in living:
                    sum_m += b.money
                    rep = b._reputation
                    sum_rep += rep
                    if rep <= -0.4:
                        outlaws_count += 1
                    elif rep >= 0.4:
                        honorable_count += 1
                    g = b.genes
                    sum_spd += g.speed
                    sum_sns += g.sense
                    sum_sz += g.size
                    sum_rew += b.cumulative_reward
                    if b.generation > max_gen:
                        max_gen = b.generation
                    if b._group_id is not None:
                        grouped_count += 1
                inv_pop = 1.0 / pop
                avg_money = sum_m * inv_pop
                avg_reputation = sum_rep * inv_pop
                avg_speed = sum_spd * inv_pop
                avg_sense = sum_sns * inv_pop
                avg_size = sum_sz * inv_pop
                avg_reward = sum_rew * inv_pop
        else:
            avg_money = 0.0
            avg_reputation = 0.0
            outlaws_count = 0
            honorable_count = 0
            avg_speed = 1.0
            avg_sense = 1.0
            avg_size = 1.0
            avg_reward = 0.0
            max_gen = 1
            grouped_count = 0

        snap = {
            "day": self.day_number,
            "population": pop,
            "births": self.births_today,
            "deaths": self.deaths_today,
            "trades": self.trades_today,
            "raids": self.raids_today,
            "exiles": self.exiles_today,
            "bounties": self.bounties_today,
            "avg_money": round(avg_money, 2),
            "avg_reputation": round(avg_reputation, 3),
            "outlaws": outlaws_count,
            "honorable": honorable_count,
            "avg_speed": round(avg_speed, 3),
            "avg_sense": round(avg_sense, 3),
            "avg_size": round(avg_size, 3),
            "avg_reward": round(avg_reward, 2),
            "max_gen": max_gen,
            "groups": len(self.groups),
            "grouped_blobs": grouped_count,
        }
        if self.history and self.history[-1]["day"] == self.day_number:
            self.history[-1] = snap
        else:
            self.history.append(snap)

        if len(self.history) > 120:
            self.history = self.history[-120:]
        self.telemetry.record_daily_snapshot(self)

    # --- Interactive God-Mode & Inspection Helpers ---
    def spawn_custom_food_cluster(self, x: float, y: float, count: int = 8) -> None:
        nearest_oasis = (
            min(self.oases.values(), key=lambda o: math.hypot(o.x - x, o.y - y))
            if self.oases
            else None
        )
        oid = (
            nearest_oasis.id
            if (nearest_oasis is not None and math.hypot(nearest_oasis.x - x, nearest_oasis.y - y) <= nearest_oasis.radius * 1.4)
            else None
        )
        for _ in range(count):
            fx = float(np.clip(x + self.rng.normal(0.0, 26.0), 15.0, self.world_cfg.width - 15.0))
            fy = float(np.clip(y + self.rng.normal(0.0, 26.0), 15.0, self.world_cfg.height - 15.0))
            if self._point_hits_rock(fx, fy, pad=6.0):
                continue
            is_golden = bool(self.rng.random() < 0.25)
            self.foods.append(
                FoodItem(
                    id=self._next_food_id,
                    x=fx,
                    y=fy,
                    nutrition=1.2 if is_golden else 0.6,
                    golden=is_golden,
                    oasis_id=oid,
                )
            )
            self._next_food_id += 1
        self.food_version += 1
        self.recent_eaten_indices.clear()
        self._sync_food_arrays()
        self.log_event("god", f"Nahrungs-Cluster ({count}x) bei ({int(x)}, {int(y)}) gespawnt.")

    def toggle_oasis_at(self, x: float, y: float) -> None:
        """
        God-Tool: If clicking inside an existing Oasis, toggle its state between Lush (100% fertility + food burst)
        and Drought/Overgrazed (10% fertility). If clicking in empty wasteland, create a new Oasis!
        """
        if self.oases:
            nearest = min(self.oases.values(), key=lambda o: math.hypot(o.x - x, o.y - y))
            dist = math.hypot(nearest.x - x, nearest.y - y)
            if dist <= nearest.radius * 1.15:
                if nearest.fertility >= 0.55:
                    nearest.fertility = 0.10
                    nearest.harvest_pressure = 8.0
                    self.foods = [f for f in self.foods if f.oasis_id != nearest.id]
                    nearest.current_food_count = 0
                    self._refresh_all_home_locations()
                    self.log_event("god", f"🏜️ Oase '{nearest.name}' ausgedörrt (Fruchtbarkeit 10%)!")
                else:
                    nearest.fertility = 1.0
                    nearest.harvest_pressure = 0.0
                    self.spawn_custom_food_cluster(nearest.x, nearest.y, count=12)
                    self._refresh_all_home_locations()
                    self.log_event("god", f"🌴 Oase '{nearest.name}' voll aufgeblüht (Fruchtbarkeit 100%)!")
                return

        oid = self._next_oasis_id
        self._next_oasis_id += 1
        name = OASIS_NAMES[(oid - 1) % len(OASIS_NAMES)]
        new_oasis = OasisNode(
            id=oid,
            name=name,
            x=float(np.clip(x, 180.0, self.world_cfg.width - 180.0)),
            y=float(np.clip(y, 180.0, self.world_cfg.height - 180.0)),
            radius=220.0,
            fertility=1.0,
            max_capacity=28,
        )
        self.oases[oid] = new_oasis
        self.spawn_custom_food_cluster(new_oasis.x, new_oasis.y, count=12)
        self._refresh_all_home_locations()
        self.log_event("god", f"🌴 Neue Oase '{name}' bei ({int(x)}, {int(y)}) erschaffen!")

    def toggle_rock_at(self, x: float, y: float) -> None:
        """
        God-Tool: If clicking on an existing Rock obstacle, remove it; otherwise place a new Rock ridge barrier.
        """
        for idx, ob in enumerate(self.obstacles):
            if ob.kind == "rock":
                d = self._point_to_seg_dist(x, y, ob.x1, ob.y1, ob.x2, ob.y2)
                if d <= ob.radius + 45.0:
                    removed = self.obstacles.pop(idx)
                    self._rebuild_obstacle_arrays()
                    self._refresh_all_home_locations()
                    self.log_event("god", f"⛏️ Fels-Riegel #{removed.id} bei ({int(x)}, {int(y)}) abgetragen.")
                    return

        angle = float(self.rng.uniform(0.0, math.pi))
        half_len = 135.0
        x1 = float(np.clip(x - math.cos(angle) * half_len, 60.0, self.world_cfg.width - 60.0))
        y1 = float(np.clip(y - math.sin(angle) * half_len, 60.0, self.world_cfg.height - 60.0))
        x2 = float(np.clip(x + math.cos(angle) * half_len, 60.0, self.world_cfg.width - 60.0))
        y2 = float(np.clip(y + math.sin(angle) * half_len, 60.0, self.world_cfg.height - 60.0))
        obs_id = self._next_obstacle_id
        self._next_obstacle_id += 1
        self.obstacles.append(
            TerrainObstacle(id=obs_id, kind="rock", x1=x1, y1=y1, x2=x2, y2=y2, radius=38.0)
        )
        self._rebuild_obstacle_arrays()
        self._refresh_all_home_locations()
        self.log_event("god", f"🪨 Neuer Fels-Riegel #{obs_id} bei ({int(x)}, {int(y)}) errichtet.")

    def spawn_custom_blob(self, x: float, y: float) -> Blob:
        nearest_home = None
        min_d = 999999.0
        for h in self.homes.values():
            if not h.is_full:
                d = math.hypot(h.x - x, h.y - y)
                if d < min_d:
                    min_d = d
                    nearest_home = h

        if nearest_home is None or min_d > 320.0:
            hid = self._next_home_id
            self._next_home_id += 1
            nearest_home = Home(
                id=hid,
                x=x,
                y=y,
                radius=48.0,
                max_residents=self.world_cfg.max_residents_per_home,
                food_depot=2.0,
                gold_vault=15.0,
            )
            self._evaluate_home_location(nearest_home)
            self.homes[hid] = nearest_home

        if self.hall_of_fame:
            champ = self.hall_of_fame[0]
            brain = BlobBrain.from_dict(champ["brain"], rng=self.rng).adapt_to_config(self.net_cfg)
            brain = brain.clone_and_mutate(self.prefs.mutation_rate, self.prefs.mutation_strength)
        else:
            innate_seed = float(self.get_active_paradigm()["innate_seed"])
            brain = BlobBrain(config=self.net_cfg, rng=self.rng, innate_seed=innate_seed)

        genes = Genes(
            speed=round(float(self.rng.uniform(0.9, 1.25)), 3),
            sense=round(float(self.rng.uniform(0.9, 1.25)), 3),
            size=round(float(self.rng.uniform(0.9, 1.15)), 3),
            hue=round(float(self.rng.uniform(0, 360)), 1),
        )
        blob = Blob(
            blob_id=self._next_blob_id,
            x=x,
            y=y,
            home_id=nearest_home.id,
            home_x=nearest_home.x,
            home_y=nearest_home.y,
            brain=brain,
            genes=genes,
            generation=1,
            group_id=nearest_home.group_id,
            money=10.0,
        )
        self._next_blob_id += 1
        nearest_home.resident_ids.append(blob.id)
        if nearest_home.group_id is not None and nearest_home.group_id in self.groups:
            self.groups[nearest_home.group_id].member_ids.append(blob.id)
        self._register_blob(blob)
        self.blobs.append(blob)
        blob._sim_idx = len(self.blobs) - 1
        self.selected_blob_id = blob.id
        self._kin_static_dirty = True
        self.log_event("god", f"Neuer Blob #{blob.id} bei ({int(x)}, {int(y)}) eingesetzt.")
        return blob

    def get_state_payload(self) -> Dict[str, Any]:
        """Build compact real-time state dictionary for the WebSocket frontend."""
        living = [b for b in self.blobs if b.alive]

        selected_blob: Optional[Blob] = None
        if self.selected_blob_id is not None:
            for b in living:
                if b.id == self.selected_blob_id:
                    selected_blob = b
                    break
        if selected_blob is None and living:
            selected_blob = max(
                living,
                key=lambda b: b.food_eaten_lifetime * 1.5 + b.children_count * 3.0 + b.cumulative_reward * 0.3,
            )
            self.selected_blob_id = selected_blob.id

        selected_inspector = None
        if selected_blob is not None:
            grp_color = (
                self.groups[selected_blob.group_id].color
                if (selected_blob.group_id is not None and selected_blob.group_id in self.groups)
                else None
            )
            grp_name = (
                self.groups[selected_blob.group_id].name
                if (selected_blob.group_id is not None and selected_blob.group_id in self.groups)
                else "Keine Siedlung (Einzelgänger)"
            )
            eff_vision = (
                selected_blob.vision_radius
                * (0.65 if self.is_night else 1.0)
                * (1.28 if selected_blob.group_id is not None else 1.0)
            )
            selected_inspector = {
                "blob": selected_blob.to_summary_dict(group_color=grp_color),
                "group_name": grp_name,
                "ticks_since_meal": selected_blob.ticks_since_last_meal,
                "vision_radius": round(eff_vision, 1),
                "brain": selected_blob.brain.to_inspector_payload(),
            }

        sid, slabel, w_mult, _g, _c = self.season_info
        avg_rep = round(float(np.mean([b.reputation for b in living])), 2) if living else 0.0
        outlaws_cnt = sum(1 for b in living if b.is_outlaw)
        honorable_cnt = sum(1 for b in living if b.is_honorable)
        return {
            "tick": self.tick_count,
            "day": self.day_number,
            "tick_in_day": self.tick_in_day,
            "ticks_per_day": self.world_cfg.ticks_per_day,
            "time_of_day": round(self.time_of_day, 3),
            "is_night": self.is_night,
            "season": {
                "id": sid,
                "label": slabel,
                "day_in_season": ((self.day_number - 1) % 4) + 1,
                "food_mult": w_mult,
            },
            "world": {
                "width": self.world_cfg.width,
                "height": self.world_cfg.height,
                "scale_preset": getattr(self.world_cfg, "scale_preset", "classic"),
                "num_cradles": getattr(self.world_cfg, "num_cradles", 1),
                "initial_blobs": self.world_cfg.initial_blobs,
                "max_population_cap": getattr(self.world_cfg, "max_population_cap", 1200),
                "daily_food_spawn": self.world_cfg.daily_food_spawn,
                "survival_need": self.world_cfg.survival_food_need,
                "repro_need": self.world_cfg.reproduction_food_need,
                "food_price": self.world_cfg.base_food_price,
                "max_residents_per_home": self.world_cfg.max_residents_per_home,
                "start_paradigm": getattr(self.world_cfg, "start_paradigm", "curriculum"),
                "paradigm": self.get_active_paradigm(),
                "zones": getattr(self, "map_zones", []),
                "skill_gates": getattr(self, "skill_gates", []),
            },
            "stats": {
                "population": len(living),
                "foods_count": len(self.foods),
                "oases_count": len(self.oases),
                "avg_oasis_fertility": (
                    round(float(np.mean([o.fertility for o in self.oases.values()])), 2)
                    if self.oases
                    else 1.0
                ),
                "groups_count": len(self.groups),
                "homes_count": len(self.homes),
                "avg_reputation": avg_rep,
                "outlaws_count": outlaws_cnt,
                "honorable_count": honorable_cnt,
                "births_today": self.births_today,
                "deaths_today": self.deaths_today,
                "trades_today": self.trades_today,
                "raids_today": self.raids_today,
                "raids_repelled_today": self.raids_repelled_today,
                "exiles_today": self.exiles_today,
                "bounties_today": self.bounties_today,
                "total_trades": self.total_trades_all_time,
                "total_births": self.total_births_all_time,
                "total_deaths": self.total_deaths_all_time,
                "total_raids": self.total_raids_all_time,
                "total_repelled": self.total_raids_repelled,
                "total_exiles": self.total_exiles_all_time,
                "total_bounties": self.total_bounties_all_time,
                "upgrades_built": self.total_upgrades_built,
                "total_migrations": self.total_migrations_all_time,
            },
            "blobs": [
                b.to_summary_dict(
                    group_color=self.groups[b.group_id].color
                    if (b.group_id is not None and b.group_id in self.groups)
                    else None
                )
                for b in living
            ],
            "foods": [
                {"id": f.id, "x": round(f.x, 1), "y": round(f.y, 1), "n": f.nutrition, "g": f.golden}
                for f in self.foods
            ],
            "oases": [o.to_dict() for o in self.oases.values()],
            "obstacles": getattr(self, "_obstacles_dict_cache", None) or [ob.to_dict() for ob in self.obstacles],
            "homes": [h.to_dict() for h in self.homes.values()],
            "groups": [g.to_dict() for g in self.groups.values()],
            "trade_lines": self.recent_trade_lines,
            "raid_lines": self.recent_raid_lines,
            "selected": selected_inspector,
            "history": self.history[-50:],
            "events": self.event_log[-18:],
            "net_cfg": self.net_cfg.to_dict(),
            "prefs": self.prefs.to_dict(),
        }

    def get_ws_state_and_binary(
        self,
        include_topology: bool = True,
        include_foods: bool = True,
        viewport: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[Dict[str, Any], bytes]:
        """
        Build lightweight JSON metadata + packed Float32 binary entity buffer for 60 FPS WebSocket streaming.
        Binary Header (16 bytes, little-endian uint32):
          [magic=0x424C4F42, n_blobs, n_foods, tick_count]
        Followed by:
          - `blobs_f32`: (n_blobs, 8) float32 -> [x, y, heading, r, food_today, id, rgb_int, flags]
          - `foods_f32`: (n_foods, 3) float32 -> [x, y, golden] (omitted when include_foods=False, setting n_foods=0)
        """
        living = [b for b in self.blobs if b.alive]
        n_blobs = len(living)
        n_foods = len(self.foods)

        # Synchronize static arrays if dirty or if population / living list mutated
        if (
            getattr(self, "_kin_ids", None) is None
            or len(self._kin_ids) < n_blobs
            or n_blobs != len(self.blobs)
            or getattr(self, "_kin_static_dirty", True)
        ):
            self._sync_static_blob_kin_arrays(living)
            self._kin_static_dirty = False
        self._update_group_rgb_table()

        selected_blob: Optional[Blob] = None
        if self.selected_blob_id is not None:
            selected_blob = self._blobs_by_id.get(self.selected_blob_id)
            if selected_blob is not None and not selected_blob.alive:
                selected_blob = None
        if selected_blob is None and living:
            selected_blob = max(
                living,
                key=lambda b: b.food_eaten_lifetime * 1.5 + b.children_count * 3.0 + b.cumulative_reward * 0.3,
            )
            self.selected_blob_id = selected_blob.id

        selected_inspector = None
        if selected_blob is not None:
            grp_color = (
                self.groups[selected_blob.group_id].color
                if (selected_blob.group_id is not None and selected_blob.group_id in self.groups)
                else None
            )
            grp_name = (
                self.groups[selected_blob.group_id].name
                if (selected_blob.group_id is not None and selected_blob.group_id in self.groups)
                else "Keine Siedlung (Einzelgänger)"
            )
            eff_vision = (
                selected_blob.vision_radius
                * (0.65 if self.is_night else 1.0)
                * (1.28 if selected_blob.group_id is not None else 1.0)
            )
            selected_inspector = {
                "blob": selected_blob.to_summary_dict(group_color=grp_color),
                "group_name": grp_name,
                "ticks_since_meal": selected_blob.ticks_since_last_meal,
                "vision_radius": round(eff_vision, 1),
                "brain": selected_blob.brain.to_inspector_payload(),
            }

        # Vectorized Blob Packing (Feature 16) with optional Viewport Culling (Feature 17)
        if viewport is not None and n_blobs > 0:
            min_x, min_y, max_x, max_y = viewport
            xs = self._kin_xs[:n_blobs]
            ys = self._kin_ys[:n_blobs]
            vis_mask = (xs >= min_x) & (xs <= max_x) & (ys >= min_y) & (ys <= max_y)
            vis_idx = np.flatnonzero(vis_mask)
            n_blobs_packed = len(vis_idx)
            blobs_f32 = np.empty((n_blobs_packed, 8), dtype=np.float32)
            if n_blobs_packed > 0:
                rep_sum, outlaws_cnt, honorable_cnt = pack_binary_blobs_kernel(
                    blobs_f32,
                    self._kin_xs[vis_idx],
                    self._kin_ys[vis_idx],
                    self._kin_headings[vis_idx],
                    self._kin_body_radii[vis_idx],
                    self._kin_foods_today[vis_idx],
                    self._kin_ids[vis_idx],
                    self._kin_gids[vis_idx],
                    self._group_rgb_table,
                    self._kin_gene_rgbs[vis_idx],
                    self._kin_is_at_homes[vis_idx],
                    self._kin_in_swamps[vis_idx],
                    self._kin_surplus_foods[vis_idx],
                    self._kin_sprints[vis_idx],
                    self._kin_reputations[vis_idx],
                    n_blobs_packed,
                )
            else:
                rep_sum = 0.0
                outlaws_cnt = 0
                honorable_cnt = 0
        else:
            n_blobs_packed = n_blobs
            blobs_f32 = np.empty((n_blobs_packed, 8), dtype=np.float32)
            if n_blobs_packed > 0:
                rep_sum, outlaws_cnt, honorable_cnt = pack_binary_blobs_kernel(
                    blobs_f32,
                    self._kin_xs,
                    self._kin_ys,
                    self._kin_headings,
                    self._kin_body_radii,
                    self._kin_foods_today,
                    self._kin_ids,
                    self._kin_gids,
                    self._group_rgb_table,
                    self._kin_gene_rgbs,
                    self._kin_is_at_homes,
                    self._kin_in_swamps,
                    self._kin_surplus_foods,
                    self._kin_sprints,
                    self._kin_reputations,
                    n_blobs_packed,
                )
            else:
                rep_sum = 0.0
                outlaws_cnt = 0
                honorable_cnt = 0

        # Vectorized Food Packing & Stationary Food Caching (Features 15 & 16)
        if include_foods and n_foods > 0:
            if getattr(self, "_food_dirty", True) or getattr(self, "_food_xs", None) is None or len(self._food_xs) != n_foods:
                self._sync_food_arrays()
            foods_f32 = np.empty((n_foods, 3), dtype=np.float32)
            foods_f32[:, 0] = self._food_xs
            foods_f32[:, 1] = self._food_ys
            foods_f32[:, 2] = (self._food_base_reaches > 9.0).astype(np.float32)
            food_bytes = foods_f32.tobytes()
            n_foods_packed = n_foods
        else:
            food_bytes = b""
            n_foods_packed = 0

        header = struct.pack("<IIII", 0x424C4F42, n_blobs_packed, n_foods_packed, self.tick_count)
        binary_payload = header + blobs_f32.tobytes() + food_bytes

        sid, slabel, w_mult, _g, _c = self.season_info
        world_dict: Dict[str, Any] = {
            "width": self.world_cfg.width,
            "height": self.world_cfg.height,
            "scale_preset": getattr(self.world_cfg, "scale_preset", "classic"),
            "num_cradles": getattr(self.world_cfg, "num_cradles", 1),
            "initial_blobs": self.world_cfg.initial_blobs,
            "max_population_cap": getattr(self.world_cfg, "max_population_cap", 1200),
            "daily_food_spawn": self.world_cfg.daily_food_spawn,
            "survival_need": self.world_cfg.survival_food_need,
            "repro_need": self.world_cfg.reproduction_food_need,
            "food_price": self.world_cfg.base_food_price,
            "max_residents_per_home": self.world_cfg.max_residents_per_home,
            "start_paradigm": getattr(self.world_cfg, "start_paradigm", "curriculum"),
            "paradigm": self.get_active_paradigm(),
        }
        if include_topology:
            world_dict["zones"] = getattr(self, "map_zones", [])
            world_dict["skill_gates"] = getattr(self, "skill_gates", [])

        eaten_copy = list(getattr(self, "recent_eaten_indices", []))
        if hasattr(self, "recent_eaten_indices"):
            self.recent_eaten_indices.clear()

        meta: Dict[str, Any] = {
            "binary_frame": True,
            "topology_version": self.topology_version,
            "food_version": getattr(self, "food_version", 1),
            "eaten_foods": eaten_copy,
            "tick": self.tick_count,
            "day": self.day_number,
            "tick_in_day": self.tick_in_day,
            "ticks_per_day": self.world_cfg.ticks_per_day,
            "time_of_day": round(self.time_of_day, 3),
            "is_night": self.is_night,
            "season": {
                "id": sid,
                "label": slabel,
                "day_in_season": ((self.day_number - 1) % 4) + 1,
                "food_mult": w_mult,
            },
            "world": world_dict,
            "stats": {
                "population": n_blobs,
                "foods_count": n_foods,
                "oases_count": len(self.oases),
                "avg_oasis_fertility": (
                    round(float(np.mean([o.fertility for o in self.oases.values()])), 2)
                    if self.oases
                    else 1.0
                ),
                "groups_count": len(self.groups),
                "homes_count": len(self.homes),
                "avg_reputation": round(rep_sum / n_blobs_packed, 2) if n_blobs_packed > 0 else 0.0,
                "outlaws_count": outlaws_cnt,
                "honorable_count": honorable_cnt,
                "births_today": self.births_today,
                "deaths_today": self.deaths_today,
                "trades_today": self.trades_today,
                "raids_today": self.raids_today,
                "raids_repelled_today": self.raids_repelled_today,
                "exiles_today": self.exiles_today,
                "bounties_today": self.bounties_today,
                "total_trades": self.total_trades_all_time,
                "total_births": self.total_births_all_time,
                "total_deaths": self.total_deaths_all_time,
                "total_raids": self.total_raids_all_time,
                "total_repelled": self.total_raids_repelled,
                "total_exiles": self.total_exiles_all_time,
                "total_bounties": self.total_bounties_all_time,
                "upgrades_built": self.total_upgrades_built,
                "total_migrations": self.total_migrations_all_time,
            },
            "oases": [o.to_dict() for o in self.oases.values()],
            "homes": [h.to_dict() for h in self.homes.values()],
            "groups": [g.to_dict() for g in self.groups.values()],
            "trade_lines": self.recent_trade_lines[-40:],
            "raid_lines": self.recent_raid_lines[-40:],
            "selected": selected_inspector,
            "history": self.history[-50:],
            "events": self.event_log[-18:],
            "net_cfg": self.net_cfg.to_dict(),
            "prefs": self.prefs.to_dict(),
        }
        if include_topology:
            meta["obstacles"] = getattr(self, "_obstacles_dict_cache", None) or [
                ob.to_dict() for ob in self.obstacles
            ]

        return meta, binary_payload

    def export_population_json(self) -> Dict[str, Any]:
        """Export full simulation configuration, Hall of Fame, and living blobs as JSON."""
        living = [b for b in self.blobs if b.alive]
        hof_serialized = []
        for entry in self.hall_of_fame:
            b_dict = entry.get("brain", {})
            hof_serialized.append(
                {
                    **entry,
                    "brain": {
                        **b_dict,
                        "weights": [
                            w.tolist() if isinstance(w, np.ndarray) else w
                            for w in b_dict.get("weights", [])
                        ],
                        "biases": [
                            b.tolist() if isinstance(b, np.ndarray) else b
                            for b in b_dict.get("biases", [])
                        ],
                    },
                }
            )
        return {
            "version": "1.1.0",
            "day": self.day_number,
            "net_cfg": self.net_cfg.to_dict(),
            "prefs": self.prefs.to_dict(),
            "world_cfg": self.world_cfg.to_dict(),
            "hall_of_fame": hof_serialized,
            "blobs": [
                {
                    "generation": b.generation,
                    "money": round(b.money, 2),
                    "age_days": b.age_days,
                    "children": b.children_count,
                    "genes": b.genes.to_dict(),
                    "brain": b.brain.to_dict(),
                }
                for b in living[:100]
            ],
        }

    def _apply_imported_configs(self, data: Dict[str, Any]) -> None:
        if "net_cfg" in data:
            nc = data["net_cfg"]
            self.net_cfg = NetworkConfig(
                depth=nc.get("depth", 2),
                width=nc.get("width", 24),
                custom_layers=nc.get("custom_layers"),
                activation=nc.get("activation", "tanh"),
                recurrent_dim=nc.get("recurrent_dim", 0),
            )
        if "prefs" in data:
            for k, v in data["prefs"].items():
                if hasattr(self.prefs, k):
                    setattr(self.prefs, k, float(v))
        if "world_cfg" in data:
            wc = data["world_cfg"]
            for field_name, caster in (
                ("daily_food_spawn", int),
                ("ticks_per_day", int),
                ("survival_food_need", float),
                ("base_food_price", float),
                ("max_residents_per_home", int),
                ("start_paradigm", str),
            ):
                if field_name in wc:
                    setattr(self.world_cfg, field_name, caster(wc[field_name]))

    def import_population_json(self, data: Dict[str, Any]) -> None:
        """Import a saved population JSON into the simulation."""
        self._apply_imported_configs(data)
        self.reset_world()
        if "day" in data:
            self.day_number = max(1, int(data["day"]))
        if "hall_of_fame" in data and isinstance(data["hall_of_fame"], list):
            self.hall_of_fame = data["hall_of_fame"][:12]

        imported_blobs = data.get("blobs", [])
        if imported_blobs:
            self.blobs.clear()
            self._blobs_by_id.clear()
            for h in self.homes.values():
                h.resident_ids.clear()
            home_list = list(self.homes.values())
            for i, bdata in enumerate(imported_blobs):
                home = home_list[i % len(home_list)]
                if home.is_full:
                    home = self._find_or_create_settlement_home(None, home.x, home.y)
                g_dict = bdata.get("genes", {})
                genes = Genes(
                    speed=g_dict.get("speed", 1.0),
                    sense=g_dict.get("sense", 1.0),
                    size=g_dict.get("size", 1.0),
                    metabolism=g_dict.get("metabolism", 1.0),
                    hue=g_dict.get("hue", 195.0),
                    drive_energy=g_dict.get("drive_energy", 1.0),
                    drive_pain=g_dict.get("drive_pain", 1.0),
                    drive_social=g_dict.get("drive_social", 0.3),
                    plasticity_lr=g_dict.get("plasticity_lr", 1.0),
                )
                brain = BlobBrain.from_dict(bdata["brain"], rng=self.rng).adapt_to_config(self.net_cfg)
                blob = Blob(
                    blob_id=self._next_blob_id,
                    x=float(home.x + self.rng.normal(0.0, 12.0)),
                    y=float(home.y + self.rng.normal(0.0, 12.0)),
                    home_id=home.id,
                    home_x=home.x,
                    home_y=home.y,
                    brain=brain,
                    genes=genes,
                    generation=bdata.get("generation", 1),
                    money=bdata.get("money", 6.0),
                )
                blob.age_days = int(bdata.get("age_days", 0))
                blob.children_count = int(bdata.get("children", 0))
                self._next_blob_id += 1
                home.resident_ids.append(blob.id)
                self._register_blob(blob)
                self.blobs.append(blob)
                blob._sim_idx = len(self.blobs) - 1
            if self.blobs:
                self.selected_blob_id = self.blobs[0].id
        self.record_daily_snapshot()
        self.log_event("world", f"Population aus JSON geladen ({len(self.blobs)} Blobs).")

    def close(self) -> None:
        """Break circular references and detach simulation buffers for rapid GC reclamation."""
        if hasattr(self, "_brain_pool") and self._brain_pool is not None:
            self._brain_pool.close()
            self._brain_pool = None
        for b in self.blobs:
            b._kin_ref = None
            b._on_rep_change = None
            b._on_group_change = None
            b.home_obj = None
            if hasattr(b, "brain") and b.brain is not None:
                b.brain._pool_ref = None
                if hasattr(b.brain, "weights"):
                    b.brain.weights.clear()
                if hasattr(b.brain, "biases"):
                    b.brain.biases.clear()
        self.blobs.clear()
        self._blobs_by_id.clear()
        self.homes.clear()
        self.groups.clear()
        self.foods.clear()
        self.oases.clear()
        self.obstacles.clear()
        self.history.clear()
        self.event_log.clear()
        self.recent_trade_lines.clear()
        self.recent_raid_lines.clear()
        self.hall_of_fame.clear()
        self.cradle_centers.clear()
        if hasattr(self, "map_zones"):
            self.map_zones.clear()
        if hasattr(self, "skill_gates"):
            self.skill_gates.clear()

        # Deallocate large kinematics numpy arrays
        self._kin_cap = 0
        self._kin_xs = np.empty(0, dtype=np.float64)
        self._kin_ys = np.empty(0, dtype=np.float64)
        self._kin_vxs = np.empty(0, dtype=np.float64)
        self._kin_vys = np.empty(0, dtype=np.float64)
        self._kin_headings = np.empty(0, dtype=np.float32)
        self._kin_energies = np.empty(0, dtype=np.float32)
        self._kin_in_swamps = np.empty(0, dtype=np.bool_)
        self._kin_is_at_homes = np.empty(0, dtype=np.bool_)
        self._kin_foods_today = np.empty(0, dtype=np.float32)
        self._kin_max_speeds = np.empty(0, dtype=np.float32)
        self._kin_body_radii = np.empty(0, dtype=np.float32)
        self._kin_basal_costs = np.empty(0, dtype=np.float32)
        self._kin_has_groups = np.empty(0, dtype=np.bool_)
        self._kin_home_xs = np.empty(0, dtype=np.float64)
        self._kin_home_ys = np.empty(0, dtype=np.float64)
        self._kin_home_radii = np.empty(0, dtype=np.float32)
        self._kin_hit_rocks = np.empty(0, dtype=np.uint8)
        self._kin_speed_norms = np.empty(0, dtype=np.float32)
        self._kin_gene_sizes = np.empty(0, dtype=np.float32)
        self._kin_moneys = np.empty(0, dtype=np.float32)
        self._kin_reputations = np.empty(0, dtype=np.float32)
        self._kin_peer_reps = np.empty(0, dtype=np.float32)
        self._kin_is_at_home_int = np.empty(0, dtype=np.int32)
        self._kin_sprints = np.empty(0, dtype=np.float32)
        self._kin_mem_xs = np.empty(0, dtype=np.float64)
        self._kin_mem_ys = np.empty(0, dtype=np.float64)
        self._kin_mem_strengths = np.empty(0, dtype=np.float64)
        self._kin_surplus_foods = np.empty(0, dtype=np.float32)
        self._kin_gids = np.empty(0, dtype=np.int32)
        self._kin_hids = np.empty(0, dtype=np.int32)
        self._kin_tcooldowns = np.empty(0, dtype=np.int32)
        self._kin_ccooldowns = np.empty(0, dtype=np.int32)
        self._kin_rcooldowns = np.empty(0, dtype=np.int32)
        self._kin_age_ticks = np.empty(0, dtype=np.int32)
        self._kin_ticks_since_meal = np.empty(0, dtype=np.int32)
        self._kin_vision_radii = np.empty(0, dtype=np.float32)
        self._kin_drive_energy = np.empty(0, dtype=np.float32)
        self._kin_drive_pain = np.empty(0, dtype=np.float32)
        self._kin_drive_social = np.empty(0, dtype=np.float32)
        self._kin_plasticity_lr = np.empty(0, dtype=np.float32)

        # Deallocate obstacle CSR arrays
        self._rock_offsets = np.empty(0, dtype=np.int32)
        self._rock_indices = np.empty(0, dtype=np.int32)
        self._swamp_offsets = np.empty(0, dtype=np.int32)
        self._swamp_indices = np.empty(0, dtype=np.int32)

        # Deallocate step buffers
        self._step_eff_visions = np.empty(0, dtype=np.float32)
        self._step_blob_foods_ok = np.empty(0, dtype=np.float32)
        self._step_blob_moneys_ok = np.empty(0, dtype=np.float32)
        self._step_blob_groups = np.empty(0, dtype=np.int32)
        if hasattr(self, "_step_home_objs"):
            self._step_home_objs.clear()
        if hasattr(self, "_step_home_radii_list"):
            self._step_home_radii_list.clear()
        self._last_min_p_idxs = np.empty(0, dtype=np.int32)

        # Deallocate food and oasis caches
        self._food_xs = None
        self._food_ys = None
        self._food_base_reaches = None
        self._cached_oasis_ox = np.empty(0, dtype=np.float32)
        self._cached_oasis_oy = np.empty(0, dtype=np.float32)
        self._cached_oasis_orad = np.empty(0, dtype=np.float32)
        self._cached_oasis_ofert = np.empty(0, dtype=np.float32)
        self._cached_oasis_oring_bonus = np.empty(0, dtype=np.float32)

        if hasattr(self, "telemetry") and self.telemetry is not None:
            self.telemetry.daily_logs.clear()
            self.telemetry.collapse_logs.clear()
            self.telemetry.community_events.clear()
            self.telemetry.tribes.clear()
            self.telemetry._initial_mean_flat = None
            self.telemetry._prev_mean_flat = None
            self.telemetry = None
        import gc
        gc.collect()


