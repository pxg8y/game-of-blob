"""Comprehensive Telemetry, Community Action, Genetic & Neural Weight Logging System."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np

from .neural import INPUT_LABELS, OUTPUT_LABELS

if TYPE_CHECKING:
    from .entities import Blob, FoodItem, Group, Home, OasisNode
    from .world import SimulationWorld


LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

PATHWAY_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "id": "food_nav",
        "label": "🍏 Nahrungs-Navigation",
        "in_idx": [0, 1, 2],
        "out_idx": [0, 1],
        "color": "#10b981",
        "desc": "Kopplung von Nahrungs-Richtung/Nähe (0..2) auf Bewegung X/Y (0..1)",
    },
    {
        "id": "home_nav",
        "label": "🏠 Heimkehr-Disziplin",
        "in_idx": [3, 4, 5],
        "out_idx": [0, 1],
        "color": "#0ea5e9",
        "desc": "Kopplung von Zuhause-Vektor/Nähe (3..5) auf Bewegung X/Y (0..1)",
    },
    {
        "id": "oasis_nav",
        "label": "🌴 Oasen-Fernsinn",
        "in_idx": [16, 17, 18],
        "out_idx": [0, 1],
        "color": "#14b8a6",
        "desc": "Kopplung von Oasen-Witterung & Fruchtbarkeit (16..18) auf Bewegung X/Y",
    },
    {
        "id": "memory_nav",
        "label": "🧭 Gedächtnis-Kompass",
        "in_idx": [19, 20],
        "out_idx": [0, 1],
        "color": "#a855f7",
        "desc": "Kopplung des räumlichen Oasen-Gedächtnisses (19..20) auf Bewegung X/Y",
    },
    {
        "id": "feeler_avoid",
        "label": "🪨 Fühler-Reflex",
        "in_idx": [21, 22, 23],
        "out_idx": [0, 1, 2],
        "color": "#f59e0b",
        "desc": "Kopplung der 3 Hindernis-Fühler (21..23) auf Ausweichbewegung & Tempo",
    },
    {
        "id": "sprint_drive",
        "label": "⚡ Sprint-Aktivierung",
        "in_idx": list(range(len(INPUT_LABELS))),
        "out_idx": [2],
        "color": "#eab308",
        "desc": "Gesamte sensorische Ansteuerung des Sprint-/Tempo-Outputs (2)",
    },
    {
        "id": "trade_drive",
        "label": "🪙 Handels-Trieb",
        "in_idx": [6, 7, 8, 9, 10, 11, 12, 13, 14, 24, 25],
        "out_idx": [3],
        "color": "#f97316",
        "desc": "Kopplung von Nachbar-, Sättigungs-, Geld- & Ruf-Sensoren auf Handels-Wunsch (3)",
    },
    {
        "id": "social_drive",
        "label": "🤝 Gruppen-Sozialität",
        "in_idx": [6, 7, 8, 9, 10, 11, 24, 25],
        "out_idx": [4],
        "color": "#3b82f6",
        "desc": "Kopplung von Nachbar- & Ruf-Sensoren (6..11, 24..25) auf Siedlungs-Sozialität (4)",
    },
    {
        "id": "reputation_drive",
        "label": "⚖️ Ruf- & Ächtungs-Sinn",
        "in_idx": [24, 25],
        "out_idx": [3, 4],
        "color": "#ec4899",
        "desc": "Kopplung von Nachbar- & Eigen-Reputation (24..25) auf Handel & Sozialität (3..4)",
    },
    {
        "id": "colonize_drive",
        "label": "🏕️ Pionier-Siedlungsdrang",
        "in_idx": [5, 15, 16, 17, 18],
        "out_idx": [5],
        "color": "#f43f5e",
        "desc": "Kopplung von Heim-Distanz, Tageszeit & Oase auf Siedeln/Umziehen (5)",
    },
]


def _empty_daily_counters() -> Dict[str, Any]:
    return {
        # Foraging
        "foraged_count": 0,
        "foraged_golden_count": 0,
        "foraged_food_units": 0.0,
        "foraged_cradle_count": 0,
        "foraged_frontier_count": 0,
        "foraged_garden_count": 0,
        # Peer & Market Trading
        "trades_peer_same_group": 0,
        "trades_peer_cross_group": 0,
        "trades_peer_stomach_sell": 0,
        "trades_peer_rescue_buy": 0,
        "trades_market_sell_surplus": 0,
        "trades_market_sell_stomach": 0,
        "trades_market_buy_day": 0,
        "trades_emergency_midday": 0,
        "trades_eod_deficit_buy": 0,
        "eod_surplus_sold_units": 0.0,
        "eod_deficit_bought_units": 0.0,
        "eod_leftover_stored_units": 0.0,
        # Conflict & Raids
        "raids_won": 0,
        "raids_repelled": 0,
        "bounties_won": 0,
        "exiles": 0,
        "raid_loot_food_count": 0,
        "raid_loot_depot_count": 0,
        "raid_loot_gold_count": 0,
        "houses_conquered": 0,
        # Social, Expansion & Upgrades
        "groups_founded": 0,
        "group_joins": 0,
        "homes_federated": 0,
        "secessions": 0,
        "groups_extinct": 0,
        "migrations_existing": 0,
        "pioneer_homes_built": 0,
        "partner_homes_built": 0,
        "upgrades_farm": 0,
        "upgrades_granary": 0,
        "upgrades_defense": 0,
        "taxes_collected_gold": 0.0,
        # Births & Deaths Breakdown
        "births_sexual": 0,
        "births_asexual": 0,
        "deaths_midday_starve": 0,
        "deaths_eod_starve": 0,
        "deaths_other": 0,
        "deaths_outdoors": 0,
        "deaths_at_home": 0,
        "deaths_with_money_far": 0,
        "deaths_with_money_empty_depot": 0,
        "deaths_in_swamp": 0,
        "deaths_in_cradle": 0,
        "deaths_in_frontier": 0,
        "deaths_grouped": 0,
        "deaths_solo": 0,
        "hof_rescue_events": 0,
        "hof_rescued_blobs": 0,
        # End-of-Day Pre-Accounting Distribution
        "eod_at_home_count": 0,
        "eod_outdoors_count": 0,
        "eod_repro_ready_count": 0,
        "eod_surv_only_count": 0,
        "eod_deficit_count": 0,
        # Neural Action & Physics Accumulators
        "action_blob_samples": 0,
        "sum_move_mag": 0.0,
        "sum_sprint": 0.0,
        "sum_trade_desire": 0.0,
        "sum_social_desire": 0.0,
        "sum_colonize_desire": 0.0,
        "sum_active_sprint": 0,
        "sum_active_trade": 0,
        "sum_active_social": 0,
        "sum_active_colonize": 0,
        "rock_collisions": 0,
        "swamp_immersions": 0,
    }


def _empty_tribe_daily() -> Dict[str, Any]:
    return {
        "births": 0,
        "deaths": 0,
        "deaths_midday": 0,
        "deaths_eod": 0,
        "deaths_outdoors": 0,
        "foraged_units": 0.0,
        "trades_internal": 0,
        "trades_external": 0,
        "trades_market": 0,
        "stomach_sells": 0,
        "raids_won": 0,
        "raids_repelled": 0,
        "bounties_won": 0,
        "exiles": 0,
        "conquests": 0,
        "secessions": 0,
        "migrations": 0,
        "upgrades": 0,
        "taxes_gold": 0.0,
    }


class SimulationTelemetryLogger:
    """
    Non-intrusive historical logger for analyzing Rise & Collapse cycles,
    collective tribe actions, genetic trade-offs, and neural weight evolution.
    """

    def __init__(self, persist_to_disk: bool = True) -> None:
        self.persist_to_disk = persist_to_disk
        self.session_id: str = time.strftime("%Y%m%d_%H%M%S")
        self.daily_counters: Dict[str, Any] = _empty_daily_counters()
        self.tribe_daily: Dict[int, Dict[str, Any]] = {}

        self.daily_logs: List[Dict[str, Any]] = []
        self.collapse_logs: List[Dict[str, Any]] = []
        self.community_events: List[Dict[str, Any]] = []
        self.tribes: Dict[int, Dict[str, Any]] = {}

        self._initial_mean_flat: Optional[np.ndarray] = None
        self._prev_mean_flat: Optional[np.ndarray] = None
        self._initial_coupling_matrix: Optional[List[List[float]]] = None
        self._peak_population: int = 0
        self._peak_pop_day: int = 1
        self._recent_window_peak: int = 0
        self._recent_window_peak_day: int = 1
        self._last_collapse_day: int = -999

        if self.persist_to_disk:
            try:
                LOG_DIR.mkdir(parents=True, exist_ok=True)
            except Exception:
                self.persist_to_disk = False

    def reset(self, world: "SimulationWorld") -> None:
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.daily_counters = _empty_daily_counters()
        self.tribe_daily.clear()
        self.daily_logs.clear()
        self.collapse_logs.clear()
        self.community_events.clear()
        self.tribes.clear()
        self._initial_mean_flat = None
        self._prev_mean_flat = None
        self._initial_coupling_matrix = None
        self._peak_population = len(world.blobs)
        self._peak_pop_day = 1
        self._recent_window_peak = len(world.blobs)
        self._recent_window_peak_day = 1
        self._last_collapse_day = -999

        if self.persist_to_disk:
            try:
                log_file = LOG_DIR / "latest_telemetry.jsonl"
                log_file.write_text("", encoding="utf-8")
            except Exception:
                pass

    def _get_tribe_daily(self, group_id: Optional[int]) -> Optional[Dict[str, Any]]:
        if group_id is None:
            return None
        if group_id not in self.tribe_daily:
            self.tribe_daily[group_id] = _empty_tribe_daily()
        return self.tribe_daily[group_id]

    def _ensure_tribe_entry(self, grp: "Group", day: int) -> Dict[str, Any]:
        if grp.id not in self.tribes:
            self.tribes[grp.id] = {
                "id": grp.id,
                "name": grp.name,
                "color": grp.color,
                "created_day": grp.created_day or day,
                "extinct_day": None,
                "alive": True,
                "parent_group_id": grp.parent_group_id,
                "peak_members": len(grp.member_ids),
                "peak_day": day,
                "peak_homes": len(grp.home_ids),
                "current_members": len(grp.member_ids),
                "current_homes": len(grp.home_ids),
                "total_births": 0,
                "total_deaths": 0,
                "total_deaths_midday": 0,
                "total_deaths_eod": 0,
                "total_deaths_outdoors": 0,
                "total_foraged_units": 0.0,
                "total_trades_internal": 0,
                "total_trades_external": 0,
                "total_trades_market": 0,
                "total_stomach_sells": 0,
                "total_raids_won": 0,
                "total_raids_repelled": 0,
                "total_conquests": 0,
                "total_secessions": 0,
                "total_migrations": 0,
                "total_upgrades": 0,
                "total_taxes_gold": 0.0,
                "collapse_count": 0,
                "last_genes": {"speed": 1.0, "sense": 1.0, "size": 1.0, "basal_cost": 0.06},
                "last_actions": {
                    "sprint": 0.0,
                    "trade": 0.0,
                    "social": 0.0,
                    "colonize": 0.0,
                },
                "history": [],
            }
        return self.tribes[grp.id]

    def log_event(
        self,
        day: int,
        tick: int,
        kind: str,
        message: str,
        group_id: Optional[int] = None,
        group_name: Optional[str] = None,
        group_color: Optional[str] = None,
        severity: str = "info",
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {
            "id": len(self.community_events) + 1,
            "day": day,
            "tick": tick,
            "kind": kind,
            "severity": severity,
            "group_id": group_id,
            "group_name": group_name,
            "group_color": group_color,
            "msg": message,
            "metrics": metrics or {},
        }
        self.community_events.append(entry)
        if len(self.community_events) > 2000:
            self.community_events = self.community_events[-2000:]

    # --- Real-Time Step & Interaction Hooks ---

    def record_step_actions(
        self, living_blobs: List["Blob"], actions_batch: np.ndarray, dt: int
    ) -> None:
        n = len(living_blobs)
        if n == 0 or actions_batch.shape[0] != n:
            return
        c = self.daily_counters
        c["action_blob_samples"] += n

        move_x = actions_batch[:, 0]
        move_y = actions_batch[:, 1]
        move_mag = np.hypot(move_x, move_y)
        sprint = (actions_batch[:, 2] + 1.0) * 0.5
        trade = actions_batch[:, 3]
        social = actions_batch[:, 4]
        colonize = actions_batch[:, 5]

        c["sum_move_mag"] += float(np.sum(move_mag))
        c["sum_sprint"] += float(np.sum(sprint))
        c["sum_trade_desire"] += float(np.sum(trade))
        c["sum_social_desire"] += float(np.sum(social))
        c["sum_colonize_desire"] += float(np.sum(colonize))

        c["sum_active_sprint"] += int(np.count_nonzero(sprint > 0.65))
        c["sum_active_trade"] += int(np.count_nonzero(trade > 0.25))
        c["sum_active_social"] += int(np.count_nonzero(social > 0.05))
        c["sum_active_colonize"] += int(np.count_nonzero(colonize > 0.08))

        ref = living_blobs[0]._kin_ref if n > 0 else None
        if ref is not None and hasattr(ref, "_kin_in_swamps"):
            swamp_count = int(np.count_nonzero(ref._kin_in_swamps[:n]))
        else:
            swamp_count = sum(1 for b in living_blobs if b.in_swamp)
        c["swamp_immersions"] += swamp_count

    def on_rock_collision(self, blob: "Blob") -> None:
        self.daily_counters["rock_collisions"] += 1

    def on_food_foraged(
        self,
        blob: "Blob",
        food: "FoodItem",
        effective_nutr: float,
        oasis: Optional["OasisNode"],
    ) -> None:
        c = self.daily_counters
        c["foraged_count"] += 1
        if food.golden:
            c["foraged_golden_count"] += 1
        c["foraged_food_units"] += float(effective_nutr)

        if oasis is None:
            c["foraged_garden_count"] += 1
        elif getattr(oasis, "ring", "outer") == "inner":
            c["foraged_cradle_count"] += 1
        else:
            c["foraged_frontier_count"] += 1

        td = self._get_tribe_daily(blob.group_id)
        if td is not None:
            td["foraged_units"] += float(effective_nutr)

    def on_peer_trade(
        self,
        seller: "Blob",
        buyer: "Blob",
        amount: float,
        price: float,
        same_group: bool,
        from_stomach: float,
        rescued_buyer: bool,
    ) -> None:
        c = self.daily_counters
        if same_group:
            c["trades_peer_same_group"] += 1
        else:
            c["trades_peer_cross_group"] += 1
        if from_stomach > 0.0:
            c["trades_peer_stomach_sell"] += 1
        if rescued_buyer:
            c["trades_peer_rescue_buy"] += 1

        td_s = self._get_tribe_daily(seller.group_id)
        if td_s is not None:
            if same_group:
                td_s["trades_internal"] += 1
            else:
                td_s["trades_external"] += 1
            if from_stomach > 0.0:
                td_s["stomach_sells"] += 1

        if not same_group and buyer.group_id is not None:
            td_b = self._get_tribe_daily(buyer.group_id)
            if td_b is not None:
                td_b["trades_external"] += 1

    def on_market_sell(
        self, blob: "Blob", home: "Home", amount: float, price: float, from_stomach: float
    ) -> None:
        c = self.daily_counters
        if from_stomach > 0.0:
            c["trades_market_sell_stomach"] += 1
        else:
            c["trades_market_sell_surplus"] += 1
        td = self._get_tribe_daily(blob.group_id or home.group_id)
        if td is not None:
            td["trades_market"] += 1
            if from_stomach > 0.0:
                td["stomach_sells"] += 1

    def on_market_buy(
        self, blob: "Blob", home: "Home", amount: float, price: float
    ) -> None:
        self.daily_counters["trades_market_buy_day"] += 1
        td = self._get_tribe_daily(blob.group_id or home.group_id)
        if td is not None:
            td["trades_market"] += 1

    def on_emergency_ration_buy(
        self, blob: "Blob", home: "Home", amount: float, cost: float
    ) -> None:
        self.daily_counters["trades_emergency_midday"] += 1
        td = self._get_tribe_daily(blob.group_id or home.group_id)
        if td is not None:
            td["trades_market"] += 1

    def on_eod_surplus_sold(
        self, blob: "Blob", home: "Home", amount: float, earned: float
    ) -> None:
        self.daily_counters["eod_surplus_sold_units"] += float(amount)
        td = self._get_tribe_daily(blob.group_id or home.group_id)
        if td is not None:
            td["trades_market"] += 1

    def on_eod_deficit_bought(
        self, blob: "Blob", home: "Home", amount: float, cost: float
    ) -> None:
        self.daily_counters["trades_eod_deficit_buy"] += 1
        self.daily_counters["eod_deficit_bought_units"] += float(amount)
        td = self._get_tribe_daily(blob.group_id or home.group_id)
        if td is not None:
            td["trades_market"] += 1

    def on_eod_leftover_stored(
        self, blob: "Blob", home: "Home", amount: float
    ) -> None:
        self.daily_counters["eod_leftover_stored_units"] += float(amount)

    def on_raid_outcome(
        self,
        aggressor: "Blob",
        defender: "Blob",
        repelled: bool,
        loot_kind: str = "food",
        bounty: bool = False,
    ) -> None:
        c = self.daily_counters
        if repelled:
            c["raids_repelled"] += 1
            td_def = self._get_tribe_daily(defender.group_id)
            if td_def is not None:
                td_def["raids_repelled"] += 1
        else:
            c["raids_won"] += 1
            if bounty:
                c["bounties_won"] += 1
            if loot_kind == "food":
                c["raid_loot_food_count"] += 1
            elif loot_kind == "depot":
                c["raid_loot_depot_count"] += 1
            else:
                c["raid_loot_gold_count"] += 1
            td_atk = self._get_tribe_daily(aggressor.group_id)
            if td_atk is not None:
                td_atk["raids_won"] += 1
                if bounty:
                    td_atk["bounties_won"] += 1

    def on_blob_exiled(self, blob: "Blob", grp: "Group") -> None:
        self.daily_counters["exiles"] += 1
        td = self._get_tribe_daily(grp.id)
        if td is not None:
            td["exiles"] += 1

    def on_house_conquered(
        self, atk_grp: "Group", old_gid: Optional[int], home: "Home"
    ) -> None:
        self.daily_counters["houses_conquered"] += 1
        td = self._get_tribe_daily(atk_grp.id)
        if td is not None:
            td["conquests"] += 1

    def on_group_founded(
        self, grp: "Group", a: "Blob", b: "Blob"
    ) -> None:
        self.daily_counters["groups_founded"] += 1
        self._ensure_tribe_entry(grp, grp.created_day)

    def on_group_join(self, grp: "Group", blob: "Blob") -> None:
        self.daily_counters["group_joins"] += 1

    def on_home_federated(self, grp: "Group", home: "Home") -> None:
        self.daily_counters["homes_federated"] += 1

    def on_secession(
        self,
        parent_grp: "Group",
        daughter_grp: "Group",
        cluster_homes: List["Home"],
        cluster_blobs: List["Blob"],
    ) -> None:
        self.daily_counters["secessions"] += 1
        self._ensure_tribe_entry(daughter_grp, daughter_grp.created_day)
        td = self._get_tribe_daily(parent_grp.id)
        if td is not None:
            td["secessions"] += 1

    def on_group_extinct(self, grp: "Group", day: int, tick: int) -> None:
        self.daily_counters["groups_extinct"] += 1
        if grp.id in self.tribes:
            t_entry = self.tribes[grp.id]
            t_entry["alive"] = False
            t_entry["extinct_day"] = day
            t_entry["current_members"] = 0
            t_entry["current_homes"] = 0
        self.log_event(
            day=day,
            tick=tick,
            kind="extinction",
            group_id=grp.id,
            group_name=grp.name,
            group_color=grp.color,
            severity="warning",
            message=f"💀 Stamm ausgestorben: {grp.name} ist an Tag {day} erloschen.",
        )

    def on_migration(
        self,
        blob: "Blob",
        target_home: "Home",
        oasis: "OasisNode",
        pioneer: bool,
    ) -> None:
        if pioneer:
            self.daily_counters["pioneer_homes_built"] += 1
        else:
            self.daily_counters["migrations_existing"] += 1
        td = self._get_tribe_daily(blob.group_id)
        if td is not None:
            td["migrations"] += 1

    def on_home_constructed(self, home: "Home", grp: Optional["Group"]) -> None:
        self.daily_counters["partner_homes_built"] += 1

    def on_tax_paid(self, blob: "Blob", home: "Home", amount: float) -> None:
        self.daily_counters["taxes_collected_gold"] += float(amount)
        td = self._get_tribe_daily(blob.group_id or home.group_id)
        if td is not None:
            td["taxes_gold"] += float(amount)

    def on_home_upgrade(self, home: "Home", choice: str) -> None:
        if choice == "farm":
            self.daily_counters["upgrades_farm"] += 1
        elif choice == "granary":
            self.daily_counters["upgrades_granary"] += 1
        elif choice == "defense":
            self.daily_counters["upgrades_defense"] += 1
        td = self._get_tribe_daily(home.group_id)
        if td is not None:
            td["upgrades"] += 1

    def on_birth(
        self,
        parent: "Blob",
        partner: Optional["Blob"],
        child: "Blob",
        home: "Home",
    ) -> None:
        if partner is not None:
            self.daily_counters["births_sexual"] += 1
        else:
            self.daily_counters["births_asexual"] += 1
        td = self._get_tribe_daily(parent.group_id)
        if td is not None:
            td["births"] += 1

    def on_blob_death(
        self, world: "SimulationWorld", blob: "Blob", reason: str
    ) -> None:
        c = self.daily_counters
        is_midday = "Energie leer" in reason
        is_eod = "Tagesende" in reason
        if is_midday:
            c["deaths_midday_starve"] += 1
        elif is_eod:
            c["deaths_eod_starve"] += 1
        else:
            c["deaths_other"] += 1

        if blob.is_at_home:
            c["deaths_at_home"] += 1
        else:
            c["deaths_outdoors"] += 1

        if blob.in_swamp:
            c["deaths_in_swamp"] += 1

        home = world.homes.get(blob.home_id)
        zone = getattr(home, "zone", "cradle") if home is not None else "cradle"
        if zone == "cradle":
            c["deaths_in_cradle"] += 1
        else:
            c["deaths_in_frontier"] += 1

        if blob.group_id is not None:
            c["deaths_grouped"] += 1
        else:
            c["deaths_solo"] += 1

        min_ration_cost = world.world_cfg.base_food_price * 0.4
        if blob.money >= min_ration_cost:
            if not blob.is_at_home:
                c["deaths_with_money_far"] += 1
            else:
                c["deaths_with_money_empty_depot"] += 1

        td = self._get_tribe_daily(blob.group_id)
        if td is not None:
            td["deaths"] += 1
            if is_midday:
                td["deaths_midday"] += 1
            if is_eod:
                td["deaths_eod"] += 1
            if not blob.is_at_home:
                td["deaths_outdoors"] += 1

    def on_eod_pre_accounting(
        self,
        living_blobs: List["Blob"],
        survival_need: float,
        repro_need: float,
    ) -> None:
        c = self.daily_counters
        at_home = 0
        outdoors = 0
        repro_ready = 0
        surv_only = 0
        deficit = 0
        for b in living_blobs:
            if b.is_at_home:
                at_home += 1
            else:
                outdoors += 1
            tot = b.food_today + b.surplus_food
            if tot >= repro_need - 1e-4:
                repro_ready += 1
            elif tot >= survival_need - 1e-4:
                surv_only += 1
            else:
                deficit += 1
        c["eod_at_home_count"] = at_home
        c["eod_outdoors_count"] = outdoors
        c["eod_repro_ready_count"] = repro_ready
        c["eod_surv_only_count"] = surv_only
        c["eod_deficit_count"] = deficit

    def on_hof_rescue(self, rescued_count: int, day: int, tick: int) -> None:
        self.daily_counters["hof_rescue_events"] += 1
        self.daily_counters["hof_rescued_blobs"] += rescued_count
        self.log_event(
            day=day,
            tick=tick,
            kind="hof_rescue",
            severity="danger",
            message=(
                f"🛟 Hall-of-Fame Notrettung an Tag {day}: Population unter Minimum gefallen "
                f"→ +{rescued_count} Nachkommen aus Champion-Genpool eingesetzt!"
            ),
            metrics={"rescued_blobs": rescued_count},
        )

    # --- Daily Snapshot & Deep Analysis Computation ---

    def _compute_genetic_stats(self, living: List["Blob"]) -> Dict[str, Any]:
        n = len(living)
        if n == 0:
            return {
                "speed": {"mean": 1.0, "std": 0.0, "p10": 1.0, "p50": 1.0, "p90": 1.0, "min": 1.0, "max": 1.0},
                "sense": {"mean": 1.0, "std": 0.0, "p10": 1.0, "p50": 1.0, "p90": 1.0, "min": 1.0, "max": 1.0},
                "size": {"mean": 1.0, "std": 0.0, "p10": 1.0, "p50": 1.0, "p90": 1.0, "min": 1.0, "max": 1.0},
                "metabolism": {"mean": 1.0, "std": 0.0},
                "basal_cost": {"mean": 0.06, "std": 0.0, "p10": 0.06, "p50": 0.06, "p90": 0.06, "max": 0.06},
                "generation": {"mean": 1.0, "max": 1, "min": 1},
                "age_days": {"mean": 0.0, "max": 0, "p90": 0.0},
                "money": {"mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0, "total": 0.0},
                "reputation": {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0, "outlaws": 0, "honorable": 0},
            }

        # Sample up to 256 blobs evenly if population is large to accelerate midnight snapshot
        if n > 256:
            step = max(1, n // 256)
            sampled = living[::step]
            m = len(sampled)
        else:
            sampled = living
            m = n

        speeds = np.fromiter((b.genes.speed for b in sampled), dtype=np.float32, count=m)
        senses = np.fromiter((b.genes.sense for b in sampled), dtype=np.float32, count=m)
        sizes = np.fromiter((b.genes.size for b in sampled), dtype=np.float32, count=m)
        metabs = np.fromiter((b.genes.metabolism for b in sampled), dtype=np.float32, count=m)
        costs = np.fromiter((b.genes._basal_cost for b in sampled), dtype=np.float32, count=m)
        gens = np.fromiter((b.generation for b in sampled), dtype=np.int32, count=m)
        ages = np.fromiter((b.age_days for b in sampled), dtype=np.int32, count=m)
        moneys = np.fromiter((b.money for b in sampled), dtype=np.float32, count=m)
        reps = np.fromiter((b.reputation for b in sampled), dtype=np.float32, count=m)

        def _dist(arr: np.ndarray, decimals: int = 3) -> Dict[str, float]:
            p10, p50, p90 = np.percentile(arr, [10, 50, 90])
            return {
                "mean": round(float(np.mean(arr)), decimals),
                "std": round(float(np.std(arr)), decimals),
                "p10": round(float(p10), decimals),
                "p50": round(float(p50), decimals),
                "p90": round(float(p90), decimals),
                "min": round(float(np.min(arr)), decimals),
                "max": round(float(np.max(arr)), decimals),
            }

        m_p50, m_p90 = np.percentile(moneys, [50, 90])
        rep_dist = _dist(reps, 3)
        scale_factor = float(n) / float(m)
        rep_dist["outlaws"] = int(round(np.count_nonzero(reps <= -0.35) * scale_factor))
        rep_dist["honorable"] = int(round(np.count_nonzero(reps >= 0.35) * scale_factor))
        return {
            "speed": _dist(speeds, 3),
            "sense": _dist(senses, 3),
            "size": _dist(sizes, 3),
            "metabolism": {
                "mean": round(float(np.mean(metabs)), 3),
                "std": round(float(np.std(metabs)), 3),
            },
            "basal_cost": _dist(costs, 4),
            "generation": {
                "mean": round(float(np.mean(gens)), 2),
                "max": int(np.max(gens)),
                "min": int(np.min(gens)),
            },
            "age_days": {
                "mean": round(float(np.mean(ages)), 2),
                "max": int(np.max(ages)),
                "p90": round(float(np.percentile(ages, 90)), 1),
            },
            "money": {
                "mean": round(float(np.mean(moneys)), 2),
                "p50": round(float(m_p50), 2),
                "p90": round(float(m_p90), 2),
                "max": round(float(np.max(moneys)), 2),
                "total": round(float(np.sum(moneys)) * scale_factor, 1),
            },
            "reputation": rep_dist,
        }

    def _compute_neural_weight_stats(self, living: List["Blob"]) -> Dict[str, Any]:
        n = len(living)
        if n == 0:
            return {
                "weight_l2_norm": 0.0,
                "weight_mean_abs": 0.0,
                "weight_max_abs": 0.0,
                "saturation_pct": 0.0,
                "brain_diversity_std": 0.0,
                "weight_drift_day1": 0.0,
                "weight_velocity_daily": 0.0,
                "layers": [],
                "pathways": {p["id"]: 0.0 for p in PATHWAY_DEFINITIONS},
                "pathway_signed": {p["id"]: 0.0 for p in PATHWAY_DEFINITIONS},
                "coupling_matrix": [[0.0] * 6 for _ in range(len(INPUT_LABELS))],
            }

        # Sample up to 64 brains evenly if population is huge so daily snapshot stays <1ms
        step = max(1, n // 64)
        sampled = living[::step]
        n_samp = len(sampled)
        num_layers = len(sampled[0].brain.weights)

        layer_stats: List[Dict[str, Any]] = []
        mean_weights_per_layer: List[np.ndarray] = []
        flat_means: List[np.ndarray] = []
        diversity_stds: List[float] = []
        total_params = 0
        saturated_count = 0
        sum_abs_all = 0.0
        max_abs_all = 0.0
        sum_sq_norm = 0.0

        for lay_idx in range(num_layers):
            w_stack = np.stack([b.brain.weights[lay_idx] for b in sampled], axis=0)
            b_stack = np.stack([b.brain.biases[lay_idx] for b in sampled], axis=0)

            w_mean = np.mean(w_stack, axis=0)
            b_mean = np.mean(b_stack, axis=0)
            mean_weights_per_layer.append(w_mean)
            flat_means.append(w_mean.ravel())
            flat_means.append(b_mean.ravel())

            w_std_across = float(np.mean(np.std(w_stack, axis=0))) if n_samp > 1 else 0.0
            diversity_stds.append(w_std_across)

            abs_w = np.abs(w_stack)
            lay_mean_abs = float(np.mean(abs_w))
            lay_std = float(np.std(w_stack))
            lay_max_abs = float(np.max(abs_w))
            lay_sat = float(np.mean(abs_w >= 3.0) * 100.0)
            lay_l2 = float(np.mean(np.linalg.norm(w_stack.reshape(n_samp, -1), axis=1)))

            n_elems = w_stack.size
            total_params += n_elems
            saturated_count += int(np.count_nonzero(abs_w >= 3.0))
            sum_abs_all += float(np.sum(abs_w))
            if lay_max_abs > max_abs_all:
                max_abs_all = lay_max_abs
            sum_sq_norm += lay_l2 * lay_l2

            layer_stats.append(
                {
                    "layer": lay_idx,
                    "shape": list(w_mean.shape),
                    "mean_abs_w": round(lay_mean_abs, 4),
                    "std_w": round(lay_std, 4),
                    "max_abs_w": round(lay_max_abs, 3),
                    "l2_norm": round(lay_l2, 3),
                    "saturation_pct": round(lay_sat, 2),
                    "diversity_std": round(w_std_across, 4),
                    "mean_abs_b": round(float(np.mean(np.abs(b_stack))), 4),
                }
            )

        full_mean_flat = np.concatenate(flat_means)
        if self._initial_mean_flat is None or self._initial_mean_flat.shape != full_mean_flat.shape:
            self._initial_mean_flat = full_mean_flat.copy()
            drift_day1 = 0.0
        else:
            drift_day1 = float(np.linalg.norm(full_mean_flat - self._initial_mean_flat))

        if self._prev_mean_flat is None or self._prev_mean_flat.shape != full_mean_flat.shape:
            velocity_daily = 0.0
        else:
            velocity_daily = float(np.linalg.norm(full_mean_flat - self._prev_mean_flat))
        self._prev_mean_flat = full_mean_flat.copy()

        # Compute Effective Sensor-to-Action Coupling Matrix
        # Signed product W_0 @ W_1 @ ... @ W_{L-1} and RMS magnitude product
        m_signed = mean_weights_per_layer[0].astype(np.float64)
        m_abs = np.abs(mean_weights_per_layer[0]).astype(np.float64)
        for lay_idx in range(1, num_layers):
            w_next = mean_weights_per_layer[lay_idx].astype(np.float64)
            scale = 1.0 / math.sqrt(max(1, w_next.shape[0]))
            m_signed = (m_signed @ w_next) * scale
            m_abs = (m_abs @ np.abs(w_next)) * scale

        # Slice to standard sensory and motor block (26x6) if recurrent units are present
        if m_signed.shape[0] > len(INPUT_LABELS) or m_signed.shape[1] > len(OUTPUT_LABELS):
            m_signed = m_signed[:len(INPUT_LABELS), :len(OUTPUT_LABELS)]
            m_abs = m_abs[:len(INPUT_LABELS), :len(OUTPUT_LABELS)]

        coupling_matrix = np.round(m_signed, 3).tolist()
        if self._initial_coupling_matrix is None:
            self._initial_coupling_matrix = coupling_matrix

        pathways_mag: Dict[str, float] = {}
        pathways_signed: Dict[str, float] = {}
        for pdef in PATHWAY_DEFINITIONS:
            pid = pdef["id"]
            in_i = pdef["in_idx"]
            out_i = pdef["out_idx"]
            sub_abs = m_abs[np.ix_(in_i, out_i)]
            sub_sig = m_signed[np.ix_(in_i, out_i)]
            pathways_mag[pid] = round(float(np.mean(sub_abs)), 4)
            if pid in ("food_nav", "home_nav", "oasis_nav", "memory_nav") and len(in_i) >= 2 and len(out_i) >= 2:
                # Diagonal alignment (dx -> move_x, dy -> move_y)
                diag_val = 0.5 * (m_signed[in_i[0], out_i[0]] + m_signed[in_i[1], out_i[1]])
                pathways_signed[pid] = round(float(diag_val), 4)
            else:
                pathways_signed[pid] = round(float(np.mean(sub_sig)), 4)

        return {
            "weight_l2_norm": round(math.sqrt(sum_sq_norm), 3),
            "weight_mean_abs": round(sum_abs_all / max(1, total_params), 4),
            "weight_max_abs": round(max_abs_all, 3),
            "saturation_pct": round((saturated_count / max(1, total_params)) * 100.0, 2),
            "brain_diversity_std": round(float(np.mean(diversity_stds)), 4),
            "weight_drift_day1": round(drift_day1, 3),
            "weight_velocity_daily": round(velocity_daily, 4),
            "layers": layer_stats,
            "pathways": pathways_mag,
            "pathway_signed": pathways_signed,
            "coupling_matrix": coupling_matrix,
        }

    def _update_tribes_snapshot(
        self, world: "SimulationWorld", living: List["Blob"]
    ) -> List[Dict[str, Any]]:
        day = world.day_number
        # Single-pass accumulation over living blobs in O(N)
        grp_stats: Dict[int, List[float]] = {}
        for b in living:
            gid = b.group_id
            if gid is not None:
                entry = grp_stats.get(gid)
                if entry is None:
                    g = b.genes
                    entry = [
                        1.0,
                        g.speed,
                        g.sense,
                        g.size,
                        g._basal_cost,
                        b.money,
                        1.0 if b.is_at_home else 0.0,
                        b.sprint_level,
                        b.trade_desire,
                        b.social_desire,
                        b.colonize_desire,
                    ]
                    grp_stats[gid] = entry
                else:
                    g = b.genes
                    entry[0] += 1.0
                    entry[1] += g.speed
                    entry[2] += g.sense
                    entry[3] += g.size
                    entry[4] += g._basal_cost
                    entry[5] += b.money
                    if b.is_at_home:
                        entry[6] += 1.0
                    entry[7] += b.sprint_level
                    entry[8] += b.trade_desire
                    entry[9] += b.social_desire
                    entry[10] += b.colonize_desire

        active_tribe_summaries: List[Dict[str, Any]] = []
        for gid, grp in world.groups.items():
            t_entry = self._ensure_tribe_entry(grp, day)
            t_entry["alive"] = True
            t_entry["extinct_day"] = None
            stats = grp_stats.get(gid)
            m_count = int(stats[0]) if stats is not None else 0
            prev_members = t_entry["current_members"]

            # Detect single-tribe collapse (>32% loss in a day when previously >= 10 members)
            if prev_members >= 10 and m_count <= int(prev_members * 0.68):
                t_entry["collapse_count"] += 1

            t_entry["current_members"] = m_count
            if m_count > t_entry["peak_members"]:
                t_entry["peak_members"] = m_count
                t_entry["peak_day"] = day

            grp_homes = [world.homes[hid] for hid in grp.home_ids if hid in world.homes]
            h_count = len(grp_homes)
            t_entry["current_homes"] = h_count
            if h_count > t_entry["peak_homes"]:
                t_entry["peak_homes"] = h_count

            td = self.tribe_daily.get(gid, _empty_tribe_daily())
            t_entry["total_births"] += td["births"]
            t_entry["total_deaths"] += td["deaths"]
            t_entry["total_deaths_midday"] += td["deaths_midday"]
            t_entry["total_deaths_eod"] += td["deaths_eod"]
            t_entry["total_deaths_outdoors"] += td["deaths_outdoors"]
            t_entry["total_foraged_units"] = round(t_entry["total_foraged_units"] + td["foraged_units"], 2)
            t_entry["total_trades_internal"] += td["trades_internal"]
            t_entry["total_trades_external"] += td["trades_external"]
            t_entry["total_trades_market"] += td["trades_market"]
            t_entry["total_stomach_sells"] += td["stomach_sells"]
            t_entry["total_raids_won"] += td["raids_won"]
            t_entry["total_raids_repelled"] += td["raids_repelled"]
            t_entry["total_conquests"] = grp.conquests
            t_entry["total_secessions"] = grp.secessions
            t_entry["total_migrations"] += td["migrations"]
            t_entry["total_upgrades"] += td["upgrades"]
            t_entry["total_taxes_gold"] = round(t_entry["total_taxes_gold"] + td["taxes_gold"], 2)

            depot_food = round(sum(h.food_depot for h in grp_homes), 2)
            vault_gold = round(sum(h.gold_vault for h in grp_homes), 1)
            cradle_homes = sum(1 for h in grp_homes if getattr(h, "zone", "cradle") == "cradle")
            frontier_homes = h_count - cradle_homes
            upg_farm = sum(h.farm_level for h in grp_homes)
            upg_granary = sum(h.granary_level for h in grp_homes)
            upg_def = sum(h.defense_level for h in grp_homes)

            if m_count > 0 and stats is not None:
                inv_m = 1.0 / m_count
                avg_speed = round(stats[1] * inv_m, 3)
                avg_sense = round(stats[2] * inv_m, 3)
                avg_size = round(stats[3] * inv_m, 3)
                avg_cost = round(stats[4] * inv_m, 4)
                avg_money = round(stats[5] * inv_m, 2)
                at_home_pct = round(stats[6] * inv_m * 100.0, 1)
                avg_sprint = round(stats[7] * inv_m, 3)
                avg_trade_d = round(stats[8] * inv_m, 3)
                avg_soc_d = round(stats[9] * inv_m, 3)
                avg_col_d = round(stats[10] * inv_m, 3)
                t_entry["last_genes"] = {
                    "speed": avg_speed,
                    "sense": avg_sense,
                    "size": avg_size,
                    "basal_cost": avg_cost,
                }
                t_entry["last_actions"] = {
                    "sprint": avg_sprint,
                    "trade": avg_trade_d,
                    "social": avg_soc_d,
                    "colonize": avg_col_d,
                }
            else:
                avg_speed = t_entry["last_genes"]["speed"]
                avg_sense = t_entry["last_genes"]["sense"]
                avg_size = t_entry["last_genes"]["size"]
                avg_cost = t_entry["last_genes"]["basal_cost"]
                avg_money = 0.0
                at_home_pct = 0.0
                avg_sprint = 0.0
                avg_trade_d = 0.0
                avg_soc_d = 0.0
                avg_col_d = 0.0

            hist_pt = {
                "day": day,
                "members": m_count,
                "homes": h_count,
                "cradle_homes": cradle_homes,
                "frontier_homes": frontier_homes,
                "depot_food": depot_food,
                "vault_gold": vault_gold,
                "births": td["births"],
                "deaths": td["deaths"],
                "deaths_midday": td["deaths_midday"],
                "deaths_eod": td["deaths_eod"],
                "deaths_outdoors": td["deaths_outdoors"],
                "foraged_units": round(td["foraged_units"], 2),
                "trades": td["trades_internal"] + td["trades_external"] + td["trades_market"],
                "stomach_sells": td["stomach_sells"],
                "raids_won": td["raids_won"],
                "migrations": td["migrations"],
                "speed": avg_speed,
                "sense": avg_sense,
                "size": avg_size,
                "basal_cost": avg_cost,
                "avg_money": avg_money,
                "at_home_pct": at_home_pct,
                "act_sprint": avg_sprint,
                "act_trade": avg_trade_d,
                "act_social": avg_soc_d,
                "act_colonize": avg_col_d,
            }
            if t_entry["history"] and t_entry["history"][-1]["day"] == day:
                t_entry["history"][-1] = hist_pt
            else:
                t_entry["history"].append(hist_pt)
                if len(t_entry["history"]) > 2500:
                    t_entry["history"] = t_entry["history"][-2500:]

            active_tribe_summaries.append(
                {
                    "id": grp.id,
                    "name": grp.name,
                    "color": grp.color,
                    "members": m_count,
                    "homes": h_count,
                    "cradle_homes": cradle_homes,
                    "frontier_homes": frontier_homes,
                    "depot_food": depot_food,
                    "vault_gold": vault_gold,
                    "upgrades": {"farm": upg_farm, "granary": upg_granary, "defense": upg_def},
                    "births": td["births"],
                    "deaths": td["deaths"],
                    "trades": td["trades_internal"] + td["trades_external"] + td["trades_market"],
                    "stomach_sells": td["stomach_sells"],
                    "speed": avg_speed,
                    "sense": avg_sense,
                    "size": avg_size,
                    "basal_cost": avg_cost,
                    "at_home_pct": at_home_pct,
                }
            )

        active_tribe_summaries.sort(key=lambda x: x["members"], reverse=True)
        return active_tribe_summaries

    def _detect_and_record_collapse(
        self, world: "SimulationWorld", snap: Dict[str, Any]
    ) -> None:
        day = snap["day"]
        pop = snap["population"]
        if pop > self._peak_population:
            self._peak_population = pop
            self._peak_pop_day = day

        if pop > self._recent_window_peak:
            self._recent_window_peak = pop
            self._recent_window_peak_day = day

        prev_pop = self.daily_logs[-1]["population"] if self.daily_logs else pop
        daily_drop_pct = ((prev_pop - pop) / max(1, prev_pop)) * 100.0 if prev_pop > pop else 0.0
        window_drop_pct = (
            ((self._recent_window_peak - pop) / max(1, self._recent_window_peak)) * 100.0
            if self._recent_window_peak > pop
            else 0.0
        )
        hof_rescued = snap["community"]["hof_rescued_blobs"] > 0

        is_sharp_daily_crash = prev_pop >= 25 and daily_drop_pct >= 26.0
        is_sustained_collapse = (
            self._recent_window_peak >= 35
            and window_drop_pct >= 42.0
            and (day - self._last_collapse_day) >= 3
        )

        if not (is_sharp_daily_crash or is_sustained_collapse or (hof_rescued and day > 1)):
            return

        self._last_collapse_day = day
        peak_before = self._recent_window_peak
        peak_day_before = self._recent_window_peak_day
        # Reset recent window peak to current pop so next cycle is tracked cleanly
        self._recent_window_peak = pop
        self._recent_window_peak_day = day

        comm = snap["community"]
        ecol = snap["ecology"]
        genes = snap["genetics"]
        weights = snap["weights"]
        dom = snap["monoculture"]

        causes: List[str] = []
        if snap["season_id"] == "winter":
            causes.append("❄️ Winter-Nahrungsengpass & Kälte-Metabolismus (1.14x Drain)")
        if ecol["overgrazed_oases_pct"] >= 45.0 or ecol["food_per_blob"] < 0.75:
            causes.append(
                f"🏜️ Oasen-Überweidung ({ecol['overgrazed_oases_pct']:.0f}% erschöpft, {ecol['food_per_blob']:.2f}🍏/Blob)"
            )
        if comm["deaths_midday_starve"] > comm["deaths_eod_starve"]:
            causes.append(
                f"⚡ Akuter Tages-Energie-Kollaps ({comm['deaths_midday_starve']} Mittags-Tode vs. {comm['deaths_eod_starve']} Abend-Tode)"
            )
        else:
            causes.append(
                f"🌙 Abend-Defizit-Verhungern ({comm['deaths_eod_starve']} Abend-Tode vs. {comm['deaths_midday_starve']} Mittags-Tode)"
            )
        if comm["eod_outdoors_pct"] >= 38.0 or comm["deaths_outdoors"] > comm["deaths_at_home"]:
            causes.append(
                f"🧭 Heimkehr-Versagen ({comm['eod_outdoors_pct']:.0f}% nachts im Freien, {comm['deaths_outdoors']} Draußen-Tode)"
            )
        if comm["deaths_with_money_far"] >= 3 or comm["deaths_with_money_empty_depot"] >= 3:
            causes.append(
                f"🪙 Tod trotz Gold ({comm['deaths_with_money_far']} fern vom Haus, {comm['deaths_with_money_empty_depot']} vor leerem Depot)"
            )
        if comm["stomach_sells_total"] >= 3:
            causes.append(
                f"📉 Magen-Ausverkauf ({comm['stomach_sells_total']}x Überlebensnahrung für Gold verkauft)"
            )
        if genes["basal_cost"]["mean"] >= 0.068:
            causes.append(
                f"🔥 Hoher genetischer Grundumsatz (Ø {genes['basal_cost']['mean']:.4f}/Tick durch Speed {genes['speed']['mean']:.2f}x / Size {genes['size']['mean']:.2f}x)"
            )
        if dom["dominant_share_pct"] >= 80.0:
            causes.append(
                f"👑 Ein-Stamm-Monokultur ({dom['dominant_tribe_name']}: {dom['dominant_share_pct']:.0f}% Anteil, Diversität σ={weights['brain_diversity_std']:.3f})"
            )
        if hof_rescued:
            causes.append(f"🛟 Hall-of-Fame Notrettung (+{comm['hof_rescued_blobs']} Blobs)")

        report = {
            "id": len(self.collapse_logs) + 1,
            "day": day,
            "season_id": snap["season_id"],
            "season_label": snap["season_label"],
            "peak_day": peak_day_before,
            "peak_population": peak_before,
            "post_population": pop,
            "drop_pct": round(max(daily_drop_pct, window_drop_pct), 1),
            "daily_drop_pct": round(daily_drop_pct, 1),
            "deaths_total": snap["deaths"],
            "deaths_midday": comm["deaths_midday_starve"],
            "deaths_eod": comm["deaths_eod_starve"],
            "deaths_outdoors": comm["deaths_outdoors"],
            "deaths_with_money_far": comm["deaths_with_money_far"],
            "deaths_with_money_empty_depot": comm["deaths_with_money_empty_depot"],
            "stomach_sells": comm["stomach_sells_total"],
            "eod_outdoors_pct": comm["eod_outdoors_pct"],
            "food_per_blob": ecol["food_per_blob"],
            "avg_oasis_fertility": ecol["avg_oasis_fertility"],
            "total_depot_food": ecol["total_depot_food"],
            "dominant_tribe": dom["dominant_tribe_name"],
            "dominant_share_pct": dom["dominant_share_pct"],
            "avg_speed": genes["speed"]["mean"],
            "avg_sense": genes["sense"]["mean"],
            "avg_size": genes["size"]["mean"],
            "avg_basal_cost": genes["basal_cost"]["mean"],
            "brain_diversity_std": weights["brain_diversity_std"],
            "hof_rescued": comm["hof_rescued_blobs"],
            "causes": causes,
        }
        self.collapse_logs.append(report)
        self.log_event(
            day=day,
            tick=world.tick_count,
            kind="collapse",
            severity="danger",
            group_name=dom["dominant_tribe_name"],
            message=(
                f"💥 Kollaps-Ereignis an Tag {day} ({snap['season_label']}): "
                f"Population fiel um -{report['drop_pct']}% ({peak_before} → {pop} Blobs). "
                f"Hauptfaktoren: {'; '.join(causes[:3])}"
            ),
            metrics=report,
        )

    def record_daily_snapshot(self, world: "SimulationWorld") -> Dict[str, Any]:
        living = [b for b in world.blobs if b.alive]
        pop = len(living)
        day = world.day_number
        sid, slabel, w_mult, _g_chance, cold_mult = world.season_info

        c = self.daily_counters
        grouped_blobs = sum(1 for b in living if b.group_id is not None)
        solo_blobs = pop - grouped_blobs

        # Spatial distribution of blobs & homes
        cradle_homes_ids = {
            h.id for h in world.homes.values() if getattr(h, "zone", "cradle") == "cradle"
        }
        blobs_in_cradle = sum(1 for b in living if b.home_id in cradle_homes_ids)
        blobs_in_frontier = pop - blobs_in_cradle

        oases_list = list(world.oases.values())
        cradle_oases = [o for o in oases_list if getattr(o, "ring", "outer") == "inner"]
        frontier_oases = [o for o in oases_list if getattr(o, "ring", "outer") != "inner"]
        avg_fert_all = (
            round(float(np.mean([o.fertility for o in oases_list])), 3) if oases_list else 1.0
        )
        avg_fert_cradle = (
            round(float(np.mean([o.fertility for o in cradle_oases])), 3) if cradle_oases else 1.0
        )
        avg_fert_frontier = (
            round(float(np.mean([o.fertility for o in frontier_oases])), 3)
            if frontier_oases
            else 1.0
        )
        overgrazed_count = sum(1 for o in oases_list if o.is_overgrazed)
        overgrazed_pct = (
            round((overgrazed_count / len(oases_list)) * 100.0, 1) if oases_list else 0.0
        )

        homes_list = list(world.homes.values())
        total_depot_food = round(sum(h.food_depot for h in homes_list), 2)
        total_vault_gold = round(sum(h.gold_vault for h in homes_list), 1)
        cradle_homes_count = len(cradle_homes_ids)
        frontier_homes_count = len(homes_list) - cradle_homes_count

        # Action means across the day
        samp = max(1, c["action_blob_samples"])
        if c["action_blob_samples"] > 0:
            mean_move = round(c["sum_move_mag"] / samp, 3)
            mean_sprint = round(c["sum_sprint"] / samp, 3)
            mean_trade_d = round(c["sum_trade_desire"] / samp, 3)
            mean_social_d = round(c["sum_social_desire"] / samp, 3)
            mean_col_d = round(c["sum_colonize_desire"] / samp, 3)
            active_sprint_pct = round((c["sum_active_sprint"] / samp) * 100.0, 1)
            active_trade_pct = round((c["sum_active_trade"] / samp) * 100.0, 1)
            active_social_pct = round((c["sum_active_social"] / samp) * 100.0, 1)
            active_col_pct = round((c["sum_active_colonize"] / samp) * 100.0, 1)
        elif pop > 0:
            mean_move = round(float(np.mean([math.hypot(b.vx, b.vy) for b in living])), 3)
            mean_sprint = round(float(np.mean([b.sprint_level for b in living])), 3)
            mean_trade_d = round(float(np.mean([b.trade_desire for b in living])), 3)
            mean_social_d = round(float(np.mean([b.social_desire for b in living])), 3)
            mean_col_d = round(float(np.mean([b.colonize_desire for b in living])), 3)
            active_sprint_pct = 0.0
            active_trade_pct = 0.0
            active_social_pct = 0.0
            active_col_pct = 0.0
        else:
            mean_move = mean_sprint = mean_trade_d = mean_social_d = mean_col_d = 0.0
            active_sprint_pct = active_trade_pct = active_social_pct = active_col_pct = 0.0

        eod_total = max(1, c["eod_at_home_count"] + c["eod_outdoors_count"])
        if c["eod_at_home_count"] + c["eod_outdoors_count"] == 0 and pop > 0:
            at_home_now = sum(1 for b in living if b.is_at_home)
            eod_at_home_pct = round((at_home_now / pop) * 100.0, 1)
            eod_outdoors_pct = round(100.0 - eod_at_home_pct, 1)
        else:
            eod_at_home_pct = round((c["eod_at_home_count"] / eod_total) * 100.0, 1)
            eod_outdoors_pct = round((c["eod_outdoors_count"] / eod_total) * 100.0, 1)

        genetics = self._compute_genetic_stats(living)
        weights = self._compute_neural_weight_stats(living)
        active_tribes = self._update_tribes_snapshot(world, living)

        # Monoculture & Dominance metrics
        if active_tribes and pop > 0:
            dom_tribe = active_tribes[0]
            dom_share_pop = round((dom_tribe["members"] / max(1, pop)) * 100.0, 1)
            dom_share_grouped = (
                round((dom_tribe["members"] / max(1, grouped_blobs)) * 100.0, 1)
                if grouped_blobs > 0
                else 0.0
            )
            dom_name = dom_tribe["name"]
            dom_id = dom_tribe["id"]
            dom_color = dom_tribe["color"]
        else:
            dom_share_pop = 0.0
            dom_share_grouped = 0.0
            dom_name = "Keine Fraktion"
            dom_id = None
            dom_color = "#64748b"

        snap: Dict[str, Any] = {
            "day": day,
            "tick": world.tick_count,
            "season_id": sid,
            "season_label": slabel,
            "season_food_mult": w_mult,
            "season_cold_mult": cold_mult,
            "population": pop,
            "grouped_blobs": grouped_blobs,
            "solo_blobs": solo_blobs,
            "groups_count": len(world.groups),
            "homes_count": len(homes_list),
            "births": world.births_today,
            "deaths": world.deaths_today,
            "trades": world.trades_today,
            "raids": world.raids_today,
            "raids_repelled": world.raids_repelled_today,
            "exiles": world.exiles_today,
            "bounties": world.bounties_today,
            "avg_reputation": genetics["reputation"]["mean"],
            "outlaws": genetics["reputation"]["outlaws"],
            "honorable": genetics["reputation"]["honorable"],
            "avg_reward": (
                round(float(np.mean([b.cumulative_reward for b in living])), 2) if pop > 0 else 0.0
            ),
            "ecology": {
                "wild_food_count": len(world.foods),
                "food_per_blob": round(len(world.foods) / max(1, pop), 2),
                "avg_oasis_fertility": avg_fert_all,
                "cradle_oasis_fertility": avg_fert_cradle,
                "frontier_oasis_fertility": avg_fert_frontier,
                "overgrazed_oases_count": overgrazed_count,
                "overgrazed_oases_pct": overgrazed_pct,
                "total_depot_food": total_depot_food,
                "total_vault_gold": total_vault_gold,
                "cradle_homes": cradle_homes_count,
                "frontier_homes": frontier_homes_count,
                "blobs_in_cradle": blobs_in_cradle,
                "blobs_in_frontier": blobs_in_frontier,
            },
            "community": {
                "foraged_count": c["foraged_count"],
                "foraged_golden_count": c["foraged_golden_count"],
                "foraged_food_units": round(c["foraged_food_units"], 2),
                "foraged_cradle_count": c["foraged_cradle_count"],
                "foraged_frontier_count": c["foraged_frontier_count"],
                "foraged_garden_count": c["foraged_garden_count"],
                "trades_peer_same_group": c["trades_peer_same_group"],
                "trades_peer_cross_group": c["trades_peer_cross_group"],
                "trades_peer_stomach_sell": c["trades_peer_stomach_sell"],
                "trades_peer_rescue_buy": c["trades_peer_rescue_buy"],
                "trades_market_sell_surplus": c["trades_market_sell_surplus"],
                "trades_market_sell_stomach": c["trades_market_sell_stomach"],
                "stomach_sells_total": c["trades_peer_stomach_sell"] + c["trades_market_sell_stomach"],
                "trades_market_buy_day": c["trades_market_buy_day"],
                "trades_emergency_midday": c["trades_emergency_midday"],
                "trades_eod_deficit_buy": c["trades_eod_deficit_buy"],
                "eod_surplus_sold_units": round(c["eod_surplus_sold_units"], 2),
                "eod_deficit_bought_units": round(c["eod_deficit_bought_units"], 2),
                "eod_leftover_stored_units": round(c["eod_leftover_stored_units"], 2),
                "raids_won": c["raids_won"],
                "raids_repelled": c["raids_repelled"],
                "bounties_won": c["bounties_won"],
                "exiles": c["exiles"],
                "raid_loot_food_count": c["raid_loot_food_count"],
                "raid_loot_depot_count": c["raid_loot_depot_count"],
                "raid_loot_gold_count": c["raid_loot_gold_count"],
                "houses_conquered": c["houses_conquered"],
                "groups_founded": c["groups_founded"],
                "group_joins": c["group_joins"],
                "homes_federated": c["homes_federated"],
                "secessions": c["secessions"],
                "groups_extinct": c["groups_extinct"],
                "migrations_existing": c["migrations_existing"],
                "pioneer_homes_built": c["pioneer_homes_built"],
                "partner_homes_built": c["partner_homes_built"],
                "upgrades_farm": c["upgrades_farm"],
                "upgrades_granary": c["upgrades_granary"],
                "upgrades_defense": c["upgrades_defense"],
                "upgrades_total": c["upgrades_farm"] + c["upgrades_granary"] + c["upgrades_defense"],
                "taxes_collected_gold": round(c["taxes_collected_gold"], 2),
                "births_sexual": c["births_sexual"],
                "births_asexual": c["births_asexual"],
                "deaths_midday_starve": c["deaths_midday_starve"],
                "deaths_eod_starve": c["deaths_eod_starve"],
                "deaths_outdoors": c["deaths_outdoors"],
                "deaths_at_home": c["deaths_at_home"],
                "deaths_with_money_far": c["deaths_with_money_far"],
                "deaths_with_money_empty_depot": c["deaths_with_money_empty_depot"],
                "deaths_in_swamp": c["deaths_in_swamp"],
                "deaths_in_cradle": c["deaths_in_cradle"],
                "deaths_in_frontier": c["deaths_in_frontier"],
                "deaths_grouped": c["deaths_grouped"],
                "deaths_solo": c["deaths_solo"],
                "hof_rescue_events": c["hof_rescue_events"],
                "hof_rescued_blobs": c["hof_rescued_blobs"],
                "eod_at_home_count": c["eod_at_home_count"],
                "eod_outdoors_count": c["eod_outdoors_count"],
                "eod_at_home_pct": eod_at_home_pct,
                "eod_outdoors_pct": eod_outdoors_pct,
                "eod_repro_ready_count": c["eod_repro_ready_count"],
                "eod_surv_only_count": c["eod_surv_only_count"],
                "eod_deficit_count": c["eod_deficit_count"],
                "rock_collisions": c["rock_collisions"],
                "swamp_immersions": c["swamp_immersions"],
            },
            "actions": {
                "mean_move_mag": mean_move,
                "mean_sprint": mean_sprint,
                "mean_trade_desire": mean_trade_d,
                "mean_social_desire": mean_social_d,
                "mean_colonize_desire": mean_col_d,
                "active_sprint_pct": active_sprint_pct,
                "active_trade_pct": active_trade_pct,
                "active_social_pct": active_social_pct,
                "active_colonize_pct": active_col_pct,
            },
            "monoculture": {
                "dominant_tribe_id": dom_id,
                "dominant_tribe_name": dom_name,
                "dominant_tribe_color": dom_color,
                "dominant_share_pct": dom_share_pop,
                "dominant_grouped_share_pct": dom_share_grouped,
                "active_tribes_count": len(active_tribes),
            },
            "genetics": genetics,
            "weights": weights,
            "active_tribes": active_tribes[:16],
        }

        if self.daily_logs and self.daily_logs[-1]["day"] == day:
            self.daily_logs[-1] = snap
        else:
            self._detect_and_record_collapse(world, snap)
            self.daily_logs.append(snap)
            if len(self.daily_logs) > 5000:
                self.daily_logs = self.daily_logs[-5000:]

            if self.persist_to_disk:
                try:
                    log_file = LOG_DIR / "latest_telemetry.jsonl"
                    with log_file.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(snap, ensure_ascii=False) + "\n")
                except Exception:
                    pass

        # Reset daily counters for the new day
        self.daily_counters = _empty_daily_counters()
        self.tribe_daily.clear()
        return snap

    def get_analytics_payload(
        self, world: "SimulationWorld", since_day: int = 0
    ) -> Dict[str, Any]:
        """Build complete or incremental analytics payload for the /analyse cockpit."""
        living = [b for b in world.blobs if b.alive]
        filtered_days = (
            [d for d in self.daily_logs if d["day"] >= since_day]
            if since_day > 0
            else self.daily_logs
        )

        # Sample up to 300 living blobs for the live bio-energetics scatter plot
        step = max(1, len(living) // 300)
        scatter_blobs = [
            {
                "id": b.id,
                "gen": b.generation,
                "age": b.age_days,
                "speed": b.genes.speed,
                "sense": b.genes.sense,
                "size": b.genes.size,
                "cost": round(b.genes._basal_cost, 4),
                "energy": round(b.energy, 1),
                "food": round(b.total_food, 2),
                "money": round(b.money, 1),
                "reputation": round(b.reputation, 2),
                "reward": round(b.cumulative_reward, 2),
                "group_id": b.group_id,
                "group_name": (
                    world.groups[b.group_id].name
                    if (b.group_id is not None and b.group_id in world.groups)
                    else "Einzelgänger"
                ),
                "color": (
                    world.groups[b.group_id].color
                    if (b.group_id is not None and b.group_id in world.groups)
                    else f"hsl({int(b.genes.hue)}, 75%, 56%)"
                ),
            }
            for b in living[::step]
        ]

        # Selected / Champion blob inspector
        selected_blob = None
        if world.selected_blob_id is not None:
            selected_blob = next(
                (b for b in living if b.id == world.selected_blob_id), None
            )
        if selected_blob is None and living:
            selected_blob = max(
                living,
                key=lambda b: b.food_eaten_lifetime * 1.5
                + b.children_count * 3.0
                + b.cumulative_reward * 0.3,
            )

        inspector_data = None
        if selected_blob is not None:
            grp_name = (
                world.groups[selected_blob.group_id].name
                if (selected_blob.group_id is not None and selected_blob.group_id in world.groups)
                else "Einzelgänger"
            )
            grp_color = (
                world.groups[selected_blob.group_id].color
                if (selected_blob.group_id is not None and selected_blob.group_id in world.groups)
                else None
            )
            inspector_data = {
                "blob": selected_blob.to_summary_dict(group_color=grp_color),
                "group_name": grp_name,
                "basal_cost": round(selected_blob.genes._basal_cost, 4),
                "food_eaten_lifetime": round(selected_blob.food_eaten_lifetime, 2),
                "brain": selected_blob.brain.to_inspector_payload(),
            }

        all_tribes = sorted(
            self.tribes.values(),
            key=lambda t: (t["alive"], t["current_members"], t["peak_members"]),
            reverse=True,
        )

        sid, slabel, w_mult, _g, _c = world.season_info
        return {
            "session_id": self.session_id,
            "day": world.day_number,
            "tick": world.tick_count,
            "tick_in_day": world.tick_in_day,
            "ticks_per_day": world.world_cfg.ticks_per_day,
            "time_of_day": round(world.time_of_day, 3),
            "is_night": world.is_night,
            "season": {
                "id": sid,
                "label": slabel,
                "day_in_season": ((world.day_number - 1) % 4) + 1,
                "food_mult": w_mult,
            },
            "summary": {
                "current_population": len(living),
                "peak_population": self._peak_population,
                "peak_pop_day": self._peak_pop_day,
                "active_groups": len(world.groups),
                "total_tribes_ever": len(self.tribes),
                "extinct_tribes": sum(1 for t in self.tribes.values() if not t["alive"]),
                "collapse_events_count": len(self.collapse_logs),
                "total_births": world.total_births_all_time,
                "total_deaths": world.total_deaths_all_time,
                "total_trades": world.total_trades_all_time,
                "total_raids": world.total_raids_all_time,
                "total_repelled": world.total_raids_repelled,
                "total_exiles": world.total_exiles_all_time,
                "total_bounties": world.total_bounties_all_time,
                "avg_reputation": round(float(np.mean([b.reputation for b in living])), 2) if living else 0.0,
                "outlaws_count": sum(1 for b in living if b.is_outlaw),
                "honorable_count": sum(1 for b in living if b.is_honorable),
                "total_migrations": world.total_migrations_all_time,
                "total_upgrades": world.total_upgrades_built,
            },
            "input_labels": INPUT_LABELS,
            "output_labels": OUTPUT_LABELS,
            "pathway_defs": PATHWAY_DEFINITIONS,
            "initial_coupling_matrix": self._initial_coupling_matrix,
            "daily_logs": filtered_days,
            "live_today_counters": self.daily_counters,
            "collapse_logs": self.collapse_logs[-100:],
            "tribes": all_tribes[:120],
            "events": self.community_events[-250:],
            "scatter_blobs": scatter_blobs,
            "selected": inspector_data,
            "net_cfg": world.net_cfg.to_dict(),
            "prefs": world.prefs.to_dict(),
        }
