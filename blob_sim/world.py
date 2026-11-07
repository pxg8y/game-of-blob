"""Core 2D Artificial Life & Economy Simulation World."""

from typing import List, Dict, Any, Optional, Tuple
import math
import numpy as np
from .config import NetworkConfig, TrainingPreferences, WorldConfig
from .neural import BlobBrain, INPUT_DIM
from .entities import (
    Genes,
    FoodItem,
    Home,
    Group,
    Blob,
    GROUP_PALETTE,
    GROUP_NAMES,
)


class SimulationWorld:
    """Manages the 2D arena, blobs, homes, groups, trading economy, and neuroevolution."""

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

        self.tick_count: int = 0
        self.day_number: int = 1
        self.tick_in_day: int = 0

        self._next_blob_id: int = 1
        self._next_food_id: int = 1
        self._next_home_id: int = 1
        self._next_group_id: int = 1

        self.blobs: List[Blob] = []
        self.foods: List[FoodItem] = []
        self.homes: Dict[int, Home] = {}
        self.groups: Dict[int, Group] = {}

        # Daily counters
        self.births_today: int = 0
        self.deaths_today: int = 0
        self.trades_today: int = 0
        self.total_trades_all_time: int = 0
        self.total_births_all_time: int = 0
        self.total_deaths_all_time: int = 0

        # Hall of Fame (best brains by cumulative reward / longevity)
        self.hall_of_fame: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self.event_log: List[Dict[str, Any]] = []
        self.recent_trade_lines: List[Dict[str, Any]] = []

        self.selected_blob_id: Optional[int] = None

        self.reset_world()

    @property
    def time_of_day(self) -> float:
        """0.0 = dawn, 0.72 = dusk/nightfall, 1.0 = end of night."""
        return float(self.tick_in_day) / max(1, self.world_cfg.ticks_per_day)

    @property
    def is_night(self) -> bool:
        return self.time_of_day >= self.world_cfg.daylight_ratio

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

    def reset_world(self) -> None:
        """Initialize a fresh world with homes, initial blobs, and food clusters."""
        self.tick_count = 0
        self.day_number = 1
        self.tick_in_day = 0
        self._next_blob_id = 1
        self._next_food_id = 1
        self._next_home_id = 1
        self._next_group_id = 1

        self.blobs.clear()
        self.foods.clear()
        self.homes.clear()
        self.groups.clear()
        self.history.clear()
        self.event_log.clear()
        self.recent_trade_lines.clear()

        self.births_today = 0
        self.deaths_today = 0
        self.trades_today = 0
        self.total_trades_all_time = 0
        self.total_births_all_time = 0
        self.total_deaths_all_time = 0

        # Create initial neighborhood homes arranged around safe rings + settlements
        num_initial_homes = max(6, self.world_cfg.initial_blobs // 3)
        cx, cy = self.world_cfg.width * 0.5, self.world_cfg.height * 0.5
        for i in range(num_initial_homes):
            angle = (2.0 * math.pi * i) / num_initial_homes + self.rng.uniform(-0.15, 0.15)
            rx = self.world_cfg.width * self.rng.uniform(0.30, 0.42)
            ry = self.world_cfg.height * self.rng.uniform(0.30, 0.42)
            hx = float(np.clip(cx + math.cos(angle) * rx, 60.0, self.world_cfg.width - 60.0))
            hy = float(np.clip(cy + math.sin(angle) * ry, 60.0, self.world_cfg.height - 60.0))
            hid = self._next_home_id
            self._next_home_id += 1
            self.homes[hid] = Home(id=hid, x=hx, y=hy, radius=30.0, food_depot=1.5, gold_vault=15.0)

        home_list = list(self.homes.values())
        for i in range(self.world_cfg.initial_blobs):
            home = home_list[i % len(home_list)]
            genes = Genes(
                speed=round(float(self.rng.uniform(0.85, 1.20)), 3),
                sense=round(float(self.rng.uniform(0.85, 1.25)), 3),
                size=round(float(self.rng.uniform(0.85, 1.15)), 3),
                metabolism=1.0,
                hue=round(float(self.rng.uniform(0, 360)), 1),
            )
            brain = BlobBrain(config=self.net_cfg, rng=self.rng)
            bx = float(np.clip(home.x + self.rng.normal(0.0, 14.0), 20.0, self.world_cfg.width - 20.0))
            by = float(np.clip(home.y + self.rng.normal(0.0, 14.0), 20.0, self.world_cfg.height - 20.0))
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
                money=round(float(self.rng.uniform(5.0, 10.0)), 1),
            )
            self._next_blob_id += 1
            home.resident_ids.append(blob.id)
            self.blobs.append(blob)

        self.spawn_daily_food(self.world_cfg.daily_food_spawn)
        if self.blobs:
            self.selected_blob_id = self.blobs[0].id
        self.record_daily_snapshot()
        self.log_event("world", f"Simulation gestartet mit {len(self.blobs)} Blobs & {len(self.homes)} Häusern.")

    def spawn_daily_food(self, count: int) -> None:
        """Spawn food in fertile oasis clusters plus scattered items across the 2D arena."""
        max_food_on_map = max(250, count * 2)
        if len(self.foods) >= max_food_on_map:
            return
        to_spawn = min(count, max_food_on_map - len(self.foods))
        if to_spawn <= 0:
            return

        # 4 fertile oasis centers that shift gently with the days
        num_clusters = 4
        clusters: List[Tuple[float, float]] = []
        for c in range(num_clusters):
            angle = (2.0 * math.pi * c / num_clusters) + (self.day_number * 0.25)
            ox = self.world_cfg.width * 0.5 + math.cos(angle) * (self.world_cfg.width * 0.22)
            oy = self.world_cfg.height * 0.5 + math.sin(angle) * (self.world_cfg.height * 0.22)
            clusters.append((ox, oy))

        for i in range(to_spawn):
            if self.rng.random() < 0.68:
                cx, cy = clusters[i % num_clusters]
                fx = float(np.clip(cx + self.rng.normal(0.0, 95.0), 25.0, self.world_cfg.width - 25.0))
                fy = float(np.clip(cy + self.rng.normal(0.0, 85.0), 25.0, self.world_cfg.height - 25.0))
            else:
                fx = float(self.rng.uniform(25.0, self.world_cfg.width - 25.0))
                fy = float(np.clip(self.rng.uniform(25.0, self.world_cfg.height - 25.0), 25.0, self.world_cfg.height - 25.0))

            is_golden = bool(self.rng.random() < 0.16)
            nutrition = 1.2 if is_golden else 0.6
            self.foods.append(
                FoodItem(
                    id=self._next_food_id,
                    x=fx,
                    y=fy,
                    nutrition=nutrition,
                    golden=is_golden,
                )
            )
            self._next_food_id += 1

    def update_network_architecture(self, new_cfg: NetworkConfig, reset_weights: bool = False) -> None:
        """Update neural network depth/width/activation for all living blobs."""
        self.net_cfg = new_cfg
        for blob in self.blobs:
            if reset_weights:
                blob.brain = BlobBrain(config=new_cfg, rng=self.rng)
            else:
                blob.brain = blob.brain.adapt_to_config(new_cfg)
        self.log_event(
            "brain",
            f"Netz-Architektur aktualisiert: Tiefe={len(new_cfg.get_hidden_dims())}, "
            f"Hidden={new_cfg.get_hidden_dims()}, Aktivierung={new_cfg.activation.upper()}",
        )

    def step(self) -> None:
        """Advance the simulation by 1 tick."""
        self.tick_count += 1
        self.tick_in_day += 1

        # Expire old visual trade lines
        self.recent_trade_lines = [
            t for t in self.recent_trade_lines if (self.tick_count - t["tick"]) <= 18
        ]

        # Mid-day food replenishment
        if self.tick_in_day == self.world_cfg.ticks_per_day // 2:
            self.spawn_daily_food(max(25, self.world_cfg.daily_food_spawn // 3))

        living_blobs = [b for b in self.blobs if b.alive]
        if not living_blobs:
            self._respawn_from_hall_of_fame()
            living_blobs = [b for b in self.blobs if b.alive]

        # Precompute arrays for fast spatial queries
        n_blobs = len(living_blobs)
        blob_xs = np.array([b.x for b in living_blobs], dtype=np.float32)
        blob_ys = np.array([b.y for b in living_blobs], dtype=np.float32)

        if self.foods:
            food_xs = np.array([f.x for f in self.foods], dtype=np.float32)
            food_ys = np.array([f.y for f in self.foods], dtype=np.float32)
        else:
            food_xs = np.empty((0,), dtype=np.float32)
            food_ys = np.empty((0,), dtype=np.float32)

        tod = self.time_of_day
        night_mode = self.is_night
        repro_target = max(0.5, self.world_cfg.reproduction_food_need)

        # 1. Sensory perception + Neural Network Forward Pass + Movement
        for idx, blob in enumerate(living_blobs):
            blob.age_ticks += 1
            blob.ticks_since_last_meal += 1
            if blob.trade_cooldown > 0:
                blob.trade_cooldown -= 1

            vision = blob.vision_radius * (0.65 if night_mode else 1.0)

            # Nearest food within vision (or shared group vision!)
            dx_food, dy_food, dist_food_norm = 0.0, 0.0, 1.0
            if len(food_xs) > 0:
                dfx = food_xs - blob.x
                dfy = food_ys - blob.y
                dists_f = np.hypot(dfx, dfy)
                min_f_idx = int(np.argmin(dists_f))
                min_f_dist = float(dists_f[min_f_idx])
                effective_vision = vision * (1.28 if blob.group_id is not None else 1.0)
                if min_f_dist <= effective_vision and min_f_dist > 1e-4:
                    dx_food = float(dfx[min_f_idx] / min_f_dist)
                    dy_food = float(dfy[min_f_idx] / min_f_dist)
                    dist_food_norm = float(min(1.0, min_f_dist / effective_vision))

            # Vector to Home
            dhx = blob.home_x - blob.x
            dhy = blob.home_y - blob.y
            dist_home = math.hypot(dhx, dhy)
            home_obj = self.homes.get(blob.home_id)
            home_radius = home_obj.radius if home_obj else 28.0
            blob.is_at_home = dist_home <= home_radius

            if dist_home > 1e-4:
                dx_home = dhx / dist_home
                dy_home = dhy / dist_home
            else:
                dx_home, dy_home = 0.0, 0.0
            dist_home_norm = min(1.0, dist_home / 450.0)

            # Nearest peer blob
            dx_peer, dy_peer, dist_peer_norm = 0.0, 0.0, 1.0
            peer_has_food = 0.0
            peer_has_money = 0.0
            peer_same_group = 0.0
            nearest_peer: Optional[Blob] = None
            nearest_peer_dist: float = 999999.0

            if n_blobs > 1:
                dpx = blob_xs - blob.x
                dpy = blob_ys - blob.y
                dists_p = np.hypot(dpx, dpy)
                dists_p[idx] = 999999.0  # exclude self
                min_p_idx = int(np.argmin(dists_p))
                min_p_dist = float(dists_p[min_p_idx])
                if min_p_dist <= vision:
                    nearest_peer = living_blobs[min_p_idx]
                    nearest_peer_dist = min_p_dist
                    if min_p_dist > 1e-4:
                        dx_peer = float(dpx[min_p_idx] / min_p_dist)
                        dy_peer = float(dpy[min_p_idx] / min_p_dist)
                    dist_peer_norm = float(min(1.0, min_p_dist / vision))
                    peer_has_food = 1.0 if (nearest_peer.surplus_food >= 0.25 or nearest_peer.food_today >= 1.5) else -1.0
                    peer_has_money = 1.0 if nearest_peer.money >= self.world_cfg.base_food_price * 0.5 else -1.0
                    peer_same_group = (
                        1.0
                        if (blob.group_id is not None and blob.group_id == nearest_peer.group_id)
                        else -1.0
                    )

            obs = np.array(
                [
                    dx_food,
                    dy_food,
                    1.0 - dist_food_norm,  # Proximity to food (1 = right here, 0 = far)
                    dx_home,
                    dy_home,
                    dist_home_norm,
                    dx_peer,
                    dy_peer,
                    1.0 - dist_peer_norm,  # Proximity to peer
                    peer_has_food,
                    peer_has_money,
                    peer_same_group,
                    min(1.5, blob.food_today / repro_target),
                    (blob.energy / 100.0) * 2.0 - 1.0,
                    math.tanh(blob.money / 20.0),
                    tod * 2.0 - 1.0,
                ],
                dtype=np.float32,
            )

            actions = blob.brain.forward(obs, explore_noise=0.04)
            move_x = float(actions[0])
            move_y = float(actions[1])
            sprint = float((actions[2] + 1.0) * 0.5)  # 0..1
            blob.sprint_level = sprint
            blob.trade_desire = float(actions[3])
            blob.social_desire = float(actions[4])

            # Normalize direction vector
            mag = math.hypot(move_x, move_y)
            if mag > 1.0:
                move_x /= mag
                move_y /= mag

            speed_mult = 1.0 + 0.55 * sprint
            if night_mode and blob.is_at_home:
                # Rest calmly inside home at night if already fed
                if blob.food_today >= self.world_cfg.survival_food_need:
                    speed_mult *= 0.25

            target_vx = move_x * blob.max_speed * speed_mult
            target_vy = move_y * blob.max_speed * speed_mult

            # Smooth inertia
            blob.vx = 0.72 * blob.vx + 0.28 * target_vx
            blob.vy = 0.72 * blob.vy + 0.28 * target_vy
            blob.x += blob.vx
            blob.y += blob.vy

            # Arena boundary reflection
            pad = blob.body_radius + 4.0
            if blob.x < pad:
                blob.x = pad
                blob.vx *= -0.6
            elif blob.x > self.world_cfg.width - pad:
                blob.x = self.world_cfg.width - pad
                blob.vx *= -0.6

            if blob.y < pad:
                blob.y = pad
                blob.vy *= -0.6
            elif blob.y > self.world_cfg.height - pad:
                blob.y = self.world_cfg.height - pad
                blob.vy *= -0.6

            # Re-check home status after movement
            dist_home_after = math.hypot(blob.home_x - blob.x, blob.home_y - blob.y)
            blob.is_at_home = dist_home_after <= home_radius

            # Energy metabolism cost
            move_speed_norm = math.hypot(blob.vx, blob.vy) / max(0.1, blob.max_speed)
            basal = blob.genes.basal_energy_cost()
            movement_cost = basal * (0.45 + 0.75 * (move_speed_norm ** 1.4) + 0.45 * (sprint ** 2))
            if blob.is_at_home:
                # Resting at home saves energy; group villages save even more!
                shelter_factor = 0.38 if blob.group_id is not None else 0.48
                movement_cost *= shelter_factor
            elif night_mode:
                # Being outside at night costs extra energy
                movement_cost *= 1.45

            blob.energy = max(0.0, blob.energy - movement_cost)

            # Calculate instantaneous step reward based on user's TrainingPreferences!
            step_reward = 0.0
            # 1) Exploration / approaching food when hungry
            if len(food_xs) > 0 and blob.food_today < repro_target:
                approach = (move_x * dx_food + move_y * dy_food)
                if approach > 0:
                    step_reward += self.prefs.pref_exploration * 0.06 * approach

            # 2) Returning home at dusk/night or when already at 2x reproduction food
            if (tod >= 0.62 or blob.food_today >= repro_target):
                if blob.is_at_home:
                    step_reward += self.prefs.pref_home * 0.08
                else:
                    home_align = (move_x * dx_home + move_y * dy_home)
                    step_reward += self.prefs.pref_home * 0.05 * home_align

            # 3) Social interaction & group cohesion
            if nearest_peer is not None and nearest_peer_dist < 42.0:
                if blob.social_desire > 0.1 and nearest_peer.social_desire > 0.0:
                    self._handle_social_encounter(blob, nearest_peer)
                    step_reward += self.prefs.pref_social * 0.12

                # 4) Peer-to-Peer Trading (Essen gegen Geld)
                if blob.trade_cooldown == 0 and nearest_peer.trade_cooldown == 0:
                    if blob.trade_desire > -0.1 or nearest_peer.trade_desire > -0.1:
                        traded = self._try_peer_trade(blob, nearest_peer)
                        if traded:
                            step_reward += self.prefs.pref_money * 0.45

            # 5) Village Market / Home Depot trade when at home
            if blob.is_at_home and home_obj is not None and blob.trade_cooldown == 0:
                market_traded = self._try_village_market_trade(blob, home_obj)
                if market_traded:
                    step_reward += self.prefs.pref_money * 0.35

            blob.last_reward = step_reward
            blob.cumulative_reward += step_reward
            blob.brain.apply_online_learning(step_reward, self.prefs.learning_rate)

        # 2. Food collection (Foraging)
        self._process_foraging(living_blobs)

        # 3. Mid-day Starvation Check (blobs that haven't eaten for too long or energy == 0)
        for blob in living_blobs:
            if not blob.alive:
                continue
            if blob.energy <= 0.0 or blob.ticks_since_last_meal >= self.world_cfg.max_starvation_ticks:
                # Emergency attempt: can the blob buy instant emergency ration with money if near home/peer?
                if blob.money >= self.world_cfg.base_food_price and blob.is_at_home:
                    home_obj = self.homes.get(blob.home_id)
                    if home_obj and home_obj.food_depot >= 0.4:
                        home_obj.food_depot -= 0.4
                        blob.money -= self.world_cfg.base_food_price * 0.8
                        blob.food_today += 0.4
                        blob.energy = min(100.0, blob.energy + 38.0)
                        blob.ticks_since_last_meal = 0
                        blob.trades_made += 1
                        self.trades_today += 1
                        self.total_trades_all_time += 1
                        continue

                # Negative terminal reward before dying
                blob.brain.apply_online_learning(-1.2 * self.prefs.pref_food, self.prefs.learning_rate)
                self._kill_blob(blob, reason="verhungert (Energie leer)")

        # 4. End of Day Accounting, Reproduction (2x Food), and Dawn of New Day
        if self.tick_in_day >= self.world_cfg.ticks_per_day:
            self._end_of_day_accounting()

    def _process_foraging(self, living_blobs: List[Blob]) -> None:
        """Check collisions between living blobs and food items."""
        if not self.foods or not living_blobs:
            return

        remaining_foods: List[FoodItem] = []
        repro_need = self.world_cfg.reproduction_food_need

        for food in self.foods:
            eaten = False
            for blob in living_blobs:
                if not blob.alive:
                    continue
                reach = blob.body_radius + (10.0 if food.golden else 8.0)
                if abs(blob.x - food.x) <= reach and abs(blob.y - food.y) <= reach:
                    dist = math.hypot(blob.x - food.x, blob.y - food.y)
                    if dist <= reach:
                        eaten = True
                        nutr = food.nutrition
                        blob.food_eaten_lifetime += nutr
                        blob.ticks_since_last_meal = 0
                        blob.energy = min(100.0, blob.energy + nutr * 44.0)

                        # Fill food_today; any excess beyond 1.5 also builds tradable surplus_food!
                        blob.food_today += nutr
                        if blob.food_today > self.world_cfg.survival_food_need:
                            # Gather extra tradable food reserve in addition to stomach
                            blob.surplus_food = round(blob.surplus_food + nutr * 0.5, 3)

                        # Apply online RL reward for eating food!
                        eat_reward = self.prefs.pref_food * (1.1 if food.golden else 0.75)
                        blob.last_reward += eat_reward
                        blob.cumulative_reward += eat_reward
                        blob.brain.apply_online_learning(eat_reward, self.prefs.learning_rate)
                        break
            if not eaten:
                remaining_foods.append(food)

        self.foods = remaining_foods

    def _handle_social_encounter(self, a: Blob, b: Blob) -> None:
        """Allow two nearby friendly blobs to form or join a Group."""
        if a.group_id is not None and b.group_id == a.group_id:
            return

        # Case 1: One is already in a group, the other is ungrouped -> join group!
        if a.group_id is not None and b.group_id is None:
            grp = self.groups.get(a.group_id)
            if grp:
                b.group_id = grp.id
                if b.id not in grp.member_ids:
                    grp.member_ids.append(b.id)
                self._relocate_blob_home(b, grp.home_id)
                reward = self.prefs.pref_social * 0.5
                a.brain.apply_online_learning(reward, self.prefs.learning_rate)
                b.brain.apply_online_learning(reward, self.prefs.learning_rate)
                return

        if b.group_id is not None and a.group_id is None:
            grp = self.groups.get(b.group_id)
            if grp:
                a.group_id = grp.id
                if a.id not in grp.member_ids:
                    grp.member_ids.append(a.id)
                self._relocate_blob_home(a, grp.home_id)
                reward = self.prefs.pref_social * 0.5
                a.brain.apply_online_learning(reward, self.prefs.learning_rate)
                b.brain.apply_online_learning(reward, self.prefs.learning_rate)
                return

        # Case 2: Both ungrouped -> found a new Group (up to 8 active groups)
        if a.group_id is None and b.group_id is None:
            if len(self.groups) < len(GROUP_PALETTE):
                gid = self._next_group_id
                self._next_group_id += 1
                color = GROUP_PALETTE[(gid - 1) % len(GROUP_PALETTE)]
                name = GROUP_NAMES[(gid - 1) % len(GROUP_NAMES)]
                village_home_id = a.home_id
                home_obj = self.homes.get(village_home_id)
                if home_obj:
                    home_obj.group_id = gid
                    home_obj.radius = 38.0  # Group village is larger!
                grp = Group(
                    id=gid,
                    name=name,
                    color=color,
                    home_id=village_home_id,
                    member_ids=[a.id, b.id],
                    created_day=self.day_number,
                )
                self.groups[gid] = grp
                a.group_id = gid
                b.group_id = gid
                self._relocate_blob_home(b, village_home_id)
                self.log_event("group", f"Neue Gruppe gegründet: {name} (Blob #{a.id} & #{b.id})")
            else:
                # Join the nearest existing group
                nearest_grp = min(
                    self.groups.values(),
                    key=lambda g: math.hypot(
                        self.homes[g.home_id].x - a.x, self.homes[g.home_id].y - a.y
                    )
                    if g.home_id in self.homes
                    else 9999.0,
                )
                for member in (a, b):
                    member.group_id = nearest_grp.id
                    if member.id not in nearest_grp.member_ids:
                        nearest_grp.member_ids.append(member.id)
                    self._relocate_blob_home(member, nearest_grp.home_id)

    def _relocate_blob_home(self, blob: Blob, new_home_id: int) -> None:
        if blob.home_id in self.homes:
            old_h = self.homes[blob.home_id]
            if blob.id in old_h.resident_ids:
                old_h.resident_ids.remove(blob.id)
        if new_home_id in self.homes:
            new_h = self.homes[new_home_id]
            blob.home_id = new_h.id
            blob.home_x = new_h.x
            blob.home_y = new_h.y
            if blob.id not in new_h.resident_ids:
                new_h.resident_ids.append(blob.id)

    def _try_peer_trade(self, a: Blob, b: Blob) -> bool:
        """
        Execute peer-to-peer trade (Food <-> Money) if one blob has surplus food
        and the other needs food and has money.
        """
        same_group = a.group_id is not None and a.group_id == b.group_id
        discount = (1.0 - self.world_cfg.group_trade_discount) if same_group else 1.0
        unit_amount = 0.5
        price = round(self.world_cfg.base_food_price * unit_amount * discount, 2)

        # Check if A can sell to B, or B can sell to A
        for seller, buyer in ((a, b), (b, a)):
            has_sellable = seller.surplus_food >= unit_amount or (
                seller.food_today >= self.world_cfg.survival_food_need + unit_amount
                and seller.food_today > buyer.food_today + 0.5
            )
            needs_food = (
                buyer.food_today < self.world_cfg.reproduction_food_need
                or buyer.energy < 75.0
            )
            if has_sellable and needs_food and buyer.money >= price:
                if seller.surplus_food >= unit_amount:
                    seller.surplus_food = round(seller.surplus_food - unit_amount, 3)
                else:
                    seller.food_today = round(seller.food_today - unit_amount, 3)

                buyer.food_today = round(buyer.food_today + unit_amount, 3)
                buyer.energy = min(100.0, buyer.energy + unit_amount * 40.0)
                buyer.ticks_since_last_meal = 0

                buyer.money = round(buyer.money - price, 2)
                seller.money = round(seller.money + price, 2)

                seller.trades_made += 1
                buyer.trades_made += 1
                seller.trade_cooldown = 30
                buyer.trade_cooldown = 30
                seller.last_trade_tick = seller.age_ticks
                buyer.last_trade_tick = buyer.age_ticks

                self.trades_today += 1
                self.total_trades_all_time += 1
                if same_group and seller.group_id in self.groups:
                    self.groups[seller.group_id].total_trades += 1

                # Visual trade line for canvas rendering
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

                trade_reward = self.prefs.pref_money * 0.65 + (
                    self.prefs.pref_social * 0.35 if same_group else 0.0
                )
                seller.cumulative_reward += trade_reward
                buyer.cumulative_reward += trade_reward
                seller.brain.apply_online_learning(trade_reward, self.prefs.learning_rate)
                buyer.brain.apply_online_learning(trade_reward, self.prefs.learning_rate)
                return True

        return False

    def _try_village_market_trade(self, blob: Blob, home: Home) -> bool:
        """
        At home or group village, blobs can sell surplus_food for money into the depot,
        or buy food from the depot if hungry / close to 2x reproduction threshold.
        """
        unit = 0.5
        discount = (
            (1.0 - self.world_cfg.group_trade_discount * 0.5)
            if (blob.group_id is not None and home.group_id == blob.group_id)
            else 1.0
        )
        price = round(self.world_cfg.base_food_price * unit * discount, 2)

        # 1) Sell surplus food for money
        if blob.surplus_food >= unit and blob.trade_desire >= -0.2:
            blob.surplus_food = round(blob.surplus_food - unit, 3)
            blob.money = round(blob.money + price, 2)
            home.food_depot = round(home.food_depot + unit, 2)
            blob.trades_made += 1
            blob.trade_cooldown = 25
            blob.last_trade_tick = blob.age_ticks
            self.trades_today += 1
            self.total_trades_all_time += 1
            if blob.group_id and blob.group_id in self.groups:
                self.groups[blob.group_id].total_trades += 1
            return True

        # 2) Buy food with money if hungry (< survival need) or close to 2x reproduction threshold!
        repro_need = self.world_cfg.reproduction_food_need
        needs_survival = blob.food_today < self.world_cfg.survival_food_need and (
            self.time_of_day > 0.55 or blob.energy < 55.0
        )
        close_to_repro = (
            blob.food_today >= self.world_cfg.survival_food_need
            and blob.food_today < repro_need
            and (repro_need - blob.food_today) <= unit
        )

        if (needs_survival or close_to_repro) and blob.money >= price and home.food_depot >= unit:
            home.food_depot = round(home.food_depot - unit, 2)
            blob.money = round(blob.money - price, 2)
            blob.food_today = round(blob.food_today + unit, 3)
            blob.energy = min(100.0, blob.energy + unit * 42.0)
            blob.ticks_since_last_meal = 0
            blob.trades_made += 1
            blob.trade_cooldown = 25
            blob.last_trade_tick = blob.age_ticks
            self.trades_today += 1
            self.total_trades_all_time += 1
            if blob.group_id and blob.group_id in self.groups:
                self.groups[blob.group_id].total_trades += 1
            return True

        return False

    def _end_of_day_accounting(self) -> None:
        """
        Execute end-of-day rules:
        - Sell any remaining surplus_food at home or lose spoiled surplus
        - Blobs with < 1x survival_food_need try to buy missing food with money; else suffer starvation
        - Blobs with >= 2x survival_food_need reproduce a new blob with their genes!
        """
        living = [b for b in self.blobs if b.alive]
        survival_need = self.world_cfg.survival_food_need
        repro_need = self.world_cfg.reproduction_food_need

        new_babies: List[Blob] = []

        for blob in living:
            blob.age_days += 1
            home = self.homes.get(blob.home_id)

            # Convert leftover surplus_food if at home into money, or into stomach if needed
            if blob.surplus_food > 0.0:
                if blob.food_today < repro_need:
                    transfer = min(blob.surplus_food, repro_need - blob.food_today)
                    blob.food_today = round(blob.food_today + transfer, 3)
                    blob.surplus_food = round(blob.surplus_food - transfer, 3)
                if blob.surplus_food > 0.0 and blob.is_at_home and home is not None:
                    earned = round(blob.surplus_food * self.world_cfg.base_food_price * 0.8, 2)
                    blob.money = round(blob.money + earned, 2)
                    home.food_depot = round(home.food_depot + blob.surplus_food, 2)
                    blob.trades_made += 1
                    self.trades_today += 1
                    self.total_trades_all_time += 1
                # Remaining unstored surplus food spoils overnight
                blob.surplus_food = 0.0

            # Overnight exposure penalty if not at home
            if not blob.is_at_home:
                blob.energy -= 22.0
                blob.brain.apply_online_learning(-0.45 * self.prefs.pref_home, self.prefs.learning_rate)
            else:
                blob.energy = min(100.0, blob.energy + 12.0)
                blob.brain.apply_online_learning(0.45 * self.prefs.pref_home, self.prefs.learning_rate)

            # Check if blob met daily survival food requirement (1.0x)
            if blob.food_today < survival_need:
                deficit = survival_need - blob.food_today
                cost = round(deficit * self.world_cfg.base_food_price, 2)
                # Can the blob buy its way out of starvation using saved money?
                if blob.money >= cost:
                    blob.money = round(blob.money - cost, 2)
                    blob.food_today = survival_need
                    blob.energy = min(100.0, blob.energy + deficit * 35.0)
                    blob.trades_made += 1
                    self.trades_today += 1
                    self.total_trades_all_time += 1
                    if home is not None:
                        home.food_depot = max(0.0, round(home.food_depot - deficit, 2))
                else:
                    # Heavy starvation drain
                    blob.energy -= 55.0 * (deficit / max(0.1, survival_need))

            if blob.energy <= 0.0:
                self._kill_blob(blob, reason="verhungert am Tagesende")
                continue

            # Check for Reproduction: ate >= 2x survival requirement today!
            if blob.food_today >= repro_need - 1e-5:
                repro_reward = self.prefs.pref_food * 1.4
                blob.cumulative_reward += repro_reward
                blob.brain.apply_online_learning(repro_reward, self.prefs.learning_rate)
                child = self._reproduce_blob(blob)
                new_babies.append(child)

            # Reset daily stomach for the new day (money persists!)
            blob.food_today = 0.0

            # Replenish a bit of village depot naturally from local economy
            if home is not None:
                home.food_depot = min(12.0, round(home.food_depot + 0.15, 2))

        self.blobs.extend(new_babies)
        self._update_hall_of_fame()

        # Remove dead blobs from active list & clean up empty groups
        self.blobs = [b for b in self.blobs if b.alive]
        self._cleanup_groups_and_homes()

        # Ensure population floor so ecosystem keeps evolving
        if len(self.blobs) < self.world_cfg.min_population_floor:
            self._respawn_from_hall_of_fame()

        # Record statistics snapshot for the completed day
        self.record_daily_snapshot()

        # Advance to next day
        self.day_number += 1
        self.tick_in_day = 0
        self.births_today = 0
        self.deaths_today = 0
        self.trades_today = 0

        # Spawn new daily food
        self.spawn_daily_food(self.world_cfg.daily_food_spawn)

    def _reproduce_blob(self, parent: Blob) -> Blob:
        """Produce a new child blob inheriting parent's trained neural network & physical genes."""
        partner: Optional[Blob] = None
        if parent.group_id is not None and parent.group_id in self.groups:
            grp = self.groups[parent.group_id]
            mates = [
                b
                for b in self.blobs
                if b.alive and b.id != parent.id and b.group_id == grp.id
            ]
            if mates:
                # Prefer high-reward group mate for crossover
                partner = max(mates, key=lambda m: m.cumulative_reward)

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

        # Endow child with a small starter gift of money from parent if parent is wealthy
        starter_money = 4.0
        if parent.money > 10.0:
            gift = round(parent.money * 0.2, 1)
            parent.money = round(parent.money - gift, 2)
            starter_money += gift

        bx = float(np.clip(parent.home_x + self.rng.normal(0.0, 14.0), 20.0, self.world_cfg.width - 20.0))
        by = float(np.clip(parent.home_y + self.rng.normal(0.0, 14.0), 20.0, self.world_cfg.height - 20.0))

        child = Blob(
            blob_id=self._next_blob_id,
            x=bx,
            y=by,
            home_id=parent.home_id,
            home_x=parent.home_x,
            home_y=parent.home_y,
            brain=child_brain,
            genes=child_genes,
            generation=parent.generation + 1,
            parent_id=parent.id,
            group_id=parent.group_id,
            money=starter_money,
        )
        self._next_blob_id += 1
        parent.children_count += 1
        self.births_today += 1
        self.total_births_all_time += 1

        if parent.home_id in self.homes:
            self.homes[parent.home_id].resident_ids.append(child.id)
        if parent.group_id is not None and parent.group_id in self.groups:
            self.groups[parent.group_id].member_ids.append(child.id)

        self.log_event(
            "birth",
            f"Blob #{parent.id} hat 2x Tagesnahrung erreicht → Kind #{child.id} (Gen {child.generation}) geboren!",
        )
        return child

    def _kill_blob(self, blob: Blob, reason: str) -> None:
        if not blob.alive:
            return
        blob.alive = False
        blob.cause_of_death = reason
        self.deaths_today += 1
        self.total_deaths_all_time += 1
        self._consider_for_hall_of_fame(blob)
        if self.selected_blob_id == blob.id:
            living = [b for b in self.blobs if b.alive]
            self.selected_blob_id = living[0].id if living else None

    def _consider_for_hall_of_fame(self, blob: Blob) -> None:
        score = float(blob.cumulative_reward + blob.age_days * 2.5 + blob.children_count * 4.0 + blob.money * 0.15)
        entry = {
            "blob_id": blob.id,
            "generation": blob.generation,
            "score": round(score, 2),
            "age_days": blob.age_days,
            "children": blob.children_count,
            "money": round(blob.money, 1),
            "genes": blob.genes.to_dict(),
            "brain": blob.brain.to_dict(),
        }
        self.hall_of_fame.append(entry)
        self.hall_of_fame.sort(key=lambda e: e["score"], reverse=True)
        self.hall_of_fame = self.hall_of_fame[:12]

    def _update_hall_of_fame(self) -> None:
        for b in self.blobs:
            if b.alive and (b.age_days >= 1 or b.children_count >= 1):
                self._consider_for_hall_of_fame(b)

    def _respawn_from_hall_of_fame(self) -> None:
        """Seed mutated offspring from the Hall of Fame if population drops below floor."""
        needed = max(0, self.world_cfg.min_population_floor - len([b for b in self.blobs if b.alive]))
        if needed <= 0:
            return
        home_list = list(self.homes.values())
        if not home_list:
            return

        for i in range(needed):
            home = home_list[i % len(home_list)]
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
                )
                genes = base_genes.mutate(self.rng, self.prefs.mutation_rate, self.prefs.mutation_strength)
                gen = champ.get("generation", 1) + 1
            else:
                brain = BlobBrain(config=self.net_cfg, rng=self.rng)
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
            self.blobs.append(blob)

    def _cleanup_groups_and_homes(self) -> None:
        alive_ids = {b.id for b in self.blobs if b.alive}
        for home in self.homes.values():
            home.resident_ids = [rid for rid in home.resident_ids if rid in alive_ids]

        empty_groups = []
        for gid, grp in self.groups.items():
            grp.member_ids = [mid for mid in grp.member_ids if mid in alive_ids]
            if len(grp.member_ids) == 0:
                empty_groups.append(gid)

        for gid in empty_groups:
            grp = self.groups.pop(gid)
            if grp.home_id in self.homes:
                self.homes[grp.home_id].group_id = None
                self.homes[grp.home_id].radius = 28.0

    def record_daily_snapshot(self) -> None:
        living = [b for b in self.blobs if b.alive]
        pop = len(living)
        if pop > 0:
            avg_money = float(np.mean([b.money for b in living]))
            avg_speed = float(np.mean([b.genes.speed for b in living]))
            avg_sense = float(np.mean([b.genes.sense for b in living]))
            avg_size = float(np.mean([b.genes.size for b in living]))
            avg_reward = float(np.mean([b.cumulative_reward for b in living]))
            max_gen = int(max(b.generation for b in living))
            grouped_count = sum(1 for b in living if b.group_id is not None)
        else:
            avg_money = 0.0
            avg_speed = 1.0
            avg_sense = 1.0
            avg_size = 1.0
            avg_reward = 0.0
            max_gen = 1
            grouped_count = 0

        self.history.append(
            {
                "day": self.day_number,
                "population": pop,
                "births": self.births_today,
                "deaths": self.deaths_today,
                "trades": self.trades_today,
                "avg_money": round(avg_money, 2),
                "avg_speed": round(avg_speed, 3),
                "avg_sense": round(avg_sense, 3),
                "avg_size": round(avg_size, 3),
                "avg_reward": round(avg_reward, 2),
                "max_gen": max_gen,
                "groups": len(self.groups),
                "grouped_blobs": grouped_count,
            }
        )
        if len(self.history) > 120:
            self.history = self.history[-120:]

    # --- Interactive God-Mode & Inspection Helpers ---
    def spawn_custom_food_cluster(self, x: float, y: float, count: int = 8) -> None:
        for _ in range(count):
            fx = float(np.clip(x + self.rng.normal(0.0, 22.0), 15.0, self.world_cfg.width - 15.0))
            fy = float(np.clip(y + self.rng.normal(0.0, 22.0), 15.0, self.world_cfg.height - 15.0))
            self.foods.append(
                FoodItem(
                    id=self._next_food_id,
                    x=fx,
                    y=fy,
                    nutrition=0.5,
                    golden=bool(self.rng.random() < 0.25),
                )
            )
            self._next_food_id += 1
        self.log_event("god", f"Nahrungs-Cluster ({count}x) bei ({int(x)}, {int(y)}) gespawnt.")

    def spawn_custom_blob(self, x: float, y: float) -> Blob:
        # Find nearest home or create one if far
        nearest_home = None
        min_d = 999999.0
        for h in self.homes.values():
            d = math.hypot(h.x - x, h.y - y)
            if d < min_d:
                min_d = d
                nearest_home = h

        if nearest_home is None or min_d > 220.0:
            hid = self._next_home_id
            self._next_home_id += 1
            nearest_home = Home(id=hid, x=x, y=y, radius=28.0, food_depot=2.0)
            self.homes[hid] = nearest_home

        if self.hall_of_fame:
            champ = self.hall_of_fame[0]
            brain = BlobBrain.from_dict(champ["brain"], rng=self.rng).adapt_to_config(self.net_cfg)
            brain = brain.clone_and_mutate(self.prefs.mutation_rate, self.prefs.mutation_strength)
        else:
            brain = BlobBrain(config=self.net_cfg, rng=self.rng)

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
        self.blobs.append(blob)
        self.selected_blob_id = blob.id
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
            # Automatically select highest-reward living blob
            selected_blob = max(living, key=lambda b: b.cumulative_reward)
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
                else "Keine Gruppe (Einzelgänger)"
            )
            selected_inspector = {
                "blob": selected_blob.to_summary_dict(group_color=grp_color),
                "group_name": grp_name,
                "ticks_since_meal": selected_blob.ticks_since_last_meal,
                "vision_radius": round(selected_blob.vision_radius, 1),
                "brain": selected_blob.brain.to_inspector_payload(),
            }

        return {
            "tick": self.tick_count,
            "day": self.day_number,
            "tick_in_day": self.tick_in_day,
            "ticks_per_day": self.world_cfg.ticks_per_day,
            "time_of_day": round(self.time_of_day, 3),
            "is_night": self.is_night,
            "world": {
                "width": self.world_cfg.width,
                "height": self.world_cfg.height,
                "survival_need": self.world_cfg.survival_food_need,
                "repro_need": self.world_cfg.reproduction_food_need,
                "food_price": self.world_cfg.base_food_price,
            },
            "stats": {
                "population": len(living),
                "foods_count": len(self.foods),
                "groups_count": len(self.groups),
                "homes_count": len(self.homes),
                "births_today": self.births_today,
                "deaths_today": self.deaths_today,
                "trades_today": self.trades_today,
                "total_trades": self.total_trades_all_time,
                "total_births": self.total_births_all_time,
                "total_deaths": self.total_deaths_all_time,
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
            "homes": [h.to_dict() for h in self.homes.values()],
            "groups": [g.to_dict() for g in self.groups.values()],
            "trade_lines": self.recent_trade_lines,
            "selected": selected_inspector,
            "history": self.history[-50:],
            "events": self.event_log[-18:],
            "net_cfg": self.net_cfg.to_dict(),
            "prefs": self.prefs.to_dict(),
        }

    def export_population_json(self) -> Dict[str, Any]:
        """Export full simulation configuration, Hall of Fame, and living blobs as JSON."""
        living = [b for b in self.blobs if b.alive]
        return {
            "version": "1.0.0",
            "day": self.day_number,
            "net_cfg": self.net_cfg.to_dict(),
            "prefs": self.prefs.to_dict(),
            "world_cfg": self.world_cfg.to_dict(),
            "hall_of_fame": self.hall_of_fame,
            "blobs": [
                {
                    "generation": b.generation,
                    "money": round(b.money, 2),
                    "genes": b.genes.to_dict(),
                    "brain": b.brain.to_dict(),
                }
                for b in living[:60]
            ],
        }

    def import_population_json(self, data: Dict[str, Any]) -> None:
        """Import a saved population JSON into the simulation."""
        if "net_cfg" in data:
            nc = data["net_cfg"]
            self.net_cfg = NetworkConfig(
                depth=nc.get("depth", 2),
                width=nc.get("width", 24),
                custom_layers=nc.get("custom_layers"),
                activation=nc.get("activation", "tanh"),
            )
        if "prefs" in data:
            p = data["prefs"]
            for k, v in p.items():
                if hasattr(self.prefs, k):
                    setattr(self.prefs, k, float(v))

        self.reset_world()
        imported_blobs = data.get("blobs", [])
        if imported_blobs:
            self.blobs.clear()
            home_list = list(self.homes.values())
            for i, bdata in enumerate(imported_blobs):
                home = home_list[i % len(home_list)]
                g_dict = bdata.get("genes", {})
                genes = Genes(
                    speed=g_dict.get("speed", 1.0),
                    sense=g_dict.get("sense", 1.0),
                    size=g_dict.get("size", 1.0),
                    metabolism=g_dict.get("metabolism", 1.0),
                    hue=g_dict.get("hue", 195.0),
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
                self._next_blob_id += 1
                home.resident_ids.append(blob.id)
                self.blobs.append(blob)
            if self.blobs:
                self.selected_blob_id = self.blobs[0].id
        self.log_event("world", f"Population aus JSON geladen ({len(self.blobs)} Blobs).")
