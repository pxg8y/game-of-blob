"""FastAPI & WebSocket server for the Blob Life Neuroevolution GUI."""

import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import threading
import time
from typing import Set, Dict, Any, AsyncIterator, Tuple, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import NetworkConfig, WorldConfig, WORLD_SCALE_PRESETS
from .world import SimulationWorld

STATIC_DIR = Path(__file__).parent / "static"


class SimulationController:
    def __init__(self) -> None:
        self.world = SimulationWorld(world_cfg=WorldConfig.from_preset("continent"))
        self.running: bool = True
        self.steps_per_frame: int = 1  # 1x .. 60x turbo speed
        self.clients: Set[WebSocket] = set()
        self.client_topo_versions: Dict[WebSocket, int] = {}
        self.client_food_versions: Dict[WebSocket, int] = {}
        self.client_viewports: Dict[WebSocket, Optional[Tuple[float, float, float, float]]] = {}
        self.client_busy: Set[WebSocket] = set()
        self.lock = threading.RLock()
        self._loop_task: asyncio.Task | None = None

    async def broadcast_precomputed(
        self, text_payload: str, binary_payload: bytes, cur_topo: int, cur_food: int
    ) -> None:
        if not self.clients:
            return

        async def _send_one(ws: WebSocket) -> bool:
            while True:
                with self.lock:
                    if ws not in self.clients:
                        return False
                    if ws not in self.client_busy:
                        self.client_busy.add(ws)
                        break
                    if self.running:
                        return True  # Drop frame under real-time streaming backpressure
                await asyncio.sleep(0.005)
            try:
                async with asyncio.timeout(2.0):
                    vp = self.client_viewports.get(ws)
                    if vp is not None:
                        with self.lock:
                            _, custom_bin = self.world.get_ws_state_and_binary(
                                include_topology=False,
                                include_foods=False,
                                viewport=vp,
                            )
                        await ws.send_text(text_payload)
                        await ws.send_bytes(custom_bin)
                    else:
                        await ws.send_text(text_payload)
                        await ws.send_bytes(binary_payload)
                with self.lock:
                    if ws in self.clients:
                        self.client_topo_versions[ws] = cur_topo
                        self.client_food_versions[ws] = cur_food
                        return True
                    return False
            except Exception:
                return False
            finally:
                with self.lock:
                    self.client_busy.discard(ws)

        clients_list = list(self.clients)
        results = await asyncio.gather(*[_send_one(ws) for ws in clients_list], return_exceptions=True)
        stale: Set[WebSocket] = {
            ws for ws, ok in zip(clients_list, results) if ok is not True
        }
        if stale:
            with self.lock:
                for ws in stale:
                    self.clients.discard(ws)
                    self.client_topo_versions.pop(ws, None)
                    self.client_food_versions.pop(ws, None)
                    self.client_viewports.pop(ws, None)
                    self.client_busy.discard(ws)

    async def broadcast_state(self) -> None:
        if not self.clients:
            return
        with self.lock:
            cur_topo = self.world.topology_version
            cur_food = self.world.food_version
            need_topo = any(
                self.client_topo_versions.get(ws, -1) != cur_topo for ws in self.clients
            )
            need_food = any(
                self.client_food_versions.get(ws, -1) != cur_food for ws in self.clients
            )
            meta, binary_payload = self.world.get_ws_state_and_binary(
                include_topology=need_topo,
                include_foods=need_food,
            )
            text_payload = json.dumps(
                {
                    "type": "state",
                    "data": meta,
                    "running": self.running,
                    "speed": self.steps_per_frame,
                }
            )
        await self.broadcast_precomputed(text_payload, binary_payload, cur_topo, cur_food)

    def _cmd_select_blob(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        blob_id = msg.get("blob_id")
        if blob_id is not None:
            self.world.selected_blob_id = int(blob_id)
        else:
            x = float(msg.get("x", 0.0))
            y = float(msg.get("y", 0.0))
            pick_r = float(msg.get("pick_radius", 95.0))
            living = [b for b in self.world.blobs if b.alive]
            if living:
                nearest = min(living, key=lambda b: (b.x - x) ** 2 + (b.y - y) ** 2)
                if ((nearest.x - x) ** 2 + (nearest.y - y) ** 2) <= max(95.0, pick_r) ** 2:
                    self.world.selected_blob_id = nearest.id
        return {"ok": True, "selected_blob_id": self.world.selected_blob_id}

    def _cmd_update_network(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        depth = int(msg.get("depth", self.world.net_cfg.depth))
        width = int(msg.get("width", self.world.net_cfg.width))
        activation = str(msg.get("activation", self.world.net_cfg.activation)).lower()
        custom_str = str(msg.get("custom_layers", "")).strip()
        custom_layers = None
        if custom_str:
            try:
                parsed = [int(x.strip()) for x in custom_str.split(",") if x.strip()]
                if parsed:
                    custom_layers = parsed
            except ValueError:
                custom_layers = None

        reset_weights = bool(msg.get("reset_weights", False))
        new_cfg = NetworkConfig(
            depth=depth,
            width=width,
            custom_layers=custom_layers,
            activation=activation,
        )
        self.world.update_network_architecture(new_cfg, reset_weights=reset_weights)
        return {"ok": True, "net_cfg": new_cfg.to_dict()}

    def _cmd_update_prefs(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        prefs_data = msg.get("prefs", {})
        for key, val in prefs_data.items():
            if hasattr(self.world.prefs, key):
                setattr(self.world.prefs, key, float(val))
        return {"ok": True, "prefs": self.world.prefs.to_dict()}

    def _cmd_update_world(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        w_data = msg.get("world", {})
        wcfg = self.world.world_cfg
        preset_changed = False
        if "scale_preset" in w_data:
            new_preset = str(w_data["scale_preset"]).strip().lower()
            if new_preset in WORLD_SCALE_PRESETS and new_preset != getattr(wcfg, "scale_preset", "classic"):
                wcfg.apply_scale_preset(new_preset)
                preset_changed = True
        if "daily_food_spawn" in w_data and not preset_changed:
            wcfg.daily_food_spawn = max(10, min(25000, int(w_data["daily_food_spawn"])))
        if "ticks_per_day" in w_data:
            wcfg.ticks_per_day = max(120, min(1200, int(w_data["ticks_per_day"])))
        if "survival_food_need" in w_data:
            wcfg.survival_food_need = max(0.25, min(5.0, float(w_data["survival_food_need"])))
        if "base_food_price" in w_data:
            wcfg.base_food_price = max(1.0, min(25.0, float(w_data["base_food_price"])))
        if "initial_blobs" in w_data and not preset_changed:
            wcfg.initial_blobs = max(8, min(12000, int(w_data["initial_blobs"])))
        if "max_population_cap" in w_data and not preset_changed:
            wcfg.max_population_cap = max(100, min(25000, int(w_data["max_population_cap"])))
        if "max_residents_per_home" in w_data:
            max_res = max(2, min(20, int(w_data["max_residents_per_home"])))
            wcfg.max_residents_per_home = max_res
            for home in self.world.homes.values():
                home.max_residents = max_res
        if "start_paradigm" in w_data:
            new_p = str(w_data["start_paradigm"]).strip()
            if new_p in ("curriculum", "tabula_rasa", "rapid_exodus") and new_p != wcfg.start_paradigm:
                wcfg.start_paradigm = new_p
                preset_changed = True
        if preset_changed:
            self.world.reset_world()
        return {"ok": True, "world_cfg": wcfg.to_dict()}

    def _cmd_god_action(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        tool = msg.get("tool", "inspect")
        x = float(msg.get("x", 2400.0))
        y = float(msg.get("y", 1800.0))
        if tool == "spawn_food":
            self.world.spawn_custom_food_cluster(x, y, count=8)
        elif tool == "spawn_blob":
            self.world.spawn_custom_blob(x, y)
        elif tool == "toggle_oasis":
            self.world.toggle_oasis_at(x, y)
        elif tool == "toggle_rock":
            self.world.toggle_rock_at(x, y)
        elif tool == "gift_gold":
            living = [b for b in self.world.blobs if b.alive]
            if living:
                nearest = min(living, key=lambda b: (b.x - x) ** 2 + (b.y - y) ** 2)
                nearest.money = round(nearest.money + 15.0, 2)
                self.world.selected_blob_id = nearest.id
                self.world.log_event("god", f"Blob #{nearest.id} erhielt +15.0 Gold-Geschenk!")
        return {"ok": True}

    def handle_command(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            cmd = msg.get("cmd")

            if cmd == "toggle_pause":
                self.running = not self.running
                return {"ok": True, "running": self.running}

            if cmd == "set_running":
                self.running = bool(msg.get("running", True))
                return {"ok": True, "running": self.running}

            if cmd == "step_once":
                self.running = False
                max_steps = max(1200, self.world.world_cfg.ticks_per_day)
                steps = max(1, min(max_steps, int(msg.get("steps", 1))))
                self.world.advance_ticks(steps)
                return {"ok": True, "running": self.running}

            if cmd == "set_speed":
                self.steps_per_frame = max(1, min(500, int(msg.get("speed", 1))))
                return {"ok": True, "speed": self.steps_per_frame}

            if cmd == "reset_world":
                if "paradigm" in msg:
                    p = str(msg["paradigm"]).strip()
                    if p in ("curriculum", "tabula_rasa", "rapid_exodus"):
                        self.world.world_cfg.start_paradigm = p
                self.world.reset_world()
                return {"ok": True, "paradigm": self.world.get_active_paradigm()}

            if cmd == "set_paradigm":
                p = str(msg.get("paradigm", "curriculum")).strip()
                if p in ("curriculum", "tabula_rasa", "rapid_exodus"):
                    self.world.world_cfg.start_paradigm = p
                    self.world.reset_world()
                return {"ok": True, "paradigm": self.world.get_active_paradigm()}

            if cmd == "set_scale_preset":
                preset = str(msg.get("scale_preset", "continent")).strip().lower()
                if preset in WORLD_SCALE_PRESETS:
                    self.world.world_cfg.apply_scale_preset(preset)
                    self.world.reset_world()
                return {"ok": True, "world_cfg": self.world.world_cfg.to_dict()}

            if cmd == "set_viewport":
                enabled = bool(msg.get("enabled", False))
                bounds = msg.get("bounds")
                return {"ok": True, "viewport": bounds if enabled else None}

            dispatch = {
                "select_blob": self._cmd_select_blob,
                "update_network": self._cmd_update_network,
                "update_prefs": self._cmd_update_prefs,
                "update_world": self._cmd_update_world,
                "god_action": self._cmd_god_action,
            }
            handler = dispatch.get(str(cmd))
            if handler is not None:
                return handler(msg)

            return {"ok": False, "error": f"Unknown command: {cmd}"}


    def _step_worker(
        self, steps: int, need_topo: bool, need_food: bool, running: bool, speed: int
    ) -> Tuple[str, bytes, int, int]:
        with self.lock:
            self.world.advance_ticks(steps)
            meta, binary_payload = self.world.get_ws_state_and_binary(
                include_topology=need_topo,
                include_foods=need_food,
            )
            text_payload = json.dumps(
                {
                    "type": "state",
                    "data": meta,
                    "running": running,
                    "speed": speed,
                }
            )
            return text_payload, binary_payload, self.world.topology_version, self.world.food_version


controller = SimulationController()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    target_frame_dt = 1.0 / 30.0

    async def simulation_loop() -> None:
        while True:
            if controller.running:
                t0 = time.perf_counter()
                with controller.lock:
                    cur_topo = controller.world.topology_version
                    cur_food = controller.world.food_version
                    need_topo = any(
                        controller.client_topo_versions.get(ws, -1) != cur_topo
                        for ws in controller.clients
                    )
                    need_food = any(
                        controller.client_food_versions.get(ws, -1) != cur_food
                        for ws in controller.clients
                    )
                    steps = controller.steps_per_frame
                    running = controller.running

                text_payload, binary_payload, topo_ver, food_ver = await asyncio.to_thread(
                    controller._step_worker,
                    steps,
                    need_topo,
                    need_food,
                    running,
                    steps,
                )
                await controller.broadcast_precomputed(text_payload, binary_payload, topo_ver, food_ver)
                elapsed = time.perf_counter() - t0
                await asyncio.sleep(max(0.002, target_frame_dt - elapsed))
            else:
                await asyncio.sleep(0.025)

    controller._loop_task = asyncio.create_task(simulation_loop())
    try:
        yield
    finally:
        if controller._loop_task is not None:
            controller._loop_task.cancel()


app = FastAPI(title="Blob Life: Neuroevolution & Economy Simulator", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def get_index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/analyse", response_class=HTMLResponse)
async def get_analyse() -> HTMLResponse:
    analyse_path = STATIC_DIR / "analyse.html"
    return HTMLResponse(analyse_path.read_text(encoding="utf-8"))


@app.get("/api/analytics")
async def get_analytics(since_day: int = 0) -> JSONResponse:
    with controller.lock:
        payload = controller.world.telemetry.get_analytics_payload(
            controller.world, since_day=since_day
        )
        payload["running"] = controller.running
        payload["speed"] = controller.steps_per_frame
    return JSONResponse(payload)


@app.get("/api/analytics/export")
async def export_analytics() -> JSONResponse:
    with controller.lock:
        payload = controller.world.telemetry.get_analytics_payload(
            controller.world, since_day=0
        )
    return JSONResponse(payload)


@app.get("/api/state")
async def get_state() -> JSONResponse:
    with controller.lock:
        payload = controller.world.get_state_payload()
    return JSONResponse(payload)


@app.get("/api/export")
async def export_population() -> JSONResponse:
    with controller.lock:
        payload = controller.world.export_population_json()
    return JSONResponse(payload)


@app.post("/api/import")
async def import_population(request: Request) -> JSONResponse:
    data = await request.json()
    with controller.lock:
        controller.world.import_population_json(data)
        pop = len(controller.world.blobs)
    await controller.broadcast_state()
    return JSONResponse({"ok": True, "population": pop})


@app.post("/api/command")
async def post_command(request: Request) -> JSONResponse:
    msg = await request.json()
    res = controller.handle_command(msg)
    if not controller.running:
        await controller.broadcast_state()
    return JSONResponse(res)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    try:
        await ws.accept()
        with controller.lock:
            controller.client_busy.add(ws)
            cur_topo = controller.world.topology_version
            cur_food = controller.world.food_version
            meta, binary_payload = controller.world.get_ws_state_and_binary(
                include_topology=True, include_foods=True
            )
            init_payload = json.dumps(
                {
                    "type": "state",
                    "data": meta,
                    "running": controller.running,
                    "speed": controller.steps_per_frame,
                }
            )
        await ws.send_text(init_payload)
        await ws.send_bytes(binary_payload)
        with controller.lock:
            controller.clients.add(ws)
            controller.client_topo_versions[ws] = cur_topo
            controller.client_food_versions[ws] = cur_food
            controller.client_viewports[ws] = None
            controller.client_busy.discard(ws)
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            cmd = msg.get("cmd")
            if cmd == "set_viewport":
                enabled = bool(msg.get("enabled", False))
                bounds = msg.get("bounds")
                with controller.lock:
                    controller.client_viewports[ws] = (
                        tuple(bounds) if (enabled and bounds and len(bounds) == 4) else None
                    )
            else:
                controller.handle_command(msg)
                if not controller.running:
                    await controller.broadcast_state()
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception:
        pass
    finally:
        with controller.lock:
            controller.clients.discard(ws)
            controller.client_topo_versions.pop(ws, None)
            controller.client_food_versions.pop(ws, None)
            controller.client_viewports.pop(ws, None)
            controller.client_busy.discard(ws)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
