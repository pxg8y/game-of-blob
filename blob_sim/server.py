"""FastAPI & WebSocket server for the Blob Life Neuroevolution GUI."""

import asyncio
import json
from pathlib import Path
from typing import Set, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import NetworkConfig, TrainingPreferences, WorldConfig
from .world import SimulationWorld

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Blob Life: Neuroevolution & Economy Simulator")


class SimulationController:
    def __init__(self) -> None:
        self.world = SimulationWorld()
        self.running: bool = True
        self.steps_per_frame: int = 1  # 1x .. 50x turbo speed
        self.clients: Set[WebSocket] = set()
        self._loop_task: asyncio.Task | None = None

    async def broadcast_state(self) -> None:
        if not self.clients:
            return
        payload = json.dumps({"type": "state", "data": self.world.get_state_payload()})
        stale: Set[WebSocket] = set()
        for ws in list(self.clients):
            try:
                await ws.send_text(payload)
            except Exception:
                stale.add(ws)
        for ws in stale:
            self.clients.discard(ws)

    def handle_command(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        cmd = msg.get("cmd")

        if cmd == "toggle_pause":
            self.running = not self.running
            return {"ok": True, "running": self.running}

        if cmd == "set_running":
            self.running = bool(msg.get("running", True))
            return {"ok": True, "running": self.running}

        if cmd == "step_once":
            self.running = False
            steps = int(msg.get("steps", 1))
            for _ in range(max(1, min(360, steps))):
                self.world.step()
            return {"ok": True, "running": self.running}

        if cmd == "set_speed":
            self.steps_per_frame = max(1, min(60, int(msg.get("speed", 1))))
            return {"ok": True, "speed": self.steps_per_frame}

        if cmd == "reset_world":
            self.world.reset_world()
            return {"ok": True}

        if cmd == "select_blob":
            blob_id = msg.get("blob_id")
            if blob_id is not None:
                self.world.selected_blob_id = int(blob_id)
            else:
                # Find nearest blob to (x, y)
                x = float(msg.get("x", 0.0))
                y = float(msg.get("y", 0.0))
                living = [b for b in self.world.blobs if b.alive]
                if living:
                    nearest = min(living, key=lambda b: (b.x - x) ** 2 + (b.y - y) ** 2)
                    if ((nearest.x - x) ** 2 + (nearest.y - y) ** 2) <= 45.0 ** 2:
                        self.world.selected_blob_id = nearest.id
            return {"ok": True, "selected_blob_id": self.world.selected_blob_id}

        if cmd == "update_network":
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

        if cmd == "update_prefs":
            prefs_data = msg.get("prefs", {})
            for key, val in prefs_data.items():
                if hasattr(self.world.prefs, key):
                    setattr(self.world.prefs, key, float(val))
            return {"ok": True, "prefs": self.world.prefs.to_dict()}

        if cmd == "update_world":
            w_data = msg.get("world", {})
            if "daily_food_spawn" in w_data:
                self.world.world_cfg.daily_food_spawn = max(10, min(300, int(w_data["daily_food_spawn"])))
            if "ticks_per_day" in w_data:
                self.world.world_cfg.ticks_per_day = max(120, min(1200, int(w_data["ticks_per_day"])))
            if "survival_food_need" in w_data:
                self.world.world_cfg.survival_food_need = max(0.25, min(5.0, float(w_data["survival_food_need"])))
            if "base_food_price" in w_data:
                self.world.world_cfg.base_food_price = max(1.0, min(25.0, float(w_data["base_food_price"])))
            if "initial_blobs" in w_data:
                self.world.world_cfg.initial_blobs = max(8, min(150, int(w_data["initial_blobs"])))
            return {"ok": True, "world_cfg": self.world.world_cfg.to_dict()}

        if cmd == "god_action":
            tool = msg.get("tool", "inspect")
            x = float(msg.get("x", 600.0))
            y = float(msg.get("y", 400.0))
            if tool == "spawn_food":
                self.world.spawn_custom_food_cluster(x, y, count=8)
            elif tool == "spawn_blob":
                self.world.spawn_custom_blob(x, y)
            elif tool == "gift_gold":
                living = [b for b in self.world.blobs if b.alive]
                if living:
                    nearest = min(living, key=lambda b: (b.x - x) ** 2 + (b.y - y) ** 2)
                    nearest.money = round(nearest.money + 15.0, 2)
                    self.world.selected_blob_id = nearest.id
                    self.world.log_event("god", f"Blob #{nearest.id} erhielt +15.0 Gold-Geschenk!")
            return {"ok": True}

        return {"ok": False, "error": f"Unknown command: {cmd}"}


controller = SimulationController()


@app.on_event("startup")
async def on_startup() -> None:
    async def simulation_loop() -> None:
        while True:
            if controller.running:
                for _ in range(controller.steps_per_frame):
                    controller.world.step()
            await controller.broadcast_state()
            await asyncio.sleep(0.033)  # ~30 FPS state broadcast

    controller._loop_task = asyncio.create_task(simulation_loop())


@app.get("/", response_class=HTMLResponse)
async def get_index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/state")
async def get_state() -> JSONResponse:
    return JSONResponse(controller.world.get_state_payload())


@app.get("/api/export")
async def export_population() -> JSONResponse:
    return JSONResponse(controller.world.export_population_json())


@app.post("/api/import")
async def import_population(request: Request) -> JSONResponse:
    data = await request.json()
    controller.world.import_population_json(data)
    return JSONResponse({"ok": True, "population": len(controller.world.blobs)})


@app.post("/api/command")
async def post_command(request: Request) -> JSONResponse:
    msg = await request.json()
    res = controller.handle_command(msg)
    return JSONResponse(res)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    controller.clients.add(ws)
    try:
        # Send immediate initial snapshot
        await ws.send_text(
            json.dumps(
                {
                    "type": "state",
                    "data": controller.world.get_state_payload(),
                    "running": controller.running,
                    "speed": controller.steps_per_frame,
                }
            )
        )
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            controller.handle_command(msg)
            if not controller.running:
                await controller.broadcast_state()
    except WebSocketDisconnect:
        controller.clients.discard(ws)
    except Exception:
        controller.clients.discard(ws)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
