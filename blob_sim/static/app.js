/**
 * Blob Life — Neuroevolution, Settlement & Economy Simulator Frontend
 */

(() => {
  const worldCanvas = document.getElementById("worldCanvas");
  const wCtx = worldCanvas.getContext("2d");
  const minimapCanvas = document.getElementById("minimapCanvas");
  const mCtx = minimapCanvas ? minimapCanvas.getContext("2d") : null;
  const nnCanvas = document.getElementById("nnCanvas");
  const nnCtx = nnCanvas ? nnCanvas.getContext("2d") : null;
  const chartCanvas = document.getElementById("chartCanvas");
  const chartCtx = chartCanvas ? chartCanvas.getContext("2d") : null;

  let simState = null;
  let blobsF32 = null;
  let foodsF32 = null;
  let nBlobsCount = 0;
  let nFoodsCount = 0;
  let cachedObstacles = [];
  let cachedZones = [];
  let cachedSkillGates = [];
  let cachedOases = null;
  let cachedHomes = null;
  let lastWorldW = 0;
  let lastWorldH = 0;
  const colorHexCache = new Map();

  // Feature 18: Zero-Allocation Typed Array Double-Buffering & LERP State (Up to 25,000+ blobs)
  const MAX_BLOBS_CAP = 25000;
  let maxBlobsCap = MAX_BLOBS_CAP;
  let blobBufA = new Float32Array(maxBlobsCap * 8);
  let blobBufB = new Float32Array(maxBlobsCap * 8);
  let currBlobBuf = blobBufA;
  let prevBlobBuf = blobBufB;
  let lastTickTime = performance.now();

  function ensureBlobBufferCapacity(requiredBlobs) {
    if (requiredBlobs <= maxBlobsCap) return;
    const newCap = Math.max(requiredBlobs, Math.ceil((maxBlobsCap * 1.5) / 1000) * 1000);
    const newBufA = new Float32Array(newCap * 8);
    const newBufB = new Float32Array(newCap * 8);
    if (currBlobBuf) {
      newBufA.set(currBlobBuf.subarray(0, Math.min(currBlobBuf.length, newBufA.length)));
    }
    if (prevBlobBuf) {
      newBufB.set(prevBlobBuf.subarray(0, Math.min(prevBlobBuf.length, newBufB.length)));
    }
    blobBufA = newBufA;
    blobBufB = newBufB;
    currBlobBuf = blobBufA;
    prevBlobBuf = blobBufB;
    maxBlobsCap = newCap;
  }

  // Feature 18: Color-Bucket Path Batching Data Structures
  const colorBuckets = new Map();
  const outlawCoords = [];
  const foodRingTiers = {
    repro: [],
    survival: [],
    under: [],
  };

  // Feature 18: Offscreen Static Minimap Canvas & 5 Hz Throttling
  const staticMinimapCanvas = document.createElement("canvas");
  staticMinimapCanvas.width = 208;
  staticMinimapCanvas.height = 156;
  const smCtx = staticMinimapCanvas.getContext("2d");
  let staticMinimapDirty = true;
  let lastMinimapRenderTime = 0;

  // Feature 17: Client Viewport Bounds Command Debounce
  let viewportDebounceTimer = null;
  let lastSentViewport = null;

  let lastStateTime = performance.now();
  let smoothedIntervalMs = 34.0;
  let stateDirty = false;
  let lastHeavyUiUpdateTime = 0;
  let isRunning = true;
  let currentSpeed = 1;
  let currentTool = "inspect";
  let showVisionLines = false;
  let showZoneOverlay = true;
  let ws = null;

  // --- Camera & Viewport State (Supports up to 40,000x30,000 Continent Worlds) ---
  let camX = 20000.0;
  let camY = 15000.0;
  let camZoom = 0.028;
  let followSelected = false;
  let isPanning = false;
  let panStartScreenX = 0;
  let panStartScreenY = 0;
  let panStartCamX = 0;
  let panStartCamY = 0;
  let pointerMovedDist = 0;
  let isMinimapDragging = false;

  function rgbIntToHex(rgbInt) {
    const key = rgbInt | 0;
    let hex = colorHexCache.get(key);
    if (!hex) {
      hex = "#" + (key & 0xffffff).toString(16).padStart(6, "0");
      colorHexCache.set(key, hex);
    }
    return hex;
  }

  function lerpAngle(a, b, t) {
    let diff = (b - a) % (Math.PI * 2);
    if (diff < -Math.PI) diff += Math.PI * 2;
    if (diff > Math.PI) diff -= Math.PI * 2;
    return a + diff * t;
  }

  function getWorldSize() {
    return {
      w: (simState && simState.world && simState.world.width) || 40000,
      h: (simState && simState.world && simState.world.height) || 30000,
    };
  }

  function getMinZoom() {
    const { w, h } = getWorldSize();
    return Math.min(0.18, Math.min(worldCanvas.width / w, worldCanvas.height / h) * 0.88);
  }

  function getLoDLabel() {
    if (camZoom < 0.12) return "MAKRO";
    if (camZoom < 0.45) return "MESO";
    return "MIKRO";
  }

  function updateCameraHud() {
    const zoomBadge = document.getElementById("camZoomBadge");
    if (zoomBadge) {
      const pct = camZoom < 0.1 ? `${(camZoom * 100).toFixed(1)}%` : `${Math.round(camZoom * 100)}%`;
      zoomBadge.textContent = `${pct} · ${getLoDLabel()}`;
    }
    const coordsEl = document.getElementById("minimapCoords");
    if (coordsEl) {
      coordsEl.textContent = `(${Math.round(camX)}, ${Math.round(camY)}) · ${getLoDLabel()}`;
    }
    const btnFollow = document.getElementById("btnCamFollow");
    if (btnFollow) {
      btnFollow.classList.toggle("btn-active", followSelected);
    }
  }

  function sendViewportBounds() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const isCulled = camZoom >= 0.45;
    if (!isCulled) {
      if (lastSentViewport !== null) {
        lastSentViewport = null;
        ws.send(JSON.stringify({ cmd: "set_viewport", enabled: false }));
      }
      return;
    }
    const halfViewW = (worldCanvas.width * 0.5) / camZoom;
    const halfViewH = (worldCanvas.height * 0.5) / camZoom;
    const margin = Math.max(250, 40 / camZoom);
    const bounds = [
      camX - halfViewW - margin,
      camY - halfViewH - margin,
      camX + halfViewW + margin,
      camY + halfViewH + margin,
    ];
    lastSentViewport = bounds;
    ws.send(JSON.stringify({ cmd: "set_viewport", enabled: true, bounds }));
  }

  function queueViewportUpdate() {
    if (viewportDebounceTimer) clearTimeout(viewportDebounceTimer);
    viewportDebounceTimer = setTimeout(sendViewportBounds, 150);
  }

  function clampCamera() {
    const { w, h } = getWorldSize();
    const minZ = getMinZoom();
    camZoom = Math.max(minZ, Math.min(2.8, camZoom));
    camX = Math.max(0, Math.min(w, camX));
    camY = Math.max(0, Math.min(h, camY));
    updateCameraHud();
    queueViewportUpdate();
  }

  function fitCameraToWorld() {
    followSelected = false;
    const { w, h } = getWorldSize();
    camX = w * 0.5;
    camY = h * 0.5;
    camZoom = Math.min(worldCanvas.width / w, worldCanvas.height / h) * 0.95;
    clampCamera();
  }

  function canvasToWorld(clientX, clientY) {
    const rect = worldCanvas.getBoundingClientRect();
    const scaleX = worldCanvas.width / rect.width;
    const scaleY = worldCanvas.height / rect.height;
    const cx = (clientX - rect.left) * scaleX;
    const cy = (clientY - rect.top) * scaleY;
    const wx = camX + (cx - worldCanvas.width * 0.5) / camZoom;
    const wy = camY + (cy - worldCanvas.height * 0.5) / camZoom;
    return { cx, cy, wx, wy, scaleX, scaleY };
  }

  // --- WebSocket Connection (Hybrid JSON Metadata + Packed Float32 Binary Frames) ---
  function connectWebSocket() {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${window.location.host}/ws`;
    ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";

    ws.onmessage = (event) => {
      try {
        if (event.data instanceof ArrayBuffer) {
          const buf = event.data;
          if (buf.byteLength < 16) return;
          const header = new Uint32Array(buf, 0, 4);
          if (header[0] !== 0x424c4f42) return;
          const nBlobs = header[1];
          const nFoods = header[2];
          const expectedBytes = 16 + nBlobs * 32 + nFoods * 12;
          if (buf.byteLength < expectedBytes) return;

          if (nBlobs > maxBlobsCap) {
            ensureBlobBufferCapacity(nBlobs);
          }

          // Zero-allocation double-buffer swap
          const temp = prevBlobBuf;
          prevBlobBuf = currBlobBuf;
          currBlobBuf = temp;

          const newBlobs = new Float32Array(buf, 16, nBlobs * 8);
          if (currBlobBuf.length >= newBlobs.length) {
            currBlobBuf.set(newBlobs);
          }
          blobsF32 = currBlobBuf;
          nBlobsCount = nBlobs;
          lastTickTime = performance.now();

          // Feature 15: Stationary Food Caching (store on nFoods > 0, retain on nFoods == 0)
          if (nFoods > 0) {
            foodsF32 = new Float32Array(buf.slice(16 + nBlobs * 32, 16 + nBlobs * 32 + nFoods * 12));
            nFoodsCount = nFoods;
          }
          stateDirty = true;
          return;
        }

        const msg = JSON.parse(event.data);
        if (msg.type === "state" && msg.data) {
          const now = performance.now();
          const dt = now - lastStateTime;
          if (dt > 8 && dt < 250) {
            smoothedIntervalMs = smoothedIntervalMs * 0.8 + dt * 0.2;
          }
          lastStateTime = now;

          const incoming = msg.data;
          if (incoming.obstacles) {
            cachedObstacles = incoming.obstacles;
            staticMinimapDirty = true;
          } else {
            incoming.obstacles = cachedObstacles;
          }
          if (incoming.oases) {
            cachedOases = incoming.oases;
            staticMinimapDirty = true;
          } else if (cachedOases) {
            incoming.oases = cachedOases;
          }
          if (incoming.homes) {
            cachedHomes = incoming.homes;
            staticMinimapDirty = true;
          } else if (cachedHomes) {
            incoming.homes = cachedHomes;
          }
          if (incoming.world) {
            if (incoming.world.zones) {
              cachedZones = incoming.world.zones;
              staticMinimapDirty = true;
            } else {
              incoming.world.zones = cachedZones;
            }
            if (incoming.world.skill_gates) {
              cachedSkillGates = incoming.world.skill_gates;
              staticMinimapDirty = true;
            } else {
              incoming.world.skill_gates = cachedSkillGates;
            }
          }

          // Feature 15: Eaten Foods Masking in Cached Buffer
          if (incoming.eaten_foods && foodsF32 && incoming.eaten_foods.length > 0) {
            for (let j = 0; j < incoming.eaten_foods.length; j++) {
              const eIdx = incoming.eaten_foods[j];
              const off = eIdx * 3;
              if (off + 2 < foodsF32.length) {
                foodsF32[off + 2] = -1.0;
              }
            }
          }

          simState = incoming;
          if (incoming.world && (incoming.world.width !== lastWorldW || incoming.world.height !== lastWorldH)) {
            lastWorldW = incoming.world.width;
            lastWorldH = incoming.world.height;
            fitCameraToWorld();
          }
          stateDirty = true;

          if (typeof msg.running === "boolean" && msg.running !== isRunning) {
            isRunning = msg.running;
            updatePlayPauseButton();
          }
          if (typeof msg.speed === "number" && msg.speed !== currentSpeed) {
            currentSpeed = msg.speed;
            updateActiveSpeedButton(msg.speed);
          }
        }
      } catch (err) {
        console.error("WS parse error:", err);
      }
    };

    ws.onclose = () => {
      setTimeout(connectWebSocket, 1500);
    };
  }

  function sendCommand(payload) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    } else {
      fetch("/api/command", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }).catch(() => {});
    }
  }

  // --- UI Controls Binding ---
  const btnPlayPause = document.getElementById("btnPlayPause");
  function updatePlayPauseButton() {
    btnPlayPause.textContent = isRunning ? "⏸ Pause" : "▶ Start";
  }

  function updateActiveSpeedButton(speed) {
    currentSpeed = speed;
    document.querySelectorAll(".speed-btn").forEach((b) => {
      const btnSpeed = parseInt(b.getAttribute("data-speed"), 10);
      b.classList.toggle("active", btnSpeed === speed);
    });
  }

  btnPlayPause.addEventListener("click", () => {
    isRunning = !isRunning;
    updatePlayPauseButton();
    sendCommand({ cmd: "set_running", running: isRunning });
  });

  document.getElementById("btnStepTick").addEventListener("click", () => {
    isRunning = false;
    updatePlayPauseButton();
    sendCommand({ cmd: "step_once", steps: 1 });
  });

  document.getElementById("btnStepDay").addEventListener("click", () => {
    isRunning = false;
    updatePlayPauseButton();
    const tpd = simState ? simState.ticks_per_day : 360;
    const rem = simState && simState.tick_in_day > 0 ? tpd - simState.tick_in_day : tpd;
    sendCommand({ cmd: "step_once", steps: rem });
  });

  document.getElementById("btnResetWorld").addEventListener("click", () => {
    sendCommand({ cmd: "reset_world" });
  });

  document.querySelectorAll(".speed-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const speed = parseInt(btn.getAttribute("data-speed"), 10);
      updateActiveSpeedButton(speed);
      sendCommand({ cmd: "set_speed", speed });
    });
  });

  // God tools
  document.querySelectorAll(".god-tool-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".god-tool-btn").forEach((b) => b.classList.remove("btn-active"));
      btn.classList.add("btn-active");
      currentTool = btn.getAttribute("data-tool");
    });
  });

  const btnToggleVision = document.getElementById("btnToggleVision");
  btnToggleVision.addEventListener("click", () => {
    showVisionLines = !showVisionLines;
    btnToggleVision.classList.toggle("btn-active", showVisionLines);
  });

  const btnToggleZones = document.getElementById("btnToggleZones");
  if (btnToggleZones) {
    btnToggleZones.addEventListener("click", () => {
      showZoneOverlay = !showZoneOverlay;
      btnToggleZones.classList.toggle("btn-active", showZoneOverlay);
    });
  }

  // Camera HUD Controls
  const btnCamFollow = document.getElementById("btnCamFollow");
  if (btnCamFollow) {
    btnCamFollow.addEventListener("click", () => {
      followSelected = !followSelected;
      if (followSelected && camZoom < 0.55) {
        camZoom = 0.65;
      }
      clampCamera();
    });
  }

  const btnCamFit = document.getElementById("btnCamFit");
  if (btnCamFit) {
    btnCamFit.addEventListener("click", () => {
      followSelected = false;
      const { w, h } = getWorldSize();
      camX = w * 0.5;
      camY = h * 0.5;
      camZoom = Math.min(worldCanvas.width / w, worldCanvas.height / h) * 0.96;
      clampCamera();
    });
  }

  const btnCamZoomIn = document.getElementById("btnCamZoomIn");
  if (btnCamZoomIn) {
    btnCamZoomIn.addEventListener("click", () => {
      camZoom *= 1.25;
      clampCamera();
    });
  }

  const btnCamZoomOut = document.getElementById("btnCamZoomOut");
  if (btnCamZoomOut) {
    btnCamZoomOut.addEventListener("click", () => {
      camZoom /= 1.25;
      clampCamera();
    });
  }

  // Canvas Drag Panning & Wheel Zoom
  worldCanvas.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 && e.button !== 1) return;
    isPanning = true;
    pointerMovedDist = 0;
    panStartScreenX = e.clientX;
    panStartScreenY = e.clientY;
    panStartCamX = camX;
    panStartCamY = camY;
  });

  window.addEventListener("pointermove", (e) => {
    if (!isPanning) return;
    const rect = worldCanvas.getBoundingClientRect();
    const scaleX = worldCanvas.width / Math.max(1, rect.width);
    const scaleY = worldCanvas.height / Math.max(1, rect.height);
    const dx = (e.clientX - panStartScreenX) * scaleX;
    const dy = (e.clientY - panStartScreenY) * scaleY;
    pointerMovedDist = Math.hypot(dx, dy);
    if (pointerMovedDist > 4) {
      followSelected = false;
      camX = panStartCamX - dx / camZoom;
      camY = panStartCamY - dy / camZoom;
      clampCamera();
    }
  });

  window.addEventListener("pointerup", () => {
    isPanning = false;
    isMinimapDragging = false;
  });

  worldCanvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      const before = canvasToWorld(e.clientX, e.clientY);
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      camZoom = Math.max(getMinZoom(), Math.min(2.8, camZoom * factor));
      if (!followSelected) {
        camX = before.wx - (before.cx - worldCanvas.width * 0.5) / camZoom;
        camY = before.wy - (before.cy - worldCanvas.height * 0.5) / camZoom;
      }
      clampCamera();
    },
    { passive: false }
  );

  // Keyboard Panning (WASD / Arrow keys)
  window.addEventListener("keydown", (e) => {
    const tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : "";
    if (tag === "input" || tag === "select" || tag === "textarea") return;
    const step = 110 / camZoom;
    if (e.key === "w" || e.key === "W" || e.key === "ArrowUp") {
      followSelected = false;
      camY -= step;
      clampCamera();
    } else if (e.key === "s" || e.key === "S" || e.key === "ArrowDown") {
      followSelected = false;
      camY += step;
      clampCamera();
    } else if (e.key === "a" || e.key === "A" || e.key === "ArrowLeft") {
      followSelected = false;
      camX -= step;
      clampCamera();
    } else if (e.key === "d" || e.key === "D" || e.key === "ArrowRight") {
      followSelected = false;
      camX += step;
      clampCamera();
    } else if (e.key === "f" || e.key === "F") {
      followSelected = !followSelected;
      clampCamera();
    }
  });

  // Minimap Click & Drag Navigation
  function jumpCameraFromMinimap(e) {
    if (!minimapCanvas) return;
    const rect = minimapCanvas.getBoundingClientRect();
    const rx = Math.max(0, Math.min(1, (e.clientX - rect.left) / Math.max(1, rect.width)));
    const ry = Math.max(0, Math.min(1, (e.clientY - rect.top) / Math.max(1, rect.height)));
    const { w, h } = getWorldSize();
    followSelected = false;
    camX = rx * w;
    camY = ry * h;
    clampCamera();
  }

  if (minimapCanvas) {
    minimapCanvas.addEventListener("pointerdown", (e) => {
      isMinimapDragging = true;
      jumpCameraFromMinimap(e);
    });
    minimapCanvas.addEventListener("pointermove", (e) => {
      if (isMinimapDragging) {
        jumpCameraFromMinimap(e);
      }
    });
  }

  // Canvas Click Interaction (World Space Unprojection)
  worldCanvas.addEventListener("click", (e) => {
    if (pointerMovedDist > 6) return;
    const { wx, wy } = canvasToWorld(e.clientX, e.clientY);

    if (currentTool === "inspect") {
      sendCommand({
        cmd: "select_blob",
        x: wx,
        y: wy,
        pick_radius: Math.max(85, 32 / camZoom),
      });
    } else {
      sendCommand({ cmd: "god_action", tool: currentTool, x: wx, y: wy });
    }
  });

  updateCameraHud();

  // Neural Network Depth & Width Sliders
  const sliderNetDepth = document.getElementById("sliderNetDepth");
  const sliderNetWidth = document.getElementById("sliderNetWidth");
  const selectActivation = document.getElementById("selectActivation");
  const inputCustomLayers = document.getElementById("inputCustomLayers");

  sliderNetDepth.addEventListener("input", () => {
    document.getElementById("valNetDepth").textContent = sliderNetDepth.value;
  });
  sliderNetWidth.addEventListener("input", () => {
    document.getElementById("valNetWidth").textContent = sliderNetWidth.value;
  });

  function applyNetworkConfig(resetWeights) {
    sendCommand({
      cmd: "update_network",
      depth: parseInt(sliderNetDepth.value, 10),
      width: parseInt(sliderNetWidth.value, 10),
      activation: selectActivation.value,
      custom_layers: inputCustomLayers.value,
      reset_weights: resetWeights,
    });
  }

  document.getElementById("btnApplyNetMigrate").addEventListener("click", () => applyNetworkConfig(false));
  document.getElementById("btnApplyNetReset").addEventListener("click", () => applyNetworkConfig(true));

  // Training Preference Sliders
  const prefDisplayMap = {
    pref_food: "valPrefFood",
    pref_money: "valPrefMoney",
    pref_social: "valPrefSocial",
    pref_home: "valPrefHome",
    pref_exploration: "valPrefExplore",
    learning_rate: "valLearningRate",
    mutation_rate: "valMutRate",
    mutation_strength: "valMutStrength",
  };

  document.querySelectorAll(".pref-slider").forEach((slider) => {
    slider.addEventListener("input", () => {
      const key = slider.getAttribute("data-key");
      const val = parseFloat(slider.value);
      const labelId = prefDisplayMap[key];
      if (labelId) {
        document.getElementById(labelId).textContent =
          key === "learning_rate" ? val.toFixed(3) : val.toFixed(2);
      }
      sendCommand({ cmd: "update_prefs", prefs: { [key]: val } });
    });
  });

  // World, Scale Preset, Settlement & Economy Sliders
  const selectScalePreset = document.getElementById("selectScalePreset");
  const selectStartParadigm = document.getElementById("selectStartParadigm");
  const paradigmDescEl = document.getElementById("paradigmDescText");
  const paradigmBadgeEl = document.getElementById("paradigmBadge");
  const sliderFoodSpawn = document.getElementById("sliderFoodSpawn");
  const sliderSurvivalNeed = document.getElementById("sliderSurvivalNeed");
  const sliderFoodPrice = document.getElementById("sliderFoodPrice");
  const sliderMaxResidents = document.getElementById("sliderMaxResidents");

  if (selectScalePreset) {
    selectScalePreset.addEventListener("change", () => {
      sendCommand({
        cmd: "set_scale_preset",
        scale_preset: selectScalePreset.value,
        preset: selectScalePreset.value,
      });
    });
  }

  if (selectStartParadigm) {
    selectStartParadigm.addEventListener("change", () => {
      sendCommand({ cmd: "set_paradigm", paradigm: selectStartParadigm.value });
    });
  }

  sliderFoodSpawn.addEventListener("input", () => {
    const v = parseInt(sliderFoodSpawn.value, 10);
    document.getElementById("valFoodSpawn").textContent = String(v);
    sendCommand({ cmd: "update_world", world: { daily_food_spawn: v } });
  });

  sliderSurvivalNeed.addEventListener("input", () => {
    const v = parseFloat(sliderSurvivalNeed.value);
    document.getElementById("valSurvivalNeed").textContent = `${v.toFixed(2)} (${(v * 2).toFixed(2)} = Kind)`;
    sendCommand({ cmd: "update_world", world: { survival_food_need: v } });
  });

  sliderFoodPrice.addEventListener("input", () => {
    const v = parseFloat(sliderFoodPrice.value);
    document.getElementById("valFoodPrice").textContent = `${v.toFixed(1)} 🪙`;
    sendCommand({ cmd: "update_world", world: { base_food_price: v } });
  });

  if (sliderMaxResidents) {
    sliderMaxResidents.addEventListener("input", () => {
      const v = parseInt(sliderMaxResidents.value, 10);
      document.getElementById("valMaxResidents").textContent = `${v} 🏠`;
      sendCommand({ cmd: "update_world", world: { max_residents_per_home: v } });
    });
  }

  // Export & Import JSON
  document.getElementById("btnExportJson").addEventListener("click", async () => {
    const res = await fetch("/api/export");
    const data = await res.json();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `blob_life_tag_${data.day || 1}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });

  const fileImportInput = document.getElementById("fileImportInput");
  document.getElementById("btnImportJson").addEventListener("click", () => {
    fileImportInput.click();
  });

  fileImportInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      const text = await file.text();
      const json = JSON.parse(text);
      await fetch("/api/import", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(json),
      });
    } catch (err) {
      console.error("Failed to import JSON:", err);
    } finally {
      e.target.value = "";
    }
  });

  const selectChartModeEl = document.getElementById("selectChartMode");
  if (selectChartModeEl) {
    selectChartModeEl.addEventListener("change", () => {
      lastChartSignature = "";
      if (simState) renderHistoryChart(simState.history || []);
    });
  }

  // --- Update UI Readouts ---
  let lastEventTick = -1;
  let lastGroupsSignature = "";
  let lastChartSignature = "";

  function syncSidebarControls(state) {
    if (state.world) {
      if (
        selectScalePreset &&
        state.world.scale_preset &&
        document.activeElement !== selectScalePreset
      ) {
        selectScalePreset.value = state.world.scale_preset;
      }
      if (
        selectStartParadigm &&
        state.world.start_paradigm &&
        document.activeElement !== selectStartParadigm
      ) {
        selectStartParadigm.value = state.world.start_paradigm;
      }
      if (state.world.paradigm) {
        const pDesc =
          state.world.paradigm.short_desc ||
          state.world.paradigm.description ||
          state.world.paradigm.desc;
        if (paradigmDescEl && pDesc) {
          paradigmDescEl.textContent = pDesc;
        }
        if (paradigmBadgeEl) {
          const cradles = state.world.num_cradles || 1;
          const pId = state.world.paradigm.id || "curriculum";
          const modeStr =
            pId === "tabula_rasa" ? "Tabula Rasa" : (pId === "rapid_exodus" ? "Exodus" : "Curriculum");
          paradigmBadgeEl.textContent = `${cradles} Wiegen · ${modeStr}`;
        }
      }
      if (
        typeof state.world.daily_food_spawn === "number" &&
        document.activeElement !== sliderFoodSpawn
      ) {
        sliderFoodSpawn.value = String(state.world.daily_food_spawn);
        document.getElementById("valFoodSpawn").textContent = String(state.world.daily_food_spawn);
      }
      if (
        typeof state.world.survival_need === "number" &&
        document.activeElement !== sliderSurvivalNeed
      ) {
        const sn = state.world.survival_need;
        sliderSurvivalNeed.value = String(sn);
        document.getElementById("valSurvivalNeed").textContent = `${sn.toFixed(2)} (${(sn * 2).toFixed(2)} = Kind)`;
      }
      if (
        typeof state.world.food_price === "number" &&
        document.activeElement !== sliderFoodPrice
      ) {
        sliderFoodPrice.value = String(state.world.food_price);
        document.getElementById("valFoodPrice").textContent = `${state.world.food_price.toFixed(1)} 🪙`;
      }
      if (
        sliderMaxResidents &&
        typeof state.world.max_residents_per_home === "number" &&
        document.activeElement !== sliderMaxResidents
      ) {
        sliderMaxResidents.value = String(state.world.max_residents_per_home);
        document.getElementById("valMaxResidents").textContent = `${state.world.max_residents_per_home} 🏠`;
      }
    }

    if (state.prefs) {
      document.querySelectorAll(".pref-slider").forEach((slider) => {
        if (document.activeElement === slider) return;
        const key = slider.getAttribute("data-key");
        if (typeof state.prefs[key] === "number") {
          const val = state.prefs[key];
          slider.value = String(val);
          const labelId = prefDisplayMap[key];
          if (labelId) {
            document.getElementById(labelId).textContent =
              key === "learning_rate" ? val.toFixed(3) : val.toFixed(2);
          }
        }
      });
    }
  }

  function renderGroupListSafe(groups) {
    const grpContainer = document.getElementById("groupListContainer");
    const countBadge = document.getElementById("groupsCountBadge");
    if (countBadge) {
      countBadge.textContent = `${groups ? groups.length : 0} aktiv`;
    }
    if (!grpContainer) return;
    const sig = JSON.stringify(groups || []);
    if (sig === lastGroupsSignature) return;
    lastGroupsSignature = sig;

    grpContainer.replaceChildren();
    if (groups && groups.length > 0) {
      const sorted = groups.slice().sort((a, b) => (b.member_count || 0) - (a.member_count || 0));
      for (const g of sorted) {
        const row = document.createElement("div");
        row.className = "group-row";
        row.tabIndex = 0;
        row.title = "Klicken, um zur Hauptstadt dieser Fraktion zu springen";

        const left = document.createElement("div");
        const dot = document.createElement("span");
        dot.className = "group-dot";
        dot.style.backgroundColor = g.color || "#10b981";
        const nameEl = document.createElement("strong");
        const secTag = g.parent_group_id ? " 🚩" : "";
        nameEl.textContent = `${g.name}${secTag}`;
        left.append(dot, nameEl);

        const right = document.createElement("span");
        const homesCount = g.homes_count || (g.home_ids ? g.home_ids.length : 1);
        const conqStr = g.conquests ? ` · ${g.conquests}⚔️` : "";
        right.textContent = `${g.member_count} Blobs · ${homesCount}🏠 · ${g.total_trades}🤝${conqStr}`;

        row.append(left, right);
        row.addEventListener("click", () => {
          if (!simState || !simState.homes) return;
          const cap = simState.homes.find((h) => h.id === g.home_id) ||
            simState.homes.find((h) => h.group_id === g.id);
          if (cap) {
            followSelected = false;
            camX = cap.x;
            camY = cap.y;
            camZoom = Math.max(camZoom, 0.28);
            clampCamera();
          }
        });
        grpContainer.appendChild(row);
      }
    } else {
      const empty = document.createElement("div");
      empty.className = "event-item";
      empty.textContent = "Noch keine Siedlungen — Häuser verbinden sich bei Begegnung!";
      grpContainer.appendChild(empty);
    }
  }

  function renderEventFeedSafe(events) {
    if (!events || events.length === 0) return;
    const evContainer = document.getElementById("eventFeedContainer");
    if (!evContainer) return;
    const newestTick = events[events.length - 1].tick;
    if (newestTick === lastEventTick && events.length > 1) return;
    lastEventTick = newestTick;
    evContainer.replaceChildren();
    const reversed = events.slice().reverse();
    for (const ev of reversed) {
      const item = document.createElement("div");
      item.className = "event-item";
      const daySpan = document.createElement("span");
      daySpan.className = "ev-day";
      daySpan.textContent = `[T${ev.day}]`;
      item.append(daySpan, document.createTextNode(ev.msg));
      evContainer.appendChild(item);
    }
  }

  function updateDashboardUI(state, now = performance.now()) {
    document.getElementById("dayBadge").textContent = `Tag ${state.day}`;
    const seasonEl = document.getElementById("seasonBadge");
    if (seasonEl && state.season) {
      seasonEl.textContent = `${state.season.label} (${state.season.day_in_season}/4)`;
      const sColors = {
        spring: ["#34d399", "rgba(16, 185, 129, 0.12)", "rgba(16, 185, 129, 0.25)"],
        summer: ["#fbbf24", "rgba(245, 158, 11, 0.14)", "rgba(245, 158, 11, 0.28)"],
        autumn: ["#fb923c", "rgba(249, 115, 22, 0.14)", "rgba(249, 115, 22, 0.28)"],
        winter: ["#38bdf8", "rgba(56, 189, 248, 0.14)", "rgba(56, 189, 248, 0.28)"],
      };
      const sc = sColors[state.season.id] || sColors.spring;
      seasonEl.style.color = sc[0];
      seasonEl.style.background = sc[1];
      seasonEl.style.borderColor = sc[2];
    }

    const pct = Math.min(100, Math.round(state.time_of_day * 100));
    const fillEl = document.getElementById("timeProgressFill");
    fillEl.style.width = `${pct}%`;

    const phaseEl = document.getElementById("phaseLabel");
    if (state.is_night) {
      phaseEl.textContent = "🌙 Nacht (Heimkehr & Fortpflanzung)";
      fillEl.style.background = "#38bdf8";
    } else if (state.time_of_day > 0.56) {
      phaseEl.textContent = "🌇 Dämmerung (Rückweg zur Siedlung)";
      fillEl.style.background = "#f97316";
    } else {
      phaseEl.textContent = "☀️ Tag (Nahrungssuche & Handel)";
      fillEl.style.background = "#f59e0b";
    }

    document.getElementById("tickLabel").textContent = `${state.tick_in_day} / ${state.ticks_per_day}`;

    // KPIs
    const s = state.stats;
    document.getElementById("kpiPopulation").textContent = String(s.population);
    document.getElementById("kpiFood").textContent = String(s.foods_count);
    const oasesEl = document.getElementById("kpiOases");
    if (oasesEl) {
      const oases = state.oases || [];
      const lushCount = oases.filter((o) => !o.overgrazed).length;
      const avgFert = Math.round((s.avg_oasis_fertility || 0.85) * 100);
      oasesEl.textContent = `${lushCount}/${oases.length} 🌱 (${avgFert}%)`;
    }
    document.getElementById("kpiGroups").textContent = `${s.groups_count} (${s.homes_count}🏠)`;
    const migEl = document.getElementById("kpiMigrations");
    if (migEl) {
      migEl.textContent = `${s.total_migrations || 0} 🏕️`;
    }
    document.getElementById("kpiTrades").textContent = `${s.trades_today} (${s.total_trades})`;
    const raidsEl = document.getElementById("kpiRaids");
    if (raidsEl) {
      raidsEl.textContent = `${s.total_raids || 0}⚔️ / ${s.total_repelled || 0}🛡️`;
    }
    const repEl = document.getElementById("kpiReputation");
    if (repEl) {
      const avgRep = typeof s.avg_reputation === "number" ? s.avg_reputation : 0.0;
      const repSign = avgRep >= 0 ? "+" : "";
      repEl.textContent = `${repSign}${avgRep.toFixed(2)} (${s.outlaws_count || 0}☠️)`;
    }
    const upgEl = document.getElementById("kpiUpgrades");
    if (upgEl) {
      upgEl.textContent = `${s.upgrades_built || 0}🏗️`;
    }
    document.getElementById("kpiBirths").textContent = `${s.births_today} (${s.total_births})`;

    // Throttle heavy sidebar DOM & secondary canvas redraws during high-speed runs
    const heavyInterval = currentSpeed >= 45 ? 90 : (currentSpeed >= 15 ? 55 : 0);
    if (isRunning && now - lastHeavyUiUpdateTime < heavyInterval) {
      return;
    }
    lastHeavyUiUpdateTime = now;

    // Network badge (26 inputs -> hidden_dims -> 6 outputs)
    if (state.net_cfg && state.net_cfg.hidden_dims) {
      const dims = [26, ...state.net_cfg.hidden_dims, 6];
      document.getElementById("netTopologyBadge").textContent = dims.join(" → ");
    }

    syncSidebarControls(state);

    // Selected Blob Inspector (if inspector panel exists in DOM)
    const inspTitleEl = document.getElementById("inspectorTitle");
    if (inspTitleEl && state.selected && state.selected.blob) {
      const b = state.selected.blob;
      const reproNeed = state.world ? state.world.repro_need : 2.0;
      inspTitleEl.textContent = `🔬 Blob #${b.id} (Gen ${b.gen})`;
      document.getElementById("inspectorGroupBadge").textContent = state.selected.group_name;
      document.getElementById("inspFoodToday").textContent = `${b.food_today.toFixed(2)} / ${reproNeed.toFixed(2)}`;
      document.getElementById("inspMoney").textContent = `${b.money.toFixed(1)} 🪙 (+${b.surplus_food.toFixed(1)}🍏)`;
      document.getElementById("inspEnergyAge").textContent = `${Math.round(b.energy)}/${Math.round(b.max_energy || 100)} · ${b.age_days}d`;
      const migTag = b.migrations ? ` · ${b.migrations}🏕️` : "";
      document.getElementById("inspGenChildren").textContent = `G${b.gen} · ${b.children}👶 · ${b.trades}🤝${migTag}`;
      document.getElementById("inspGenesText").textContent =
        `${b.genes.speed.toFixed(2)}x / ${b.genes.sense.toFixed(2)}x / ${b.genes.size.toFixed(2)}x`;

      if (state.selected.brain && nnCtx) {
        document.getElementById("inspParamCount").textContent =
          `${state.selected.brain.param_count} Synapsen (${state.selected.brain.activation.toUpperCase()})`;
        renderNeuralNetwork(state.selected.brain);
      }
    }

    renderGroupListSafe(state.groups || []);
    renderEventFeedSafe(state.events || []);
    renderMinimap(state, now);

    if (chartCtx && selectChartModeEl) {
      const history = state.history || [];
      const chartMode = selectChartModeEl.value;
      const lastHist = history.length > 0 ? history[history.length - 1] : null;
      const chartSig = `${chartMode}:${history.length}:${lastHist ? lastHist.day : 0}:${lastHist ? lastHist.population : 0}:${lastHist ? lastHist.trades : 0}`;
      if (chartSig !== lastChartSignature) {
        lastChartSignature = chartSig;
        renderHistoryChart(history);
      }
    }
  }

  // --- Offscreen Static Geography Minimap Cache ---
  function updateStaticMinimap(state) {
    if (!smCtx || !state) return;
    const mw = staticMinimapCanvas.width;
    const mh = staticMinimapCanvas.height;
    const { w: worldW, h: worldH } = getWorldSize();
    const sx = mw / worldW;
    const sy = mh / worldH;

    smCtx.fillStyle = "#080d13";
    smCtx.fillRect(0, 0, mw, mh);

    // 0. Concentric Training Zones
    if (showZoneOverlay && state.world && state.world.zones) {
      for (const z of state.world.zones) {
        const zcx = typeof z.cx === "number" ? z.cx : worldW * 0.5;
        const zcy = typeof z.cy === "number" ? z.cy : worldH * 0.5;
        smCtx.save();
        smCtx.beginPath();
        smCtx.ellipse(zcx * sx, zcy * sy, z.rx * sx, z.ry * sy, 0, 0, Math.PI * 2);
        smCtx.setLineDash([3, 3]);
        smCtx.lineWidth = 1;
        smCtx.strokeStyle =
          z.id === "cradle" || String(z.id).startsWith("cradle")
            ? "rgba(52, 211, 153, 0.35)"
            : z.id === "canyon_gates" || String(z.id).startsWith("canyon")
            ? "rgba(148, 163, 184, 0.35)"
            : "rgba(56, 189, 248, 0.30)";
        smCtx.stroke();
        smCtx.restore();
      }
    }

    // 1. Oases
    if (state.oases) {
      for (const o of state.oases) {
        const ox = o.x * sx;
        const oy = o.y * sy;
        const rVal = o.radius || o.r || 220;
        const orad = Math.max(2.2, rVal * sx);
        const biome = o.biome || "gold_grove";
        smCtx.beginPath();
        smCtx.arc(ox, oy, orad, 0, Math.PI * 2);
        if (o.overgrazed) {
          smCtx.fillStyle = "rgba(245, 158, 11, 0.25)";
          smCtx.strokeStyle = "rgba(245, 158, 11, 0.72)";
        } else if (biome === "thermal_spring") {
          smCtx.fillStyle = `rgba(34, 211, 238, ${0.18 + o.fertility * 0.22})`;
          smCtx.strokeStyle = "rgba(34, 211, 238, 0.75)";
        } else if (biome === "gold_grove") {
          smCtx.fillStyle = `rgba(250, 204, 21, ${0.16 + o.fertility * 0.22})`;
          smCtx.strokeStyle = "rgba(251, 191, 36, 0.72)";
        } else {
          smCtx.fillStyle = `rgba(16, 185, 129, ${0.16 + o.fertility * 0.22})`;
          smCtx.strokeStyle = "rgba(52, 211, 153, 0.70)";
        }
        smCtx.fill();
        smCtx.lineWidth = 1;
        smCtx.stroke();
      }
    }

    // 2. Obstacles
    if (state.obstacles) {
      for (const obs of state.obstacles) {
        const obsR = obs.radius || obs.r || 42;
        smCtx.beginPath();
        smCtx.moveTo(obs.x1 * sx, obs.y1 * sy);
        smCtx.lineTo(obs.x2 * sx, obs.y2 * sy);
        smCtx.lineCap = "round";
        smCtx.lineWidth = Math.max(1.2, obsR * 2 * sx);
        smCtx.strokeStyle =
          obs.kind === "rock" ? "rgba(100, 116, 139, 0.85)" : "rgba(132, 204, 22, 0.28)";
        smCtx.stroke();
      }
    }

    // 3. Homes
    if (state.homes) {
      smCtx.fillStyle = "#f8fafc";
      for (const h of state.homes) {
        smCtx.fillRect(h.x * sx - 1.5, h.y * sy - 1.5, 3, 3);
      }
    }
    staticMinimapDirty = false;
  }

  // --- Render Minimap Overview (#minimapCanvas) with 5 Hz Throttling ---
  function renderMinimap(state, now = performance.now()) {
    if (!mCtx || !minimapCanvas || !state) return;
    if (!isMinimapDragging && now - lastMinimapRenderTime < 200) {
      return;
    }
    lastMinimapRenderTime = now;

    if (staticMinimapDirty) {
      updateStaticMinimap(state);
    }

    const mw = minimapCanvas.width;
    const mh = minimapCanvas.height;
    const { w: worldW, h: worldH } = getWorldSize();
    const sx = mw / worldW;
    const sy = mh / worldH;

    // Single blit of static geography from offscreen canvas
    mCtx.drawImage(staticMinimapCanvas, 0, 0);

    // Dynamic Blobs (Fast Typed-Array Minimap Pass)
    const selId = state.selected && state.selected.blob ? state.selected.blob.id : null;
    if (blobsF32 && nBlobsCount > 0) {
      const step = nBlobsCount > 4000 ? 2 : 1;
      for (let i = 0; i < nBlobsCount; i += step) {
        const off = i * 8;
        const bx = blobsF32[off] * sx;
        const by = blobsF32[off + 1] * sy;
        const bid = blobsF32[off + 5] | 0;
        if (bid === selId) {
          mCtx.fillStyle = "#38bdf8";
          mCtx.fillRect(bx - 2.5, by - 2.5, 5, 5);
        } else {
          mCtx.fillStyle = rgbIntToHex(blobsF32[off + 6]);
          mCtx.fillRect(bx - 1, by - 1, 2, 2);
        }
      }
    } else if (state.blobs) {
      for (const b of state.blobs) {
        const isSel = b.id === selId;
        mCtx.fillStyle = isSel ? "#38bdf8" : b.color;
        const br = isSel ? 2.8 : 1.4;
        mCtx.beginPath();
        mCtx.arc(b.x * sx, b.y * sy, br, 0, Math.PI * 2);
        mCtx.fill();
      }
    }

    // Camera Viewport Frame
    const viewW = (worldCanvas.width / camZoom) * sx;
    const viewH = (worldCanvas.height / camZoom) * sy;
    const vx = camX * sx - viewW * 0.5;
    const vy = camY * sy - viewH * 0.5;
    mCtx.strokeStyle = "#38bdf8";
    mCtx.lineWidth = 1.4;
    mCtx.strokeRect(vx, vy, viewW, viewH);
  }

  // --- Render 2D World Arena (3-Stage Semantic Zoom + Binary Typed-Array Pipeline) ---
  function renderWorld(now = performance.now()) {
    requestAnimationFrame(renderWorld);
    if (!simState) return;

    if (stateDirty) {
      stateDirty = false;
      updateDashboardUI(simState, now);
    }

    const W = worldCanvas.width;
    const H = worldCanvas.height;
    const { w: worldW, h: worldH } = getWorldSize();
    const tod = simState.time_of_day || 0;
    const seasonId = simState.season ? simState.season.id : "spring";
    const selectedBlob = simState.selected && simState.selected.blob ? simState.selected.blob : null;
    const selectedId = selectedBlob ? selectedBlob.id : null;

    const isMacro = camZoom < 0.12;
    const isMeso = camZoom >= 0.12 && camZoom < 0.45;
    const isMicro = camZoom >= 0.45;

    // Follow selected blob if active
    if (followSelected && selectedBlob) {
      camX = camX * 0.82 + selectedBlob.x * 0.18;
      camY = camY * 0.82 + selectedBlob.y * 0.18;
      clampCamera();
    }

    // Camera Frustum Bounds in World Coordinates
    const halfViewW = (W * 0.5) / camZoom;
    const halfViewH = (H * 0.5) / camZoom;
    const viewMinX = camX - halfViewW - 450;
    const viewMaxX = camX + halfViewW + 450;
    const viewMinY = camY - halfViewH - 450;
    const viewMaxY = camY + halfViewH + 450;

    // Day/Dusk/Night + Seasonal ambient background color
    let bgR = 13, bgG = 22, bgB = 29;
    if (seasonId === "summer") {
      bgR = 16; bgG = 24; bgB = 26;
    } else if (seasonId === "autumn") {
      bgR = 19; bgG = 20; bgB = 24;
    } else if (seasonId === "winter") {
      bgR = 11; bgG = 19; bgB = 31;
    }
    if (simState.is_night) {
      bgR = Math.max(6, bgR - 5);
      bgG = Math.max(10, bgG - 8);
      bgB = Math.max(18, bgB - 6);
    } else if (tod > 0.56) {
      bgR += 4;
      bgG -= 1;
    }
    wCtx.fillStyle = `rgb(${bgR}, ${bgG}, ${bgB})`;
    wCtx.fillRect(0, 0, W, H);

    // Apply Camera Transform
    wCtx.save();
    wCtx.translate(W * 0.5, H * 0.5);
    wCtx.scale(camZoom, camZoom);
    wCtx.translate(-camX, -camY);

    // Adaptive spatial sector grid (every 2000px in Macro, 400px in Meso/Micro)
    const gridStep = isMacro ? 2000 : 400;
    const gStartX = Math.max(gridStep, Math.floor(viewMinX / gridStep) * gridStep);
    const gEndX = Math.min(worldW, Math.ceil(viewMaxX / gridStep) * gridStep);
    const gStartY = Math.max(gridStep, Math.floor(viewMinY / gridStep) * gridStep);
    const gEndY = Math.min(worldH, Math.ceil(viewMaxY / gridStep) * gridStep);
    wCtx.strokeStyle = "rgba(148, 163, 184, 0.045)";
    wCtx.lineWidth = Math.max(1.5, 1.0 / camZoom);
    wCtx.beginPath();
    for (let gx = gStartX; gx < gEndX; gx += gridStep) {
      wCtx.moveTo(gx, Math.max(0, viewMinY));
      wCtx.lineTo(gx, Math.min(worldH, viewMaxY));
    }
    for (let gy = gStartY; gy < gEndY; gy += gridStep) {
      wCtx.moveTo(Math.max(0, viewMinX), gy);
      wCtx.lineTo(Math.min(worldW, viewMaxX), gy);
    }
    wCtx.stroke();

    // Subtle seasonal world boundary
    const borderColors = {
      spring: "rgba(52, 211, 153, 0.28)",
      summer: "rgba(251, 191, 36, 0.30)",
      autumn: "rgba(251, 146, 60, 0.30)",
      winter: "rgba(56, 189, 248, 0.36)",
    };
    wCtx.strokeStyle = borderColors[seasonId] || borderColors.spring;
    wCtx.lineWidth = Math.max(4, 2.5 / camZoom);
    wCtx.strokeRect(8, 8, worldW - 16, worldH - 16);

    // -3. Draw Multi-Cradle Evolutionary Training Rings & Skill-Gate Markers
    if (showZoneOverlay && simState.world) {
      if (simState.world.zones) {
        for (const z of simState.world.zones) {
          const zcx = typeof z.cx === "number" ? z.cx : worldW * 0.5;
          const zcy = typeof z.cy === "number" ? z.cy : worldH * 0.5;
          if (zcx + z.rx < viewMinX || zcx - z.rx > viewMaxX || zcy + z.ry < viewMinY || zcy - z.ry > viewMaxY) {
            continue;
          }
          const isCradleRing = z.id === "cradle" || String(z.id).startsWith("cradle");
          const isCanyonRing = z.id === "canyon_gates" || String(z.id).startsWith("canyon");
          wCtx.save();
          wCtx.beginPath();
          wCtx.ellipse(zcx, zcy, z.rx, z.ry, 0, 0, Math.PI * 2);
          if (isCradleRing) {
            wCtx.fillStyle = "rgba(16, 185, 129, 0.028)";
            wCtx.fill();
            wCtx.strokeStyle = "rgba(52, 211, 153, 0.28)";
            wCtx.setLineDash([16, 10]);
          } else if (isCanyonRing) {
            wCtx.strokeStyle = "rgba(148, 163, 184, 0.24)";
            wCtx.setLineDash([10, 12]);
          } else {
            wCtx.strokeStyle = "rgba(56, 189, 248, 0.20)";
            wCtx.setLineDash([16, 12]);
          }
          wCtx.lineWidth = Math.max(2.2, 1.5 / camZoom);
          wCtx.stroke();
          wCtx.setLineDash([]);

          if (!isMacro || isCradleRing) {
            const fontPx = Math.max(13, Math.round(11 / camZoom));
            wCtx.font = `600 ${fontPx}px JetBrains Mono, monospace`;
            wCtx.textAlign = "center";
            wCtx.fillStyle = isCradleRing
              ? "rgba(110, 231, 183, 0.72)"
              : isCanyonRing
              ? "rgba(203, 213, 225, 0.62)"
              : "rgba(125, 211, 252, 0.65)";
            const zTitle = String(z.title || z.label || z.id || "").toUpperCase();
            const zSub = !isMacro && z.subtitle ? ` — ${z.subtitle}` : "";
            wCtx.fillText(`▲ ${zTitle}${zSub} ▲`, zcx, zcy - z.ry + fontPx * 1.4);
          }
          wCtx.restore();
        }
      }

      if (!isMacro && simState.world.skill_gates) {
        for (const g of simState.world.skill_gates) {
          if (g.x < viewMinX || g.x > viewMaxX || g.y < viewMinY || g.y > viewMaxY) continue;
          wCtx.save();
          wCtx.beginPath();
          wCtx.arc(g.x, g.y, 68, 0, Math.PI * 2);
          wCtx.strokeStyle =
            g.kind === "chicane"
              ? "rgba(251, 191, 36, 0.32)"
              : g.kind === "swamp_ford"
              ? "rgba(163, 230, 53, 0.35)"
              : "rgba(56, 189, 248, 0.35)";
          wCtx.lineWidth = 1.8;
          wCtx.setLineDash([5, 5]);
          wCtx.stroke();
          wCtx.setLineDash([]);

          if (isMicro) {
            wCtx.font = "600 13px Inter, sans-serif";
            wCtx.textAlign = "center";
            wCtx.fillStyle =
              g.kind === "chicane"
                ? "rgba(253, 224, 71, 0.85)"
                : g.kind === "swamp_ford"
                ? "rgba(190, 242, 100, 0.88)"
                : "rgba(125, 211, 252, 0.88)";
            wCtx.fillText(g.title || g.label || "Gate", g.x, g.y - 76);
          }
          wCtx.restore();
        }
      }
    }

    // -2. Draw Fertile Oases (with distinct Biome Styles: Cradle / Gold Grove / Thermal Spring)
    if (simState.oases) {
      for (const o of simState.oases) {
        const oRad = o.radius || o.r || 220;
        if (o.x + oRad < viewMinX || o.x - oRad > viewMaxX || o.y + oRad < viewMinY || o.y - oRad > viewMaxY) {
          continue;
        }
        const fert = typeof o.fertility === "number" ? o.fertility : 1.0;
        const isOvergrazed = Boolean(o.overgrazed || fert < 0.38);
        const biome = o.biome || "gold_grove";

        wCtx.save();
        wCtx.beginPath();
        wCtx.arc(o.x, o.y, oRad, 0, Math.PI * 2);
        if (isOvergrazed) {
          wCtx.fillStyle = "rgba(245, 158, 11, 0.10)";
        } else if (biome === "thermal_spring") {
          wCtx.fillStyle = `rgba(34, 211, 238, ${(0.08 + fert * 0.14).toFixed(3)})`;
        } else if (biome === "gold_grove") {
          wCtx.fillStyle = `rgba(234, 179, 8, ${(0.07 + fert * 0.14).toFixed(3)})`;
        } else {
          wCtx.fillStyle = `rgba(16, 185, 129, ${(0.08 + fert * 0.14).toFixed(3)})`;
        }
        wCtx.fill();

        wCtx.lineWidth = Math.max(2.6, 1.6 / camZoom);
        wCtx.setLineDash(isOvergrazed ? [10, 8] : []);
        if (isOvergrazed) {
          wCtx.strokeStyle = "rgba(245, 158, 11, 0.62)";
        } else if (biome === "thermal_spring") {
          wCtx.strokeStyle = `rgba(34, 211, 238, ${(0.32 + fert * 0.38).toFixed(2)})`;
        } else if (biome === "gold_grove") {
          wCtx.strokeStyle = `rgba(250, 204, 21, ${(0.30 + fert * 0.36).toFixed(2)})`;
        } else {
          wCtx.strokeStyle = `rgba(52, 211, 153, ${(0.28 + fert * 0.35).toFixed(2)})`;
        }
        wCtx.stroke();
        wCtx.setLineDash([]);

        // Inner water / geothermal / golden spring pool
        const poolRad = Math.max(18, oRad * 0.17 * (0.5 + 0.5 * fert));
        wCtx.beginPath();
        wCtx.arc(o.x, o.y, poolRad, 0, Math.PI * 2);
        if (isOvergrazed) {
          wCtx.fillStyle = "rgba(180, 83, 9, 0.38)";
        } else if (biome === "thermal_spring") {
          wCtx.fillStyle = "rgba(6, 182, 212, 0.42)";
        } else if (biome === "gold_grove") {
          wCtx.fillStyle = "rgba(245, 158, 11, 0.34)";
        } else {
          wCtx.fillStyle = "rgba(16, 185, 129, 0.34)";
        }
        wCtx.fill();

        // Oasis name & fertility label in Meso and Micro views
        if (!isMacro) {
          const biomeIcon =
            biome === "thermal_spring" ? "♨️" : biome === "gold_grove" ? "☀️" : "🌿";
          wCtx.fillStyle = isOvergrazed
            ? "rgba(251, 191, 36, 0.92)"
            : biome === "thermal_spring"
            ? "rgba(165, 243, 252, 0.92)"
            : biome === "gold_grove"
            ? "rgba(254, 240, 138, 0.92)"
            : "rgba(167, 243, 208, 0.92)";
          wCtx.font = "600 15px Inter, sans-serif";
          wCtx.textAlign = "center";
          wCtx.fillText(
            `${biomeIcon} ${o.name} (${Math.round(fert * 100)}%)`,
            o.x,
            o.y - oRad - 10
          );
        }
        wCtx.restore();
      }
    }

    // -1. Draw Terrain Obstacles (Frustum-Culled Swamps & Impassable Rock Canyon Walls)
    if (simState.obstacles) {
      for (const obs of simState.obstacles) {
        if (obs.kind !== "swamp") continue;
        if (
          Math.max(obs.x1, obs.x2) < viewMinX ||
          Math.min(obs.x1, obs.x2) > viewMaxX ||
          Math.max(obs.y1, obs.y2) < viewMinY ||
          Math.min(obs.y1, obs.y2) > viewMaxY
        ) {
          continue;
        }
        const obsR = obs.radius || obs.r || 64;
        wCtx.save();
        wCtx.beginPath();
        wCtx.moveTo(obs.x1, obs.y1);
        wCtx.lineTo(obs.x2, obs.y2);
        wCtx.lineCap = "round";
        wCtx.lineWidth = obsR * 2;
        wCtx.strokeStyle = "rgba(101, 163, 13, 0.18)";
        wCtx.stroke();
        wCtx.restore();
      }

      for (const obs of simState.obstacles) {
        if (obs.kind !== "rock") continue;
        if (
          Math.max(obs.x1, obs.x2) < viewMinX ||
          Math.min(obs.x1, obs.x2) > viewMaxX ||
          Math.max(obs.y1, obs.y2) < viewMinY ||
          Math.min(obs.y1, obs.y2) > viewMaxY
        ) {
          continue;
        }
        const obsR = obs.radius || obs.r || 42;
        wCtx.save();
        wCtx.lineCap = "round";
        wCtx.beginPath();
        wCtx.moveTo(obs.x1, obs.y1);
        wCtx.lineTo(obs.x2, obs.y2);
        wCtx.lineWidth = obsR * 2;
        wCtx.strokeStyle = "#334155";
        wCtx.stroke();

        if (!isMacro) {
          wCtx.beginPath();
          wCtx.moveTo(obs.x1, obs.y1);
          wCtx.lineTo(obs.x2, obs.y2);
          wCtx.lineWidth = Math.max(6, obsR * 0.85);
          wCtx.strokeStyle = "#475569";
          wCtx.stroke();
        }
        wCtx.restore();
      }
    }

    const groupMap = {};
    if (simState.groups) {
      for (const g of simState.groups) groupMap[g.id] = g;
    }
    const homeMap = {};
    if (simState.homes) {
      for (const h of simState.homes) homeMap[h.id] = h;
    }

    // 0. Draw Empire Territory Halos (Macro/Meso) & Multi-Home Settlement Roads
    if (simState.groups) {
      const sortedGroupsForBanners = simState.groups
        .slice()
        .sort((a, b) => (b.member_count || 0) - (a.member_count || 0));
      const drawnBannerCenters = [];
      const minBannerDist = isMacro ? (155 / camZoom) : (75 / camZoom);

      for (const g of sortedGroupsForBanners) {
        const capHome = homeMap[g.home_id] || (g.home_ids && homeMap[g.home_ids[0]]);
        if (!capHome) continue;

        // Macro Territory Halo around Empire Houses
        if (isMacro && g.home_ids) {
          wCtx.save();
          wCtx.fillStyle = `${g.color}1e`;
          wCtx.strokeStyle = `${g.color}66`;
          wCtx.lineWidth = Math.max(3, 1.8 / camZoom);
          for (const hid of g.home_ids) {
            const sh = homeMap[hid];
            if (!sh) continue;
            if (sh.x < viewMinX || sh.x > viewMaxX || sh.y < viewMinY || sh.y > viewMaxY) continue;
            wCtx.beginPath();
            wCtx.arc(sh.x, sh.y, Math.max(240, sh.radius * 4.5), 0, Math.PI * 2);
            wCtx.fill();
            wCtx.stroke();
          }
          wCtx.restore();
        }

        if (g.home_ids && g.home_ids.length >= 2) {
          wCtx.save();
          wCtx.strokeStyle = `${g.color}77`;
          wCtx.lineWidth = Math.max(2.2, 1.5 / camZoom);
          wCtx.setLineDash([8, 6]);
          for (const hid of g.home_ids) {
            const satHome = homeMap[hid];
            if (!satHome || satHome.id === capHome.id) continue;
            wCtx.beginPath();
            wCtx.moveTo(capHome.x, capHome.y);
            wCtx.lineTo(satHome.x, satHome.y);
            wCtx.stroke();
          }
          wCtx.restore();
        }

        // Macro / Meso Empire Capital Banner (collision-filtered by dominance so labels never overlap)
        if ((isMacro || isMeso) && capHome.x >= viewMinX && capHome.x <= viewMaxX && capHome.y >= viewMinY && capHome.y <= viewMaxY) {
          const tooClose = drawnBannerCenters.some(
            (pt) => Math.hypot(pt.x - capHome.x, pt.y - capHome.y) < minBannerDist
          );
          if (!tooClose) {
            drawnBannerCenters.push({ x: capHome.x, y: capHome.y });
            wCtx.save();
            wCtx.translate(capHome.x, capHome.y - Math.max(90, 18 / camZoom));
            const invZ = 1 / camZoom;
            wCtx.scale(invZ, invZ);
            wCtx.font = "600 11px Inter, sans-serif";
            wCtx.textAlign = "center";
            const txt = `${g.name} (${g.member_count} · ${g.homes_count || 1}🏠)`;
            const tw = wCtx.measureText(txt).width + 12;
            wCtx.fillStyle = "rgba(15, 23, 42, 0.88)";
            wCtx.strokeStyle = g.color || "#10b981";
            wCtx.lineWidth = 1.4;
            wCtx.beginPath();
            wCtx.roundRect(-tw * 0.5, -10, tw, 18, 4);
            wCtx.fill();
            wCtx.stroke();
            wCtx.fillStyle = "#f8fafc";
            wCtx.fillText(txt, 0, 3);
            wCtx.restore();
          }
        }
      }
    }

    // 1. Draw Homes & Multi-House Settlements (Frustum-Culled + LoD)
    if (simState.homes) {
      for (const h of simState.homes) {
        if (h.x + h.radius < viewMinX || h.x - h.radius > viewMaxX || h.y + h.radius < viewMinY || h.y - h.radius > viewMaxY) {
          continue;
        }
        const grp = h.group_id ? groupMap[h.group_id] : null;
        const baseColor = grp ? grp.color : "#64748b";
        const maxRes = h.max_residents || 6;
        const drawHRad = Math.max(h.radius, 3.5 / camZoom);

        wCtx.save();
        wCtx.beginPath();
        wCtx.arc(h.x, h.y, drawHRad, 0, Math.PI * 2);
        wCtx.fillStyle = grp ? `${baseColor}2e` : "rgba(100, 116, 139, 0.16)";
        wCtx.fill();
        wCtx.lineWidth = Math.max(grp ? 2.4 : 1.5, 1.2 / camZoom);
        wCtx.setLineDash(grp ? [] : [4, 4]);
        wCtx.strokeStyle = grp ? baseColor : "rgba(148, 163, 184, 0.55)";
        wCtx.stroke();
        wCtx.setLineDash([]);

        if (!isMacro && (h.defense_level > 0 || (h.terrain_defense_bonus && h.terrain_defense_bonus >= 0.2))) {
          wCtx.beginPath();
          wCtx.arc(h.x, h.y, h.radius + 5.0, 0, Math.PI * 2);
          wCtx.strokeStyle = "rgba(56, 189, 248, 0.48)";
          wCtx.lineWidth = 1.4 + (h.defense_level || 0) * 0.5;
          wCtx.setLineDash([7, 4]);
          wCtx.stroke();
          wCtx.setLineDash([]);
        }

        // Inner hearth icon
        wCtx.beginPath();
        wCtx.arc(h.x, h.y, Math.max(5.5, 2.0 / camZoom), 0, Math.PI * 2);
        wCtx.fillStyle = grp ? baseColor : "#94a3b8";
        wCtx.fill();

        // Detailed House Label only in Micro view
        if (isMicro) {
          wCtx.fillStyle = "rgba(226, 232, 240, 0.9)";
          wCtx.font = "11.5px Inter, sans-serif";
          wCtx.textAlign = "center";
          const prefix = grp ? `${grp.name} #${h.id}` : `Haus #${h.id}`;
          const upgBadges = [];
          if (h.farm_level > 0) upgBadges.push(`🌾${h.farm_level}`);
          if (h.granary_level > 0) upgBadges.push(`🏛️${h.granary_level}`);
          if (h.defense_level > 0) upgBadges.push(`🛡️${h.defense_level}`);
          if (typeof h.fertility_bonus === "number" && h.fertility_bonus >= 0.5) {
            upgBadges.push(`🌱${Math.round(h.fertility_bonus * 100)}%`);
          }
          const upgStr = upgBadges.length > 0 ? ` · ${upgBadges.join(" ")}` : "";
          const label = `${prefix} (${h.residents}/${maxRes}🏠 · ${h.food_depot.toFixed(1)}🍏${upgStr})`;
          wCtx.fillText(label, h.x, h.y - h.radius - (h.defense_level > 0 ? 9 : 6));
        }
        wCtx.restore();
      }
    }

    // 2. Draw Food Items (Fast Binary Float32Array Pass with Frustum Culling)
    const minFoodDrawRad = Math.max(4.6, 1.7 / camZoom);
    if (foodsF32 && nFoodsCount > 0) {
      wCtx.fillStyle = "#22c55e";
      wCtx.beginPath();
      for (let i = 0; i < nFoodsCount; i++) {
        const off = i * 3;
        if (foodsF32[off + 2] < 0.0 || foodsF32[off + 2] > 0.5) continue;
        const fx = foodsF32[off];
        const fy = foodsF32[off + 1];
        if (fx < viewMinX || fx > viewMaxX || fy < viewMinY || fy > viewMaxY) continue;
        wCtx.moveTo(fx + minFoodDrawRad, fy);
        wCtx.arc(fx, fy, minFoodDrawRad, 0, Math.PI * 2);
      }
      wCtx.fill();

      const goldRad = Math.max(6.6, 2.3 / camZoom);
      wCtx.fillStyle = "#fbbf24";
      wCtx.beginPath();
      for (let i = 0; i < nFoodsCount; i++) {
        const off = i * 3;
        if (foodsF32[off + 2] <= 0.5) continue;
        const fx = foodsF32[off];
        const fy = foodsF32[off + 1];
        if (fx < viewMinX || fx > viewMaxX || fy < viewMinY || fy > viewMaxY) continue;
        wCtx.moveTo(fx + goldRad, fy);
        wCtx.arc(fx, fy, goldRad, 0, Math.PI * 2);
      }
      wCtx.fill();
    } else if (simState.foods) {
      for (const f of simState.foods) {
        if (f.x < viewMinX || f.x > viewMaxX || f.y < viewMinY || f.y > viewMaxY) continue;
        wCtx.beginPath();
        const r = f.g ? Math.max(6.6, 2.3 / camZoom) : minFoodDrawRad;
        wCtx.arc(f.x, f.y, r, 0, Math.PI * 2);
        wCtx.fillStyle = f.g ? "#fbbf24" : "#22c55e";
        wCtx.fill();
      }
    }

    // 3. Draw Active Trade & Raid Lines (Meso & Micro views)
    if (!isMacro && simState.trade_lines) {
      for (const tl of simState.trade_lines) {
        if (tl.x1 < viewMinX || tl.x1 > viewMaxX || tl.y1 < viewMinY || tl.y1 > viewMaxY) continue;
        wCtx.save();
        wCtx.beginPath();
        wCtx.moveTo(tl.x1, tl.y1);
        wCtx.lineTo(tl.x2, tl.y2);
        wCtx.strokeStyle = tl.same_group ? "#34d399" : "#fbbf24";
        wCtx.lineWidth = 2.2;
        wCtx.setLineDash([5, 3]);
        wCtx.stroke();
        if (isMicro) {
          const mx = (tl.x1 + tl.x2) * 0.5;
          const my = (tl.y1 + tl.y2) * 0.5;
          wCtx.fillStyle = "#fde047";
          wCtx.font = "11px sans-serif";
          wCtx.textAlign = "center";
          wCtx.fillText("🪙⇄🍏", mx, my - 4);
        }
        wCtx.restore();
      }
    }

    if (!isMacro && simState.raid_lines) {
      for (const rl of simState.raid_lines) {
        if (rl.x1 < viewMinX || rl.x1 > viewMaxX || rl.y1 < viewMinY || rl.y1 > viewMaxY) continue;
        wCtx.save();
        wCtx.beginPath();
        wCtx.moveTo(rl.x1, rl.y1);
        wCtx.lineTo(rl.x2, rl.y2);
        wCtx.strokeStyle = rl.repelled ? "#38bdf8" : "#f43f5e";
        wCtx.lineWidth = 2.4;
        wCtx.setLineDash([4, 3]);
        wCtx.stroke();
        if (isMicro) {
          const mx = (rl.x1 + rl.x2) * 0.5;
          const my = (rl.y1 + rl.y2) * 0.5;
          wCtx.fillStyle = rl.repelled ? "#7dd3fc" : "#fda4af";
          wCtx.font = "11px sans-serif";
          wCtx.textAlign = "center";
          wCtx.fillText(rl.repelled ? "🛡️⚔️" : "⚔️🍏", mx, my - 4);
        }
        wCtx.restore();
      }
    }

    // 4. Draw Blobs (3-Stage Semantic LoD over Binary Float32Array Buffer)
    const survivalNeed = simState.world ? simState.world.survival_need : 1.0;
    const reproNeed = simState.world ? simState.world.repro_need : 2.0;
    const minMacroBlobRad = Math.max(8.0, 2.7 / camZoom);

    if (blobsF32 && nBlobsCount > 0) {
      const dtTick = now - lastTickTime;
      const alpha = (currentSpeed === 1) ? Math.min(1.0, Math.max(0.0, dtTick / smoothedIntervalMs)) : 1.0;
      let selectedBlobRender = null;

      if (isMacro) {
        // Macro View (< 0.12 Zoom): Color-Bucket Path Batching
        for (const bucket of colorBuckets.values()) bucket.length = 0;

        for (let i = 0; i < nBlobsCount; i++) {
          const off = i * 8;
          let bx = blobsF32[off];
          let by = blobsF32[off + 1];
          const bid = blobsF32[off + 5] | 0;

          if (alpha < 1.0 && prevBlobBuf && off + 5 < prevBlobBuf.length && (prevBlobBuf[off + 5] | 0) === bid) {
            bx = prevBlobBuf[off] + (bx - prevBlobBuf[off]) * alpha;
            by = prevBlobBuf[off + 1] + (by - prevBlobBuf[off + 1]) * alpha;
          }

          if (bx < viewMinX || bx > viewMaxX || by < viewMinY || by > viewMaxY) continue;

          if (bid === selectedId) {
            selectedBlobRender = { x: bx, y: by, r: minMacroBlobRad * 1.65, color: "#38bdf8", isMacro: true };
            continue;
          }

          const rgbKey = blobsF32[off + 6] | 0;
          let bucket = colorBuckets.get(rgbKey);
          if (!bucket) {
            bucket = [];
            colorBuckets.set(rgbKey, bucket);
          }
          bucket.push(bx, by, minMacroBlobRad);
        }

        // Single fill per distinct color bucket
        for (const [rgbKey, coords] of colorBuckets) {
          const len = coords.length;
          if (len === 0) continue;
          wCtx.fillStyle = rgbIntToHex(rgbKey);
          wCtx.beginPath();
          for (let j = 0; j < len; j += 3) {
            const x = coords[j], y = coords[j + 1], r = coords[j + 2];
            wCtx.moveTo(x + r, y);
            wCtx.arc(x, y, r, 0, Math.PI * 2);
          }
          wCtx.fill();
        }
      } else if (isMeso) {
        // Meso View (0.12 - 0.45 Zoom): Batched Bodies, Outlines, Food Rings & Outlaw Rings
        for (const bucket of colorBuckets.values()) bucket.length = 0;
        outlawCoords.length = 0;
        foodRingTiers.repro.length = 0;
        foodRingTiers.survival.length = 0;
        foodRingTiers.under.length = 0;

        for (let i = 0; i < nBlobsCount; i++) {
          const off = i * 8;
          let bx = blobsF32[off];
          let by = blobsF32[off + 1];
          const bid = blobsF32[off + 5] | 0;

          if (alpha < 1.0 && prevBlobBuf && off + 5 < prevBlobBuf.length && (prevBlobBuf[off + 5] | 0) === bid) {
            bx = prevBlobBuf[off] + (bx - prevBlobBuf[off]) * alpha;
            by = prevBlobBuf[off + 1] + (by - prevBlobBuf[off + 1]) * alpha;
          }

          if (bx < viewMinX || bx > viewMaxX || by < viewMinY || by > viewMaxY) continue;

          const bRad = blobsF32[off + 3];
          const foodToday = blobsF32[off + 4];
          const rgbKey = blobsF32[off + 6] | 0;
          const flags = blobsF32[off + 7] | 0;

          if (bid === selectedId) {
            selectedBlobRender = { x: bx, y: by, r: bRad, color: rgbIntToHex(rgbKey), foodToday, flags, isMacro: false };
            continue;
          }

          let bucket = colorBuckets.get(rgbKey);
          if (!bucket) {
            bucket = [];
            colorBuckets.set(rgbKey, bucket);
          }
          bucket.push(bx, by, bRad);

          if (foodToday > 0.02) {
            const ratio = Math.min(1.0, foodToday / reproNeed);
            const ringRad = bRad + 3.2;
            const ea = -Math.PI / 2 + ratio * Math.PI * 2;
            const target = foodToday >= reproNeed ? foodRingTiers.repro
                         : foodToday >= survivalNeed ? foodRingTiers.survival
                         : foodRingTiers.under;
            target.push(bx, by, ringRad, -Math.PI / 2, ea);
          }

          if (flags & 32) {
            outlawCoords.push(bx, by, bRad + 5.8);
          }
        }

        // 1. Batched Body Fills (1 fill per distinct color)
        for (const [rgbKey, coords] of colorBuckets) {
          const len = coords.length;
          if (len === 0) continue;
          wCtx.fillStyle = rgbIntToHex(rgbKey);
          wCtx.beginPath();
          for (let j = 0; j < len; j += 3) {
            const x = coords[j], y = coords[j + 1], r = coords[j + 2];
            wCtx.moveTo(x + r, y);
            wCtx.arc(x, y, r, 0, Math.PI * 2);
          }
          wCtx.fill();
        }

        // 2. Batched Body Outlines (1 stroke for all unselected blobs)
        wCtx.strokeStyle = "rgba(15, 23, 42, 0.85)";
        wCtx.lineWidth = 1.2;
        wCtx.beginPath();
        for (const coords of colorBuckets.values()) {
          const len = coords.length;
          for (let j = 0; j < len; j += 3) {
            const x = coords[j], y = coords[j + 1], r = coords[j + 2];
            wCtx.moveTo(x + r, y);
            wCtx.arc(x, y, r, 0, Math.PI * 2);
          }
        }
        wCtx.stroke();

        // 3. Batched Food Progress Rings (EXACTLY 3 strokes)
        const tierStyles = [
          { bucket: foodRingTiers.repro, color: "#fbbf24" },
          { bucket: foodRingTiers.survival, color: "#10b981" },
          { bucket: foodRingTiers.under, color: "#38bdf8" }
        ];
        wCtx.lineWidth = 2.3;
        for (let t = 0; t < tierStyles.length; t++) {
          const { bucket, color } = tierStyles[t];
          const len = bucket.length;
          if (len === 0) continue;
          wCtx.strokeStyle = color;
          wCtx.beginPath();
          for (let j = 0; j < len; j += 5) {
            const x = bucket[j], y = bucket[j + 1], r = bucket[j + 2];
            const sa = bucket[j + 3], ea = bucket[j + 4];
            wCtx.moveTo(x + Math.cos(sa) * r, y + Math.sin(sa) * r);
            wCtx.arc(x, y, r, sa, ea);
          }
          wCtx.stroke();
        }

        // 4. Batched Outlaw Rings (EXACTLY 1 stroke)
        if (outlawCoords.length > 0) {
          wCtx.strokeStyle = "rgba(244, 63, 94, 0.88)";
          wCtx.lineWidth = 1.9;
          wCtx.setLineDash([3, 2.5]);
          wCtx.beginPath();
          for (let j = 0; j < outlawCoords.length; j += 3) {
            const x = outlawCoords[j], y = outlawCoords[j + 1], r = outlawCoords[j + 2];
            wCtx.moveTo(x + r, y);
            wCtx.arc(x, y, r, 0, Math.PI * 2);
          }
          wCtx.stroke();
          wCtx.setLineDash([]);
        }
      } else {
        // Micro View (>= 0.45 Zoom): Detailed Individual Blobs with LERP + Eyes + Labels
        for (let i = 0; i < nBlobsCount; i++) {
          const off = i * 8;
          let bx = blobsF32[off];
          let by = blobsF32[off + 1];
          let heading = blobsF32[off + 2];
          const bid = blobsF32[off + 5] | 0;

          if (alpha < 1.0 && prevBlobBuf && off + 5 < prevBlobBuf.length && (prevBlobBuf[off + 5] | 0) === bid) {
            bx = prevBlobBuf[off] + (bx - prevBlobBuf[off]) * alpha;
            by = prevBlobBuf[off + 1] + (by - prevBlobBuf[off + 1]) * alpha;
            heading = lerpAngle(prevBlobBuf[off + 2], heading, alpha);
          }

          if (bx < viewMinX || bx > viewMaxX || by < viewMinY || by > viewMaxY) continue;

          const bRad = blobsF32[off + 3];
          const foodToday = blobsF32[off + 4];
          const colorHex = rgbIntToHex(blobsF32[off + 6]);
          const flags = blobsF32[off + 7] | 0;
          const isSelected = bid === selectedId;

          if (isSelected) {
            selectedBlobRender = { x: bx, y: by, r: bRad, color: colorHex, foodToday, flags, isMacro: false };
          }

          wCtx.beginPath();
          wCtx.arc(bx, by, bRad, 0, Math.PI * 2);
          wCtx.fillStyle = colorHex;
          wCtx.fill();
          wCtx.lineWidth = isSelected ? 2.8 : 1.2;
          wCtx.strokeStyle = isSelected ? "#ffffff" : "rgba(15, 23, 42, 0.85)";
          wCtx.stroke();

          if (foodToday > 0.02) {
            const ratio = Math.min(1.0, foodToday / reproNeed);
            wCtx.beginPath();
            wCtx.arc(bx, by, bRad + 3.2, -Math.PI / 2, -Math.PI / 2 + ratio * Math.PI * 2);
            wCtx.strokeStyle =
              foodToday >= reproNeed ? "#fbbf24" : foodToday >= survivalNeed ? "#10b981" : "#38bdf8";
            wCtx.lineWidth = 2.3;
            wCtx.stroke();
          }

          const isOutlaw = (flags & 32) !== 0;
          const isHonorable = (flags & 64) !== 0;
          if (isOutlaw) {
            wCtx.save();
            wCtx.beginPath();
            wCtx.arc(bx, by, bRad + 5.8, 0, Math.PI * 2);
            wCtx.strokeStyle = "rgba(244, 63, 94, 0.88)";
            wCtx.lineWidth = 1.9;
            wCtx.setLineDash([3, 2.5]);
            wCtx.stroke();
            wCtx.restore();
          } else if (isHonorable) {
            wCtx.beginPath();
            wCtx.arc(bx, by, bRad + 5.2, 0, Math.PI * 2);
            wCtx.strokeStyle = "rgba(52, 211, 153, 0.45)";
            wCtx.lineWidth = 1.4;
            wCtx.stroke();
          }

          if (flags & 2) {
            wCtx.beginPath();
            wCtx.arc(bx, by, bRad + 5, 0, Math.PI * 2);
            wCtx.strokeStyle = "rgba(163, 230, 53, 0.55)";
            wCtx.lineWidth = 1.5;
            wCtx.stroke();
          }

          const eyeDist = bRad * 0.48;
          const eyeOffset = 0.48;
          const ex1 = bx + Math.cos(heading - eyeOffset) * eyeDist;
          const ey1 = by + Math.sin(heading - eyeOffset) * eyeDist;
          const ex2 = bx + Math.cos(heading + eyeOffset) * eyeDist;
          const ey2 = by + Math.sin(heading + eyeOffset) * eyeDist;

          wCtx.fillStyle = "#090d11";
          wCtx.beginPath();
          wCtx.arc(ex1, ey1, Math.max(1.5, bRad * 0.2), 0, Math.PI * 2);
          wCtx.arc(ex2, ey2, Math.max(1.5, bRad * 0.2), 0, Math.PI * 2);
          wCtx.fill();

          if (isSelected || showVisionLines || foodToday >= reproNeed || isOutlaw) {
            wCtx.fillStyle = isOutlaw ? "#fda4af" : "#e2e8f0";
            wCtx.font = "10px JetBrains Mono, monospace";
            wCtx.textAlign = "center";
            const tag = isOutlaw
              ? `☠️#${bid}`
              : foodToday >= reproNeed
                ? `★#${bid}`
                : `#${bid}`;
            wCtx.fillText(tag, bx, by - bRad - 6);
          }
        }
      }

      // Draw Highlighted Selected Blob on Top (if visible in Macro or Meso)
      if (selectedBlobRender && (isMacro || isMeso)) {
        wCtx.beginPath();
        wCtx.arc(selectedBlobRender.x, selectedBlobRender.y, selectedBlobRender.r, 0, Math.PI * 2);
        wCtx.fillStyle = selectedBlobRender.color;
        wCtx.fill();
        wCtx.strokeStyle = "#ffffff";
        wCtx.lineWidth = Math.max(2.6, 2.0 / camZoom);
        wCtx.stroke();
      }

      // Selected Blob Sensory Overlay (Vision ring, Home tether, Memory tether, 3 Feelers)
      if (selectedBlob && simState.selected) {
        const sx = selectedBlob.x;
        const sy = selectedBlob.y;
        const sAngle = selectedBlob.heading || 0;
        wCtx.save();
        wCtx.beginPath();
        wCtx.moveTo(sx, sy);
        wCtx.lineTo(selectedBlob.home_x, selectedBlob.home_y);
        wCtx.strokeStyle = "rgba(56, 189, 248, 0.52)";
        wCtx.lineWidth = Math.max(1.6, 1.2 / camZoom);
        wCtx.setLineDash([4, 4]);
        wCtx.stroke();

        if (selectedBlob.mem_s && selectedBlob.mem_s > 0.15) {
          wCtx.beginPath();
          wCtx.moveTo(sx, sy);
          wCtx.lineTo(selectedBlob.mem_x, selectedBlob.mem_y);
          wCtx.strokeStyle = "rgba(192, 132, 252, 0.65)";
          wCtx.lineWidth = Math.max(1.6, 1.2 / camZoom);
          wCtx.setLineDash([2, 5]);
          wCtx.stroke();
        }

        wCtx.beginPath();
        wCtx.arc(sx, sy, simState.selected.vision_radius || 215, 0, Math.PI * 2);
        wCtx.strokeStyle = "rgba(56, 189, 248, 0.32)";
        wCtx.setLineDash([]);
        wCtx.stroke();

        const feelerLen = 115;
        for (const fOff of [-0.65, 0.0, 0.65]) {
          const fa = sAngle + fOff;
          wCtx.beginPath();
          wCtx.moveTo(sx, sy);
          wCtx.lineTo(sx + Math.cos(fa) * feelerLen, sy + Math.sin(fa) * feelerLen);
          wCtx.strokeStyle = "rgba(251, 191, 36, 0.45)";
          wCtx.lineWidth = 1.4;
          wCtx.stroke();
        }
        wCtx.restore();
      }
    } else if (simState.blobs) {
      for (const b of simState.blobs) {
        if (b.x < viewMinX || b.x > viewMaxX || b.y < viewMinY || b.y > viewMaxY) continue;
        const isSelected = b.id === selectedId;
        wCtx.beginPath();
        wCtx.arc(b.x, b.y, Math.max(b.r, minMacroBlobRad), 0, Math.PI * 2);
        wCtx.fillStyle = b.color;
        wCtx.fill();
        if (isSelected) {
          wCtx.strokeStyle = "#ffffff";
          wCtx.lineWidth = 2;
          wCtx.stroke();
        }
      }
    }

    wCtx.restore();
  }

  // --- Render Live Neural Network Brain on #nnCanvas ---
  function renderNeuralNetwork(brain) {
    const W = nnCanvas.width;
    const H = nnCanvas.height;
    nnCtx.clearRect(0, 0, W, H);

    const acts = brain.activations || [];
    const weights = brain.weights || [];
    if (acts.length === 0) return;

    const numLayers = acts.length;
    const padX = 68;
    const padY = 10;
    const usableW = W - padX * 2;
    const usableH = H - padY * 2;

    // Render all 26 sensory inputs on layer 0, and up to 26 hidden nodes on internal layers
    const maxDrawNodes = 26;
    const nodePositions = [];

    for (let l = 0; l < numLayers; l++) {
      const layerActs = acts[l];
      const count = Math.min(layerActs.length, maxDrawNodes);
      const x = padX + (numLayers === 1 ? usableW / 2 : (l / (numLayers - 1)) * usableW);
      const layerCoords = [];
      for (let i = 0; i < count; i++) {
        const y = padY + (count === 1 ? usableH / 2 : (i / (count - 1)) * usableH);
        layerCoords.push({ x, y, val: layerActs[i] });
      }
      nodePositions.push(layerCoords);
    }

    // 1. Draw Synaptic Weights
    for (let l = 0; l < weights.length; l++) {
      const wMat = weights[l];
      const srcNodes = nodePositions[l];
      const dstNodes = nodePositions[l + 1];
      if (!srcNodes || !dstNodes) continue;

      for (let i = 0; i < srcNodes.length; i++) {
        const row = wMat[i];
        if (!row) continue;
        for (let j = 0; j < dstNodes.length; j++) {
          const wVal = row[j] || 0;
          const signal = Math.abs(wVal * srcNodes[i].val);
          if (signal < 0.06 && Math.abs(wVal) < 0.25) continue;

          const alpha = Math.min(0.75, Math.max(0.08, signal * 0.65));
          nnCtx.strokeStyle = wVal >= 0
            ? `rgba(16, 185, 129, ${alpha})`
            : `rgba(244, 63, 94, ${alpha})`;
          nnCtx.lineWidth = Math.min(2.2, 0.5 + signal * 1.4);
          nnCtx.beginPath();
          nnCtx.moveTo(srcNodes[i].x, srcNodes[i].y);
          nnCtx.lineTo(dstNodes[j].x, dstNodes[j].y);
          nnCtx.stroke();
        }
      }
    }

    // 2. Draw Neurons & Labels
    const outLabels = ["Bew X", "Bew Y", "Sprint", "Handel", "Gruppe", "Siedeln"];
    for (let l = 0; l < nodePositions.length; l++) {
      const nodes = nodePositions[l];
      const r = l === 0 ? 3.1 : (l === nodePositions.length - 1 ? 3.8 : 3.2);

      for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        const v = Math.max(-1, Math.min(1, n.val));
        nnCtx.beginPath();
        nnCtx.arc(n.x, n.y, r, 0, Math.PI * 2);
        if (v >= 0) {
          const intensity = Math.round(40 + v * 195);
          nnCtx.fillStyle = `rgb(16, ${intensity}, 129)`;
        } else {
          const intensity = Math.round(40 + Math.abs(v) * 195);
          nnCtx.fillStyle = `rgb(${intensity}, 63, 94)`;
        }
        nnCtx.fill();
        nnCtx.strokeStyle = "#1e293b";
        nnCtx.lineWidth = 1;
        nnCtx.stroke();

        // Output layer labels
        if (l === nodePositions.length - 1 && outLabels[i]) {
          nnCtx.fillStyle = "#cbd5e1";
          nnCtx.font = "9.5px Inter, sans-serif";
          nnCtx.textAlign = "left";
          nnCtx.fillText(outLabels[i], n.x + 7, n.y + 3);
        }
      }
    }

    // Input summary labels aligned accurately with INPUT_LABELS indices (0..25)
    const keyInLabels = [
      { idx: 0, text: "Nahrung" },
      { idx: 3, text: "Zuhause" },
      { idx: 6, text: "Nachbar" },
      { idx: 10, text: "Handel/Grp" },
      { idx: 13, text: "Essen/NRG" },
      { idx: 16, text: "Oase Δ" },
      { idx: 19, text: "Gedächtnis" },
      { idx: 22, text: "Fühler L/V/R" },
      { idx: 24, text: "Ruf N/Eig" },
    ];
    nnCtx.fillStyle = "#94a3b8";
    nnCtx.font = "9.5px Inter, sans-serif";
    nnCtx.textAlign = "right";
    const firstLayer = nodePositions[0];
    for (const item of keyInLabels) {
      if (firstLayer[item.idx]) {
        nnCtx.fillText(item.text, firstLayer[item.idx].x - 7, firstLayer[item.idx].y + 3);
      }
    }
  }

  // --- Render Evolution & Economy History Chart on #chartCanvas ---
  function renderHistoryChart(history) {
    const W = chartCanvas.width;
    const H = chartCanvas.height;
    chartCtx.clearRect(0, 0, W, H);

    if (!history || history.length === 0) return;

    const mode = document.getElementById("selectChartMode").value;
    let series = [];

    if (mode === "pop") {
      series = [
        { key: "population", color: "#10b981", label: "Blobs" },
        { key: "grouped_blobs", color: "#0ea5e9", label: "In Siedlung" },
      ];
    } else if (mode === "econ") {
      series = [
        { key: "avg_money", color: "#f59e0b", label: "Ø Gold" },
        { key: "trades", color: "#10b981", label: "Handel/Tag" },
      ];
    } else {
      series = [
        { key: "avg_speed", color: "#f43f5e", label: "Speed" },
        { key: "avg_sense", color: "#0ea5e9", label: "Sicht" },
        { key: "avg_size", color: "#10b981", label: "Größe" },
      ];
    }

    const padL = 28, padR = 10, padT = 18, padB = 16;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    let maxVal = 1.0;
    for (const s of series) {
      for (const pt of history) {
        if ((pt[s.key] || 0) > maxVal) maxVal = pt[s.key];
      }
    }
    maxVal *= 1.12;

    // Draw series lines
    series.forEach((s, sIdx) => {
      chartCtx.beginPath();
      chartCtx.strokeStyle = s.color;
      chartCtx.lineWidth = 2;
      history.forEach((pt, i) => {
        const x = padL + (history.length === 1 ? plotW / 2 : (i / (history.length - 1)) * plotW);
        const y = padT + plotH - ((pt[s.key] || 0) / maxVal) * plotH;
        if (i === 0) chartCtx.moveTo(x, y);
        else chartCtx.lineTo(x, y);
      });
      chartCtx.stroke();

      // Legend
      chartCtx.fillStyle = s.color;
      chartCtx.font = "10px Inter, sans-serif";
      chartCtx.textAlign = "left";
      const lastVal = history[history.length - 1][s.key] || 0;
      const formatted = typeof lastVal === "number" && !Number.isInteger(lastVal) ? lastVal.toFixed(2) : lastVal;
      chartCtx.fillText(`${s.label}: ${formatted}`, padL + sIdx * 102, 12);
    });
  }

  // Start rendering and connect WebSocket
  connectWebSocket();
  requestAnimationFrame(renderWorld);
})();
