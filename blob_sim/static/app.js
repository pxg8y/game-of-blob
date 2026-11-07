/**
 * Blob Life — Neuroevolution & Economy Simulator Frontend
 */

(() => {
  const worldCanvas = document.getElementById("worldCanvas");
  const wCtx = worldCanvas.getContext("2d");
  const nnCanvas = document.getElementById("nnCanvas");
  const nnCtx = nnCanvas.getContext("2d");
  const chartCanvas = document.getElementById("chartCanvas");
  const chartCtx = chartCanvas.getContext("2d");

  let simState = null;
  let isRunning = true;
  let currentTool = "inspect";
  let showVisionLines = false;
  let ws = null;

  // --- WebSocket Connection ---
  function connectWebSocket() {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${window.location.host}/ws`;
    ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "state" && msg.data) {
          simState = msg.data;
          if (typeof msg.running === "boolean") {
            isRunning = msg.running;
            updatePlayPauseButton();
          }
          updateDashboardUI(simState);
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
    sendCommand({ cmd: "step_once", steps: tpd });
  });

  document.getElementById("btnResetWorld").addEventListener("click", () => {
    sendCommand({ cmd: "reset_world" });
  });

  document.querySelectorAll(".speed-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".speed-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const speed = parseInt(btn.getAttribute("data-speed"), 10);
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

  // Canvas Click Interaction
  worldCanvas.addEventListener("click", (e) => {
    const rect = worldCanvas.getBoundingClientRect();
    const scaleX = worldCanvas.width / rect.width;
    const scaleY = worldCanvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    if (currentTool === "inspect") {
      sendCommand({ cmd: "select_blob", x, y });
    } else {
      sendCommand({ cmd: "god_action", tool: currentTool, x, y });
    }
  });

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

  // World & Economy Sliders
  const sliderFoodSpawn = document.getElementById("sliderFoodSpawn");
  const sliderSurvivalNeed = document.getElementById("sliderSurvivalNeed");
  const sliderFoodPrice = document.getElementById("sliderFoodPrice");

  sliderFoodSpawn.addEventListener("input", () => {
    const v = parseInt(sliderFoodSpawn.value, 10);
    document.getElementById("valFoodSpawn").textContent = v;
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
    const text = await file.text();
    const json = JSON.parse(text);
    await fetch("/api/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(json),
    });
  });

  document.getElementById("selectChartMode").addEventListener("change", () => {
    if (simState) renderHistoryChart(simState.history || []);
  });

  // --- Update UI Readouts ---
  let lastEventTick = -1;

  function updateDashboardUI(state) {
    document.getElementById("dayBadge").textContent = `Tag ${state.day}`;
    const pct = Math.min(100, Math.round(state.time_of_day * 100));
    const fillEl = document.getElementById("timeProgressFill");
    fillEl.style.width = `${pct}%`;

    const phaseEl = document.getElementById("phaseLabel");
    if (state.is_night) {
      phaseEl.textContent = "🌙 Nacht (Heimkehr & Fortpflanzung)";
      fillEl.style.background = "#38bdf8";
    } else if (state.time_of_day > 0.56) {
      phaseEl.textContent = "🌇 Dämmerung (Rückweg zum Haus)";
      fillEl.style.background = "#f97316";
    } else {
      phaseEl.textContent = "☀️ Tag (Nahrungssuche & Handel)";
      fillEl.style.background = "#f59e0b";
    }

    document.getElementById("tickLabel").textContent = `${state.tick_in_day} / ${state.ticks_per_day}`;

    // KPIs
    const s = state.stats;
    document.getElementById("kpiPopulation").textContent = s.population;
    document.getElementById("kpiFood").textContent = s.foods_count;
    document.getElementById("kpiGroups").textContent = s.groups_count;
    document.getElementById("kpiTrades").textContent = `${s.trades_today} (${s.total_trades})`;
    document.getElementById("kpiBirths").textContent = `${s.births_today} (${s.total_births})`;

    // Network badge
    if (state.net_cfg && state.net_cfg.hidden_dims) {
      const dims = [16, ...state.net_cfg.hidden_dims, 5];
      document.getElementById("netTopologyBadge").textContent = dims.join(" → ");
    }

    // Selected Blob Inspector
    if (state.selected && state.selected.blob) {
      const b = state.selected.blob;
      const reproNeed = state.world ? state.world.repro_need : 2.0;
      document.getElementById("inspectorTitle").textContent = `🔬 Blob #${b.id} (Gen ${b.gen})`;
      document.getElementById("inspectorGroupBadge").textContent = state.selected.group_name;
      document.getElementById("inspFoodToday").textContent = `${b.food_today.toFixed(2)} / ${reproNeed.toFixed(2)}`;
      document.getElementById("inspMoney").textContent = `${b.money.toFixed(1)} 🪙 (+${b.surplus_food.toFixed(1)}🍏)`;
      document.getElementById("inspEnergyAge").textContent = `${Math.round(b.energy)}% · ${b.age_days}d`;
      document.getElementById("inspGenChildren").textContent = `G${b.gen} · ${b.children}👶 · ${b.trades}🤝`;
      document.getElementById("inspGenesText").textContent =
        `${b.genes.speed.toFixed(2)}x / ${b.genes.sense.toFixed(2)}x / ${b.genes.size.toFixed(2)}x`;

      if (state.selected.brain) {
        document.getElementById("inspParamCount").textContent =
          `${state.selected.brain.param_count} Synapsen (${state.selected.brain.activation.toUpperCase()})`;
        renderNeuralNetwork(state.selected.brain);
      }
    }

    // Groups list
    const grpContainer = document.getElementById("groupListContainer");
    if (state.groups && state.groups.length > 0) {
      grpContainer.innerHTML = state.groups
        .map(
          (g) => `
        <div class="group-row">
          <div>
            <span class="group-dot" style="background:${g.color}"></span>
            <strong>${g.name}</strong>
          </div>
          <span>${g.member_count} Blobs · ${g.total_trades}🤝</span>
        </div>`
        )
        .join("");
    } else {
      grpContainer.innerHTML = `<div class="event-item">Noch keine Gruppen — Blobs gründen Clans bei Begegnung!</div>`;
    }

    // Events ticker
    if (state.events && state.events.length > 0) {
      const newestTick = state.events[state.events.length - 1].tick;
      if (newestTick !== lastEventTick) {
        lastEventTick = newestTick;
        const evContainer = document.getElementById("eventFeedContainer");
        evContainer.innerHTML = state.events
          .slice()
          .reverse()
          .map(
            (ev) => `
          <div class="event-item">
            <span class="ev-day">[T${ev.day}]</span>${ev.msg}
          </div>`
          )
          .join("");
      }
    }

    renderHistoryChart(state.history || []);
  }

  // --- Render 2D World Arena ---
  function renderWorld() {
    requestAnimationFrame(renderWorld);
    if (!simState) return;

    const W = worldCanvas.width;
    const H = worldCanvas.height;
    const tod = simState.time_of_day || 0;

    // Day/Dusk/Night ambient background color
    let bgR = 13, bgG = 22, bgB = 29;
    if (simState.is_night) {
      bgR = 8; bgG = 13; bgB = 22;
    } else if (tod > 0.56) {
      bgR = 18; bgG = 21; bgB = 28;
    }
    wCtx.fillStyle = `rgb(${bgR}, ${bgG}, ${bgB})`;
    wCtx.fillRect(0, 0, W, H);

    // Subtle organic terrain boundary
    wCtx.strokeStyle = "rgba(56, 189, 248, 0.14)";
    wCtx.lineWidth = 2;
    wCtx.strokeRect(10, 10, W - 20, H - 20);

    const selectedId = simState.selected && simState.selected.blob ? simState.selected.blob.id : null;
    const groupMap = {};
    if (simState.groups) {
      for (const g of simState.groups) groupMap[g.id] = g;
    }

    // 1. Draw Homes & Group Villages
    if (simState.homes) {
      for (const h of simState.homes) {
        const grp = h.group_id ? groupMap[h.group_id] : null;
        const baseColor = grp ? grp.color : "#64748b";

        wCtx.save();
        wCtx.beginPath();
        wCtx.arc(h.x, h.y, h.radius, 0, Math.PI * 2);
        wCtx.fillStyle = grp ? `${baseColor}1f` : "rgba(100, 116, 139, 0.12)";
        wCtx.fill();
        wCtx.lineWidth = grp ? 2.2 : 1.3;
        wCtx.setLineDash(grp ? [] : [4, 4]);
        wCtx.strokeStyle = grp ? baseColor : "rgba(148, 163, 184, 0.45)";
        wCtx.stroke();

        // Inner hearth icon
        wCtx.beginPath();
        wCtx.arc(h.x, h.y, 5, 0, Math.PI * 2);
        wCtx.fillStyle = grp ? baseColor : "#94a3b8";
        wCtx.fill();

        // Depot label
        wCtx.fillStyle = "rgba(226, 232, 240, 0.78)";
        wCtx.font = "11px Inter, sans-serif";
        wCtx.textAlign = "center";
        const label = grp ? `${grp.name} (${h.food_depot.toFixed(1)}🍏)` : `Haus #${h.id} (${h.food_depot.toFixed(1)}🍏)`;
        wCtx.fillText(label, h.x, h.y - h.radius - 5);
        wCtx.restore();
      }
    }

    // 2. Draw Food Items
    if (simState.foods) {
      for (const f of simState.foods) {
        wCtx.beginPath();
        const r = f.g ? 6.2 : 4.3;
        wCtx.arc(f.x, f.y, r, 0, Math.PI * 2);
        wCtx.fillStyle = f.g ? "#fbbf24" : "#22c55e";
        wCtx.fill();
        if (f.g) {
          wCtx.strokeStyle = "rgba(251, 191, 36, 0.45)";
          wCtx.lineWidth = 3;
          wCtx.stroke();
        }
      }
    }

    // 3. Draw Active Trade Lines (Food <-> Money transactions)
    if (simState.trade_lines) {
      for (const tl of simState.trade_lines) {
        wCtx.save();
        wCtx.beginPath();
        wCtx.moveTo(tl.x1, tl.y1);
        wCtx.lineTo(tl.x2, tl.y2);
        wCtx.strokeStyle = tl.same_group ? "#34d399" : "#fbbf24";
        wCtx.lineWidth = 2.2;
        wCtx.setLineDash([5, 3]);
        wCtx.stroke();

        const mx = (tl.x1 + tl.x2) * 0.5;
        const my = (tl.y1 + tl.y2) * 0.5;
        wCtx.fillStyle = "#fde047";
        wCtx.font = "11px sans-serif";
        wCtx.textAlign = "center";
        wCtx.fillText("🪙⇄🍏", mx, my - 4);
        wCtx.restore();
      }
    }

    // 4. Draw Blobs
    const survivalNeed = simState.world ? simState.world.survival_need : 1.0;
    const reproNeed = simState.world ? simState.world.repro_need : 2.0;

    if (simState.blobs) {
      for (const b of simState.blobs) {
        const isSelected = b.id === selectedId;

        // Optional vision & home tether lines
        if (isSelected || showVisionLines) {
          wCtx.save();
          wCtx.beginPath();
          wCtx.moveTo(b.x, b.y);
          wCtx.lineTo(b.home_x, b.home_y);
          wCtx.strokeStyle = isSelected ? "rgba(56, 189, 248, 0.45)" : "rgba(148, 163, 184, 0.14)";
          wCtx.lineWidth = isSelected ? 1.5 : 1;
          wCtx.setLineDash([3, 4]);
          wCtx.stroke();

          if (isSelected && simState.selected) {
            wCtx.beginPath();
            wCtx.arc(b.x, b.y, simState.selected.vision_radius || 140, 0, Math.PI * 2);
            wCtx.strokeStyle = "rgba(56, 189, 248, 0.28)";
            wCtx.setLineDash([]);
            wCtx.stroke();
          }
          wCtx.restore();
        }

        // Blob body
        wCtx.save();
        wCtx.beginPath();
        wCtx.arc(b.x, b.y, b.r, 0, Math.PI * 2);
        wCtx.fillStyle = b.color;
        wCtx.fill();
        wCtx.lineWidth = isSelected ? 2.8 : 1.2;
        wCtx.strokeStyle = isSelected ? "#ffffff" : "rgba(15, 23, 42, 0.85)";
        wCtx.stroke();

        // Outer Food Progress Arc (Emerald = 1x survival, Gold = 2x reproduction ready!)
        if (b.food_today > 0.02) {
          const ratio = Math.min(1.0, b.food_today / reproNeed);
          wCtx.beginPath();
          wCtx.arc(b.x, b.y, b.r + 3.2, -Math.PI / 2, -Math.PI / 2 + ratio * Math.PI * 2);
          wCtx.strokeStyle = b.food_today >= reproNeed ? "#fbbf24" : (b.food_today >= survivalNeed ? "#10b981" : "#38bdf8");
          wCtx.lineWidth = 2.4;
          wCtx.stroke();
        }

        // Directional eyes
        const angle = Math.atan2(b.vy || 0.01, b.vx || 0.01);
        const eyeDist = b.r * 0.48;
        const eyeOffset = 0.48;
        const ex1 = b.x + Math.cos(angle - eyeOffset) * eyeDist;
        const ey1 = b.y + Math.sin(angle - eyeOffset) * eyeDist;
        const ex2 = b.x + Math.cos(angle + eyeOffset) * eyeDist;
        const ey2 = b.y + Math.sin(angle + eyeOffset) * eyeDist;

        wCtx.fillStyle = "#090d11";
        wCtx.beginPath();
        wCtx.arc(ex1, ey1, Math.max(1.5, b.r * 0.2), 0, Math.PI * 2);
        wCtx.arc(ex2, ey2, Math.max(1.5, b.r * 0.2), 0, Math.PI * 2);
        wCtx.fill();

        // Selection ring or 2x reproduction indicator
        if (isSelected) {
          wCtx.beginPath();
          wCtx.arc(b.x, b.y, b.r + 7, 0, Math.PI * 2);
          wCtx.strokeStyle = "#38bdf8";
          wCtx.lineWidth = 1.6;
          wCtx.stroke();
        }

        if (isSelected || showVisionLines || b.food_today >= reproNeed) {
          wCtx.fillStyle = "#e2e8f0";
          wCtx.font = "10px JetBrains Mono, monospace";
          wCtx.textAlign = "center";
          const tag = b.food_today >= reproNeed ? `★#${b.id}` : `#${b.id}`;
          wCtx.fillText(tag, b.x, b.y - b.r - 6);
        }
        wCtx.restore();
      }
    }
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
    const padX = 56;
    const padY = 14;
    const usableW = W - padX * 2;
    const usableH = H - padY * 2;

    // Compute node coordinates per layer (cap visual nodes at 14 per layer for crispness)
    const maxDrawNodes = 14;
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
    const outLabels = ["Bew X", "Bew Y", "Sprint", "Handel", "Gruppe"];
    for (let l = 0; l < nodePositions.length; l++) {
      const nodes = nodePositions[l];
      const r = l === 0 || l === nodePositions.length - 1 ? 4.2 : 3.6;

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
          nnCtx.font = "10px Inter, sans-serif";
          nnCtx.textAlign = "left";
          nnCtx.fillText(outLabels[i], n.x + 7, n.y + 3);
        }
      }
    }

    // Input summary labels on left
    const keyInLabels = [
      { idx: 0, text: "Nahrung" },
      { idx: 3, text: "Zuhause" },
      { idx: 6, text: "Nachbar" },
      { idx: 11, text: "Essen/2x" },
      { idx: 13, text: "Geld/Zeit" },
    ];
    nnCtx.fillStyle = "#94a3b8";
    nnCtx.font = "10px Inter, sans-serif";
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
        { key: "grouped_blobs", color: "#0ea5e9", label: "In Gruppen" },
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
