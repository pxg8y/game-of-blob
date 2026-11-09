/**
 * Blob Life — /analyse Telemetry, Rise-and-Collapse, Genetics & Neural Weight Cockpit
 */

(() => {
  // --- Telemetry State ---
  let dailyRecords = [];
  let collapseLogs = [];
  let tribesList = [];
  let tribesById = {};
  let summaryInfo = {};
  let inputLabels = [];
  let outputLabels = [];
  let pathwayDefs = [];
  let initialCouplingMatrix = null;
  let scatterBlobs = [];
  let selectedInspector = null;
  let eventsList = [];
  let selectedTribeId = null;
  let activeTab = "tab-cycles";
  let isRunning = true;
  let currentSpeed = 1;
  let ws = null;
  let liveSimState = null;
  let isFetchingAnalytics = false;
  let lastFetchedDay = 0;

  // Interactive Hover State
  let cycleHoverIndex = -1;
  let heatmapHoverCell = null;

  // --- DOM References ---
  const anDayBadge = document.getElementById("anDayBadge");
  const anSeasonBadge = document.getElementById("anSeasonBadge");
  const anPhaseLabel = document.getElementById("anPhaseLabel");
  const anTickLabel = document.getElementById("anTickLabel");
  const anTimeProgressFill = document.getElementById("anTimeProgressFill");
  const anBtnPlayPause = document.getElementById("anBtnPlayPause");

  // Canvases
  const cycleCanvas = document.getElementById("cycleTimelineCanvas");
  const cycleCtx = cycleCanvas ? cycleCanvas.getContext("2d") : null;

  const commCanvas = document.getElementById("communityActionsCanvas");
  const commCtx = commCanvas ? commCanvas.getContext("2d") : null;

  const tribeCanvas = document.getElementById("tribeDetailCanvas");
  const tribeCtx = tribeCanvas ? tribeCanvas.getContext("2d") : null;

  const genCanvas = document.getElementById("geneticsChartCanvas");
  const genCtx = genCanvas ? genCanvas.getContext("2d") : null;

  const scatterCanvas = document.getElementById("geneScatterCanvas");
  const scatterCtx = scatterCanvas ? scatterCanvas.getContext("2d") : null;

  const pathwayCanvas = document.getElementById("pathwayChartCanvas");
  const pathwayCtx = pathwayCanvas ? pathwayCanvas.getContext("2d") : null;

  const weightMetricsCanvas = document.getElementById("weightMetricsCanvas");
  const weightMetricsCtx = weightMetricsCanvas ? weightMetricsCanvas.getContext("2d") : null;

  const heatmapCanvas = document.getElementById("couplingHeatmapCanvas");
  const heatmapCtx = heatmapCanvas ? heatmapCanvas.getContext("2d") : null;

  const cockpitNnCanvas = document.getElementById("cockpitNnCanvas");
  const cockpitNnCtx = cockpitNnCanvas ? cockpitNnCanvas.getContext("2d") : null;

  // --- Transport & Command Helpers ---
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

  function updatePlayPauseButton() {
    if (anBtnPlayPause) {
      anBtnPlayPause.textContent = isRunning ? "⏸ Pause" : "▶ Start";
    }
  }

  function updateActiveSpeedButton(speed) {
    currentSpeed = speed;
    document.querySelectorAll(".speed-btn").forEach((b) => {
      const btnSpeed = parseInt(b.getAttribute("data-speed"), 10);
      b.classList.toggle("active", btnSpeed === speed);
    });
  }

  if (anBtnPlayPause) {
    anBtnPlayPause.addEventListener("click", () => {
      isRunning = !isRunning;
      updatePlayPauseButton();
      sendCommand({ cmd: "set_running", running: isRunning });
    });
  }

  const anBtnStepDay = document.getElementById("anBtnStepDay");
  if (anBtnStepDay) {
    anBtnStepDay.addEventListener("click", async () => {
      isRunning = false;
      updatePlayPauseButton();
      const tpd = liveSimState ? liveSimState.ticks_per_day : 360;
      const rem = liveSimState && liveSimState.tick_in_day > 0 ? tpd - liveSimState.tick_in_day : tpd;
      await fetch("/api/command", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cmd: "step_once", steps: rem }),
      });
      await fetchAnalyticsPayload(false);
    });
  }

  const anBtnStep10Days = document.getElementById("anBtnStep10Days");
  if (anBtnStep10Days) {
    anBtnStep10Days.addEventListener("click", async () => {
      isRunning = false;
      updatePlayPauseButton();
      anBtnStep10Days.disabled = true;
      anBtnStep10Days.textContent = "⏳ Berechne 10 Tage…";
      try {
        const tpd = liveSimState ? liveSimState.ticks_per_day : 360;
        const rem = liveSimState && liveSimState.tick_in_day > 0 ? tpd - liveSimState.tick_in_day : tpd;
        const totalSteps = rem + tpd * 9;
        await fetch("/api/command", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ cmd: "step_once", steps: totalSteps }),
        });
        await fetchAnalyticsPayload(false);
      } finally {
        anBtnStep10Days.disabled = false;
        anBtnStep10Days.textContent = "+10 Tage";
      }
    });
  }

  document.querySelectorAll(".speed-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const speed = parseInt(btn.getAttribute("data-speed"), 10);
      updateActiveSpeedButton(speed);
      sendCommand({ cmd: "set_speed", speed });
    });
  });

  // --- Export Buttons ---
  const anBtnExportTelemetry = document.getElementById("anBtnExportTelemetry");
  if (anBtnExportTelemetry) {
    anBtnExportTelemetry.addEventListener("click", async () => {
      const res = await fetch("/api/analytics/export");
      const data = await res.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const lastDay = dailyRecords.length > 0 ? dailyRecords[dailyRecords.length - 1].day : 1;
      a.download = `blob_telemetry_tag_${lastDay}.json`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  const anBtnExportCsv = document.getElementById("anBtnExportCsv");
  if (anBtnExportCsv) {
    anBtnExportCsv.addEventListener("click", () => {
      if (dailyRecords.length === 0) return;
      const headers = [
        "day",
        "season",
        "population",
        "grouped_blobs",
        "active_tribes",
        "dominant_tribe_name",
        "dominant_share_pct",
        "births",
        "deaths_total",
        "deaths_midday_starve",
        "deaths_eod_starve",
        "deaths_with_money_far",
        "deaths_with_money_empty_depot",
        "trades_total",
        "stomach_sells_total",
        "trades_emergency_midday",
        "trades_eod_deficit_buy",
        "raids_won",
        "raids_repelled",
        "houses_conquered",
        "secessions",
        "upgrades_total",
        "speed_mean",
        "size_mean",
        "sense_mean",
        "basal_cost_mean",
        "weight_drift_day1",
        "brain_diversity_std",
      ];
      const rows = [headers.join(",")];
      for (const r of dailyRecords) {
        const g = r.genetics || {};
        const w = r.weights || {};
        const c = r.community || {};
        const m = r.monoculture || {};
        const row = [
          r.day,
          r.season_id,
          r.population,
          r.grouped_blobs,
          m.active_tribes_count || r.groups_count || 0,
          `"${(m.dominant_tribe_name || "").replace(/"/g, '""')}"`,
          m.dominant_share_pct || 0,
          r.births || 0,
          r.deaths || 0,
          c.deaths_midday_starve || 0,
          c.deaths_eod_starve || 0,
          c.deaths_with_money_far || 0,
          c.deaths_with_money_empty_depot || 0,
          r.trades || 0,
          c.stomach_sells_total || 0,
          c.trades_emergency_midday || 0,
          c.trades_eod_deficit_buy || 0,
          c.raids_won || 0,
          c.raids_repelled || 0,
          c.houses_conquered || 0,
          c.secessions || 0,
          c.upgrades_total || 0,
          g.speed ? g.speed.mean : 1,
          g.size ? g.size.mean : 1,
          g.sense ? g.sense.mean : 1,
          g.basal_cost ? g.basal_cost.mean : 0.06,
          w.weight_drift_day1 || 0,
          w.brain_diversity_std || 0,
        ];
        rows.push(row.join(","));
      }
      const blob = new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `blob_daily_telemetry_${dailyRecords[dailyRecords.length - 1].day}d.csv`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  // --- Tab Switching ---
  document.querySelectorAll(".cockpit-tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".cockpit-tab-btn").forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      document.querySelectorAll(".cockpit-tab-panel").forEach((p) => {
        p.classList.remove("active");
      });
      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      activeTab = btn.getAttribute("data-tab");
      const targetPanel = document.getElementById(activeTab);
      if (targetPanel) targetPanel.classList.add("active");
      renderActiveTabCharts();
    });
  });

  // --- Select & Filter Event Listeners ---
  const selectCycleChartMode = document.getElementById("selectCycleChartMode");
  if (selectCycleChartMode) {
    selectCycleChartMode.addEventListener("change", () => renderCycleTimeline());
  }

  const selectCommunityActionMode = document.getElementById("selectCommunityActionMode");
  if (selectCommunityActionMode) {
    selectCommunityActionMode.addEventListener("change", () => renderCommunityActionsChart());
  }

  const selectTribeFilter = document.getElementById("selectTribeFilter");
  if (selectTribeFilter) {
    selectTribeFilter.addEventListener("change", () => renderTribesTable());
  }

  const selectGeneticsMode = document.getElementById("selectGeneticsMode");
  if (selectGeneticsMode) {
    selectGeneticsMode.addEventListener("change", () => renderGeneticsChart());
  }

  const selectPathwayGroup = document.getElementById("selectPathwayGroup");
  if (selectPathwayGroup) {
    selectPathwayGroup.addEventListener("change", () => renderPathwayChart());
  }

  const selectWeightMetricsMode = document.getElementById("selectWeightMetricsMode");
  if (selectWeightMetricsMode) {
    selectWeightMetricsMode.addEventListener("change", () => renderWeightMetricsChart());
  }

  const selectHeatmapMode = document.getElementById("selectHeatmapMode");
  if (selectHeatmapMode) {
    selectHeatmapMode.addEventListener("change", () => renderCouplingHeatmap());
  }

  const heatmapDaySlider = document.getElementById("heatmapDaySlider");
  if (heatmapDaySlider) {
    heatmapDaySlider.addEventListener("input", () => {
      updateHeatmapSliderLabel();
      renderCouplingHeatmap();
    });
  }

  const eventSearchInput = document.getElementById("eventSearchInput");
  const eventCategoryFilter = document.getElementById("eventCategoryFilter");
  if (eventSearchInput) eventSearchInput.addEventListener("input", () => renderEventLog());
  if (eventCategoryFilter) eventCategoryFilter.addEventListener("change", () => renderEventLog());

  // --- Interactive Canvas Hover Listeners ---
  if (cycleCanvas) {
    cycleCanvas.addEventListener("pointermove", (e) => {
      if (dailyRecords.length === 0) return;
      const rect = cycleCanvas.getBoundingClientRect();
      const scaleX = cycleCanvas.width / Math.max(1, rect.width);
      const cx = (e.clientX - rect.left) * scaleX;
      const padL = 52;
      const padR = 24;
      const plotW = cycleCanvas.width - padL - padR;
      const rel = Math.max(0, Math.min(1, (cx - padL) / Math.max(1, plotW)));
      const idx = Math.round(rel * (dailyRecords.length - 1));
      if (idx !== cycleHoverIndex) {
        cycleHoverIndex = idx;
        renderCycleTimeline();
      }
    });
    cycleCanvas.addEventListener("pointerleave", () => {
      cycleHoverIndex = -1;
      renderCycleTimeline();
    });
  }

  if (heatmapCanvas) {
    heatmapCanvas.addEventListener("pointermove", (e) => {
      const rect = heatmapCanvas.getBoundingClientRect();
      const scaleX = heatmapCanvas.width / Math.max(1, rect.width);
      const scaleY = heatmapCanvas.height / Math.max(1, rect.height);
      const cx = (e.clientX - rect.left) * scaleX;
      const cy = (e.clientY - rect.top) * scaleY;
      const padL = 175;
      const padT = 36;
      const padR = 16;
      const padB = 12;
      const gridW = heatmapCanvas.width - padL - padR;
      const gridH = heatmapCanvas.height - padT - padB;
      const numInputs = (inputLabels && inputLabels.length) || 26;
      const cellW = gridW / 6;
      const cellH = gridH / numInputs;
      const col = Math.floor((cx - padL) / cellW);
      const row = Math.floor((cy - padT) / cellH);
      if (col >= 0 && col < 6 && row >= 0 && row < numInputs) {
        if (!heatmapHoverCell || heatmapHoverCell.row !== row || heatmapHoverCell.col !== col) {
          heatmapHoverCell = { row, col };
          renderCouplingHeatmap();
        }
      } else if (heatmapHoverCell) {
        heatmapHoverCell = null;
        renderCouplingHeatmap();
      }
    });
    heatmapCanvas.addEventListener("pointerleave", () => {
      heatmapHoverCell = null;
      renderCouplingHeatmap();
    });
  }

  // --- Fetch Telemetry from /api/analytics ---
  async function fetchAnalyticsPayload(incremental = true) {
    if (isFetchingAnalytics) return;
    isFetchingAnalytics = true;
    try {
      const sinceDay = incremental && dailyRecords.length > 0 ? lastFetchedDay : 0;
      const res = await fetch(`/api/analytics?since_day=${sinceDay}`);
      const data = await res.json();

      if (data.input_labels) inputLabels = data.input_labels;
      if (data.output_labels) outputLabels = data.output_labels;
      if (data.pathway_defs) pathwayDefs = data.pathway_defs;
      if (data.initial_coupling_matrix) initialCouplingMatrix = data.initial_coupling_matrix;
      if (data.summary) summaryInfo = data.summary;
      if (data.scatter_blobs) scatterBlobs = data.scatter_blobs;
      if (data.selected) selectedInspector = data.selected;

      const incomingDays = data.daily_logs || [];
      if (!incremental || sinceDay === 0 || (incomingDays.length > 0 && incomingDays[0].day === 1 && sinceDay > 1)) {
        dailyRecords = incomingDays;
      } else if (incomingDays.length > 0) {
        const byDay = new Map();
        dailyRecords.forEach((r, idx) => byDay.set(r.day, idx));
        for (const rec of incomingDays) {
          if (byDay.has(rec.day)) {
            dailyRecords[byDay.get(rec.day)] = rec;
          } else {
            byDay.set(rec.day, dailyRecords.length);
            dailyRecords.push(rec);
          }
        }
      }

      if (dailyRecords.length > 0) {
        lastFetchedDay = dailyRecords[dailyRecords.length - 1].day;
      } else {
        lastFetchedDay = 0;
      }

      collapseLogs = data.collapse_logs || [];
      tribesList = data.tribes || [];
      tribesById = {};
      for (const t of tribesList) {
        tribesById[String(t.id)] = t;
      }
      if (data.events) {
        eventsList = data.events;
      }

      syncHeatmapSliderRange();
      updateTopKpisAndDiagnostics();
      renderActiveTabCharts();
    } catch (err) {
      console.error("Failed to fetch analytics:", err);
    } finally {
      isFetchingAnalytics = false;
    }
  }

  function syncHeatmapSliderRange() {
    if (!heatmapDaySlider) return;
    const maxIdx = Math.max(0, dailyRecords.length - 1);
    const prevMax = parseInt(heatmapDaySlider.max, 10) || 0;
    const prevVal = parseInt(heatmapDaySlider.value, 10) || 0;
    const wasAtEnd = prevVal >= prevMax;
    heatmapDaySlider.max = String(maxIdx);
    if (wasAtEnd || prevVal > maxIdx) {
      heatmapDaySlider.value = String(maxIdx);
    }
    updateHeatmapSliderLabel();
  }

  function updateHeatmapSliderLabel() {
    const label = document.getElementById("heatmapDayLabel");
    if (!label || !heatmapDaySlider) return;
    if (dailyRecords.length === 0) {
      label.textContent = "Tag 1";
      return;
    }
    const idx = Math.max(0, Math.min(dailyRecords.length - 1, parseInt(heatmapDaySlider.value, 10) || 0));
    const rec = dailyRecords[idx];
    label.textContent = `Tag ${rec.day} (${rec.season_label || rec.season_id})`;
  }

  // --- Update Top KPIs & Automated Root-Cause Diagnostic Banner ---
  function updateTopKpisAndDiagnostics() {
    const latest = dailyRecords.length > 0 ? dailyRecords[dailyRecords.length - 1] : null;
    if (!latest) return;

    const trackedBadge = document.getElementById("diagDaysTrackedBadge");
    if (trackedBadge) {
      trackedBadge.textContent = `${dailyRecords.length} Tage protokolliert · Aktuell Tag ${latest.day}`;
    }

    // 1. Population & Peak
    const popEl = document.getElementById("kpiAnPop");
    const popSubEl = document.getElementById("kpiAnPopSub");
    const curPop = summaryInfo.current_population ?? latest.population ?? 0;
    const peakPop = summaryInfo.peak_population ?? Math.max(...dailyRecords.map((r) => r.population || 0), curPop);
    const grpPct = latest.population > 0 ? Math.round(((latest.grouped_blobs || 0) / latest.population) * 100) : 0;
    if (popEl) popEl.textContent = String(curPop);
    if (popSubEl) popSubEl.textContent = `Peak: ${peakPop} (T${summaryInfo.peak_pop_day || 1}) · In Siedlung: ${grpPct}%`;

    // 2. Tribes
    const activeTribesCount = summaryInfo.active_groups ?? latest.groups_count ?? 0;
    const totalTribesEver = summaryInfo.total_tribes_ever ?? tribesList.length;
    const totalSecessions = dailyRecords.reduce((acc, r) => acc + ((r.community && r.community.secessions) || 0), 0);
    document.getElementById("kpiAnTribes").textContent = `${activeTribesCount} / ${totalTribesEver}`;
    document.getElementById("kpiAnTribesSub").textContent = `Häuser: ${latest.homes_count || 0} · Sezessionen: ${totalSecessions}`;

    // 3. Dominance
    const mono = latest.monoculture || {};
    const domPct = Math.round(mono.dominant_share_pct || 0);
    const domName = mono.dominant_tribe_name || "Keine Fraktion";
    const domEl = document.getElementById("kpiAnDominance");
    domEl.textContent = `${domPct}%`;
    domEl.style.color = domPct >= 80 ? "#f59e0b" : "var(--text-primary)";
    document.getElementById("kpiAnDominanceSub").textContent =
      domName !== "Keine Fraktion"
        ? `${domName} (${Math.round(mono.dominant_grouped_share_pct || 0)}% der Siedler)`
        : "Noch kein Großstamm aktiv";

    // 4. Collapses & HoF Respawns
    const hofTotal = dailyRecords.reduce((acc, r) => acc + ((r.community && r.community.hof_rescued_blobs) || 0), 0);
    const colEl = document.getElementById("kpiAnCollapses");
    colEl.textContent = String(collapseLogs.length);
    colEl.style.color = collapseLogs.length > 0 ? "#f43f5e" : "var(--text-primary)";
    document.getElementById("kpiAnCollapsesSub").textContent = `HoF-Notrettungen: ${hofTotal} Blobs`;

    // 5. Critical Stomach Sales (< survival_need)
    const stomachTotal = dailyRecords.reduce(
      (acc, r) => acc + ((r.community && r.community.stomach_sells_total) || 0),
      0
    );
    const totalTradesAll = summaryInfo.total_trades ?? dailyRecords.reduce((acc, r) => acc + (r.trades || 0), 0);
    document.getElementById("kpiAnStomachSales").textContent = String(stomachTotal);
    document.getElementById("kpiAnStomachSalesSub").textContent = `Von ${totalTradesAll} Gesamt-Trades`;

    // 6. Starvation Deaths despite holding Gold
    const deathsFar = dailyRecords.reduce((acc, r) => acc + ((r.community && r.community.deaths_with_money_far) || 0), 0);
    const deathsEmptyDepot = dailyRecords.reduce(
      (acc, r) => acc + ((r.community && r.community.deaths_with_money_empty_depot) || 0),
      0
    );
    document.getElementById("kpiAnDeathsWithGold").textContent = String(deathsFar + deathsEmptyDepot);
    document.getElementById("kpiAnDeathsWithGoldSub").textContent =
      `Fernab Haus: ${deathsFar} · Depot leer: ${deathsEmptyDepot}`;

    // 7. Basal Cost & Genes
    const genes = latest.genetics || {};
    const basalMean = genes.basal_cost ? genes.basal_cost.mean : 0.06;
    const spdMean = genes.speed ? genes.speed.mean : 1.0;
    const sizMean = genes.size ? genes.size.mean : 1.0;
    document.getElementById("kpiAnBasalCost").textContent = basalMean.toFixed(4);
    document.getElementById("kpiAnBasalCostSub").textContent =
      `Speed: ${spdMean.toFixed(2)}x · Größe: ${sizMean.toFixed(2)}x`;

    // 8. Brain Diversity & Weight Drift
    const weights = latest.weights || {};
    const divStd = weights.brain_diversity_std || 0;
    const drift = weights.weight_drift_day1 || 0;
    document.getElementById("kpiAnBrainDiv").textContent = divStd.toFixed(3);
    document.getElementById("kpiAnBrainDivSub").textContent =
      `Drift ab T1: ${drift.toFixed(3)} · L2-Norm: ${(weights.weight_l2_norm || 0).toFixed(2)}`;

    // Update Diagnostic Findings Chips
    const diagRow = document.getElementById("diagnosticFindingsRow");
    if (!diagRow) return;
    diagRow.replaceChildren();

    const chips = [];
    if (activeTribesCount === 1 && curPop >= 15) {
      chips.push({
        cls: "chip-warn",
        text: `👑 Ein-Stamm-Monopol: "${domName}" stellt ${domPct}% der Population (nur 1 aktiver Stamm übrig).`,
      });
    } else if (domPct >= 75) {
      chips.push({
        cls: "chip-warn",
        text: `👑 Hohe Stammes-Dominanz: "${domName}" hält ${domPct}% der Gesamtpopulation.`,
      });
    }

    if (divStd > 0 && divStd < 0.14 && latest.day > 3) {
      chips.push({
        cls: "chip-danger",
        text: `🧬 Klonale Konvergenz (Gehirn-Diversität σ=${divStd.toFixed(3)}): Synchrone Entscheidungen erhöhen das Risiko kollektiver Überweidungs-Kollapse.`,
      });
    }

    if (basalMean > 0.066) {
      chips.push({
        cls: "chip-danger",
        text: `🔥 Metabolische Kostenschere: Ø Grundumsatz (${basalMean.toFixed(4)}/Tick) durch Speed (${spdMean.toFixed(2)}x) & Größe (${sizMean.toFixed(2)}x) erhöht.`,
      });
    }

    if (stomachTotal > 0) {
      chips.push({
        cls: "chip-warn",
        text: `⚠️ Magen-Ausverkauf: ${stomachTotal}× verkauften Blobs eigene Überlebensnahrung (< Überlebensbedarf) gegen Gold.`,
      });
    }

    if (deathsFar + deathsEmptyDepot > 0) {
      chips.push({
        cls: "chip-danger",
        text: `🪙💀 Markt-/Distanz-Tod: ${deathsFar} Blobs verhungerten mit Gold fernab des Hauses, ${deathsEmptyDepot} vor leerem Haus-Depot.`,
      });
    }

    if (collapseLogs.length > 0) {
      const lastCol = collapseLogs[collapseLogs.length - 1];
      const topCause = (lastCol.causes && lastCol.causes[0]) || "Ressourcen-Kollaps";
      chips.push({
        cls: "chip-danger",
        text: `📉 Letzter Kollaps (Tag ${lastCol.day}): −${lastCol.drop_pct}% (${lastCol.peak_population}→${lastCol.post_population}) · ${topCause}`,
      });
    }

    if (chips.length === 0) {
      chips.push({
        cls: "chip-ok",
        text: `✅ System-Beobachtung aktiv (Tag ${latest.day}): ${activeTribesCount} aktive Stämme, Ø Grundumsatz ${basalMean.toFixed(4)}, Gehirn-Diversität σ=${divStd.toFixed(3)}.`,
      });
    }

    for (const c of chips) {
      const div = document.createElement("div");
      div.className = `diag-chip ${c.cls}`;
      div.textContent = c.text;
      diagRow.appendChild(div);
    }
  }

  // --- Render Active Tab Charts ---
  function renderActiveTabCharts() {
    if (activeTab === "tab-cycles") {
      renderCycleTimeline();
      renderCollapseAutopsyCards();
    } else if (activeTab === "tab-community") {
      renderCommunityActionsChart();
      renderTribesTable();
      renderTribeDetailChart();
    } else if (activeTab === "tab-genetics") {
      renderGeneticsChart();
      renderGeneScatterPlot();
    } else if (activeTab === "tab-weights") {
      renderPathwayChart();
      renderWeightMetricsChart();
      renderCouplingHeatmap();
      if (selectedInspector && selectedInspector.brain) {
        renderCockpitNeuralNet(selectedInspector);
      }
    } else if (activeTab === "tab-logs") {
      renderDailyTelemetryTable();
      renderEventLog();
    }
  }

  // --- Helper: Draw Multi-Series Line Chart on a Canvas ---
  function drawMultiLineChart(ctx, canvas, records, seriesList, options = {}) {
    if (!ctx || !canvas) return;
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const padL = options.padL || 52;
    const padR = options.padR || 24;
    const padT = options.padT || 24;
    const padB = options.padB || 28;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    if (!records || records.length === 0) {
      ctx.fillStyle = "#687b8f";
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Noch keine Zeitreihen-Daten vorhanden.", W / 2, H / 2);
      return;
    }

    let minVal = typeof options.minVal === "number" ? options.minVal : 0;
    let maxVal = typeof options.maxVal === "number" ? options.maxVal : -Infinity;
    for (const s of seriesList) {
      for (const r of records) {
        const v = s.getter(r);
        if (typeof v === "number" && !Number.isNaN(v)) {
          if (v > maxVal) maxVal = v;
          if (v < minVal) minVal = v;
        }
        if (s.bandLow && s.bandHigh) {
          const lo = s.bandLow(r);
          const hi = s.bandHigh(r);
          if (typeof hi === "number" && hi > maxVal) maxVal = hi;
          if (typeof lo === "number" && lo < minVal) minVal = lo;
        }
      }
    }
    if (maxVal <= minVal) maxVal = minVal + 1;
    const span = (maxVal - minVal) * 1.08;
    const yMax = minVal + span;

    const xForIdx = (i) =>
      padL + (records.length === 1 ? plotW * 0.5 : (i / (records.length - 1)) * plotW);
    const yForVal = (v) => padT + plotH - ((v - minVal) / (yMax - minVal)) * plotH;

    // Winter Season Bands
    if (options.showSeasonBands) {
      for (let i = 0; i < records.length; i++) {
        const r = records[i];
        if (r.season_id === "winter") {
          const x0 = i === 0 ? padL : (xForIdx(i - 1) + xForIdx(i)) * 0.5;
          const x1 = i === records.length - 1 ? padL + plotW : (xForIdx(i) + xForIdx(i + 1)) * 0.5;
          ctx.fillStyle = "rgba(56, 189, 248, 0.08)";
          ctx.fillRect(x0, padT, Math.max(2, x1 - x0), plotH);
        }
      }
    }

    // Horizontal Grid Lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
    ctx.lineWidth = 1;
    ctx.font = "10.5px JetBrains Mono, monospace";
    ctx.fillStyle = "#687b8f";
    ctx.textAlign = "right";
    const steps = 4;
    for (let s = 0; s <= steps; s++) {
      const frac = s / steps;
      const gy = padT + plotH - frac * plotH;
      const gVal = minVal + frac * (yMax - minVal);
      ctx.beginPath();
      ctx.moveTo(padL, gy);
      ctx.lineTo(padL + plotW, gy);
      ctx.stroke();
      const labelStr =
        Math.abs(yMax - minVal) >= 20
          ? String(Math.round(gVal))
          : Math.abs(yMax - minVal) >= 2
          ? gVal.toFixed(1)
          : gVal.toFixed(3);
      ctx.fillText(labelStr, padL - 6, gy + 3);
    }

    // X-axis Day Ticks
    ctx.textAlign = "center";
    const tickCount = Math.min(records.length, 8);
    for (let t = 0; t < tickCount; t++) {
      const idx = tickCount === 1 ? 0 : Math.round((t / (tickCount - 1)) * (records.length - 1));
      const rx = xForIdx(idx);
      ctx.fillText(`T${records[idx].day}`, rx, H - 8);
    }

    // Collapse Markers
    if (options.showCollapseMarkers && collapseLogs.length > 0) {
      const dayToIdx = new Map();
      records.forEach((r, idx) => dayToIdx.set(r.day, idx));
      for (const col of collapseLogs) {
        const idx = dayToIdx.get(col.day);
        if (typeof idx === "number") {
          const cx = xForIdx(idx);
          ctx.save();
          ctx.strokeStyle = "rgba(244, 63, 94, 0.75)";
          ctx.lineWidth = 1.5;
          ctx.setLineDash([4, 3]);
          ctx.beginPath();
          ctx.moveTo(cx, padT);
          ctx.lineTo(cx, padT + plotH);
          ctx.stroke();
          ctx.fillStyle = "#fda4af";
          ctx.font = "600 10px JetBrains Mono, monospace";
          ctx.fillText(`−${Math.round(col.drop_pct)}%`, cx, padT + 11);
          ctx.restore();
        }
      }
    }

    // HoF Markers
    if (options.showHofMarkers) {
      records.forEach((r, idx) => {
        const hof = (r.community && r.community.hof_rescued_blobs) || 0;
        if (hof > 0) {
          const cx = xForIdx(idx);
          ctx.save();
          ctx.strokeStyle = "rgba(192, 132, 252, 0.7)";
          ctx.lineWidth = 1.3;
          ctx.beginPath();
          ctx.moveTo(cx, padT);
          ctx.lineTo(cx, padT + plotH);
          ctx.stroke();
          ctx.restore();
        }
      });
    }

    // Percentile Ribbons
    for (const s of seriesList) {
      if (s.bandLow && s.bandHigh && records.length >= 2) {
        ctx.save();
        ctx.beginPath();
        for (let i = 0; i < records.length; i++) {
          const x = xForIdx(i);
          const y = yForVal(s.bandHigh(records[i]));
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        for (let i = records.length - 1; i >= 0; i--) {
          const x = xForIdx(i);
          const y = yForVal(s.bandLow(records[i]));
          ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.fillStyle = s.bandColor || "rgba(16, 185, 129, 0.14)";
        ctx.fill();
        ctx.restore();
      }
    }

    // Series Lines
    for (const s of seriesList) {
      ctx.save();
      ctx.beginPath();
      ctx.strokeStyle = s.color;
      ctx.lineWidth = s.lineWidth || 2.1;
      if (s.dashed) ctx.setLineDash([5, 4]);
      for (let i = 0; i < records.length; i++) {
        const x = xForIdx(i);
        const val = s.getter(records[i]);
        const y = yForVal(typeof val === "number" ? val : 0);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      if (records.length <= 28) {
        ctx.fillStyle = s.color;
        for (let i = 0; i < records.length; i++) {
          const x = xForIdx(i);
          const val = s.getter(records[i]);
          const y = yForVal(typeof val === "number" ? val : 0);
          ctx.beginPath();
          ctx.arc(x, y, 2.6, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.restore();
    }

    // Hover Crosshair
    if (typeof options.hoverIndex === "number" && options.hoverIndex >= 0 && options.hoverIndex < records.length) {
      const hx = xForIdx(options.hoverIndex);
      ctx.save();
      ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(hx, padT);
      ctx.lineTo(hx, padT + plotH);
      ctx.stroke();
      ctx.restore();
    }
  }

  function renderLegendDom(containerId, seriesList, latestRecord) {
    const el = document.getElementById(containerId);
    if (!el) return;
    el.replaceChildren();
    for (const s of seriesList) {
      const span = document.createElement("span");
      span.className = "legend-item";
      const sw = document.createElement("span");
      sw.className = "legend-swatch";
      sw.style.background = s.color;
      const val = latestRecord ? s.getter(latestRecord) : null;
      const valStr =
        typeof val === "number"
          ? Number.isInteger(val)
            ? String(val)
            : val.toFixed(2)
          : "";
      span.append(sw, document.createTextNode(valStr !== "" ? `${s.label}: ${valStr}` : s.label));
      el.appendChild(span);
    }
  }

  // --- TAB 1: Cycle & Collapse Timeline ---
  function renderCycleTimeline() {
    const mode = selectCycleChartMode ? selectCycleChartMode.value : "pop_dominance";
    let series = [];

    if (mode === "pop_dominance") {
      series = [
        { label: "Gesamt-Population", color: "#10b981", getter: (r) => r.population || 0, lineWidth: 2.5 },
        { label: "In Siedlung/Stamm", color: "#0ea5e9", getter: (r) => r.grouped_blobs || 0 },
        {
          label: "Größter Stamm (Mitglieder)",
          color: "#f59e0b",
          dashed: true,
          getter: (r) => {
            const top = r.active_tribes && r.active_tribes[0];
            return top ? top.members : 0;
          },
        },
      ];
    } else if (mode === "deaths_breakdown") {
      series = [
        { label: "Hunger-Tod Tagesende", color: "#f43f5e", getter: (r) => (r.community && r.community.deaths_eod_starve) || 0 },
        { label: "Energie-Tod Mittags", color: "#fb923c", getter: (r) => (r.community && r.community.deaths_midday_starve) || 0 },
        { label: "Tod im Freien", color: "#eab308", getter: (r) => (r.community && r.community.deaths_outdoors) || 0 },
        { label: "Tod trotz Gold (Fernab)", color: "#38bdf8", getter: (r) => (r.community && r.community.deaths_with_money_far) || 0 },
        { label: "Tod trotz Gold (Depot leer)", color: "#c084fc", getter: (r) => (r.community && r.community.deaths_with_money_empty_depot) || 0 },
      ];
    } else {
      series = [
        { label: "Haus-Depots Gesamt (🍏)", color: "#10b981", getter: (r) => (r.ecology && r.ecology.total_depot_food) || 0 },
        { label: "Überweidete Oasen (%)", color: "#f43f5e", getter: (r) => (r.ecology && r.ecology.overgrazed_oases_pct) || 0 },
        { label: "Ø Oasen-Fruchtbarkeit (%)", color: "#38bdf8", getter: (r) => Math.round(((r.ecology && r.ecology.avg_oasis_fertility) || 0) * 100) },
        { label: "Geburten", color: "#fbbf24", getter: (r) => r.births || 0 },
      ];
    }

    drawMultiLineChart(cycleCtx, cycleCanvas, dailyRecords, series, {
      showSeasonBands: true,
      showCollapseMarkers: true,
      showHofMarkers: true,
      hoverIndex: cycleHoverIndex,
    });

    const hoverReadout = document.getElementById("cycleTimelineHoverReadout");
    if (hoverReadout && dailyRecords.length > 0) {
      const idx = cycleHoverIndex >= 0 && cycleHoverIndex < dailyRecords.length ? cycleHoverIndex : dailyRecords.length - 1;
      const r = dailyRecords[idx];
      const c = r.community || {};
      const m = r.monoculture || {};
      hoverReadout.firstElementChild.textContent =
        `[Tag ${r.day} · ${r.season_label || r.season_id}] Pop: ${r.population} (Siedlung: ${r.grouped_blobs}) · ` +
        `Stämme: ${r.groups_count} (Dom: ${m.dominant_tribe_name || "-"} ${Math.round(m.dominant_share_pct || 0)}%) · ` +
        `Geburten: ${r.births || 0} · Tode: ${r.deaths || 0} (Abend: ${c.deaths_eod_starve || 0}, Mittag: ${c.deaths_midday_starve || 0}) · ` +
        `Magen-Verkäufe: ${c.stomach_sells_total || 0}`;
    }
  }

  function renderCollapseAutopsyCards() {
    const badge = document.getElementById("collapseCountBadge");
    if (badge) badge.textContent = `${collapseLogs.length} Kollapse protokolliert`;
    const container = document.getElementById("collapseCardsContainer");
    if (!container) return;
    container.replaceChildren();

    if (collapseLogs.length === 0) {
      const empty = document.createElement("div");
      empty.className = "empty-state-box";
      empty.textContent =
        "Noch kein Bevölkerungseinbruch aufgetreten. Nutze oben '+10 Tage' oder erhöhe die Geschwindigkeit, um Rise-&-Collapse-Zyklen im Zeitraffer aufzuzeichnen.";
      container.appendChild(empty);
      return;
    }

    const reversed = collapseLogs.slice().reverse();
    for (const c of reversed) {
      const card = document.createElement("div");
      card.className = "collapse-card";

      const top = document.createElement("div");
      top.className = "collapse-card-top";
      const title = document.createElement("strong");
      title.textContent = `Peak Tag ${c.peak_day} → Kollaps Tag ${c.day} (${c.season_label})`;
      const dropBadge = document.createElement("span");
      dropBadge.className = "collapse-drop-badge";
      dropBadge.textContent = `−${c.drop_pct}% (${c.peak_population} → ${c.post_population})`;
      top.append(title, dropBadge);

      const causePill = document.createElement("div");
      causePill.className = "collapse-cause-pill";
      causePill.textContent = `Hauptfaktoren: ${(c.causes || []).join(" · ")}`;

      const grid = document.createElement("div");
      grid.className = "collapse-metrics-grid";
      const metrics = [
        ["Dominanter Stamm", `${c.dominant_tribe} (${Math.round(c.dominant_share_pct || 0)}%)`],
        ["Gehirn-Diversität σ", String(c.brain_diversity_std)],
        ["Tode (Abend / Mittag)", `${c.deaths_eod} / ${c.deaths_midday}`],
        ["Tode trotz Gold", `${c.deaths_with_money_far} fern / ${c.deaths_with_money_empty_depot} leer`],
        ["Magen-Notverkäufe", String(c.stomach_sells)],
        ["Nachts im Freien", `${c.eod_outdoors_pct}%`],
        ["Ø Speed / Grundumsatz", `${c.avg_speed}x / ${c.avg_basal_cost}`],
        ["HoF Not-Respawns", String(c.hof_rescued || 0)],
      ];
      for (const [k, v] of metrics) {
        const item = document.createElement("div");
        item.textContent = `${k}: `;
        const st = document.createElement("strong");
        st.textContent = v;
        item.appendChild(st);
        grid.appendChild(item);
      }

      card.append(top, causePill, grid);
      container.appendChild(card);
    }
  }

  // --- TAB 2: Community Actions & Tribes ---
  function renderCommunityActionsChart() {
    const mode = selectCommunityActionMode ? selectCommunityActionMode.value : "trade_ration";
    let series = [];

    if (mode === "trade_ration") {
      series = [
        {
          label: "Peer-Handel",
          color: "#10b981",
          getter: (r) =>
            ((r.community && r.community.trades_peer_same_group) || 0) +
            ((r.community && r.community.trades_peer_cross_group) || 0),
        },
        {
          label: "Markt-Verkäufe",
          color: "#0ea5e9",
          getter: (r) =>
            ((r.community && r.community.trades_market_sell_surplus) || 0) +
            ((r.community && r.community.trades_market_sell_stomach) || 0),
        },
        {
          label: "⚠️ Magen-Verkauf (< Survival)",
          color: "#f43f5e",
          getter: (r) => (r.community && r.community.stomach_sells_total) || 0,
        },
        {
          label: "Depot-Rettungen (Tag+Abend)",
          color: "#f59e0b",
          getter: (r) =>
            ((r.community && r.community.trades_emergency_midday) || 0) +
            ((r.community && r.community.trades_eod_deficit_buy) || 0),
        },
      ];
    } else if (mode === "foraging_biomes") {
      series = [
        { label: "Wiegen-Oasen", color: "#10b981", getter: (r) => (r.community && r.community.foraged_cradle_count) || 0 },
        { label: "Frontier-Oasen", color: "#38bdf8", getter: (r) => (r.community && r.community.foraged_frontier_count) || 0 },
        { label: "Goldene Äpfel", color: "#fbbf24", getter: (r) => (r.community && r.community.foraged_golden_count) || 0 },
        { label: "Hausgärten", color: "#a3e635", getter: (r) => (r.community && r.community.foraged_garden_count) || 0 },
      ];
    } else if (mode === "conflict_settle") {
      series = [
        { label: "Raids gewonnen", color: "#f43f5e", getter: (r) => (r.community && r.community.raids_won) || 0 },
        { label: "Raids abgewehrt", color: "#38bdf8", getter: (r) => (r.community && r.community.raids_repelled) || 0 },
        { label: "Verbannungen (Exil)", color: "#e11d48", getter: (r) => (r.community && r.community.exiles) || 0 },
        { label: "Häuser erobert", color: "#fb923c", getter: (r) => (r.community && r.community.houses_conquered) || 0 },
        { label: "Sezessionen", color: "#c084fc", getter: (r) => (r.community && r.community.secessions) || 0 },
        { label: "Haus-Upgrades", color: "#10b981", getter: (r) => (r.community && r.community.upgrades_total) || 0 },
      ];
    } else {
      series = [
        { label: "Aktiver Sprint (%)", color: "#f43f5e", getter: (r) => (r.actions && r.actions.active_sprint_pct) || 0 },
        { label: "Aktiver Handel (%)", color: "#f59e0b", getter: (r) => (r.actions && r.actions.active_trade_pct) || 0 },
        { label: "Aktive Sozialität (%)", color: "#0ea5e9", getter: (r) => (r.actions && r.actions.active_social_pct) || 0 },
        { label: "Aktives Siedeln (%)", color: "#c084fc", getter: (r) => (r.actions && r.actions.active_colonize_pct) || 0 },
        { label: "Abends im Haus (%)", color: "#10b981", getter: (r) => (r.community && r.community.eod_at_home_pct) || 0 },
      ];
    }

    drawMultiLineChart(commCtx, commCanvas, dailyRecords, series, { showSeasonBands: true });
    renderLegendDom("communityActionsLegend", series, dailyRecords[dailyRecords.length - 1]);
  }

  function renderTribesTable() {
    const tbody = document.getElementById("tribesTableBody");
    if (!tbody) return;
    const filterMode = selectTribeFilter ? selectTribeFilter.value : "all";
    let list = tribesList.slice();
    if (filterMode === "alive") list = list.filter((t) => t.alive);
    if (filterMode === "extinct") list = list.filter((t) => !t.alive);

    tbody.replaceChildren();
    if (list.length === 0) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 9;
      td.style.textAlign = "center";
      td.style.color = "var(--text-muted)";
      td.style.padding = "18px";
      td.textContent = "Noch keine Stämme für diesen Filter vorhanden — Stämme entstehen sobald Blobs Siedlungen gründen.";
      tr.appendChild(td);
      tbody.appendChild(tr);
      return;
    }

    if (!selectedTribeId || !tribesById[String(selectedTribeId)]) {
      selectedTribeId = list[0].id;
    }

    for (const t of list) {
      const tr = document.createElement("tr");
      tr.tabIndex = 0;
      if (String(t.id) === String(selectedTribeId)) {
        tr.classList.add("selected-row");
      }

      const th = document.createElement("th");
      th.scope = "row";
      const dot = document.createElement("span");
      dot.className = "group-dot";
      dot.style.backgroundColor = t.color || "#10b981";
      const secFlag = t.parent_group_id ? " 🚩" : "";
      th.append(dot, document.createTextNode(`#${t.id} ${t.name}${secFlag}`));

      const tdStatus = document.createElement("td");
      const pill = document.createElement("span");
      pill.className = `status-pill ${t.alive ? "alive" : "extinct"}`;
      pill.textContent = t.alive ? "AKTIV" : `ERLOSCHEN (T${t.extinct_day})`;
      tdStatus.appendChild(pill);

      const tdSpan = document.createElement("td");
      tdSpan.textContent = t.alive ? `ab T${t.created_day} → heute` : `T${t.created_day} → T${t.extinct_day}`;

      const tdPop = document.createElement("td");
      tdPop.textContent = `${t.peak_members} Peak (${t.current_members || 0} akt.)`;

      const tdHomes = document.createElement("td");
      tdHomes.textContent = `${t.peak_homes} Peak (${t.current_homes || 0} akt.)`;

      const tdVital = document.createElement("td");
      tdVital.textContent = `${t.total_births || 0}👶 / ${t.total_deaths || 0}💀 (${t.collapse_count || 0}📉)`;

      const tdTrades = document.createElement("td");
      const totTr = (t.total_trades_internal || 0) + (t.total_trades_external || 0) + (t.total_trades_market || 0);
      tdTrades.textContent = `${totTr}🤝 (${t.total_stomach_sells || 0}⚠️)`;

      const tdConflict = document.createElement("td");
      tdConflict.textContent = `${t.total_raids_won || 0}⚔️ / ${t.total_conquests || 0}🏴 / ${t.total_secessions || 0}🚩`;

      const tdGenes = document.createElement("td");
      const lg = t.last_genes || { speed: 1, size: 1, basal_cost: 0.06 };
      tdGenes.textContent = `${(lg.speed || 1).toFixed(2)}x · ${(lg.size || 1).toFixed(2)}x · ${(lg.basal_cost || 0.06).toFixed(4)}`;

      tr.append(th, tdStatus, tdSpan, tdPop, tdHomes, tdVital, tdTrades, tdConflict, tdGenes);
      tr.addEventListener("click", () => {
        selectedTribeId = t.id;
        renderTribesTable();
        renderTribeDetailChart();
      });
      tbody.appendChild(tr);
    }
  }

  function renderTribeDetailChart() {
    const t = selectedTribeId ? tribesById[String(selectedTribeId)] : null;
    const titleEl = document.getElementById("tribeDetailTitle");
    const badgeEl = document.getElementById("tribeDetailBadge");

    if (!t || !t.history || t.history.length === 0) {
      if (tribeCtx && tribeCanvas) {
        tribeCtx.clearRect(0, 0, tribeCanvas.width, tribeCanvas.height);
        tribeCtx.fillStyle = "#687b8f";
        tribeCtx.font = "12px Inter, sans-serif";
        tribeCtx.textAlign = "center";
        tribeCtx.fillText(
          "Sobald Stämme gegründet werden, erscheint hier die Stammes-Zeitreihe.",
          tribeCanvas.width / 2,
          tribeCanvas.height / 2
        );
      }
      return;
    }

    if (titleEl) titleEl.textContent = `🏛️ Stammes-Chronik: #${t.id} ${t.name}`;
    if (badgeEl) {
      badgeEl.textContent = t.alive
        ? `Aktiv (${t.current_members} Blobs · Peak ${t.peak_members})`
        : `Ausgestorben an Tag ${t.extinct_day} (Peak ${t.peak_members})`;
    }

    const series = [
      { label: "Mitglieder", color: t.color || "#10b981", getter: (r) => r.members || 0, lineWidth: 2.5 },
      { label: "Häuser (×4)", color: "#0ea5e9", getter: (r) => (r.homes || 0) * 4 },
      { label: "Depot-Futter (🍏)", color: "#fbbf24", getter: (r) => r.depot_food || 0 },
      { label: "Tode/Tag", color: "#f43f5e", getter: (r) => r.deaths || 0 },
    ];
    drawMultiLineChart(tribeCtx, tribeCanvas, t.history, series, {});
    renderLegendDom("tribeDetailSummary", series, t.history[t.history.length - 1]);
  }

  // --- TAB 3: Genetic Evolution & Bio-Energetic Scatter ---
  function renderGeneticsChart() {
    const mode = selectGeneticsMode ? selectGeneticsMode.value : "all_means";
    let series = [];

    if (mode === "all_means") {
      series = [
        { label: "Ø Speed", color: "#f43f5e", getter: (r) => (r.genetics && r.genetics.speed ? r.genetics.speed.mean : 1) },
        { label: "Ø Sichtweite", color: "#0ea5e9", getter: (r) => (r.genetics && r.genetics.sense ? r.genetics.sense.mean : 1) },
        { label: "Ø Größe", color: "#10b981", getter: (r) => (r.genetics && r.genetics.size ? r.genetics.size.mean : 1) },
        { label: "Ø Metabolismus-Faktor", color: "#f59e0b", getter: (r) => (r.genetics && r.genetics.metabolism ? r.genetics.metabolism.mean : 1) },
      ];
    } else if (mode === "basal_cost") {
      series = [
        {
          label: "Basaler Grundumsatz Mean (p10–p90 Band)",
          color: "#f43f5e",
          bandColor: "rgba(244, 63, 94, 0.16)",
          getter: (r) => (r.genetics && r.genetics.basal_cost ? r.genetics.basal_cost.mean : 0.06),
          bandLow: (r) => (r.genetics && r.genetics.basal_cost ? r.genetics.basal_cost.p10 : 0.05),
          bandHigh: (r) => (r.genetics && r.genetics.basal_cost ? r.genetics.basal_cost.p90 : 0.07),
        },
        {
          label: "Median (p50)",
          color: "#fbbf24",
          dashed: true,
          getter: (r) => (r.genetics && r.genetics.basal_cost ? r.genetics.basal_cost.p50 : 0.06),
        },
      ];
    } else if (mode === "speed" || mode === "size" || mode === "sense") {
      const colors = { speed: "#f43f5e", size: "#10b981", sense: "#0ea5e9" };
      const col = colors[mode];
      series = [
        {
          label: `${mode.toUpperCase()} Mean (p10–p90 Band)`,
          color: col,
          bandColor: "rgba(56, 189, 248, 0.15)",
          getter: (r) => (r.genetics && r.genetics[mode] ? r.genetics[mode].mean : 1),
          bandLow: (r) => (r.genetics && r.genetics[mode] ? r.genetics[mode].p10 : 0.9),
          bandHigh: (r) => (r.genetics && r.genetics[mode] ? r.genetics[mode].p90 : 1.1),
        },
        {
          label: "Median (p50)",
          color: "#fbbf24",
          dashed: true,
          getter: (r) => (r.genetics && r.genetics[mode] ? r.genetics[mode].p50 : 1),
        },
      ];
    } else {
      series = [
        { label: "Ø Generation", color: "#10b981", getter: (r) => (r.genetics && r.genetics.generation ? r.genetics.generation.mean : 1) },
        { label: "Max Generation", color: "#38bdf8", getter: (r) => (r.genetics && r.genetics.generation ? r.genetics.generation.max : 1) },
        { label: "Ø Alter (Tage)", color: "#f59e0b", getter: (r) => (r.genetics && r.genetics.age_days ? r.genetics.age_days.mean : 0) },
      ];
    }

    drawMultiLineChart(genCtx, genCanvas, dailyRecords, series, { showSeasonBands: true });
    renderLegendDom("geneticsChartLegend", series, dailyRecords[dailyRecords.length - 1]);
  }

  function renderGeneScatterPlot() {
    if (!scatterCtx || !scatterCanvas) return;
    const W = scatterCanvas.width;
    const H = scatterCanvas.height;
    scatterCtx.clearRect(0, 0, W, H);

    const countBadge = document.getElementById("scatterCountBadge");
    if (countBadge) countBadge.textContent = `${scatterBlobs.length} Blobs`;

    const padL = 46, padR = 18, padT = 20, padB = 30;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const minX = 0.55, maxX = 2.25;
    const minY = 0.65, maxY = 1.95;

    // Background Metabolic Danger Zone Shading
    const dangerX = padL + ((1.35 - minX) / (maxX - minX)) * plotW;
    const dangerY = padT + plotH - ((1.35 - minY) / (maxY - minY)) * plotH;
    scatterCtx.fillStyle = "rgba(244, 63, 94, 0.08)";
    scatterCtx.fillRect(dangerX, padT, padL + plotW - dangerX, dangerY - padT);

    // Axes & Grid
    scatterCtx.strokeStyle = "rgba(255, 255, 255, 0.08)";
    scatterCtx.lineWidth = 1;
    scatterCtx.strokeRect(padL, padT, plotW, plotH);

    scatterCtx.fillStyle = "#687b8f";
    scatterCtx.font = "10.5px JetBrains Mono, monospace";
    scatterCtx.textAlign = "center";
    scatterCtx.fillText("Speed-Gen (0.6x → 2.2x)", padL + plotW * 0.5, H - 7);

    scatterCtx.save();
    scatterCtx.translate(14, padT + plotH * 0.5);
    scatterCtx.rotate(-Math.PI / 2);
    scatterCtx.fillText("Größen-Gen (0.7x → 1.9x)", 0, 0);
    scatterCtx.restore();

    for (const b of scatterBlobs) {
      const px = padL + Math.max(0, Math.min(1, (b.speed - minX) / (maxX - minX))) * plotW;
      const py = padT + plotH - Math.max(0, Math.min(1, (b.size - minY) / (maxY - minY))) * plotH;
      const cost = b.cost || 0.06;
      scatterCtx.beginPath();
      scatterCtx.arc(px, py, 3.6, 0, Math.PI * 2);
      scatterCtx.fillStyle = cost > 0.075 ? "#f43f5e" : cost > 0.055 ? "#f59e0b" : "#10b981";
      scatterCtx.fill();
      scatterCtx.strokeStyle = "rgba(15, 23, 42, 0.85)";
      scatterCtx.lineWidth = 1;
      scatterCtx.stroke();
    }
  }

  // --- TAB 4: Neural Weights, Pathways & 26x6 Coupling Heatmap ---
  function renderPathwayChart() {
    const groupMode = selectPathwayGroup ? selectPathwayGroup.value : "core";
    const defs =
      pathwayDefs && pathwayDefs.length > 0
        ? pathwayDefs
        : [
            { id: "food_nav", label: "🍏 Nahrungs-Navigation", color: "#10b981" },
            { id: "home_nav", label: "🏠 Heimkehr-Disziplin", color: "#0ea5e9" },
            { id: "sprint_drive", label: "⚡ Sprint-Aktivierung", color: "#eab308" },
            { id: "trade_drive", label: "🪙 Handels-Trieb", color: "#f97316" },
            { id: "social_drive", label: "🤝 Gruppen-Sozialität", color: "#3b82f6" },
            { id: "colonize_drive", label: "🏕️ Pionier-Siedlungsdrang", color: "#f43f5e" },
            { id: "reputation_drive", label: "⚖️ Reputations-Sensitivität", color: "#ec4899" },
          ];

    const coreIds = new Set([
      "food_nav",
      "home_nav",
      "sprint_drive",
      "trade_drive",
      "social_drive",
      "colonize_drive",
      "reputation_drive",
    ]);
    const filtered = defs.filter((p) => {
      if (groupMode === "core") return coreIds.has(p.id);
      if (groupMode === "navigation") return !coreIds.has(p.id);
      return true;
    });

    const series = filtered.map((p) => ({
      label: p.label,
      color: p.color || "#10b981",
      getter: (r) =>
        r.weights && r.weights.pathways && typeof r.weights.pathways[p.id] === "number"
          ? r.weights.pathways[p.id]
          : 0,
    }));

    drawMultiLineChart(pathwayCtx, pathwayCanvas, dailyRecords, series, { showSeasonBands: true });
    renderLegendDom("pathwayChartLegend", series, dailyRecords[dailyRecords.length - 1]);
  }

  function renderWeightMetricsChart() {
    const mode = selectWeightMetricsMode ? selectWeightMetricsMode.value : "drift_diversity";
    let series = [];

    if (mode === "drift_diversity") {
      series = [
        { label: "Weight-Drift ab Tag 1", color: "#38bdf8", getter: (r) => (r.weights && r.weights.weight_drift_day1) || 0 },
        { label: "Gehirn-Diversität (σ)", color: "#10b981", getter: (r) => (r.weights && r.weights.brain_diversity_std) || 0 },
        { label: "Täglicher Weight-Wandel (×10)", color: "#f59e0b", getter: (r) => ((r.weights && r.weights.weight_velocity_daily) || 0) * 10 },
      ];
    } else if (mode === "layer_norms") {
      series = [0, 1, 2].map((lIdx) => ({
        label: `Layer ${lIdx + 1} L2-Norm`,
        color: lIdx === 0 ? "#10b981" : lIdx === 1 ? "#0ea5e9" : "#f59e0b",
        getter: (r) => {
          const layers = (r.weights && r.weights.layers) || [];
          return layers[lIdx] ? layers[lIdx].l2_norm : 0;
        },
      }));
    } else {
      series = [0, 1, 2].map((lIdx) => ({
        label: `Layer ${lIdx + 1} Sättigung (% |W|≥3.0)`,
        color: lIdx === 0 ? "#f43f5e" : lIdx === 1 ? "#fb923c" : "#c084fc",
        getter: (r) => {
          const layers = (r.weights && r.weights.layers) || [];
          return layers[lIdx] ? layers[lIdx].saturation_pct : 0;
        },
      }));
    }

    drawMultiLineChart(weightMetricsCtx, weightMetricsCanvas, dailyRecords, series, { showSeasonBands: true });
    renderLegendDom("weightMetricsLegend", series, dailyRecords[dailyRecords.length - 1]);
  }

  function renderCouplingHeatmap() {
    if (!heatmapCtx || !heatmapCanvas || dailyRecords.length === 0) return;
    const W = heatmapCanvas.width;
    const H = heatmapCanvas.height;
    heatmapCtx.clearRect(0, 0, W, H);

    const sliderIdx = heatmapDaySlider
      ? Math.max(0, Math.min(dailyRecords.length - 1, parseInt(heatmapDaySlider.value, 10) || 0))
      : dailyRecords.length - 1;
    const targetRec = dailyRecords[sliderIdx];
    const day1Rec = dailyRecords[0];

    const matNow = (targetRec && targetRec.weights && targetRec.weights.coupling_matrix) || [];
    const matDay1 =
      initialCouplingMatrix ||
      (day1Rec && day1Rec.weights && day1Rec.weights.coupling_matrix) ||
      matNow;
    if (matNow.length === 0) return;

    const isDelta = selectHeatmapMode && selectHeatmapMode.value === "delta_day1";

    const numInputs = matNow.length || 26;
    const inLabels =
      inputLabels && inputLabels.length === numInputs
        ? inputLabels
        : Array.from({ length: numInputs }, (_, i) => `Input #${i}`);
    const outLabels =
      outputLabels && outputLabels.length === 6
        ? outputLabels
        : ["Bewegung X", "Bewegung Y", "Sprint", "Handel", "Gruppe", "Siedeln"];

    const padL = 175, padT = 36, padR = 16, padB = 12;
    const gridW = W - padL - padR;
    const gridH = H - padT - padB;
    const cellW = gridW / 6;
    const cellH = gridH / numInputs;

    let maxAbs = 0.15;
    for (let i = 0; i < numInputs; i++) {
      for (let j = 0; j < 6; j++) {
        const vNow = (matNow[i] && matNow[i][j]) || 0;
        const vBase = (matDay1[i] && matDay1[i][j]) || 0;
        const v = isDelta ? vNow - vBase : vNow;
        if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
      }
    }

    // Column Headers (6 Action Outputs)
    heatmapCtx.font = "600 11px Inter, sans-serif";
    heatmapCtx.fillStyle = "#eef2f6";
    heatmapCtx.textAlign = "center";
    for (let j = 0; j < 6; j++) {
      heatmapCtx.fillText(outLabels[j], padL + (j + 0.5) * cellW, 22);
    }

    // Rows (26 Sensory Inputs) & Cells
    for (let i = 0; i < numInputs; i++) {
      const cy = padT + i * cellH;
      heatmapCtx.font = "10.5px Inter, sans-serif";
      heatmapCtx.fillStyle = "#9fb0c2";
      heatmapCtx.textAlign = "right";
      heatmapCtx.fillText(`${i}. ${inLabels[i]}`, padL - 8, cy + cellH * 0.72);

      for (let j = 0; j < 6; j++) {
        const cx = padL + j * cellW;
        const vNow = (matNow[i] && matNow[i][j]) || 0;
        const vBase = (matDay1[i] && matDay1[i][j]) || 0;
        const val = isDelta ? vNow - vBase : vNow;
        const norm = Math.max(-1, Math.min(1, val / maxAbs));

        if (norm >= 0) {
          const alpha = 0.08 + norm * 0.78;
          heatmapCtx.fillStyle = `rgba(16, 185, 129, ${alpha.toFixed(3)})`;
        } else {
          const alpha = 0.08 + Math.abs(norm) * 0.78;
          heatmapCtx.fillStyle = `rgba(244, 63, 94, ${alpha.toFixed(3)})`;
        }
        heatmapCtx.fillRect(cx + 1, cy + 1, cellW - 2, cellH - 2);

        if (heatmapHoverCell && heatmapHoverCell.row === i && heatmapHoverCell.col === j) {
          heatmapCtx.strokeStyle = "#38bdf8";
          heatmapCtx.lineWidth = 2;
          heatmapCtx.strokeRect(cx + 1, cy + 1, cellW - 2, cellH - 2);
        }

        heatmapCtx.font = "9.5px JetBrains Mono, monospace";
        heatmapCtx.fillStyle = Math.abs(norm) > 0.35 ? "#ffffff" : "#94a3b8";
        heatmapCtx.textAlign = "center";
        heatmapCtx.fillText(val >= 0 ? `+${val.toFixed(2)}` : val.toFixed(2), cx + cellW * 0.5, cy + cellH * 0.74);
      }
    }

    const readout = document.getElementById("heatmapHoverReadout");
    if (readout && heatmapHoverCell) {
      const { row, col } = heatmapHoverCell;
      const vNow = (matNow[row] && matNow[row][col]) || 0;
      const vBase = (matDay1[row] && matDay1[row][col]) || 0;
      const diff = vNow - vBase;
      readout.firstElementChild.textContent =
        `Sensor [${row}] "${inLabels[row]}" → Aktion [${col}] "${outLabels[col]}" | ` +
        `Tag ${targetRec.day}: ${vNow >= 0 ? "+" : ""}${vNow.toFixed(4)} | Tag 1: ${vBase >= 0 ? "+" : ""}${vBase.toFixed(4)} | ` +
        `Δ Drift: ${diff >= 0 ? "+" : ""}${diff.toFixed(4)}`;
    }
  }

  // --- Render Live Champion Brain Topology on #cockpitNnCanvas ---
  function renderCockpitNeuralNet(selectedObj) {
    if (!cockpitNnCtx || !cockpitNnCanvas || !selectedObj || !selectedObj.brain) return;
    const b = selectedObj.blob;
    const brain = selectedObj.brain;

    if (b) {
      document.getElementById("cockpitBrainTitle").textContent = `🧠 Live Gehirn: Blob #${b.id} (Gen ${b.gen})`;
      document.getElementById("ckInspGroup").textContent = selectedObj.group_name || "Einzelgänger";
      document.getElementById("ckInspFoodAge").textContent = `${b.food_today.toFixed(2)}🍏 · ${b.age_days}d`;
      document.getElementById("ckInspMoney").textContent = `${b.money.toFixed(1)}🪙 (+${b.surplus_food.toFixed(1)}🍏)`;
      document.getElementById("ckInspGenes").textContent =
        `${b.genes.speed.toFixed(2)}x / ${b.genes.sense.toFixed(2)}x / ${b.genes.size.toFixed(2)}x`;
    }
    document.getElementById("cockpitBrainParamBadge").textContent =
      `${brain.param_count} Synapsen (${(brain.activation || "tanh").toUpperCase()})`;

    const W = cockpitNnCanvas.width;
    const H = cockpitNnCanvas.height;
    cockpitNnCtx.clearRect(0, 0, W, H);

    const acts = brain.activations || [];
    const weights = brain.weights || [];
    if (acts.length === 0) return;

    const numLayers = acts.length;
    const padX = 64, padY = 12;
    const usableW = W - padX * 2;
    const usableH = H - padY * 2;
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
          cockpitNnCtx.strokeStyle =
            wVal >= 0 ? `rgba(16, 185, 129, ${alpha})` : `rgba(244, 63, 94, ${alpha})`;
          cockpitNnCtx.lineWidth = Math.min(2.2, 0.5 + signal * 1.4);
          cockpitNnCtx.beginPath();
          cockpitNnCtx.moveTo(srcNodes[i].x, srcNodes[i].y);
          cockpitNnCtx.lineTo(dstNodes[j].x, dstNodes[j].y);
          cockpitNnCtx.stroke();
        }
      }
    }

    const outLabels = ["Bew X", "Bew Y", "Sprint", "Handel", "Gruppe", "Siedeln"];
    for (let l = 0; l < nodePositions.length; l++) {
      const nodes = nodePositions[l];
      const r = l === 0 ? 3.2 : l === nodePositions.length - 1 ? 4.0 : 3.4;
      for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        const v = Math.max(-1, Math.min(1, n.val));
        cockpitNnCtx.beginPath();
        cockpitNnCtx.arc(n.x, n.y, r, 0, Math.PI * 2);
        cockpitNnCtx.fillStyle =
          v >= 0
            ? `rgb(16, ${Math.round(40 + v * 195)}, 129)`
            : `rgb(${Math.round(40 + Math.abs(v) * 195)}, 63, 94)`;
        cockpitNnCtx.fill();
        if (l === nodePositions.length - 1 && outLabels[i]) {
          cockpitNnCtx.fillStyle = "#cbd5e1";
          cockpitNnCtx.font = "10px Inter, sans-serif";
          cockpitNnCtx.textAlign = "left";
          cockpitNnCtx.fillText(outLabels[i], n.x + 7, n.y + 3);
        }
      }
    }
  }

  // --- TAB 5: Daily Telemetry Table & Event Log ---
  function renderDailyTelemetryTable() {
    const tbody = document.getElementById("dailyLogTableBody");
    const badge = document.getElementById("dailyLogCountBadge");
    if (badge) badge.textContent = `${dailyRecords.length} Tage`;
    if (!tbody) return;
    tbody.replaceChildren();

    const reversed = dailyRecords.slice().reverse().slice(0, 180);
    for (const r of reversed) {
      const tr = document.createElement("tr");
      const c = r.community || {};
      const g = r.genetics || {};
      const w = r.weights || {};
      const m = r.monoculture || {};

      const th = document.createElement("th");
      th.scope = "row";
      th.textContent = `Tag ${r.day}`;

      const tdSeason = document.createElement("td");
      tdSeason.textContent = r.season_label || r.season_id;

      const tdPop = document.createElement("td");
      tdPop.textContent = `${r.population} (${r.grouped_blobs}🏠)`;

      const tdTribes = document.createElement("td");
      tdTribes.textContent = `${r.groups_count} (${m.dominant_tribe_name || "-"} ${Math.round(m.dominant_share_pct || 0)}%)`;

      const tdVital = document.createElement("td");
      const goldDeaths = (c.deaths_with_money_far || 0) + (c.deaths_with_money_empty_depot || 0);
      tdVital.textContent = `+${r.births || 0}👶 / −${r.deaths || 0}💀 (Abend:${c.deaths_eod_starve || 0}, 🪙:${goldDeaths})`;

      const tdTrade = document.createElement("td");
      const stomach = c.stomach_sells_total || 0;
      const rations = (c.trades_emergency_midday || 0) + (c.trades_eod_deficit_buy || 0);
      tdTrade.textContent = `${r.trades || 0}🤝 (${stomach}⚠️ / ${rations}🛟)`;

      const tdRaid = document.createElement("td");
      tdRaid.textContent = `${c.raids_won || 0}⚔️ / ${c.houses_conquered || 0}🏴 / ${c.upgrades_total || 0}🏗️`;

      const tdGene = document.createElement("td");
      const spd = g.speed ? g.speed.mean : 1;
      const siz = g.size ? g.size.mean : 1;
      const cst = g.basal_cost ? g.basal_cost.mean : 0.06;
      tdGene.textContent = `${spd.toFixed(2)}x · ${siz.toFixed(2)}x · ${cst.toFixed(4)}`;

      const tdBrain = document.createElement("td");
      tdBrain.textContent = `Δ${(w.weight_drift_day1 || 0).toFixed(3)} · σ${(w.brain_diversity_std || 0).toFixed(3)}`;

      tr.append(th, tdSeason, tdPop, tdTribes, tdVital, tdTrade, tdRaid, tdGene, tdBrain);
      tbody.appendChild(tr);
    }
  }

  function renderEventLog() {
    const container = document.getElementById("cockpitEventLogContainer");
    if (!container) return;
    const query = eventSearchInput ? eventSearchInput.value.trim().toLowerCase() : "";
    const cat = eventCategoryFilter ? eventCategoryFilter.value : "all";

    const filtered = (eventsList || []).filter((ev) => {
      const msg = (ev.msg || "").toLowerCase();
      const kind = (ev.kind || "").toLowerCase();
      if (query && !msg.includes(query)) return false;
      if (cat === "collapse") {
        return (
          kind === "collapse" ||
          kind === "hof_rescue" ||
          kind === "extinction" ||
          msg.includes("winter") ||
          msg.includes("hall of fame") ||
          msg.includes("kollaps") ||
          msg.includes("ausgestorben")
        );
      }
      if (cat === "tribe") {
        return msg.includes("stamm") || msg.includes("siedlung") || msg.includes("schließt sich");
      }
      if (cat === "conflict") {
        return msg.includes("raubzug") || msg.includes("erobert") || msg.includes("spaltet") || msg.includes("wehrt");
      }
      if (cat === "upgrade") {
        return msg.includes("baut") || msg.includes("pionier") || msg.includes("zieht") || msg.includes("oase");
      }
      return true;
    });

    container.replaceChildren();
    if (filtered.length === 0) {
      const empty = document.createElement("div");
      empty.className = "empty-state-box";
      empty.textContent = "Keine Ereignisse für diesen Filter gefunden.";
      container.appendChild(empty);
      return;
    }

    for (const ev of filtered.slice().reverse().slice(0, 200)) {
      const row = document.createElement("div");
      row.className = "cockpit-event-row";
      const daySpan = document.createElement("strong");
      daySpan.style.fontFamily = "var(--font-mono)";
      daySpan.style.color =
        ev.severity === "danger"
          ? "#f43f5e"
          : ev.severity === "warning"
          ? "#f59e0b"
          : "var(--accent-sky)";
      daySpan.textContent = `[Tag ${ev.day}]`;
      row.append(daySpan, document.createTextNode(ev.msg));
      container.appendChild(row);
    }
  }

  // --- Connect Live WebSocket for Header Status & Automatic Day Refresh ---
  function connectWebSocket() {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${window.location.host}/ws`);
    ws.binaryType = "arraybuffer";

    let lastSeenStateDay = -1;

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) return;
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "state" && msg.data) {
          const st = msg.data;
          liveSimState = st;

          if (typeof msg.running === "boolean" && msg.running !== isRunning) {
            isRunning = msg.running;
            updatePlayPauseButton();
          }
          if (typeof msg.speed === "number" && msg.speed !== currentSpeed) {
            updateActiveSpeedButton(msg.speed);
          }

          if (anDayBadge) anDayBadge.textContent = `Tag ${st.day}`;
          if (anSeasonBadge && st.season) {
            anSeasonBadge.textContent = `${st.season.label} (${st.season.day_in_season}/4)`;
          }
          if (anTickLabel) {
            anTickLabel.textContent = `${st.tick_in_day} / ${st.ticks_per_day}`;
          }
          if (anTimeProgressFill) {
            anTimeProgressFill.style.width = `${Math.min(100, Math.round((st.time_of_day || 0) * 100))}%`;
          }
          if (anPhaseLabel) {
            anPhaseLabel.textContent = st.is_night
              ? "🌙 Nacht (Fortpflanzung & Abrechnung)"
              : st.time_of_day > 0.56
              ? "🌇 Dämmerung (Heimkehr)"
              : `☀️ Tag (${st.stats ? st.stats.population : 0} Blobs aktiv)`;
          }

          if (st.selected && st.selected.brain) {
            selectedInspector = st.selected;
            if (activeTab === "tab-weights") {
              renderCockpitNeuralNet(st.selected);
            }
          }

          if (st.day !== lastSeenStateDay) {
            lastSeenStateDay = st.day;
            fetchAnalyticsPayload(true);
          }
        }
      } catch (err) {
        console.error("WS parse error in /analyse:", err);
      }
    };

    ws.onclose = () => {
      setTimeout(connectWebSocket, 1800);
    };
  }

  // Initialize Cockpit
  fetchAnalyticsPayload(false);
  connectWebSocket();
})();
