# Finished Plan: The Perfect Basic Training Map for Game of Blobs

## 1. Overview & Objective

The primary objective of the **Perfect Basic Training Map** ("The Kindergarten Arena") is to provide a stable, gradient-rich training environment where reinforcement learning (Lamarckian online RL) and neuroevolution have **ample time to acquire vital survival and navigation policies** before populations face extinction pressures.

In the baseline collapse (`blob_telemetry_tag_3.json`), 3,600 blobs faced instant overgrazing and an impassable canyon ring, resulting in an 87.7% crash on Day 2 and 98% extinction by Day 4. This plan specifies the exact architectural layout, physical topography, resource replenishment model, temporal parameters, and evaluation criteria required to guarantee continuous survival ($>85\%$ population retention) and successful multi-generational learning.

---

## 2. Geometric & Spatial Design: "The Concentric Sanctuary Arena"

### 2.1 World Dimensions & Scale Preset
- **Map Dimensions**: $W = 4,800.0$ units, $H = 3,600.0$ units (Area $= 17.28\times 10^6\text{ units}^2$).
- **Coordinate Center**: $(C_x, C_y) = (2400.0, 1800.0)$.
- **Scale Preset**: `training_kindergarten` (derived from `classic`, optimized for RL convergence).
- **Initial Blob Count**: $100$ blobs (allows high per-capita resource density and avoids overcrowding).
- **Carrying Capacity**: 1,500 blobs; 120 houses; 24 groups.

```
+-------------------------------------------------------------------------------+
|                                                                               |
|       [Frontier Biome: Golden Grove]            [Frontier Biome: Thermal]     |
|                                                                               |
|                     +-----------------------------------+                     |
|                     |     Zone 1: Transition Ring       |                     |
|                     |   (Wide convex rock guideposts)   |                     |
|                     |                                   |                     |
|                     |      +---------------------+      |                     |
|                     |      |  Zone 0: Sanctuary  |      |                     |
|                     |      |  (Bowl & 4 Oases)   |      |                     |
|                     |      |     H1   H2   H3    |      |                     |
|                     |      +---------------------+      |                     |
|                     |                                   |                     |
|                     |   ==== Radial Food Spokes ====    |                     |
|                     +-----------------------------------+                     |
|                                                                               |
|       [Frontier Biome: Thermal]                 [Frontier Biome: Golden]      |
|                                                                               |
+-------------------------------------------------------------------------------+
```

### 2.2 Zone Stratification

#### Zone 0: The Central Nursery Sanctuary ($0 \le r < 850$ units)
- **Topography**: Completely flat, smooth ground. **Zero rocks, zero swamps, zero dead-ends**.
- **Water & Nursery Springs**: 4 balanced nursery oases positioned in a cross/diamond pattern:
  - North: $(C_x, C_y - 320)$
  - South: $(C_x, C_y + 320)$
  - East: $(C_x + 380, C_y)$
  - West: $(C_x - 380, C_y)$
  - Radius per nursery oasis: $r = 220$ units.
  - Initial fertility: $1.0$ (Lush).
  - Daily nursery food quota: 220 food units distributed across the 4 oases.
- **Starter Homes**: 18 starter huts placed within $120$ units of the nursery oases.
  - Resident capacity: 6 blobs per hut.
  - Initial home food depot: 1.5 units (emergency reserve for young blobs).
  - Maximum distance from any home to food: $\le 150$ units (well within base vision $V = 215$).

#### Zone 1: The Transition Ring ($850 \le r < 1,600$ units)
- **Obstacle Layout**: Instead of continuous enclosing canyon walls with chicanes, Zone 1 features **8 radial convex boulders** (rounded pillars, length $80$ units, width $40$ units).
- **Pass Width**: The minimum clearance between any two obstacles is **$320$ units** (blobs have radius $7.5$, meaning passes are $>40\times$ blob body diameter).
- **Radial Food Spokes**: 4 continuous trails of low-density food beacons connecting Zone 0 to outer Zone 2 along the cardinal axes ($0^\circ, 90^\circ, 180^\circ, 270^\circ$).
- **Curriculum Purpose**: Enables blobs to bump into obstacles gently, receive sensor activation on feeler rays (Channels 21–23), and learn negative steerage weights without becoming trapped or suffering lethal starvation.

#### Zone 2: The Frontier Biomes ($1,600 \le r < 2,400$ units)
- **Specialized Biomes**:
  - 4 **Golden Groves** in diagonal quadrants (NE, NW, SE, SW; rich golden cluster chance $= 45\%$).
  - 4 **Thermal Springs** on outer perimeters (autumn/winter cold-resilient oases).
- **Hazards**: 4 small mud swamps ($r = 85$ units) positioned off-axis to reward spatial detour learning.
- **Settlement Expansion**: Ample space for pioneer huts, tribe federations, and trade outposts.

---

## 3. Ecological & Simulation Mechanics Configuration

### 3.1 Carrying Capacity & Resource Replenishment Formulae

1. **Daily Food Spawn Ratio**:
   $$\text{Daily Food Spawn} = 320 \text{ food units} \quad (\text{for } 100 \text{ initial blobs})$$
   - Food-to-blob ratio: **$3.20\times$ daily survival need**.
   - Guarantee: Even with random movement, random walk collision probability ensures $>92\%$ of blobs acquire $\ge 1.0$ food on Day 1.

2. **Cradle Oasis Depletion & Recovery Balancing**:
   - In `blob_telemetry_tag_3.json`, cradle depletion was $2.4\times$ and recovery was $0.25\times$.
   - **New Balanced Formula**:
     $$\text{depletion\_rate} = 0.85, \quad \text{recovery\_rate} = 0.65$$
     $$\text{harvest\_pressure\_threshold} = 4.5 \quad (\text{raised from } 3.2)$$
   - This prevents nursery springs from desertifying during Generation 1 and 2 while still applying gentle evolutionary pressure by Generation 5.

### 3.2 Starvation & Energy Smoothing

1. **Starvation Tick Grace Buffer**:
   - `max_starvation_ticks`: **Increased from 900 to 1,350 ticks** (2.25 full days).
   - *Impact*: Eliminates the catastrophic Day 2 midday cliff observed at tick 900.
2. **Daylight Ratio**:
   - `daylight_ratio`: **Increased from 0.68 to 0.74** ($444$ day ticks, $156$ dusk/night ticks).
   - Gives uncoordinated early agents more active search time to stumble upon food.
3. **Overnight Outdoor Penalty Damping**:
   - Reduce Day 1–3 outdoor overnight energy penalty from $-8.0$ to $-2.5$ energy units.
   - Satiated blobs retain $\ge 70$ energy going into Day 2 morning, preventing premature exhaustion.

### 3.3 Sensory Vector Modulation for RL Shaping

1. **Oasis Fernsinn Sensitivity**:
   - Modify the denominator in `compute_oasis_and_feelers_numba`:
     $$\text{score} = \frac{\text{fertility}^{1.5}}{d_{\text{oasis}} + 120.0} \quad (\text{improved from } d + 220.0)$$
   - Increases directional gradient sensitivity for blobs near the edge of the sanctuary.
2. **Reward Shaping Balance**:
   - `pref_food = 2.0` (encourages foraging focus)
   - `pref_home = 1.3` (strengthens dusk return instinct)
   - `pref_exploration = 0.45` (encourages active velocity when hungry)
   - `learning_rate = 0.025` (smooth, stable gradient updates)

---

## 4. Implementation Specifications for the Codebase

### 4.1 New Preset in `blob_sim/config.py`

Add the dedicated `training_kindergarten` preset to `WORLD_SCALE_PRESETS`:

```python
WORLD_SCALE_PRESETS["training_kindergarten"] = {
    "id": "training_kindergarten",
    "label": "🌱 Kindergarten · Perfekte Trainings-Arena (4.800×3.600)",
    "short_desc": "Hindernisfreie Starter-Wiege mit 4 resilienten Oasen, 3.2x Nahrungsüberschuss, 1.350 Ticks Hunger-Puffer und weiten Übergangspässen für optimales RL-Lernen ohne Frühaussterben.",
    "width": 4800.0,
    "height": 3600.0,
    "num_cradles": 1,
    "initial_blobs": 100,
    "daily_food_spawn": 320,
    "max_population_cap": 1500,
    "max_homes_cap": 120,
    "max_groups_cap": 24,
    "num_oases": 12,
    "num_rock_formations": 16,     # Reduced from 28 to eliminate narrow traps
    "num_swamps": 4,               # Reduced from 10, moved outside sanctuary
    "min_population_floor": 24,
    "daylight_ratio": 0.74,
    "max_starvation_ticks": 1350,
}
```

### 4.2 Topography Generation Routine in `blob_sim/world.py`

Implement `_generate_kindergarten_topography()`:
1. **Clear Central Zone**: Reject any rock or swamp placement within $r \le 900.0$ of map center $(W/2, H/2)$.
2. **Inner Cross Oases**: Place 4 inner oases at offset $\Delta r = 350.0$, angles $[0, \pi/2, \pi, 3\pi/2]$. Set `depletion_rate=0.85`, `recovery_rate=0.65`, `max_capacity=80`.
3. **Radial Food Corridors**: During `spawn_daily_food`, guarantee at least 65% of daily food spawns inside Zone 0 during Days 1–3, gradually shifting outward as population exceeds 150.
4. **Convex Outer Obstacles**: Place 8 tangential convex rock segments along $r = 1,250.0$ with angular gaps $\Delta \theta \ge 0.45\text{ rad}$ ($>350$ unit clearances).

---

## 5. Telemetry Validation & Success Criteria

To verify that the new training map prevents extinction and supports reinforcement learning, a 10-day evaluation simulation will be executed. Success is measured against the following strict KPIs:

| Metric | Flawed Telemetry (Tag 3) | Target KPI (Kindergarten Map) | Verification Method |
| :--- | :--- | :--- | :--- |
| **Day 2 Population Survival** | $12.3\%$ (443 / 3,600) | **$\ge 88.0\%$** ($\ge 88$ / 100) | `daily_logs[1]["population"]` |
| **Day 5 Population Survival** | $< 2.0\%$ (Extinction) | **$\ge 110\%$** (Population growth via repro) | `daily_logs[4]["population"]` |
| **Midday Starvation Deaths** | 2,917 on Day 2 | **$< 10$ blobs / day** | `community["deaths_midday_starve"]` |
| **Cradle Oasis Fertility (Day 3)** | $0.122$ (Dust bowl) | **$\ge 0.65$** (Sustainable) | `ecology["cradle_oasis_fertility"]` |
| **Homing Success Rate** | $2.1\text{--}5.2\%$ at home | **$\ge 45.0\%$** at home by Day 3 | `community["eod_at_home_pct"]` |
| **Rock Collisions (Day 2)** | 290,033 collisions | **$< 15,000$ collisions** | `community["rock_collisions"]` |
| **Food Navigation Weight Coupling** | $0.022$ (stagnant) | **$\ge 0.085$** (rapid convergence) | `weights["pathways"]["food_nav"]` |
| **Home Navigation Weight Coupling** | $-0.0003$ (broken) | **$\ge 0.040$** (positive homing) | `weights["pathways"]["home_nav"]` |
| **Frontier Emigration (Day 5)** | 0 blobs in frontier | **$\ge 20$ blobs active in frontier** | `ecology["blobs_in_frontier"]` |

---

## 6. Execution Timeline & Next Steps

1. **Step 1**: Register the `training_kindergarten` preset in `blob_sim/config.py`.
2. **Step 2**: Add the open-basin topography logic in `blob_sim/world.py`.
3. **Step 3**: Verify test suite execution and continuous simulation stability.
4. **Step 4**: Run an automated 5-day verification benchmark to log telemetry parity against the target KPIs.
