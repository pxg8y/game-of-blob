# Empirical Telemetry Findings & Architectural Ideas for Game of Blobs Training Maps

## Executive Summary

This document presents a comprehensive diagnostic analysis of the population collapse observed in `blob_telemetry_tag_3.json`, evaluates the core simulation mechanics of the *Game of Blobs* neuroevolution engine, and formulates concrete ideas and design paradigms for constructing the ideal basic training map.

In `blob_telemetry_tag_3.json`, the blob population experienced an catastrophic extinction cascade:
- **Day 1**: Starting population of **3,600 blobs** suffered 330 deaths, ending at 3,328 blobs.
- **Day 2**: Population crashed from **3,328 down to 443 blobs** (a **87.7% single-day collapse**; 2,917 midday starvation deaths).
- **Day 3**: Population plummeted further to **160 blobs** (a **63.9% collapse**; 335 midday starvation deaths), requiring an emergency Hall-of-Fame rescue (+40 clones).
- **Day 4 (Tick 2242)**: Only **72 living blobs** remained out of the original 3,600 (a **98.0% cumulative extinction rate**).

Our empirical audit reveals that this collapse was not caused by failure of the neural network learning algorithm itself, but rather by a **severe ecological mismatch between the physical map topography, the spatial resource distribution, and the RL learning timeframe**. Untrained neural networks with near-zero pre-wired weights were placed into a harsh, high-friction labyrinth with rapid resource depletion, giving them insufficient time and signal to learn basic foraging and homing behaviors before starvation eliminated them.

---

## 1. Deep Dive: Telemetry Analysis of `blob_telemetry_tag_3.json`

### 1.1 The Extinction Curve & Collapse Chronology

| Metric | Day 1 | Day 2 | Day 3 | Day 4 (Tick 2242) |
| :--- | :--- | :--- | :--- | :--- |
| **Living Population** | 3,328 | 443 | 160 | 72 |
| **Births** | 58 | 32 | 12 | 0 |
| **Deaths Total** | 330 | 2,917 | 335 | 88 (partial day) |
| **Midday Starvation Deaths** | 330 (100%) | 2,917 (100%) | 335 (100%) | 88 (100%) |
| **End-of-Day (EOD) Starvation** | 0 (0%) | 0 (0%) | 0 (0%) | - |
| **Outdoor Deaths** | 325 (98.5%) | 2,855 (97.9%) | 330 (98.5%) | 88 (100%) |
| **Deaths in Cradle Ring** | 330 (100%) | 2,917 (100%) | 335 (100%) | 88 (100%) |
| **Deaths in Frontier Biomes** | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| **Blobs in Cradle** | 3,328 (100%) | 443 (100%) | 160 (100%) | 72 (100%) |
| **Blobs in Frontier** | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| **Cradle Oasis Fertility** | 0.137 | 0.122 | 0.197 | ~0.20 |
| **Frontier Oasis Fertility** | 0.985 | 0.985 | 0.996 | ~0.99 |
| **Overgrazed Oases Count** | 32 (29.6%) | 37 (34.3%) | 36 (33.3%) | 36 (33.3%) |
| **Total Wild Food on Map** | 5,476 | 5,196 | 5,048 | 5,000+ |
| **Available Food per Blob** | 1.65 🍏 | 11.73 🍏 | 31.55 🍏 | >69.0 🍏 |
| **Rock Wall Collisions** | 187,780 | 290,033 | 34,759 | 11,913 |
| **Swamp Immersions** | 43,971 | 71,673 | 7,934 | 4,884 |
| **EOD Outdoors Percentage** | 94.8% | 96.1% | 97.9% | - |
| **Deaths Far with Gold** | 180 | 2,725 | 286 | 73 |
| **Hall of Fame Rescues** | 0 | 0 | 1 (+40 blobs) | 0 |

### 1.2 Five Root Causes of Early Extinction

#### Cause 1: The "900-Tick Hard Starvation Cliff" on Day 2 Midday
- **Simulation Rule**: A blob dies midday if its energy $\le 0$ **or** if `ticks_since_last_meal >= max_starvation_ticks` (default `max_starvation_ticks = 900`).
- **Timing Mechanics**: Each simulation day has `ticks_per_day = 600`. Day 1 runs from tick 0 to 600. Day 2 midday occurs exactly at tick 900 ($600 + 300$).
- **The Catastrophe**: Blobs spawned on Day 1 that failed to eat during Day 1 or early Day 2 hit `ticks_since_last_meal = 900` simultaneously at tick 900.
- Furthermore, blobs with $< 1.0$ food at Day 1 EOD suffered severe overnight energy penalties:
  $$\Delta E = -32.0 \times \left(\frac{1.0 - \text{food\_today}}{1.0}\right) - 8.0 \text{ (outdoor penalty)} = -40.0 \text{ energy}$$
  Starting with 85.0 energy, their energy dropped to $\approx 35\text{--}45$ energy entering Day 2.
- At tick 900, **2,917 blobs died in an instant**, causing an 87.7% population collapse.

#### Cause 2: Localized Cradle Overgrazing vs. Wasteland Trap
- The simulation used the **Continent preset** ($40,000 \times 30,000$ units, 12 regional cradles, 108 oases, 3,600 initial blobs).
- In each cradle:
  - 300 blobs were initially concentrated around 3 fragile nursery oases (`cradle_angles`).
  - Each cradle oasis had `max_capacity = 95` (total cradle capacity = 285 food units).
  - 300 blobs consumed all available cradle food in less than 200 ticks on Day 1.
  - Harvest pressure spiked, driving cradle fertility from `0.86` down to `0.122`–`0.137`.
- **The Paradox**: While total food on the map was abundant (over 5,000 food units; 11.73 to 31.55 food per blob), **all of that food was located in the outer Frontier oases**!
- The nursery had become a barren dust bowl within 24 hours, yet 100% of the population remained locked inside the cradle.

#### Cause 3: The 6-Gate Canyon Ring as an Impassable Skill Barrier
- In the continental generator, each cradle is encircled by an elliptical **Canyon Rock Ring** ($r_x \approx 1,108$, $r_y \approx 897$, wall thickness 36 units) with 6 narrow "Skill Gates":
  - 2 Chicanes (narrow S-baffles with perpendicular rock barriers)
  - 2 Swamp Fords (speed reduced by 56%, metabolic cost increased by 75%)
  - 2 Steppe Gaps (barren corridors requiring long-range scent following)
- **Obstacle Collisions**:
  - Day 1: 187,780 rock collisions, 43,971 swamp immersions.
  - Day 2: 290,033 rock collisions, 71,673 swamp immersions.
- Untrained neural networks perform pseudo-random Brownian walks with minimal directional persistence. They bounced back and forth against the canyon rock walls, unable to thread the needle through complex chicane gates.
- Consequently: **`blobs_in_frontier = 0` on every single day**. Not a single blob managed to migrate into the lush frontier.

#### Cause 4: Weight Coupling Matrix Immaturity (Insufficient RL Exploration Time)
- Blobs initialize with `innate_seed = 0.72` (under standard curriculum), which adds a small weight offset strictly connecting Food $\Delta X, \Delta Y$ (Channels 0, 1) to Movement $X, Y$ (Outputs 0, 1).
- The coupling matrix from telemetry reveals the state of the neural networks at Day 3:
  - Food Nav coupling: $\approx 0.022$ (weak initial gradient).
  - Home Nav coupling: $-0.0003$ (completely non-functional).
  - Feeler / Obstacle Avoidance: $0.0000$ (no learned rock avoidance).
  - Oasis Fernsinn / Scent: $0.0004$ (blind to distant fertile biomes).
  - Spatial Memory: $0.0003$ (no effective spatial recall).
- Lamarckian backpropagation with learning rate $\eta = 0.015$ (or $0.08$ in prefs) requires hundreds of successful foraging and homing experiences across multiple generations to shape hidden layer representations.
- Because blobs died during their very first generation before finding food or returning home, the RL algorithm received overwhelmingly negative or zero reward signals, preventing convergence.

#### Cause 5: Homing Failure and "Dying Rich in the Desert"
- Across all three days, **94.8% to 97.9% of blobs failed to return home at dusk** (`eod_outdoors_pct`).
- When night fell, blobs stranded outdoors incurred severe metabolic cold drain ($1.18\times$ movement cost multiplier + overnight deficit penalty).
- Blobs accumulated gold ($780$ gold in circulation; 4.87 gold per blob) from early peer encounters or market trades, but **2,725 blobs died on Day 2 holding gold** because emergency ration purchases require being at home or inside a settlement depot!

---

## 2. Key Concepts for the Perfect Basic Training Map

To enable reinforcement learning algorithms and evolutionary agents to acquire useful navigation, foraging, obstacle avoidance, and homing policies before extinction, the map design must adhere to **ecological scaffolding** principles.

### Concept 1: Progressive Ecological Zoning ("Concentric Curriculum Topology")
Rather than confronting nascent agents with complex chicanes and rapid overgrazing simultaneously, the training map must be structured in concentric, graduated difficulty zones:

1. **Zone 0: The Kindergarten Basin (Inner Sanctuary)**
   - **Radius**: $0 \le r < R_{\text{basin}}$ ($R \approx 800\text{--}1,000$ units).
   - **Characteristics**: Flat terrain, **zero lethal obstacles** (no impassable rock rings, no swamps).
   - **Homes & Food**: Centrally positioned starter huts surrounded by continuous, moderate-density food clusters within the blob's basic sensory radius ($215 \times \text{sense}$).
   - **Learning Objective**: Master the fundamental loop: *Sense Food $\to$ Steer $\to$ Consume $\to$ Sense Home at Dusk $\to$ Return*.

2. **Zone 1: The Transition Ring (Guided Gateways & Wide Passes)**
   - **Radius**: $R_{\text{basin}} \le r < R_{\text{frontier}}$ ($R \approx 1,000\text{--}1,800$ units).
   - **Characteristics**: Wide, gentle rock pillars (convex rounded obstacles with wide 250+ unit gaps) rather than narrow chicanes.
   - **Learning Objective**: Form positive sensory weights on feeler inputs (Channels 21, 22, 23) to steer around obstacles without getting trapped.

3. **Zone 2: The Frontier Biomes (Abundant Specialization)**
   - **Radius**: $r \ge R_{\text{frontier}}$.
   - **Characteristics**: Rich Golden Groves (high calories, fast reproduction) and Thermal Springs (winter shelter), bordered by swamps and narrower canyons.
   - **Learning Objective**: Long-range navigation, settlement founding, colonization, inter-group trade.

### Concept 2: Carrying-Capacity-Balanced Resource Dynamics
In `blob_telemetry_tag_3.json`, 300 blobs per cradle rapidly exhausted 285 units of food. The ratio was $0.95$ food units per blob, triggering immediate starvation.

The perfect basic training map must enforce a **sustainable carrying capacity ratio**:
- **Baseline Rule**: Starter food availability must equal at least $2.5\text{--}3.5\times$ the daily survival need per initial blob:
  $$\text{Daily Cradle Food Spawn} \ge \text{Initial Blobs} \times 2.5$$
- **Cradle Oasis Resilience**: Starter oases should have a higher overgrazing threshold and faster baseline recovery rate during early generations (e.g., $r_{\text{rec}} \ge 0.60$ instead of $0.25$, harvest pressure sensitivity halved during Day 1–3).

### Concept 3: Sensory Guidance Gradients (Preventing "Sensory Blindness")
A neural network cannot learn to navigate toward an oasis if the oasis is beyond its sensory horizon and emits no gradient:
- Standard blobs have vision radius $V \approx 215$ units.
- If oases are separated by $2,000+$ units of empty wasteland, a blob wandering outside vision range receives zero food sensory input (`Food ΔX=0, Food ΔY=0, Food Prox=0`).
- **Solution — "Food Stepping Stones" & Continuous Scent Attractors**:
  - Lay out food beacons or decaying nutrient density trails leading from the Kindergarten Basin outward to the transition passes.
  - Calibrate the oasis scent vector (Channels 16–18) with a gentler distance denominator ($d + 100$ rather than $d + 220$), ensuring even distant oases exert a perceptible directional gradient on the input sensors.

### Concept 4: Starvation Buffer & Daylight Ratio Calibration
- **Daylight Ratio**: In early training, blobs need more daylight to forage before dusk arrives. Raising daylight ratio from 0.68 to 0.75 gives beginner agents an extra 42 ticks of foraging time.
- **Starvation Grace Buffer**: The hard threshold `max_starvation_ticks` should scale dynamically with generation or start at $\ge 1,200\text{ ticks}$ (2 full days) for Gen 1, preventing the simultaneous tick-900 mass-death cliff.
- **Overnight Outdoors Grace**: Dampen the overnight outdoor energy penalty during Generation 1 from $-32.0$ to $-12.0$ so that blobs that foraged partial food do not start Day 2 with near-empty fuel tanks.

---

## 3. Comparison: Flawed Continental Map vs. Ideal Training Map

| Feature | Flawed Telemetry Setup (Tag 3) | Ideal Basic Training Map ("Kindergarten Basin") |
| :--- | :--- | :--- |
| **Map Preset / Scale** | Continent ($40,000 \times 30,000$) | Classic Training Arena ($4,800 \times 3,600$) |
| **Initial Population** | 3,600 blobs (300 per cradle) | 80–120 blobs (single unified cradle or 2 connected valleys) |
| **Initial Food Density** | 285 cradle food / 300 blobs ($<1.0\times$) | 350 cradle food / 100 blobs ($3.5\times$ survival threshold) |
| **Obstacle Barrier** | Enclosing 6-Gate Canyon Ring with S-chicanes | Open basin with wide peripheral gaps ($>300$ units wide) |
| **Hazard Proximity** | Swamps immediately outside gates | Swamps relegated to outer exploration zones |
| **Cradle Depletion Rate** | High ($2.4\times$), slow recovery ($0.25\times$) | Gentle ($0.8\times$), resilient recovery ($0.65\times$) |
| **Starvation Limit** | 900 ticks (instant cliff at Day 2 midday) | 1,200–1,500 ticks (2 to 2.5 full diurnal cycles) |
| **Daylight Ratio** | 0.68 (192 ticks of dusk/night) | 0.75 (150 ticks of dusk/night, ample foraging window) |
| **Distance to Nearest Food** | $>600$ units after initial patch eaten | $\le 120$ units throughout the nursery basin |
| **Day 2 Survival Probability** | $12.3\%$ (Catastrophic collapse) | $>85.0\%$ (Stable evolutionary progression) |

---

## 4. Architectural Innovations for the Training Map Generator

### 4.1 "Concentric Valley with Radial Corridors" (CVRC)
Instead of an unbroken elliptical canyon wall with narrow chicanes, CVRC generates:
1. **Central Nursery Bowl**: A circular clearing containing 3 interconnected fertile springs and distributed starter homes.
2. **Radial Food Corridors**: 4 to 6 natural corridors radiating outward toward satellite oases like spokes on a wheel. Food spawns along these spokes, naturally drawing blobs outward through reinforcement gradients.
3. **Rounded Boulders**: Obstacles placed along spoke borders are circular/convex rock pillars. When blobs bump into them, normal physics causes them to slide along the surface toward open space rather than getting wedged into concave corners.

### 4.2 Dynamic Environmental Scaffolding (Curriculum Auto-Pacing)
1. **Epoch 1 (Days 1–5 - Nursery Phase)**:
   - Cradle oases have infinite recovery; food respawns continuously.
   - Outdoor night penalty reduced by 60%.
   - Blobs easily acquire food and return home, reinforcing synaptic weights for Channels 0–5.
2. **Epoch 2 (Days 6–12 - Expansion Phase)**:
   - Cradle fertility begins normal carrying capacity depletion.
   - Outer frontier oases begin blooming with Golden Clusters (2.5x food value).
   - Blobs naturally follow the radial food spokes outward into the frontier.
3. **Epoch 3 (Days 13+ - Continental Mastery)**:
   - Full seasonal cycles (Winter frost, mud swamps, inter-tribe raids, pioneer settlements).
   - Populations possess robust, mutated neural policies capable of thriving in harsh conditions.

---

## 5. Summary of Key Ideas for Implementation

1. **Preset Selection**: Default the basic training map to an optimized `arena_training` / `kindergarten` preset ($4,800 \times 3,600$ units, 100 blobs, 300 daily food spawn).
2. **Topography Overhaul**: Replace enclosing canyon chicanes in the primary training map with open radial channels and convex guideposts.
3. **Starvation Smoothing**: Soften the `max_starvation_ticks` cliff (1,200 ticks) and reduce Day 1–2 overnight outdoor penalties.
4. **Cradle Food Buffer**: Ensure inner nursery oases produce at least $2.5\times$ the daily survival need during the initial days.
5. **Reward Shaping Alignment**: Maintain positive directional velocity rewards towards home when $t \ge 0.65$ to solidify homing reflexes before winter/scarcity strikes.
