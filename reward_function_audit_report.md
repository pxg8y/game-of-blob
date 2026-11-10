# Comprehensive Diagnostic Analysis of the Full Reward Function Paradigm

## Executive Summary

This document presents a comprehensive, rigorous investigation of the complete reward function paradigm in the *Game of Blobs* simulation engine. It audits the mathematical formulations, timing mechanics, interaction pathways, and reinforcement learning / neuroevolution credit assignment pipeline across `blob_sim/world.py`, `blob_sim/neural.py`, `blob_sim/spatial.py`, `blob_sim/entities.py`, and `blob_sim/logger.py`.

The analysis identifies:
1. **Critical Logical Inconsistencies**: Contradictory incentives, double-counting / phantom penalties, baseline ratcheting artifacts, uncalibrated scaling factors, and coordinate frame discrepancies.
2. **Key Structural Hurdles That Make Training Harder**: Eligibility trace horizon collapse vs. delayed terminal rewards, continuous shaping reward dominance over sparse goals, sub-stepping / cohort temporal distortions, and objective misalignment between individual RL updates and multi-generational evolutionary fitness.
3. **Actionable Architectural Solutions**: Concrete, mathematically sound proposals to resolve each flaw and dramatically accelerate policy convergence and evolutionary robustness.

---

## 1. Architectural Map of the Reward Paradigm

The simulation employs a hybrid learning architecture:
- **Online Lamarckian Reinforcement Learning**: Every active blob brain (an MLP) maintains low-rank backpropagated eligibility traces $\mathbf{e}_t$. In each simulation tick $t$, learning steps update weights via $\Delta \mathbf{W} \propto \eta \cdot A_t \cdot \mathbf{e}_t$, where $A_t$ is the baseline-subtracted advantage and $\eta$ is the learning rate.
- **Genetic Evolution (Selection & Mutation)**: At the end of each day, blobs that consumed surplus food reproduce. Child blobs inherit their parents' lifetime-trained weights, subjected to crossover and Gaussian mutation. Fitness for partner choice and the "Hall of Fame" is governed by an explicit heuristic formula factoring lifetime food, children count, reputation, and cumulative reward.

Rewards are generated across five primary channels:
1. **Kinematic & Movement Shaping Rewards** (`_compute_movement_shaping_reward_batch`): Continuous per-tick signals for approaching food, aligning with fertile oasis scents, avoiding swamps, and returning home at dusk/when sated.
2. **Resource Consumption Rewards** (`_process_foraging`): Discrete event rewards when a blob physically reaches and consumes food ($+1.10 \times \text{pref\_food}$ for golden food, $+0.75 \times \text{pref\_food}$ for regular food).
3. **Social & Economic Interaction Rewards** (`_try_peer_trade`, `_try_peer_raid`, `_try_village_market_trade`, `_handle_social_encounter`, `_try_colonize_or_relocate`): Rewards/penalties for peer trades, settlement founding, raids, bounties, and market transactions.
4. **End-of-Day (EOD) Diurnal Accounting** (`_stage_eod_settlements_and_learning`): Daily evaluations rewarding being sheltered at home ($+0.35 \times \text{pref\_home}$), penalizing being stranded outdoors ($-0.25 \times \text{pref\_home}$), penalizing starvation ($-1.0 \times \text{pref\_food}$), and rewarding reproduction readiness ($+1.40 \times \text{pref\_food}$).
5. **Terminal Midday Starvation Penalties** (`_check_midday_starvation`): One-off weight updates of $-1.20 \times \text{pref\_food}$ applied immediately before a blob dies of energy depletion.

---

## 2. In-Depth Audit of Logical Inconsistencies

### 2.1 Double-Counting & Temporal Desynchronization of Cumulative Reward vs. Step Reward
- **The Code**:
  In `blob_sim/world.py`:
  - Lines 3205–3221: Per-tick movement shaping reward `scaled_reward_vec = base_rewards * dt_scale` is added directly to `blob.cumulative_reward += sr`. Then, `step_rewards = (scaled_reward_vec * 2.5).astype(np.float32)`.
  - Line 3288: Social encounter bonus `bonus = pref_social * 0.45` adds `blob.cumulative_reward += bonus * dt_scale`, while `step_rewards[i] += bonus * (dt_scale * 2.5)`.
  - Discrete events (eating food line 3501, migrations line 2594, peer trades line 4132, market trades line 4434, raids line 4289) add the full event reward $R_{\text{event}}$ to **both** `blob.cumulative_reward` and `blob.pending_rl_reward`.
  - In line 3365: `step_rewards[idx] += living_blobs[idx].pending_rl_reward; blob.pending_rl_reward = 0.0`.
  - **Inconsistency**:
    1. Movement shaping is scaled by $2.5\times$ for online neural learning (`step_rewards`), but only by $1.0\times$ for `cumulative_reward`.
    2. Discrete event rewards (`pending_rl_reward`) enter `step_rewards` with an effective multiplier of $1.0\times$, completely unscaled by `dt_scale` or the $2.5\times$ factor applied to movement shaping.
    3. At high simulation speeds (`dt_scale > 1`), movement shaping accumulates in `step_rewards` as `base_rewards * dt_scale * 2.5`, while physical discrete actions (like consuming an apple or successfully trading) remain fixed at $1.0\times$. This severely distorts the agent's objective function depending on simulation speed: at $\text{dt} = 10$, wandering toward food yields $25\times$ higher gradient updates per step than at $\text{dt} = 1$, whereas actually eating the food yields the exact same fixed $+1.2$ update!

### 2.2 Asymmetric Dual-Application of End-of-Day Rewards
- **The Code**:
  In `_stage_eod_settlements_and_learning` (lines 4686–4752):
  ```python
  rew = home_rew_pos if blob.is_at_home else home_rew_neg
  if blob.energy <= 0.0:
      rew += starve_pen
  elif blob.food_today >= repro_need - 1e-5:
      rew += repro_rew
  eod_rewards[idx] = rew
  # Applied to online brain weights:
  self._brain_pool.apply_online_learning_batch(brains, eod_rewards, self.prefs.learning_rate)
  ```
  Meanwhile, in `_stage_eod_reproduction_and_rollover` (lines 4537–4538):
  ```python
  if blob.food_today >= repro_need - 1e-5:
      repro_reward = self.prefs.pref_food * 1.4
      blob.cumulative_reward += repro_reward
  ```
  - **Inconsistency**:
    - `repro_reward` is added to `cumulative_reward` in Phase 3, but the corresponding EOD online learning batch update happens in Phase 2.
    - Conversely, `home_rew_pos` ($+0.35 \times \text{pref\_home}$), `home_rew_neg` ($-0.25 \times \text{pref\_home}$), and `starve_pen` ($-1.0 \times \text{pref\_food}$) are applied to the neural network weights via `apply_online_learning_batch`, **but are never added to `blob.cumulative_reward`**!
    - As a result, `blob.cumulative_reward` omits the major diurnal consequences of succeeding or failing to return home. Evolutionary selection (partner choice and Hall of Fame ranking), which evaluates `cumulative_reward * 0.3` to `0.4`, is completely blind to whether a blob reached home or spent every night freezing outdoors!

### 2.3 The "Dead Blob Backpropagation" Phantom Learning Update
- **The Code**:
  In `_check_midday_starvation` (lines 2723–2724):
  ```python
  blob.brain.apply_online_learning(-1.2 * self.prefs.pref_food, self.prefs.learning_rate)
  self._kill_blob(blob, reason="verhungert (Energie leer)")
  ```
  - **Inconsistency**:
    - A massive negative reward ($-1.2 \times \text{pref\_food}$) is backpropagated into `blob.brain` at the precise millisecond of death, right before `_kill_blob` marks `blob.alive = False`.
    - Dead blobs never act again, and their brains are never used as parents during EOD reproduction (reproduction strictly filters for living blobs with surplus food).
    - Even worse, in `BatchedBrainPool`, `apply_single_brain_learning` updates `_accum_coef_cap` and `_B_cap` for that blob's slot. But immediately afterward, `_compact_living_blobs()` executes an in-place swap-remove, moving a surviving blob into the dead blob's slot or invalidating the slot.
    - If the dying blob was previously stored in the Hall of Fame, this penalty does not even update the Hall of Fame snapshot because the snapshot was already deep-copied at the end of the previous day. Thus, the midday starvation RL update is pure wasted computation that risks memory corruption or slot contamination during compaction.

### 2.4 Reward-Observation Coordinate Frame Discrepancies
- **The Observation**:
  - In `spatial.py` (`prep_step_observations_numba`):
    - Food observation channels 0 and 1 are unit directional vectors pointing towards the nearest food: $(\Delta x_{\text{food}}, \Delta y_{\text{food}})$, modulated by `food_gain = 0.85 + 0.25 * pref_food`.
    - Home observation channels 3 and 4: $(\Delta x_{\text{home}}, \Delta y_{\text{home}})$, modulated by `home_gain = 0.85 + 0.35 * pref_home`.
    - Oasis channels 16 and 17: unit direction to best oasis.
    - Feeler channels 21, 22, 23: distance to obstacles along headings $-40^\circ, 0^\circ, +40^\circ$ relative to the blob's current heading.
- **The Reward Formulation**:
  - In `_compute_movement_shaping_reward_batch`:
    ```python
    approach = np.maximum(0.0, move_x * dx_foods + move_y * dy_foods)
    food_reward = (pref_exp * 0.068) * approach
    ```
  - **Inconsistencies**:
    1. **`approach` rewards instantaneous action `(move_x, move_y)` instead of actual velocity `(vx, vy)`**:
       Because blobs have mass, friction, and inertia ($v_{t+1} = v_t \times (1 - \text{friction}) + a_t$), if a blob is moving away from food and outputs a positive action towards food, it receives an immediate positive shaping reward *even while it is still physically drifting farther away from the food*!
       Conversely, if a blob has already attained top speed towards food and outputs $0.0$ action to coast and conserve energy, its shaping reward drops to $0.0$, penalizing optimal coasting!
    2. **Pref Weight Inversion / Coupling Contradiction**:
       - `food_reward` in movement shaping is weighted by `self.prefs.pref_exploration`, **not** `self.prefs.pref_food`!
       - If a user increases `pref_food` in the UI to encourage blobs to seek food, the observation vectors increase via `food_gain`, but the per-step movement shaping reward does not increase at all! Blobs only receive a higher reward upon actually reaching and swallowing the food.
       - Meanwhile, `home_reward` is weighted by `self.prefs.pref_home` in both shaping and observation. This creates a severe priority skew where `pref_home` overpowers `pref_food` during daytime foraging.

### 2.5 Contradictory Incentives in Resource Trading & Raiding
- **Peer Selling vs. Hunger Penalty**:
  - In `_compute_seller_trade_reward` (lines 4029–4030):
    ```python
    if seller.food_today < surv_need - 1e-4:
        return -1.0 * self.prefs.pref_food
    ```
    While this correctly penalizes suicidal sales, the gating method `_is_seller_willing` still allows a hungry blob to sell if `seller.trade_desire > 0.25`. If an agent explores high trade desire, it sells food, receives a huge $-1.0$ penalty, but has already transferred its only food to the peer.
  - In `_try_peer_raid` (lines 4280–4295):
    - When raiding a peaceful peer, the aggressor gains $0.28 \times \text{pref\_food} - \text{social\_pen}$, and the defender loses $-0.25 \times \text{pref\_home}$.
    - But why does the defender lose `pref_home`? A raid is a physical theft of food or money occurring out in the field; penalizing the victim's *home preference* injects negative gradient into the victim's homing reflex channels (Outputs 0,1 with respect to Channels 3,4), training the victim *not* to return home!

---

## 3. Structural Hurdles That Make Training Harder

### 3.1 Eligibility Trace Horizon Collapse vs. Delayed Goal Rewards
This is the single greatest RL hurdle in the simulation:
- **Mathematical Form of the Trace**:
  In `neural.py` (lines 245–255 and 763–768):
  $$\mathbf{e}_{t} = \gamma \mathbf{e}_{t-1} + (1 - \gamma)(\mathbf{a}_{\text{pre}} \otimes \boldsymbol{\delta}_{\text{post}})$$
  where $\gamma = 0.82$ (single brain) or $\gamma = 0.75$ (batched pool).
- **The Half-Life Problem**:
  For $\gamma = 0.82$:
  $$t_{1/2} = \frac{\ln(0.5)}{\ln(0.82)} \approx 3.49 \text{ ticks}$$
  After just 15 ticks:
  $$\gamma^{15} = (0.82)^{15} \approx 0.051 \quad (95\% \text{ of the credit is lost})$$
  After 30 ticks:
  $$\gamma^{30} \approx 0.0026 \quad (99.74\% \text{ of the credit is lost})$$
- **The Physical Reality**:
  - A blob spawning in the cradle typically has to travel $60\text{--}180$ units to reach the nearest food item.
  - At average speed $\approx 2.5\text{--}3.5$ units/tick, navigation to a food item takes **$25$ to $70$ ticks**.
  - When the blob finally touches the food item at tick 60 and receives `eat_reward = +1.20`, the eligibility trace containing the steering decisions made at ticks $0\text{--}40$ has decayed to $< 0.0001$!
  - The sparse terminal reward `eat_reward` only reinforces the microscopic twitch made in the last 2 to 3 ticks immediately preceding mouth contact. It provides virtually **zero credit assignment** to the 50 ticks of sustained heading alignment that actually brought the blob to the food!

### 3.2 Continuous Shaping Dominance: The "Tail Wagging the Dog" Problem
Because the sparse terminal reward cannot bridge the 50-tick gap, the system relies entirely on per-tick movement shaping rewards:
- **Quantitative Comparison**:
  - In 600 ticks (1 day), a blob moving towards food or home accumulates:
    $$\text{Shaping Rewards} = 600 \text{ ticks} \times 0.05 \times 2.5 = 75.0 \text{ reward units}$$
  - In contrast, eating 2 food items yields:
    $$\text{Eating Rewards} = 2 \times (1.6 \times 0.75) = 2.4 \text{ reward units}$$
- **The Training Failure**:
  - Continuous movement shaping rewards exceed terminal consumption rewards by a factor of **$30\times$ to $50\times$**!
  - **Reward Hacking / Orbiting**: A blob discovers that it can maximize its reward by running circles around a cluster of food items without ever eating them. Every tick that it approaches a food item, it receives $+0.068 \times \text{pref\_exploration}$. When it passes the food, it turns around and approaches it again. Actually eating the food is a *negative* event for continued shaping, because once the food is gone, $\text{prox\_food} = 0$, and the rich $+0.068$ continuous stream instantly terminates!
  - Similarly, blobs discover they can vibrate back and forth near the boundary of the home zone at dusk, harvesting `home_align * 0.058` continuously without actually resting.

### 3.3 The Baseline Ratcheting & Advantage Cancellation Trap
- **The Mechanism**:
  In `neural.py` (lines 1151–1170):
  $$b_{t} = 0.96 b_{t-1} + 0.04 R_t$$
  $$\text{Advantage } A_t = \max(0.25 R_t, R_t - b_{t-1}) \quad (\text{for } R_t > 0)$$
- **Why This Hinders Convergence**:
  1. **Shaping Adaptation Decay**: If a blob sustains good behavior (consistently moving toward food for 25 ticks with $R_t \approx 0.10$), its baseline $b_t$ tracks up toward $0.10$. By tick 20, $R_t - b_{t-1} \approx 0$, so its advantage collapses to the artificial floor:
     $$A_t = 0.25 \times R_t$$
     This punishes consistency: the better the agent navigates over a prolonged trajectory, the lower its learning gradient becomes per tick!
  2. **The "Post-Meal Hangover" Trap**:
     When the blob eats a food item ($R_t = +1.20$), the baseline jumps:
     $$b_{t+1} \approx 0.96 \times 0.01 + 0.04 \times 1.20 = 0.058$$
     On the very next tick, the blob resumes normal movement shaping ($R_{t+1} \approx 0.02$).
     Now, $R_{t+1} - b_t = 0.02 - 0.058 = -0.038$!
     Although the code contains `raw_pos = np.maximum(0.25 * rewards, rewards - prev_b)` to clamp $A_{t+1} \ge 0.25 \times 0.02 = +0.005$, **90% of the normal shaping gradient is erased for the next 15–20 ticks** because the baseline was inflated by the sparse meal reward!

### 3.4 Temporal Distortion from 3-Cohort Strided Scheduling
- **The Implementation**:
  In `world.py` (line 3176):
  ```python
  cohort_idx = self._substep_count % 3
  do_trace = cohort_idx == 0
  ```
  `update_traces=do_trace` means that eligibility traces are only updated every 3rd simulation tick when using cohort scheduling!
  Meanwhile, physics, velocity, collisions, and food foraging execute every single tick.
- **The Inconsistency**:
  - On ticks where `do_trace = False` (ticks 1 and 2 of every 3-tick window), actions are taken and food might be eaten or rocks hit, but **no eligibility trace is recorded for those actions**.
  - When `apply_online_learning_batch` is called, it credits the weights using the trace from tick 0, attributing rewards earned on ticks 1 or 2 to the unrelated action chosen on tick 0!
  - This introduces substantial noise into gradient descent, significantly slowing convergence for dense sensory-motor loops like steering away from obstacles.

### 3.5 Disconnect Between Lamarckian RL and Evolutionary Selection Fitness
- The agent lives in a dual optimization landscape:
  - **Inner Loop (RL)**: Maximizes instantaneous discounted reward $R_t$ (dominated by movement shaping $+75.0$ vs $+2.4$ food).
  - **Outer Loop (Evolution)**: Maximizes survival and reproduction:
    $$\text{Selection Score} = \text{food\_eaten} \times 1.5 + \text{children} \times 3.0 + \text{cumulative\_reward} \times 0.3 + \text{reputation} \times 4.5$$
- **The Conflict**:
  - A blob with high movement shaping reward that never reproduces has fitness $\approx 75.0 \times 0.3 = 22.5$.
  - A quiet blob that makes straight, efficient moves to 2 food items and stays in home has fitness:
    $$\text{food} \times 1.5 + \text{children} \times 3.0 \approx 3.0 + 3.0 + (10.0 \times 0.3) = 9.0 \text{--} 15.0$$
  - Because `cumulative_reward` is dominated by sheer mileage of movement shaping, high-mileage jittery blobs that run around aimlessly frequently outscore calmer, energy-conserving foragers in mate selection and Hall of Fame preservation.
  - Furthermore, child blobs inherit parent neural network weights, but their **eligibility traces and baselines reset to zero**. If the parent learned weights that require high baseline compensation to avoid overshooting, the child begins life with zero baseline, causing extreme weight fluctuations during its first hours of life.

---

## 4. Synthesis of Logical Inconsistencies & Training Hurdles

| Component | Current Implementation | Logical Inconsistency | Impact on Training Hardness |
| :--- | :--- | :--- | :--- |
| **Movement vs. Eating Scale** | Movement shaping = $75.0$/day; Eating = $2.4$/day | Dense continuous reward is $30\times$ larger than primary ecological objective | Induces food orbiting and boundary vibrating; agents optimize for journey, not destination |
| **Trace Half-Life vs. Distance** | $\gamma = 0.82 \implies t_{1/2} = 3.5$ ticks; target distance $= 30\text{--}60$ ticks | Trace vanishes before food is reached | Zero credit assignment from terminal food consumption to early navigation steering |
| **Simulation Speed ($dt$) Scaling** | Movement shaping scaled by $dt \times 2.5$; Events scaled by $1.0\times$ | Training dynamics change drastically between $1\times$ and $10\times$ turbo speeds | High simulation speed disproportionately inflates movement shaping gradients over physical interactions |
| **End-of-Day (EOD) Rewards** | Home $+0.35$, outdoor $-0.25$, starve $-1.0$ applied to weights but NOT to `cumulative_reward` | Diurnal survival success is hidden from evolutionary selection | Evolutionary mate selection and Hall of Fame ignore homing and night survival |
| **Midday Starvation Penalty** | $-1.2 \times \text{pref\_food}$ backpropagated into dying blob immediately before deletion | Dead blobs are discarded; brain is never used again | Wasted CPU cycles; potential slot memory corruption in `BatchedBrainPool` during compaction |
| **Action vs. Velocity Shaping** | `approach = move_x * dx_food + move_y * dy_food` | Rewards instantaneous neural action output, ignoring physical inertia & current velocity | Penalizes efficient coasting; rewards twitching towards food even when drifting away |
| **Preference Weight Alignment** | Food approach shaping uses `pref_exploration`; home approach uses `pref_home` | User increasing `pref_food` does not increase movement gradient toward food | Asymmetrical control; `pref_home` overpowers food search even during peak morning foraging |
| **Raid Defender Penalty** | Defender penalized $-0.25 \times \text{pref\_home}$ upon being raided | Physical robbery in the field penalizes the victim's homing reflex | Victim's neural network learns *not* to navigate toward home |
| **Baseline Contamination** | Shared scalar baseline tracks both dense shaping ($\approx 0.05$) and discrete meals ($+1.20$) | Single sparse meal spike inflates baseline, suppressing shaping learning for 15+ ticks | Creates "post-meal learning paralysis" where good navigation actions receive zero/negative advantage |
| **3-Cohort Trace Striding** | Traces only recorded when `substep % 3 == 0` | Actions on ticks 1 and 2 have no trace; rewards earned on those ticks credit tick 0 | High gradient variance and noisy policy updates in fine obstacle avoidance |

---

## 5. Concrete Actionable Recommendations

To resolve these inconsistencies and dramatically streamline training for both the neural RL agent and the evolutionary population, the following architectural adjustments are recommended:

### 5.1 True Potential-Based Reward Shaping (PBRS)
Replace the arbitrary action dot-product `approach = move_x * dx + move_y * dy` with canonical **Potential-Based Reward Shaping** (Ng et al., 1999):
$$\Phi(s) = -\frac{\text{dist\_to\_food}}{\text{vision\_range}}$$
$$F(s, s') = \gamma \Phi(s') - \Phi(s)$$
- **Benefits**:
  1. Mathematically proven to preserve the optimal policy of the underlying MDP.
  2. Measures actual physical displacement $(\Delta x, \Delta y)$ rather than raw unclipped network outputs.
  3. Eliminates reward hacking: running in circles or orbiting a food item yields a net potential change of $0$, completely preventing agents from harvesting infinite shaping rewards without eating!

### 5.2 Decoupled Two-Stream Baselines (Dense vs. Sparse)
Decouple the baseline tracker into two separate exponential moving averages:
1. $\bar{b}_{\text{shaping}}$ for continuous per-tick movement shaping ($\alpha = 0.02$).
2. $\bar{b}_{\text{event}}$ for discrete interaction events (eating, trading, raiding).
- **Benefits**:
  Eating an apple ($+1.20$) will no longer contaminate $\bar{b}_{\text{shaping}}$, eliminating the post-meal learning paralysis and allowing the agent to immediately begin seeking the next food item without gradient suppression.

### 5.3 Adaptive Eligibility Trace Trace-Back ($\gamma$-Horizon Extension)
Increase the eligibility trace discount factor from $\gamma = 0.82$ to $\gamma = 0.94\text{--}0.96$:
$$t_{1/2}(\gamma = 0.95) \approx \frac{\ln(0.5)}{\ln(0.95)} \approx 13.5 \text{ ticks} \implies \text{trace lasts } 40\text{--}60 \text{ ticks}$$
- **Benefits**:
  Allows the discrete $+1.20$ reward from actually swallowing food to backpropagate across the entire trajectory of steering decisions that guided the blob to the food from $50$ units away.

### 5.4 Unified Reward Accounting & Simulation Speed Invariance
1. Normalize all reward inputs to `step_rewards` by the time delta:
   $$\text{step\_reward} = \left(R_{\text{shaping}} + \frac{R_{\text{event}}}{dt}\right) \times \text{scale}$$
   ensuring that whether the simulation runs at $1\times$ or $20\times$ turbo speed, total gradient steps per physical unit of game time remain constant.
2. Ensure every reward signal applied to neural weights (EOD home shelter $+0.35$, outdoor frost $-0.25$, starvation $-1.0$) is also mirrored into `blob.cumulative_reward`.
3. Remove the dead-blob backpropagation update in `_check_midday_starvation`. Instead, incorporate starvation mortality directly into the EOD fitness and reproductive culling pipeline.

### 5.5 Coherent Preference Weight Coupling
In `_compute_movement_shaping_reward_batch`:
- Scale food approach shaping by `self.prefs.pref_food * 0.068` (not `pref_exploration`).
- Reserve `pref_exploration` specifically for following distant oasis scents when no food is in sight (`oasis_align`).
- Remove the defender's `pref_home` penalty during raids; replace it with a neutral energy loss or stamina penalty that does not degrade home-seeking weights.
