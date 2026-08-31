---
layout: distill
title: DyLam
description: Learning with Decomposed and Dynamically Prioritized Reward Signals
htmlwidgets: true

# Anonymize when submitting
authors:
  - name: Anonymous
    affiliations:
      name: Anonymous

# Only add author names for camera-ready
# authors:
#   - name: Author Name
#     url: "https://[url_of_author]"
#     affiliations:
#       name: Research Center, University Name

# Must be the same name as your submission. Do not change this name, just use "submission.bib".
bibliography: submission.bib

# Add a table of contents to your submission.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the submission to work correctly.
toc:
  - name: Introduction
  - name: Mechanism
    subsections:
      - name: Sufficient Values
      - name: Q-Decomposition with Unified Action Selection
      - name: Dynamic Reward Weighting
      - name: "Stability of the Weight Update: Proof and Discussion"
  - name: Results
    subsections:
      - name: Learning-Dynamics-Oriented Environments
      - name: Robustness to Bound Misspecification
      - name: Pareto-Oriented Environments
  - name: Ablation
    subsections:
      - name: Effect of the DyLam's Update Rate
      - name: Effect of Experience Buffer Size
      - name: Effect of the Deficiency Transform
      - name: Scaling in the Number of Components
      - name: Effect of Exploration Rate Decay
---

## Introduction

Reinforcement learning agents are famously easy to reward and famously hard to reward *well*. Give an agent a single scalar signal and, given enough interaction, it will find a way to maximize it <d-cite key="sutton2018reinforcement"></d-cite> — the trouble starts when the thing you actually care about doesn't collapse into one number. Robotic manipulation <d-cite key="intelligence2025pi"></d-cite><d-cite key="todorov2012mujoco"></d-cite><d-cite key="rsoccer"></d-cite>, competitive game-playing <d-cite key="silver2016mastering"></d-cite><d-cite key="silver2018general"></d-cite><d-cite key="berner2019dota"></d-cite><d-cite key="vinyals2019grandmaster"></d-cite>, and aligning language models with human feedback <d-cite key="kaufmann2023survey"></d-cite> all end up juggling several objectives at once, and the standard move is to fold them into one reward with a set of hand-picked weights: $r = \sum_i \lambda_i r_i$.

Those weights are not cosmetic. They define an implicit curriculum <d-cite key="curriculum"></d-cite> — which sub-goal the agent is effectively being pushed toward at any given moment — and getting the curriculum wrong quietly wrecks training. Tilt the weights too far toward whichever component is easiest to farm and the agent will happily overfit to it: a soccer-playing agent rewarded generously for holding onto the ball can become excellent at holding onto the ball and never discover the sparser, harder objective of actually scoring. The usual fix is to sit down and hand-tune $\lambda$ — a slow, brittle, per-environment ritual that has to be redone from scratch the moment the reward structure changes even slightly.

People have tried to get out of this by other doors. Reward shaping <d-cite key="ng1999policy"></d-cite> adds auxiliary feedback to make learning easier, but the priorities it bakes in are static — they don't know or care how far along the agent already is. Curriculum learning <d-cite key="bengio2009curriculum"></d-cite> tackles the staging problem more directly, and automated variants <d-cite key="portelas2020automatic"></d-cite><d-cite key="li2023understanding"></d-cite><d-cite key="graves2017automated"></d-cite> — including self-paced learning, where the model's own performance decides what to emphasize next <d-cite key="kumar2010self"></d-cite> — go a long way toward removing the hand-crafting. But almost all of that machinery is built to sequence *tasks* or *environments* <d-cite key="xiao2025collaborative"></d-cite><d-cite key="lv2026metagrasp"></d-cite><d-cite key="mead2026multi"></d-cite>, not the components sitting inside a single reward. Hand-staged curricula over reward terms specifically remain a bespoke, per-domain exercise <d-cite key="xiao2023flying"></d-cite><d-cite key="lian2026curriculum"></d-cite><d-cite key="efendi2026learning"></d-cite>. Meanwhile, multi-objective RL methods that decompose the reward — Q-decomposition keeps one value function per component and coordinates them through a shared decision rule <d-cite key="russell2003q"></d-cite><d-cite key="vanseijen2017hybrid"></d-cite><d-cite key="fatemi2022orchestrated"></d-cite>, and methods like Envelope Q-learning or GPI-based prioritization <d-cite key="yang-envelope"></d-cite><d-cite key="alegre2023sample"></d-cite> push the idea further — but they're usually aimed at recovering a whole Pareto front of trade-off policies, not at handing back one well-shaped policy for a single deployment.

That's the gap DyLam sits in: nobody was treating the *weights themselves* as the curriculum. The idea is to keep the reward decomposed — one component, one signal — and let a competence-style measure decide, online, how much attention each component deserves right now, borrowing the "responsibility signal" intuition from Multiple Model-Based RL <d-cite key="doya2002multiple"></d-cite>. Concretely, DyLam tracks each component's recent return against a rough estimate of where that component saturates, and reweights toward whichever components are still underperforming while quietly turning down the ones the agent has already mastered. No fixed schedule, no per-environment retuning — just two numbers per component, a floor and a ceiling on its return, and the weights find their own way over the course of training. An interactive walkthrough of this update over one slice of real training data follows next, before the formal statement of the mechanism.

<link rel="stylesheet" href="{{ '/assets/css/mechanism.css' | relative_url }}">

Below: one worked window from a real ChickenBanana training run. Watch the recent per-Component rewards feed the sliding-window average, the average update DyLam's smoothed estimate of each component's progress, that estimate resolve into a weight, and the resulting weights produce a policy on the grid -- the same update every episode of training repeats. Play through it, or scrub to any beat and dwell on it.

<div class="l-page">
  <iframe src="{{ 'assets/html/submission/mechanism_explainer.html' | relative_url }}" frameborder="0" scrolling="no" height="420" width="100%" style="border: none;"></iframe>
</div>

## Mechanism {#mechanism}

This section formalizes DyLam: what makes a reward component safe to deprioritize ([Sufficient Values](#sufficient-values)), the Q-decomposition architecture that lets components share one policy ([Q-Decomposition with Unified Action Selection](#q-decomposition-with-unified-action-selection)), the weighting mechanism itself and the algorithm it drives ([Dynamic Reward Weighting](#dynamic-reward-weighting)), and the boundedness, Lipschitz-continuity, and bounded-variation guarantees the abstract advertises ([Stability of the Weight Update: Proof and Discussion](#stability-of-the-weight-update-proof-and-discussion)).

### Sufficient Values {#sufficient-values}

In this work, we define the decomposition through the lens of reward shaping rather than Pareto coverage. The ultimate task objective is typically sparse and hard to optimize directly. Therefore, we distinguish a *dominant component* $r_d$ from *auxiliary components* $r_i$ that provide denser feedback, guiding learning toward $r_d$. While some environments explicitly define a dominant objective, others rely on the emergent combination of auxiliary signals to induce optimal behavior <d-cite key="felten_toolkit_2023"></d-cite>.

We introduce the concept of a *sufficient value* for an auxiliary component, which formalizes when further optimization of that component becomes unnecessary or detrimental.

<div class="callout" id="def-sufficient-value" markdown="1">
<span class="callout-label">Definition 1 (Sufficient Value).</span>
Let $r_d$ denote the dominant reward, $r_i$ an auxiliary component, and write $r^\pi_j$ for the expected return of component $j$ under policy $\pi$. For a level $c$ attainable by component $i$, define the *best dominant return at level $c$*,

$$
\begin{equation}
\label{eq:level-set-frontier}
D_i(c) \;=\; \sup_{\pi \,:\, r_i^{\pi} = c} \; r_d^{\pi},
\end{equation}
$$

the largest dominant return achievable among policies whose component-$i$ return equals $c$. A value $\bar{r}_i$ is *sufficient* for component $i$ if $D_i$ is non-increasing on $[\bar{r}_i, \infty)$, that is, if $D_i(c') \leq D_i(c)$ whenever $c' \geq c \geq \bar{r}_i$.
</div>

The condition is stated on the level sets of $r_i$ rather than over arbitrary pairs of policies. The natural pairwise reading, $r_i^{\pi'} \geq r_i^{\pi} \geq \bar{r}_i \Rightarrow r_d^{\pi'} \leq r_d^{\pi}$, is strictly stronger than intended and admits no finite $\bar{r}_i$ in any environment where distinct policies share a component return while differing on the dominant one, which includes all of ours (a discussion the manuscript carries in an appendix not reproduced on this page). Eq. $\eqref{eq:level-set-frontier}$ keeps the intended meaning, that beyond the sufficient value further auxiliary return cannot buy a better dominant return, and states it without derivatives $\partial r_d^\pi / \partial r_i^\pi$, which would require a differentiable family of policies along which one component return varies while the others are held fixed.

Once an auxiliary objective reaches its sufficient value, continued optimization yields zero or negative returns on the dominant objective. This provides a principled criterion for deprioritizing components during training and motivates the dynamic weighting mechanism developed in [Dynamic Reward Weighting](#dynamic-reward-weighting).

### Q-Decomposition with Unified Action Selection {#q-decomposition-with-unified-action-selection}

Let the environment provide a vector reward $\vec{r}(s,a,s') = (r_1,\dots,r_n)$ with $n$ components. We assume each component is bounded and that the global reward is their sum. The agent learns $n$ separate action-value functions $Q_i(s,a)$, one per component.

In the original Q-decomposition framework <d-cite key="russell2003q"></d-cite>, each $Q_i$ is updated independently using its own greedy action. As discussed elsewhere in the manuscript, this leads to an *illusion of control* where components pursue conflicting policies. The remedy, adopted by the Hybrid Reward Architecture <d-cite key="vanseijen2017hybrid"></d-cite> and Orchestrated Value Mapping <d-cite key="fatemi2022orchestrated"></d-cite>, is to evaluate all components under a single globally greedy action. We refer to this established approach as *Unified Decision Control* (UDC) for brevity. Let the global action-value be:

$$
Q_G(s,a) = \sum_{i=1}^n \lambda_i Q_i(s,a),
$$

with $\lambda_i \ge 0$, normalized to $\sum_i \lambda_i = 1$ for the fixed-weight baselines and bounded in $[1, n]$ under DyLam's update ([Dynamic Reward Weighting](#dynamic-reward-weighting))<d-footnote>We use \(\lambda_i\) for component weights throughout. This should not be confused with the trace-decay parameter in TD\((\lambda)\) <d-cite key="sutton2018reinforcement"></d-cite>, which does not appear in this paper.</d-footnote>. The global policy is $\pi_G(s) = \arg\max_a Q_G(s,a)$, and each component bootstraps from this global action:

$$
\begin{equation}
\label{eq:udc-update}
Q_i(s_t,a_t) \leftarrow (1-\alpha)\, Q_i(s_t,a_t)
+ \alpha\bigl[ r_i(s_t,a_t,s_{t+1}) + \gamma\, Q_i(s_{t+1}, \pi_G(s_{t+1})) \bigr].
\end{equation}
$$

When the $\lambda_i$ are held constant, summing Eq. $\eqref{eq:udc-update}$ over $i$ recovers the standard Q-learning update for $Q_G$, and convergence to the optimal value function follows from <d-cite key="watkins1989learning"></d-cite>. For actor-critic architectures where the greedy policy is replaced by a parameterized stochastic policy $\pi_\theta$, the policy gradient extends directly by linearity of expectation and the policy gradient theorem <d-cite key="sutton1999policy"></d-cite>:

$$
\nabla_\theta J(\pi_\theta) = \mathbb{E}\bigl[\nabla_\theta \log \pi_\theta(a|s) \sum_i \lambda_i Q_i^{\pi_\theta}(s,a)\bigr],
$$

and approximation errors in each critic propagate linearly into the gradient bias. When the base learner is SAC <d-cite key="sac"></d-cite>, the entropy term is shared across component critics rather than decomposed; the manuscript's SAC appendix, not reproduced on this page, states the resulting target and the invariant it preserves.

DyLam generalizes the fixed weighting above to time-varying weights $\lambda_i^{(k)}$ that depend on the episode index $k$. All expressions in this subsection carry over verbatim with $\lambda_i$ replaced by $\lambda_i^{(k)}$; their soundness under time-varying weights is justified by the stability result of [Stability of the Weight Update](#stability-of-the-weight-update-proof-and-discussion).

### Dynamic Reward Weighting {#dynamic-reward-weighting}

Adaptive weighting of competing objectives is well studied in multi-task supervised learning <d-cite key="chen2018gradnorm, liu2019mtan, kendall2018multi"></d-cite>, under the shared principle of allocating more capacity to under-performing objectives. Three complications distinguish the RL setting, and DyLam answers each: the per-component "loss" is a non-stationary expected return under a changing policy, met by exponential moving averages of returns; the components are coupled through a shared action space, so reweighting one alters the global policy and every component's return, met by the unified action selection of [Q-Decomposition with Unified Action Selection](#q-decomposition-with-unified-action-selection); and the episodic-return signal is far noisier than a mini-batch loss, met by a smoothing factor $\tau_\lambda$ close to $1$.

A second family of methods adapts a scalar coefficient online for a different purpose, namely to enforce a constraint. The Lagrangian relaxation of a constrained MDP with per-component requirements $r_i \geq \bar{r}_i$ produces a multiplier that grows while the constraint is violated and relaxes once it is satisfied, which is the update signature of RCPO <d-cite key="tessler2019reward"></d-cite>, CPO <d-cite key="achiam2017constrained"></d-cite> and PID-Lagrangian <d-cite key="stooke2020responsive"></d-cite>, and of the automatic entropy tuning of SAC <d-cite key="sac"></d-cite> that our continuous-control instantiation uses. DyLam follows the same *direction* but not the same *dynamics*. A Lagrange multiplier integrates the violation, $\mu_i \leftarrow [\mu_i + \eta(\bar{r}_i - \overline{G}_i)]_+$, and therefore carries the history of past violations; Eq. $\eqref{eq:dylam-weights}$ maps the *current* smoothed deficiency to a weight without accumulating anything, so $\lambda_i$ is a memoryless function of $\zeta_i$ rather than an integral of it. The distinction is proportional versus integral control on a normalized constraint violation, and it has a concrete consequence: when a bound is set above what any policy can attain, a dual-ascent multiplier diverges, whereas DyLam's weight saturates at the ceiling of [Proposition 1](#prop-dylam-stability)(1) and the curriculum stalls with the weights bounded, which is the failure mode measured by the Results section's robustness experiments. A primal-dual method supplied with the same per-component thresholds is therefore a natural comparison, and one we do not run here; so is AutoRL-style evolutionary search over reward weights <d-cite key="chiang2019learning"></d-cite>, which solves the same practitioner problem by an outer loop over training runs rather than an inner loop within one.

**Mechanism.** Let $G^{(k)}_i$ be the cumulative return of component $i$ in the $k$-th completed episode. DyLam tracks each component's current performance in two stages. A ring buffer of length $E$ holds the most recently completed episodes and supplies their mean,

$$
\begin{equation}
\label{eq:dylam-window}
M^{(k)}_i = \frac{1}{E}\sum_{l=k-E+1}^{k} G^{(l)}_i,
\end{equation}
$$

which is then smoothed by an exponential moving average with factor $\tau_\lambda \in (0,1)$,

$$
\begin{equation}
\label{eq:dylam-ema}
\overline{G}_i \;\leftarrow\; \tau_\lambda\, \overline{G}_i + (1-\tau_\lambda)\, M^{(k)}_i .
\end{equation}
$$

The buffer average suppresses the episode-to-episode noise carried by a single return and the EMA sets how fast the estimate reacts. The weights are recomputed whenever Eq. $\eqref{eq:dylam-ema}$ is applied, which happens once per episode in the tabular implementation and once per environment step in the deep RL ones, where episodes terminate asynchronously across parallel environments and $M^{(k)}_i$ changes only as each one ends. This cadence difference is why the continuous-control experiments use $\tau_\lambda$ far closer to $1$ than the tabular ones (per the manuscript's hyperparameter-protocol appendix, not reproduced on this page): the effective horizon $1/(1-\tau_\lambda)$ counts weight updates, hence steps rather than episodes.

Using pre-defined return bounds $R^i_{\min}$ and $R^i_{\max}$, DyLam computes a normalized proficiency $\varrho$ and deficiency score $\zeta$ for each component:

$$
\varrho_i = \frac{\overline{G}^{(k)}_i - R^i_{\min}}
                          {R^i_{\max} - R^i_{\min}},
\qquad
\zeta_i = \operatorname{clip}\bigl(1 - \varrho_i,\; 0,\; 1\bigr).
$$

The clip ensures well-defined behavior when smoothed returns drift outside the prescribed range: a component above $R^i_{\max}$ is treated as fully optimized ($\zeta_i = 0$), and one below $R^i_{\min}$ has its deficiency capped ($\zeta_i = 1$). The deficiency is converted to a raw weight via $w_i = \mathrm{e}^{\zeta_i} - 1$, which is monotone in $\zeta_i$, equals zero exactly when the component is fully optimized, and amplifies differences between laggards and near-saturated components. The final weights are obtained by additive smoothing and normalization<d-footnote>We study the effects of different normalization methods elsewhere in the manuscript's ablation study.</d-footnote>,

$$
\begin{equation}
\label{eq:dylam-weights}
\lambda_i = \frac{w_i + \epsilon}{\sum_{j=1}^n w_j + \epsilon},
\end{equation}
$$

where $\epsilon > 0$ is a small stability constant (we use $\epsilon = 10^{-4}$) that prevents the weight of a saturated component from collapsing to exactly zero. The denominator carries a single $\epsilon$ rather than one per component, so the weights are not constrained to the simplex: $\sum_i \lambda_i = (\sum_j w_j + n\epsilon)/(\sum_j w_j + \epsilon)$, which differs from $1$ by $(n-1)\epsilon / (\sum_j w_j + \epsilon)$ and is therefore indistinguishable from $1$ while any component remains deficient. It reaches $n$ only in the degenerate case where every component has saturated ($w_j = 0$ for all $j$), which assigns $\lambda_i = 1$ to every component and recovers the plain unweighted sum of the decomposition. DyLam is thus a deficiency-routed reweighting of bounded scale rather than a strict convex combination, and [Proposition 1](#prop-dylam-stability) is stated for this form.

**Stability of the update.** The resulting weights are well behaved for any sequence of smoothed returns, with no assumption on the policy or on convergence. [Proposition 1](#prop-dylam-stability) is proved in [Stability of the Weight Update](#stability-of-the-weight-update-proof-and-discussion).

<div class="callout" id="prop-dylam-stability" markdown="1">
<span class="callout-label">Proposition 1 (Bounded and Lipschitz weight updates).</span>
Let $\lambda_i(\overline{\mathbf{G}})$ denote the weight assigned to component $i$ by Eq. $\eqref{eq:dylam-weights}$, viewed as a function of the smoothed-return vector $\overline{\mathbf{G}} = (\overline{G}^1, \dots, \overline{G}^n)$, and let $S = \sum_j w_j$. Let $\mathrm{e} \approx 2.718$ denote Euler's number. Then:

1. **(Boundedness)** For every component $i$ and every update $t$,

   $$
   \frac{\epsilon}{(n-1)(\mathrm{e}-1) + \epsilon}
   \;\le\; \lambda_i^{(t)} \;\le\; 1,
   \qquad
   1 \;\le\; \sum_{i=1}^n \lambda_i^{(t)} \;\le\; n,
   $$

   with $\lambda_i = 1$ attained only when component $i$ carries the entire deficiency, and $\sum_i \lambda_i = n$ only when all have saturated ($S = 0$).

2. **(Lipschitz continuity)** The map $\lambda_i: \mathbb{R}^n \to (0,1]$ is Lipschitz in each smoothed return, with constant

   $$
   L \;\le\; \frac{\mathrm{e}}{(\min_j (R^j_{\max} - R^j_{\min})) \cdot \epsilon}.
   $$

3. **(Bounded per-update change)** Under the smoothing rule $\overline{G}^i \leftarrow \tau_\lambda \overline{G}^i + (1-\tau_\lambda) M^i$ of Eq. $\eqref{eq:dylam-ema}$,

   $$
   \bigl|\lambda_i^{(t)} - \lambda_i^{(t-1)}\bigr|
   \;\le\; L \cdot (1-\tau_\lambda) \cdot
   \max_j \bigl|M^j - \overline{G}^{j,(t-1)}\bigr| ,
   $$

   so weight changes vanish as $\tau_\lambda \to 1$. Since $M^j$ is constant between episode terminations, $T$ updates within one such interval induce a total weight change of at most $L\,(1 - \tau_\lambda^{T}) \max_j \lvert M^j - \overline{G}^{j}\rvert$, regardless of $T$.
</div>

A component can therefore never be switched off entirely, noisy return estimates translate into proportionally noisy weights rather than jumps, and the trajectory can be made arbitrarily gradual through $\tau_\lambda$. Part (3) also licenses the per-step cadence: between episode terminations $M^j$ is fixed, so step-level updates trace a geometric approach to that target whose total displacement is bounded regardless of how many steps the interval contains.

**Algorithm.** The algorithm below summarizes DyLam for an actor-critic architecture. The value-based case is identical, with the exception that the policy update is replaced by greedy action selection over $Q_G$.

<figure class="algorithm" id="alg-dylam" markdown="1">

1. Set buffer length $E$, smoothing factor $\tau_\lambda$, stability constant $\epsilon$, learning rates $\alpha_\theta, \alpha_\phi$.
2. Set bounds $R^i_{\min}, R^i_{\max}$ for each reward $i$. *(From literature or domain knowledge.)*
3. Initialize $\theta$, critics $\phi_i$, smoothed returns $\overline{G}_i \gets 0$, weights $\lambda_i \gets 1/n$.
4. **`for`** each episode $k = 1, 2, \dots$ **`do`**
   1. Reset environment, observe $s_0$; initialize $G^{(k)}_i \gets 0$ for all $i$.
   2. **`for`** each step $t$ **`do`**
      1. Sample $a_t \sim \pi_\theta(\cdot \mid s_t)$, execute $a_t$, observe $s_{t+1}$ and $\vec{r}$.
      2. Update each $Q^{\phi_i}_i$ via Eq. $\eqref{eq:udc-update}$ using $\pi_G$ from the current $\vec{\lambda}$.
      3. Accumulate $G^{(k)}_i \gets G^{(k)}_i + r_i$; when an episode ends, push $G^{(k)}$ into the length-$E$ buffer and refresh $M^{(k)}$ by Eq. $\eqref{eq:dylam-window}$.
      4. Update $\overline{G}_i$ by Eq. $\eqref{eq:dylam-ema}$; recompute $\varrho_i$, $\zeta_i$, $w_i$ and $\lambda_i$ by Eq. $\eqref{eq:dylam-weights}$. *(Every step here; once per episode in the tabular case.)*
      5. Update actor: $\theta \gets \theta + \alpha_\theta \nabla_\theta \log\pi_\theta(a_t \mid s_t) \sum_i \lambda_i\, Q^{\phi_i}_i(s_t, a_t)$. *(Baseline subtraction omitted.)*
   3. **`end for`**
5. **`end for`**

<figcaption>Algorithm 1: DyLam Actor-Critic.</figcaption>
</figure>

DyLam introduces three hyperparameters: the buffer length $E$, the smoothing factor $\tau_\lambda$, and the component bounds $R^i_{\min}, R^i_{\max}$. The pair $(E, \tau_\lambda)$ fixes the responsiveness of the weight signal: $E$ sets how many episodes are averaged before smoothing, and $\tau_\lambda$ gives an effective horizon of $1/(1-\tau_\lambda)$ weight updates. Because the deep RL implementations update the weights at every environment step rather than at episode boundaries, that horizon is measured in steps there, which is why $\tau_\lambda = 0.9999$ on HalfCheetah and VSS ($10^4$ steps, i.e. tens of episodes) plays the same role as $\tau_\lambda = 0.995$ on the tabular tasks ($200$ episodes). We set $\tau_\lambda$ close to $1$ in both regimes to ensure gradual weight evolution: keeping the per-update weight change small is what allows the critics to treat the scalarization as quasi-static ([Proposition 1](#prop-dylam-stability)). For benchmark environments, the bounds are obtained from established baselines in the literature; for novel environments, the reward designer typically knows the theoretical bounds of each component by construction. The Results section's robustness experiments measure what moderate misspecification of each costs, and in which direction. Eliciting the bounds is nonetheless a requirement the method does not remove, and no procedure here automates it. Where a bound is uncertain, the safer direction to err in depends on the component. Setting $R^i_{\max}$ too low is safe for a component that a later one depends on, since the latter keeps the former alive, but abandons a terminal component outright; an unattainable ceiling keeps $\zeta_i$ pinned near $1$ and holds the weight mass on that component indefinitely, which stalls the curriculum upstream and is harmless on the component nearest the objective.

### Stability of the Weight Update: Proof and Discussion {#stability-of-the-weight-update-proof-and-discussion}

This section proves [Proposition 1](#prop-dylam-stability) of [Dynamic Reward Weighting](#dynamic-reward-weighting) and discusses how tight its bounds are in practice. The guarantees do not depend on the policy's convergence or on any equilibrium assumption, and are stated for Eq. $\eqref{eq:dylam-weights}$ as implemented, whose denominator carries a single stability constant, so the weights are bounded in scale rather than confined to the simplex.

<div class="callout" id="proof-dylam-stability" markdown="1">
<span class="callout-label">Proof sketch.</span>
For (1), each $\zeta_i \in [0,1]$ implies $w_i = \mathrm{e}^{\zeta_i} - 1 \in [0, \mathrm{e}-1]$, hence $S \in [0, n(\mathrm{e}-1)]$. The numerator of Eq. $\eqref{eq:dylam-weights}$ is at least $\epsilon$ and the denominator at most $(n-1)(\mathrm{e}-1) + \epsilon$ when $w_i = 0$, giving the lower bound; the ratio is at most $1$ because $w_i \le S$, with equality exactly when $w_j = 0$ for all $j \neq i$. Summing gives $\sum_i \lambda_i = (S + n\epsilon)/(S + \epsilon)$, which decreases from $n$ at $S = 0$ toward $1$ as $S$ grows. For (2), $\varrho_i$ is linear in $\overline{G}^i$ with slope $1/(R^i_{\max} - R^i_{\min})$; the clip and the exponential are $1$- and $\mathrm{e}$-Lipschitz on $[0,1]$ respectively; and for the normalization, $\partial \lambda_i / \partial w_i = (S - w_i)/(S+\epsilon)^2 \le 1/(4\epsilon)$ and $\lvert\partial \lambda_i / \partial w_j\rvert = (w_i + \epsilon)/(S+\epsilon)^2 \le 1/(S + \epsilon) \le 1/\epsilon$ for $j \neq i$. Composing these factors yields the stated bound. For (3), $\lvert\overline{G}^i_{t} - \overline{G}^i_{t-1}\rvert = (1-\tau_\lambda)\lvert M^i - \overline{G}^i_{t-1}\rvert$ by definition of the EMA, and applying (2) to each coordinate of $\overline{\mathbf{G}}$ gives the per-update bound. Iterating the EMA $T$ times with $M^i$ held fixed gives $\overline{G}^i_{t+T} = \tau_\lambda^{T}\overline{G}^i_{t} + (1-\tau_\lambda^{T})M^i$, from which the interval bound follows.
</div>

Proposition 1 provides three concrete guarantees: weights cannot collapse to $0$, and reach their upper bound only in the extreme configuration where a single component carries all remaining deficiency (preserving exploration over all components); weights are smooth functions of the observed returns (so noisy episodic returns produce only proportionally noisy weight updates); and weight changes are controlled by $\tau_\lambda$ (so weight evolution can be made arbitrarily gradual). The interval bound in (3) is what licenses the per-step update cadence of the deep RL implementations: because the buffer mean $M$ is piecewise constant between episode terminations, updating the weights every step does not make them move faster than the geometric approach to that fixed target, and the displacement accumulated over an interval is bounded independently of the number of updates it contains. Together, these properties ensure that the policy never sees abrupt scalarization shifts during training, justifying the quasi-static treatment of weights in the analysis of standard RL algorithms applied to the scalarized reward at any given episode. The bounds themselves are worst-case, corresponding to extreme configurations in which a single component dominates the entire weight budget; because $R^i_{\min}$ and $R^i_{\max}$ are hand-shaped from domain knowledge and reward components tend to be concurrent rather than fully orthogonal, the empirical Lipschitz constant and per-episode weight variation observed in our experiments (see the Results section) are substantially tighter than the worst case suggests. The stability guarantees should therefore be read as protection against pathological cases rather than as predictions of typical behavior.

**Compatibility with experience replay.** Stored transitions do not become stale as the weights evolve. Each critic $Q_i$ is updated from the raw component reward $r_i$ and its own bootstrapped value (Eq. $\eqref{eq:udc-update}$); the weights enter only at the policy level, through $\pi_G(s) = \arg\max_a \sum_i \lambda_i Q_i(s,a)$ and the weighted critic sum in the policy gradient. The buffer therefore stores unweighted transitions $(s_t, a_t, r_1, \dots, r_n, s_{t+1})$, valid under any weighting, and the only indirect effect of a weight change is through $\pi_G(s_{t+1})$, recomputed at sample time rather than stored.

<link rel="stylesheet" href="{{ '/assets/css/mechanism.css' | relative_url }}">

## Results

We evaluate along three axes: learning-dynamics-oriented environments against static-weight scalarizations ([Learning-Dynamics-Oriented Environments](#learning-dynamics-oriented-environments)), robustness to misspecified reward bounds ([Robustness to Bound Misspecification](#robustness-to-bound-misspecification), RQ3), and Pareto-oriented benchmarks against dedicated MORL methods ([Pareto-Oriented Environments](#pareto-oriented-environments)).

### Learning-Dynamics-Oriented Environments {#learning-dynamics-oriented-environments}

We begin with DyLam's primary use case: environments where the goal is a single high-performing policy and the reward decomposes into a dominant component and auxiliary signals. This section addresses RQ1 (sample efficiency relative to static scalarizations) and RQ2 (whether the weight update induces an implicit curriculum) across three environments of increasing difficulty, specified in [the bound-derivation table below](#bound-derivation): the Chicken–Banana diagnostic, HalfCheetah-v4, and VSS-v0, whose dominant objective is sparse and whose auxiliary components are required for learning to occur at all. Throughout this section, comparisons are exact two-sided Mann–Whitney $U$ tests on the per-seed mean of the final $10\%$ of training, Holm–Bonferroni corrected within the three-comparison RQ1 family (one test per environment, DyLam against the strongest baseline there) and reported with rank-biserial effect sizes $r$. Every comparison is over ten seeds per method. [Performance (RQ1)](#performance-rq1) reports every method and metric in full.

**Bound specification.** The three environments instantiate three ways of obtaining bounds, in decreasing order of prior knowledge: *algebraic* for Chicken–Banana, where each maximum follows from the reward definition; *calibrated* for HalfCheetah-v4, where the velocity ceiling is the cumulative velocity reward of a converged SAC policy <d-cite key="sac"></d-cite>; and *reference-trajectory* for VSS-v0, where each shaping component is integrated along a hand-designed trajectory. In all three $R_{\max}$ encodes a sufficient rather than an unattainable value. [Bound Derivation](#bound-derivation) gives the derivations and the values; we recommend the reference-trajectory recipe when neither algebraic bounds nor a reference policy are available.

<figure id="fig-res-all" class="results-figure fig-row">
  <div>
    <img src="{{ 'assets/img/submission/trad_reward_total.svg' | relative_url }}" alt="Chicken-Banana learning curve">
    <figcaption>(a) Chicken-Banana.</figcaption>
  </div>
  <div>
    <img src="{{ 'assets/img/submission/trad_halfcheetah_position.svg' | relative_url }}" alt="HalfCheetah-v4 final x-position learning curve">
    <figcaption>(b) HalfCheetah.</figcaption>
  </div>
  <div>
    <img src="{{ 'assets/img/submission/trad_vss_goal.svg' | relative_url }}" alt="VSS-v0 goal-rate learning curve">
    <figcaption>(c) VSS.</figcaption>
  </div>
  <figcaption>Learning curves summarized across seeds by the interquartile mean, with bands giving a $95\%$ bootstrap confidence interval over seeds ($2000$ resamples) <d-cite key="agarwal2021deep"></d-cite>; each seed's history is smoothed by a rolling mean before aggregation. Colors denote <strong>DyLam</strong> (green), <strong>UDC</strong> (blue), <strong>Q-Learning</strong>/<strong>SAC</strong> (orange), and <strong>Q-Decomp</strong>/<strong>Tuned-UDC</strong> (purple). Every series is $n = 10$ seeds. Metrics: cumulative episode reward, final $x$-position (m), and goal rate.</figcaption>
</figure>

#### Performance (RQ1) {#performance-rq1}

The table below reports final performance across all three environments with the significance tests; the discussion below walks through each in turn, and the efficiency table further down restates the same runs by interquartile mean and reports sample efficiency in full.

<div class="results-table-wrap" markdown="0">
<table>
<caption>Final performance (mean $\pm$ std over seeds of the per-seed mean of the final $10\%$ of training); every cell is $n = 10$ seeds unless noted. $^\ast$: significant improvement over the strongest baseline in that environment (exact two-sided Mann–Whitney $U$, Holm–Bonferroni corrected within the three-comparison RQ1 family, one test per environment): Chicken–Banana vs. UDC $U = 91$, $p = 0.0011$, $p_{\mathrm{Holm}} = 0.0021$, rank-biserial $r = 0.82$; HalfCheetah-v4 vs. Base SO RL $U = 81$, $p = p_{\mathrm{Holm}} = 0.019$, $r = 0.62$; VSS-v0 vs. Tuned-UDC $U = 100$, $p = 1.1 \times 10^{-5}$, $p_{\mathrm{Holm}} = 3.2 \times 10^{-5}$, $r = 1.00$. The HalfCheetah environment-return column re-scores the same runs on the environment's own scalar reward and is not a fourth family member; on it the same comparison gives $U = 81$, $p = 0.019$, $r = 0.62$. DyLam-Scalar applies DyLam's weight update to a single scalar critic with no Q-decomposition; DyLam-OpenLoop replays DyLam's own mean weight trajectory with no return feedback (both defined in the manuscript's experimental-setup section, not reproduced on this page). Neither is a baseline; both are excluded from the RQ1 family, and their test statistics and discussion are reported under Mechanism Ablations below. Entries marked — are inapplicable, not missing: Q-Decomposition needs tabular per-component values, Tuned-UDC needs an expert-tuned weighting, which exists only for VSS-v0 <d-cite key="rsoccer"></d-cite>, and each ablation is implemented for a subset of the environments (DyLam-Scalar for the tabular environment only; DyLam-OpenLoop for Chicken–Banana and VSS-v0, not HalfCheetah-v4). The significance tests above are quoted from the manuscript at $n=10$ per method; the HalfCheetah-v4 and VSS-v0 UDC/Tuned-UDC rows below are regenerated live from this repository's currently committed data, where UDC and Tuned-UDC do not yet have the full ten seeds on those two environments (see cell counts) -- a gap in seed collection, not in the method, tracked separately from this port.</caption>
<thead><tr><th>Method</th><th>Chicken–Banana</th><th>HalfCheetah-v4</th><th>HalfCheetah-v4 (env return)</th><th>VSS-v0</th></tr></thead>
<tbody>
<tr><td>Base SO RL</td><td>127.000 ± 9.487 (n=10)</td><td>409.407 ± 36.048 (n=10)</td><td>8043.4 ± 708.8 (n=10)</td><td>0.070 ± 0.023 (n=10)</td></tr>
<tr><td>Q-Decomposition</td><td>90.149 ± 17.153 (n=10)</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>UDC</td><td>129.985 ± 0.047 (n=10)</td><td>329.980 ± 80.324 (n=6)</td><td>6480.0 ± 1579.1 (n=6)</td><td>0.060 ± 0.012 (n=5)</td></tr>
<tr><td>Tuned-UDC</td><td>—</td><td>—</td><td>—</td><td>0.424 ± 0.115 (n=7)</td></tr>
<tr><td>DyLam</td><td>185.826 ± 29.428* (n=10)</td><td>464.568 ± 47.627* (n=10)</td><td>9086.5 ± 962.1* (n=10)</td><td>0.852 ± 0.021* (n=10)</td></tr>
<tr><td>DyLam-Scalar</td><td>3.985 ± 8.369 (n=10)</td><td>—</td><td>—</td><td>—</td></tr>
</tbody>
</table>
</div>

**Chicken–Banana.** Objects sit at increasing distance from the start (Banana, Gate, Chicken) and reaching the Gate ends the episode, so an agent that prioritizes the nearer terminal Gate never encounters Chicken-reaching trajectories. This failure mode is real and exploitable: standard Q-learning with $\epsilon$-greedy exploration ($\epsilon_d = 0.9988$) never collects the Chicken within 2000 episodes. Our central finding is that *every* static-weight method exhibits it and only DyLam resolves it. Q-Learning, Q-Decomposition and UDC all plateau well below saturation on the decisive Chicken component in every seed, whereas DyLam reaches full collection by roughly episode 1250 and holds it in $8$ of $10$ seeds ([panel (a) above](#fig-res-all); per-component curves in [Chicken–Banana Environment and Per-Component Detail](#chicken-banana-environment-and-per-component-detail)). The remaining two seeds plateau at the UDC level, so the primary statistic for this environment is the success rate, $8/10$ for DyLam against $0/10$ for Base SO RL, Q-Decomposition and UDC alike; the outcome is bimodal and a mean over it is a poor summary, which is why we also report the interquartile mean, $199.71\ [165.00, 200.00]$ for DyLam against $130.00\ [129.98, 130.00]$ for UDC, alongside the mean of $185.826 \pm 29.428$ against $129.985 \pm 0.047$ ($n = 10$ per method; $U = 91$, $p_{\mathrm{Holm}} = 0.0021$, $r = 0.82$). The mechanism is not a guarantee, but no static weighting solves the task in any seed. The static methods reach reasonable Banana and Gate performance only because those components never have to resolve the structural bias.

**HalfCheetah-v4.** SAC uses the environment's default scalar reward, whose implicit weighting ($\lambda_{\text{velocity}} = 1.0$, $\lambda_{\text{control}} = 0.1$) was chosen by the environment's designers; our implementation reproduces the published CleanRL reference <d-cite key="huang2022cleanrl"></d-cite>, making it a representative tuned single-objective baseline. DyLam exceeds it after a warm-up of roughly $10^5$ steps, reaching a final $x$-position of $464.568 \pm 47.627$ against SAC's $409.407 \pm 36.048$ ($U = 81$, $p_{\mathrm{Holm}} = 0.019$, $r = 0.62$; [panel (b) above](#fig-res-all)). UDC is high-variance and fails to match SAC, a useful sanity check that naive equal weighting of decomposed components is not in itself beneficial. Because final $x$-position rewards exactly the component DyLam upweights and ignores the control cost it suppresses, we also score these runs on the environment's own scalar reward, velocity minus $0.1\times$ control cost. DyLam leads there as well ($9086.5 \pm 962.1$ against SAC's $8043.4 \pm 708.8$ and UDC's $5593.5 \pm 1649.5$), so the advantage is not an artifact of the reporting axis: the curriculum buys more velocity than the extra control effort costs under the designers' own exchange rate.

**VSS-v0.** Goal-scoring contributes no reward term: it is the held-out metric, and the three shaping components are designed so that mastering them induces a goal-scoring policy, so any advantage must come from how those signals are prioritized over training. Within $5 \times 10^5$ steps DyLam is the only method that reliably learns to score ([panel (c) above](#fig-res-all)). The contrast with Tuned-UDC is the informative one: its static weights come from the reward specification the rSoccer authors hand-tuned for this exact task <d-cite key="rsoccer"></d-cite>, rebalanced to episode-level component contributions, and even that domain-expert configuration plateaus at roughly half DyLam's goal rate ($0.388 \pm 0.116$ against $0.852 \pm 0.021$; $U = 100$, $p_{\mathrm{Holm}} = 3.2 \times 10^{-5}$, $r = 1.00$). SAC and uniform UDC perform worse still, neither learning to score within the budget. This is stronger than HalfCheetah-v4, where DyLam won on a quantitative margin: here the static methods do not solve the task, and DyLam does so without access to the expert-tuned configuration.

**Sample efficiency.** RQ1 also asks about sample efficiency. DyLam reaches SAC's final HalfCheetah-v4 level in roughly two-thirds of the budget SAC itself needs and in nine seeds of ten, and reaches the expert-tuned VSS-v0 goal rate slightly faster than the expert-tuned weights themselves, while the untuned baselines only touch that level transiently and never hold it. The efficiency table below reports the per-environment budgets and learning-curve areas, including the one environment where DyLam is the slower method ($200$ episodes against $98$).

<div class="results-table-wrap" markdown="0">
<table>
<caption>The same runs summarized by the interquartile mean with a $95\%$ bootstrap confidence interval ($10^4$ resamples over seeds), following the reporting recommendations of <d-cite key="agarwal2021deep"></d-cite>. On Chicken–Banana the mean and IQM order differently for DyLam-OpenLoop, UDC and Base SO RL alone: the means order DyLam-OpenLoop ($133.985$) above UDC ($129.985$) above Base SO RL ($127.000$), whereas all three share an IQM of $130.00$. They differ most in magnitude on DyLam's own Chicken–Banana entry, where the IQM of $199.710$ against a mean of $185.826$ exposes the bimodality noted above (eight of ten seeds solve the task, two plateau at the UDC level).</caption>
<thead><tr><th>Method</th><th>Chicken–Banana</th><th>HalfCheetah-v4</th><th>HalfCheetah-v4 (env return)</th><th>VSS-v0</th></tr></thead>
<tbody>
<tr><td>Base SO RL</td><td>130.00 [125.00, 130.00]</td><td>407.1 [382.6, 436.8]</td><td>7998.9 [7516.5, 8581.7]</td><td>0.071 [0.056, 0.087]</td></tr>
<tr><td>Q-Decomposition</td><td>92.79 [76.77, 103.38]</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>UDC</td><td>130.00 [129.98, 130.00]</td><td>333.3 [260.4, 395.4]</td><td>6543.5 [5114.5, 7766.1]</td><td>0.060 [0.048, 0.073]</td></tr>
<tr><td>Tuned-UDC</td><td>—</td><td>—</td><td>—</td><td>0.394 [0.355, 0.516]</td></tr>
<tr><td>DyLam</td><td>199.71 [165.00, 200.00]</td><td>472.6 [434.0, 496.9]</td><td>9249.9 [8467.7, 9742.9]</td><td>0.848 [0.838, 0.865]</td></tr>
<tr><td>DyLam-Scalar</td><td>0.29 [0.00, 8.18]</td><td>—</td><td>—</td><td>—</td></tr>
</tbody>
</table>
</div>

**Sample-efficiency protocol.** RQ1 asks about sample efficiency as well as final performance, so the table below reports, per environment, the interaction budget each method needs to reach the *strongest baseline's* final level, together with the normalized area under the learning curve. The threshold is the strongest baseline's own final mean, so that baseline reaches it by construction and the question is how quickly the others do. Three conventions are worth making explicit. First, *reached* counts a seed whose $20$-point rolling mean attains the threshold at any point within the budget; a seed that touches the threshold and later falls back still counts, so the statistic measures arrival rather than retention. Second, the threshold is a *baseline-relative* level, not a task-completion criterion: reaching it on VSS-v0 means matching the expert-tuned static weighting's final goal rate of $0.388$, well under half of the goal rate DyLam itself attains and not the point at which the domain objective is considered solved. The environments have no absolute success threshold apart from Chicken–Banana, where full collection of all three objects is the natural criterion and is reported separately as the $8/10$ success rate above. Third, the area under the learning curve (AUC) is computed after normalizing each environment's metric to the $[\min, \max]$ range observed across the baselines and DyLam in that environment, so it is comparable within an environment and meaningless across environments; it summarizes the whole trajectory rather than its endpoint, and therefore separates a method that arrives late and keeps improving from one that arrives early and plateaus.

<div class="results-table-wrap" markdown="0">
<table>
<caption>Sample efficiency. "Reached" counts seeds whose $20$-point rolling mean attains the threshold within the budget; the median is taken over those seeds only, so a low count makes the median optimistic. AUC is the mean over seeds of the learning curve normalized to the $[\min, \max]$ range observed across all methods in that environment. Chicken–Banana is measured in episodes, the other two in environment steps.</caption>
<thead><tr><th>Environment</th><th>Method</th><th>Reached</th><th>Median budget</th><th>AUC</th></tr></thead>
<tbody>
<tr><td>Chicken–Banana (≥ 129.99)</td><td>Base SO RL</td><td>9/10</td><td>132 ep.</td><td>0.453 ± 0.066</td></tr>
<tr><td></td><td>Q-Decomposition</td><td>10/10</td><td>212 ep.</td><td>0.357 ± 0.052</td></tr>
<tr><td></td><td>UDC</td><td>10/10</td><td><strong>98 ep.</strong></td><td>0.477 ± 0.002</td></tr>
<tr><td></td><td>DyLam</td><td>10/10</td><td>200 ep.</td><td><strong>0.662 ± 0.126</strong></td></tr>
<tr><td>HalfCheetah-v4 (≥ 409.4)</td><td>Base SO RL</td><td>5/10</td><td>359k steps</td><td>0.358 ± 0.072</td></tr>
<tr><td></td><td>UDC</td><td>1/6</td><td>418k steps</td><td>0.182 ± 0.143</td></tr>
<tr><td></td><td>DyLam</td><td>9/10</td><td><strong>227k steps</strong></td><td><strong>0.523 ± 0.128</strong></td></tr>
<tr><td>VSS-v0 (≥ 0.424)</td><td>Base SO RL</td><td>0/10</td><td>never</td><td>0.068 ± 0.007</td></tr>
<tr><td></td><td>UDC</td><td>0/5</td><td>never</td><td>0.080 ± 0.008</td></tr>
<tr><td></td><td>Tuned-UDC</td><td>7/7</td><td>118k steps</td><td>0.368 ± 0.018</td></tr>
<tr><td></td><td>DyLam</td><td>10/10</td><td><strong>105k steps</strong></td><td><strong>0.670 ± 0.030</strong></td></tr>
</tbody>
</table>
</div>

The answer to RQ1 differs by environment, and the Chicken–Banana row is the honest qualification. There, DyLam is *slower* to reach UDC's plateau ($200$ episodes against $98$) because it spends its early episodes on the Chicken component that no static weighting ever learns; it crosses the threshold late and then continues past it, which is what the AUC gap records. On the two harder environments the efficiency advantage is unambiguous: DyLam reaches SAC's final HalfCheetah performance in roughly two-thirds of the budget SAC itself needs and in nine seeds of ten against five of ten, and reaches the expert-tuned VSS-v0 goal rate in roughly the same budget the expert-tuned weights themselves need. The VSS-v0 baseline counts need one qualification, because the convention above measures arrival rather than retention and the threshold is a level the untuned baselines touch only in passing: read as retention rather than arrival, no untuned baseline reaches the expert-tuned level in any seed, which is what the final-performance table above and the AUC column here both show.

#### Mechanism Ablations {#mechanism-ablations}

DyLam couples two things: a decomposed critic and a weight vector that keeps moving. UDC (above) holds the decomposition fixed and removes the dynamic weights; the two ablations below make the reverse cuts, one each, so that the three configurations together isolate both halves of the mechanism. Both are run on Chicken–Banana, the environment in which every static-weight method fails outright and the cut is therefore sharpest; DyLam-OpenLoop is additionally run on VSS-v0. Their rows appear in the performance and IQM tables above. Neither is a baseline and neither enters the RQ1 family; the three comparisons reported here form their own mechanism-ablation family, Holm–Bonferroni corrected within it.

**Isolating the decomposition.** DyLam-Scalar keeps DyLam's weight update unchanged, computing $\boldsymbol{\lambda}^{(k)}$ from the per-component episodic returns exactly as before, but trains one $Q$-table on the scalarized reward $r = \sum_i \lambda_i^{(k)} r_i$ instead of $n$ component critics. It does not merely fail to solve the task: it collapses to $3.985 \pm 8.369$ over 10 seeds, $0/10$ solved, far below the $127.000$ of plain Q-learning ($U = 100$, exact $p = 1.1 \times 10^{-5}$, $p_{\mathrm{Holm}} = 3.3 \times 10^{-5}$, $r = 1.00$ against DyLam). This realizes the prediction of the [Mechanism](#mechanism) section: a single scalar critic stores values for the scalarization in force when it was updated, and as $\boldsymbol{\lambda}$ moves those values become estimates of an objective that no longer exists. Decomposed critics avoid this because each is updated from its own raw component reward and the weights enter only at action selection, so a weight change re-mixes current values instead of invalidating stale ones: the decomposition is what makes moving weights safe. The reversed min–max sign-flip control in the [Ablation](#ablation) section gives complementary evidence for the weighting side of the mechanism, inverting only the direction of routing and collapsing performance as a result.

**Isolating continued adaptation.** DyLam-Scalar asks whether decomposed critics are necessary given moving weights; it says nothing about whether the weights need to keep moving. To test that, we recorded DyLam's own mean $\boldsymbol{\lambda}$ trajectory, averaged across all ten closed-loop seeds of each environment, and replayed it as a fixed open-loop schedule on fresh seeds with neither return feedback nor bounds. On VSS-v0, DyLam-OpenLoop reaches $0.759 \pm 0.134$ goal rate ($n = 10$), $89\%$ of closed-loop DyLam's $0.852 \pm 0.021$; the gap favours DyLam but is not significant ($U = 73.5$, $p = p_{\mathrm{Holm}} = 0.089$, $r = +0.47$). On Chicken–Banana the same replay separates clearly: $133.985 \pm 25.036$ against DyLam's $185.826 \pm 29.428$, and where DyLam solves $8/10$ seeds, the fixed replay of its own recorded trajectory solves only $1/10$, plateauing at the UDC level in the rest ($U = 86.5$, $p = 0.0052$, $p_{\mathrm{Holm}} = 0.0104$, $r = +0.73$).

**What the replay does and does not establish.** One property of the construction bounds the Chicken–Banana reading, and we state it rather than leave it implicit. The replayed schedule is a *mean* over the ten closed-loop seeds, and this environment's outcome is bimodal: eight seeds solve the task and two plateau at the UDC level, which is why the interquartile mean is reported alongside the mean throughout. The averaged trajectory is therefore a schedule no individual seed ever ran, and the two non-solving seeds enter it; consistent with this, DyLam-OpenLoop's Chicken–Banana IQM of $130.00$ is exactly UDC's. Two explanations are compatible with the outcome, that open-loop replay is deficient in itself and that averaging a bimodal population yields a schedule optimal for neither mode, and this experiment does not separate them. The replay's failure consequently *bounds* the open-loop condition from below rather than characterizing it. A per-seed replay, in which each seed's own recorded trajectory is replayed on a fresh seed, would remove the averaging entirely and is the design that would settle the question.

We therefore read the two results without adjudicating between them. On Chicken–Banana, a fixed replay of DyLam's own averaged trajectory does not reproduce closed-loop performance; on VSS-v0 the gap is not significant, and a non-significant result is not evidence of equivalence, so VSS-v0 supports no conclusion in either direction. What the pair establishes, together with DyLam-Scalar, is environment-scoped: on the hardest exploration problem in this paper the decomposition is what makes moving weights safe, and possessing the eventual weight trajectory is not by itself sufficient to reproduce the curriculum. Whether that extends across task types is not something two environments, one of them subject to the averaging caveat above, can establish. The question it raises is the one the [Introduction](#introduction) opens with: a good weighting is not knowable a priori, and the same may hold of the trajectory that weighting should follow.

#### Chicken–Banana Environment and Per-Component Detail {#chicken-banana-environment-and-per-component-detail}

**Environment layout.** The figure below shows the environment layout. The agent navigates a $4 \times 4$ grid with four discrete actions; the state space is $|\mathcal{S}| = 64$ (16 positions $\times$ 4 inventory states). The reward decomposes into three known-bounded components: Banana ($R_{\max} = 30$), Chicken ($R_{\max} = 70$) and Gate ($R_{\max} = 100$, terminal). Because the bounds follow directly from the definition of the reward function, no estimation is required and the normalization used above is exact.

<figure id="fig-chicken-banana-env" class="results-figure">
  <img src="{{ 'assets/img/submission/chicken_banana_env.png' | relative_url }}" alt="The Chicken-Banana grid world layout" style="max-width: 320px;">
  <figcaption>The Chicken–Banana grid world. The agent (blue) starts at the bottom; reaching the Gate ($G$) terminates the episode. Banana, $B$, (easiest) and Chicken, $C$, (hardest) update inventory and yield positive rewards.</figcaption>
</figure>

**Per-component performance.** The figure below shows the per-component cumulative episode reward across 2000 episodes for all four methods. DyLam is the only method that consistently learns to collect all three objects before terminating. The decisive component is Chicken (panel b): Q-Learning, Q-Decomposition and UDC all plateau well below the saturation level, while DyLam reaches full Chicken collection by approximately episode 1250 and remains there. On Banana and Gate (panels a and c), the static methods achieve reasonable performance, but only because these components do not need to resolve the structural bias. Once the Gate is reached the episode terminates, and the agent no longer needs to plan around Chicken.

A notable qualitative difference exists between UDC and standard Q-Decomposition. Under Q-Decomposition, each component is updated independently, so the Banana component only receives value backup along trajectories that are locally optimal for Banana alone. UDC, by contrast, bootstraps all components using the shared greedy policy $\pi_G$, coupling them through a single decision rule. This allows early Banana-reaching events to accumulate value even when the trajectory is not optimal for Banana in isolation, which explains the earlier Banana emergence under UDC visible in panel (a) below.

<figure id="fig-cb-components" class="results-figure fig-row">
  <div>
    <img src="{{ 'assets/img/submission/trad_cb_reward_banana.svg' | relative_url }}" alt="Banana component cumulative reward">
    <figcaption>(a) Banana component.</figcaption>
  </div>
  <div>
    <img src="{{ 'assets/img/submission/trad_cb_reward_chicken.svg' | relative_url }}" alt="Chicken component cumulative reward">
    <figcaption>(b) Chicken component.</figcaption>
  </div>
  <div>
    <img src="{{ 'assets/img/submission/trad_cb_reward_gate.svg' | relative_url }}" alt="Gate component cumulative reward">
    <figcaption>(c) Gate component.</figcaption>
  </div>
  <div>
    <img src="{{ 'assets/img/submission/trad_cb_reward_total_app.svg' | relative_url }}" alt="Total cumulative episode reward">
    <figcaption>(d) Cumulative episode reward (total).</figcaption>
  </div>
  <figcaption>Per-component cumulative episode reward in Chicken–Banana across 2000 episodes (mean over 10 seeds). Rewards are normalized by their respective $R_{\max} = \{30, 70, 100\}$ for Banana, Chicken and Gate. Only DyLam consistently solves the Chicken component (panel b).</figcaption>
</figure>

**Curriculum: per-component returns.** The figure below gives the per-component cumulative episode returns that drive the $\lambda$-weights shown in [Curriculum (RQ2)](#curriculum-rq2). The three phases described there are read off the two figures together: a component's weight rises while its return sits far below $R_{\max}$ (dashed lines) and relaxes as the return approaches it.

<figure id="fig-curr-components" class="results-figure fig-row">
  <div>
    <img src="{{ 'assets/img/submission/curriculum_cb_components.svg' | relative_url }}" alt="Chicken-Banana per-component returns">
    <figcaption>(a) Chicken–Banana.</figcaption>
  </div>
  <div>
    <img src="{{ 'assets/img/submission/curriculum_hc_components.svg' | relative_url }}" alt="HalfCheetah-v4 per-component returns">
    <figcaption>(b) HalfCheetah-v4.</figcaption>
  </div>
  <div>
    <img src="{{ 'assets/img/submission/curriculum_vss_components.svg' | relative_url }}" alt="VSS-v0 per-component returns">
    <figcaption>(c) VSS-v0.</figcaption>
  </div>
  <figcaption>Per-component cumulative episode reward over training, with $R_{\max}$ shown as dashed lines where finite. Pair with the $\lambda$-weight figure in <a href="#curriculum-rq2">Curriculum (RQ2)</a>.</figcaption>
</figure>

#### Curriculum (RQ2) {#curriculum-rq2}

<figure id="fig-curr-weights" class="results-figure fig-row">
  <div>
    <img src="{{ 'assets/img/submission/curriculum_cb_weights.svg' | relative_url }}" alt="Chicken-Banana adaptive lambda weights">
    <figcaption>(a) Chicken–Banana.</figcaption>
  </div>
  <div>
    <img src="{{ 'assets/img/submission/curriculum_hc_weights.svg' | relative_url }}" alt="HalfCheetah-v4 adaptive lambda weights">
    <figcaption>(b) HalfCheetah-v4.</figcaption>
  </div>
  <div>
    <img src="{{ 'assets/img/submission/curriculum_vss_weights.svg' | relative_url }}" alt="VSS-v0 adaptive lambda weights">
    <figcaption>(c) VSS-v0.</figcaption>
  </div>
  <figcaption>Adaptive $\lambda$-weights over training; the per-component returns they respond to are in the figure above. Chicken–Banana: banana, chicken and gate. HalfCheetah-v4: control cost (blue) and velocity (orange). VSS-v0: approach (blue), ball-to-goal (orange) and energy (green); goal-scoring is the held-out metric, not a weighted component.</figcaption>
</figure>

**Chicken–Banana.** Panel (a) above traces DyLam's adaptive weights against the per-component returns of the figure in the previous subsection, and the curriculum mechanism is visible in three phases: *weight concentration* (episodes 0–250), in which Chicken's weight rises sharply as the EMA registers that its reward remains near zero while Banana and Gate saturate; *transient instability* (250–750), in which $\pi_G$ favors Chicken-directed trajectories and the policy sacrifices performance on the mastered components, most visibly Banana ($\approx 0.95 \to 0.72$); and *convergence* (750–1250), in which Chicken rises to saturation in the seeds that solve the task, Banana and Gate recover, and Chicken's share relaxes without vanishing. The mid-training regression is the cost of the curriculum, not a failure. The phase structure is why a static allocation must fail here: the optimal scalarization is time-varying, requiring heavy Chicken weight early, tolerance of regression mid-training, and recovery of the saturated components late.

**HalfCheetah-v4.** The same signature appears in continuous control (panel b above): DyLam suppresses the control-cost weight while velocity is far from $R_{\max}$, allowing the high torques early SAC policies require, then redistributes weight back as velocity saturates, refining toward efficient locomotion rather than brute-force velocity. The transition is smooth, an artifact of the EMA in the $\lambda$ update.

**VSS-v0.** The curriculum is cleanest here (panel c above). DyLam first concentrates weight on the ball-approach component, which saturates within roughly $10^5$ steps, then transfers it to ball-to-goal, which begins to improve precisely as approach saturates; the energy penalty holds the smallest share throughout, consistent with its role as a refinement signal. Goal-scoring, which carries no reward term, rises only once the two shaping components have been mastered in sequence. This staged progression is not designed in: it emerges from the relative-progress signal, in the most demanding environment we evaluate. Across all three environments the static baselines fail not because no balanced allocation exists at convergence, but because the binding constraint changes over training; DyLam tracks the optimal *trajectory* of weightings rather than committing to a single point on the simplex.

DyLam decomposes a multi-objective reward into per-Component critics and adapts a weight $\lambda(t)$ over training to trade them off. Scrub the slider below to watch $\lambda$ move through training on ChickenBanana and see the greedy policy it induces at each moment. Click anywhere in the weight simplex to try a different, fixed weighting against the *same* learned Component Q-tables at that same moment -- the counterfactual this Embed exists to make checkable.

<div class="l-page">
  <iframe src="{{ 'assets/html/submission/lambda_simplex_scrubber.html' | relative_url }}" frameborder="0" scrolling="no" height="720" width="100%" style="border: none;"></iframe>
</div>

Left: the ChickenBanana grid, with the currently-displayed policy's greedy action drawn in each reachable cell (before either pickup) and its rollout path overlaid statically -- the path is not animated, so two weightings can be compared at a glance by switching between them. Middle: the three-Component weight simplex, coloured by the behaviour class each weighting's policy achieves at the currently selected episode, with DyLam's own $\lambda$ trajectory drawn as a trail up to that episode. Clicking elsewhere fixes that weighting (free mode, amber) so the time slider keeps scrubbing the counterfactual; the button above returns to DyLam's own trajectory (follow mode, green). Right: the rollout's per-Component returns (solid bars) against the method's own r_max ceilings (dashed), with DyLam's actual training-time return at that episode ghosted behind for reference.

#### Bound Derivation {#bound-derivation}

<div class="results-table-wrap" markdown="0">
<table>
<caption>Environment specifications. Bounds are listed per component in the order given in the components column, as $R_{\max}$ then $R_{\min}$. Bound provenance is detailed in the text.</caption>
<thead><tr><th>Environment</th><th>Components</th><th>Bounds</th><th>Bound source</th></tr></thead>
<tbody>
<tr><td>Chicken–Banana</td><td>banana / chicken / gate</td><td>$(30, 70, 100)$, $(0,0,0)$</td><td>algebraic</td></tr>
<tr><td>HalfCheetah-v4</td><td>velocity / torque</td><td>$(800, -200)$, $(0, -800)$</td><td>converged SAC reference</td></tr>
<tr><td>VSS-v0</td><td>move / ball-to-goal / energy</td><td>$(150, 40, -100)$, $(0, 0, -300)$</td><td>reference trajectory</td></tr>
</tbody>
</table>
</div>

**HalfCheetah-v4.** The bounds are identical to those used in MO-HalfCheetah, $R_{\max} = (800, -200)$ and $R_{\min} = (0, -800)$. Unlike the grid-world environments, these are not algebraic maxima. The forward-velocity bound $R_{\max} = 800$ corresponds to the cumulative velocity reward typically achieved by a converged SAC policy on the standard scalar-reward HalfCheetah task, taken from the original SAC results <d-cite key="sac"></d-cite>; the control-cost bound $R_{\min} = -800$ is the worst torque penalty observed over $10^4$ steps of a random policy, and the reverse direction ($R_{\min}$ for velocity, $R_{\max}$ for control) is set by the same rollouts under the opposite policy. Calibrating against a known converged-policy performance level, rather than against an unreachable algebraic ceiling, ensures that $R_{\max}$ encodes a realistic sufficient value: the level a strong scalar-reward agent is known to reach, beyond which further improvement in the component yields diminishing returns on the dominant objective.

**VSS-v0.** VSS-v0 <d-cite key="rsoccer"></d-cite> is a robotic soccer task with continuous state and action spaces whose objective, scoring goals, is sparse and is realized only once a chain of intermediate skills has been acquired. The reward presented to the agent decomposes into move (ball-approach) shaping, ball-to-goal progress and an energy penalty, with bounds $R_{\max} = (150, 40, -100)$ and $R_{\min} = (0, 0, -300)$. These bounds are constructed from deterministic-trajectory rollouts rather than from algebraic maxima or a converged reference policy. The move bound is the per-step shaping reward of a robot moving in a straight line toward the ball across two consecutive frames, multiplied by the episode horizon: the cumulative reward a maximally efficient approach trajectory would accumulate within an episode. The ball-to-goal bound is obtained analogously, using a straight-line trajectory from the ball to the goal under its shaping function, and the energy bound is the per-step penalty under the maximum-effort action scaled by the horizon. Because these values derive from physically grounded reference trajectories, they correspond to interpretable upper bounds on what each shaping component can realistically contribute given the task geometry and the robot's dynamics, rather than to unattainable ceilings. VSS-v0 also doubles as the continuous robustness benchmark for RQ3, where the two shaping bounds are perturbed individually ([Robustness to Bound Misspecification](#robustness-to-bound-misspecification)).

### Robustness to Bound Misspecification {#robustness-to-bound-misspecification}

DyLam's deprioritization mechanism relies on the practitioner-supplied bound $R^i_{\max}$ approximating the sufficient value $\bar{r}_i$ of [Definition 1](#def-sufficient-value). Because $\bar{r}_i$ is a property of the joint optimization landscape that the practitioner does not observe, while $R^i_{\max}$ is supplied from domain knowledge or reference-policy rollouts, the two will generally not coincide. This section asks how much that gap costs, and in which direction it is dangerous.

<figure id="fig-robustness-curves" class="results-figure">
  <img src="{{ 'assets/img/submission/robustness_curves.svg' | relative_url }}" alt="VSS-v0 goal rate under bound misspecification, by condition" style="max-width: 480px;">
  <figcaption>VSS-v0 goal rate under bound misspecification ($10$ seeds per condition; interquartile mean with $95\%$ bootstrap confidence bands over seeds). Conditions by colour: <strong>grey</strong> nominal $(150, 40, -100)$; <strong>blue</strong> move $-25\%$; <strong>red</strong> move $+50\%$; <strong>purple</strong> ball $+25\%$; <strong>brown</strong> ball $-25\%$; <strong>orange</strong> move $+50\%$, ball $+25\%$; <strong>green</strong> move $-25\%$, ball $-50\%$. The safe direction differs by component: for the move ceiling, tightening (blue) tracks or exceeds nominal while inflating (red) is damaging; for the ball-to-goal ceiling, inflating (purple) is indistinguishable from nominal while tightening (brown) is damaging.</figcaption>
</figure>

We perturb the shaping ceilings of VSS-v0 around the nominal $R_{\max} = (150, 40, -100)$, holding every other hyperparameter and the budget fixed. The goal-scoring metric is never perturbed, since it carries no reward term. Each condition uses $10$ seeds; comparisons are exact two-sided Mann–Whitney $U$ tests on the per-seed mean of the final $10\%$ of training, Holm–Bonferroni corrected within the six-comparison RQ3 family, and reported with rank-biserial effect sizes. [Full Robustness Results](#full-robustness-results) reports every condition with interquartile means and bootstrap confidence intervals and discusses what the design does and does not isolate.

Which direction of error is safe depends on the component, not on the sign. Tightening the move ceiling by $25\%$ significantly improves on nominal ($0.898 \pm 0.016$ vs. $0.852 \pm 0.021$; $U = 95$, $p_{\text{Holm}} = 0.0006$, $r = +0.90$), while inflating it by $50\%$ costs a third of the goal rate ($0.607 \pm 0.120$; $U = 0$, $p_{\text{Holm}} = 6 \times 10^{-5}$, $r = -1.00$). The ball-to-goal ceiling reverses the pattern. Inflating it by $25\%$ leaves the goal rate statistically indistinguishable from nominal ($0.870 \pm 0.023$; $p_{\text{Holm}} = 0.089$), whereas tightening it by the same quarter costs $0.098$ goal rate with every seed falling below every nominal seed ($0.754 \pm 0.070$; $U = 0$, $p_{\text{Holm}} = 6 \times 10^{-5}$, $r = -1.00$). The two compound conditions, which perturb both shaping ceilings at once, sit between the extremes ($0.737 \pm 0.111$ and $0.670 \pm 0.036$, both $p_{\text{Holm}} \leq 0.001$) and do not compose additively; [Full Robustness Results](#full-robustness-results) reads that deviation.

Eq. $\eqref{eq:dylam-weights}$ explains both asymmetries, and what orders them is the component's position in the curriculum rather than the sign of the error. Over-estimation stalls the hand-off wherever it occurs: a component whose smoothed return can never reach its stated $R^i_{\max}$ holds $\zeta_i$ near $1$ indefinitely and retains the largest weight for the whole run. That is costly for the move component, which the agent must be released from to progress, and harmless for ball-to-goal, the last shaping component before the goal itself, where pinning the weight pins it on the objective the metric rewards. Under-estimation mirrors this exactly. Retiring the approach signal early costs little because the skill it teaches is a prerequisite for the ball-to-goal component that succeeds it, so the shared policy keeps maintaining it; retiring ball-to-goal early abandons it, because no later component depends on it. The practical rule is therefore not "err low" but *err in whichever direction keeps weight nearer the objective*: bound upstream components tightly, since whatever consumes them keeps them alive, and bound terminal components generously, since an unattainable ceiling there merely holds weight where it was wanted anyway. Two shaping components in one environment is thin evidence for a rule of this form; what supports it here is that the mechanism is the one Eq. $\eqref{eq:dylam-weights}$ implies and all four one-at-a-time cells agree with it.

Critically, every perturbation preserves task success. DyLam learns a goal-scoring policy in all six conditions, from $0.607$ to $0.898$, against $0.388$ for the expert-tuned static weights and $0.054$–$0.070$ for the untuned baselines above. Misspecification changes how fast the curriculum is traversed, not whether it emerges.

#### Full Robustness Results {#full-robustness-results}

The table below reports all seven conditions in full: the per-seed mean of the final $10\%$ of training, the interquartile mean with its bootstrap interval, and the exact Mann–Whitney statistics against nominal after Holm–Bonferroni correction within the RQ3 family. The prose above quotes the goal rates and corrected $p$-values; what follows is the per-condition read-through.

<div class="results-table-wrap" markdown="0">
<table>
<caption>Robustness to bound misspecification on VSS-v0 (nominal $R_{\max} = (150, 40, -100)$). Values are per-seed means of the final $10\%$ of training episodes over $10$ seeds, reported as mean $\pm$ std and as the interquartile mean with a $95\%$ bootstrap CI over $10^4$ resamples. $U$ and $p$ are exact two-sided Mann–Whitney tests against nominal, $r$ is the rank-biserial effect size (negative means below nominal), and $p_{\text{Holm}}$ applies the Holm–Bonferroni correction within this family of six comparisons. $^\ast$ denotes significance after correction ($\alpha = 0.05$). The lower block reports compound perturbations, in which the ball-to-goal ceiling is varied on top of a move perturbation.</caption>
<thead><tr><th>Condition</th><th>$R_{\max}$</th><th>Goal rate</th><th>IQM [95% CI]</th><th>$U$</th><th>$p$</th><th>$p_{\text{Holm}}$</th><th>$r$</th></tr></thead>
<tbody>
<tr><td>Nominal</td><td>(150, 40, -100)</td><td>0.852 ± 0.021</td><td>0.848 [0.838, 0.865]</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td colspan="8"><em>Move ceiling</em></td></tr>
<tr><td>Move -25%</td><td>(112.5, 40, -100)</td><td>0.898 ± 0.016</td><td>0.899 [0.885, 0.911]</td><td>95</td><td>2.1×10⁻⁴</td><td>6.2×10⁻⁴ *</td><td>+0.90</td></tr>
<tr><td>Move +50%</td><td>(225, 40, -100)</td><td>0.607 ± 0.120</td><td>0.589 [0.524, 0.698]</td><td>0</td><td>1.1×10⁻⁵</td><td>6.5×10⁻⁵ *</td><td>-1.00</td></tr>
<tr><td colspan="8"><em>Ball-to-goal ceiling</em></td></tr>
<tr><td>Ball -25%</td><td>(150, 30, -100)</td><td>0.754 ± 0.070</td><td>0.775 [0.706, 0.800]</td><td>0</td><td>1.1×10⁻⁵</td><td>6.5×10⁻⁵ *</td><td>-1.00</td></tr>
<tr><td>Ball +25%</td><td>(150, 50, -100)</td><td>0.870 ± 0.023</td><td>0.874 [0.856, 0.885]</td><td>73</td><td>0.089</td><td>0.089 n.s.</td><td>+0.46</td></tr>
<tr><td colspan="8"><em>Compound</em></td></tr>
<tr><td>Move +50%, ball +25%</td><td>(225, 50, -100)</td><td>0.737 ± 0.111</td><td>0.757 [0.651, 0.810]</td><td>7</td><td>4.9×10⁻⁴</td><td>9.7×10⁻⁴ *</td><td>-0.86</td></tr>
<tr><td>Move -25%, ball -50%</td><td>(112.5, 20, -100)</td><td>0.670 ± 0.036</td><td>0.663 [0.646, 0.696]</td><td>0</td><td>1.1×10⁻⁵</td><td>6.5×10⁻⁵ *</td><td>-1.00</td></tr>
</tbody>
</table>
</div>

**What each condition isolates.** The design is one-at-a-time in the upper two blocks and compound in the lower one, and the two halves answer different questions. Each shaping ceiling is bracketed on both sides with the remaining ceilings held at nominal, so any difference from nominal within the upper blocks is attributable to the perturbed ceiling alone. The move bracket is asymmetric in the direction an over-estimated ceiling would predict: tightening helps, inflating costs a third of the goal rate. The ball-to-goal bracket is asymmetric in the *opposite* direction: inflating that ceiling by a quarter does not move the goal rate detectably ($p_{\text{Holm}} = 0.089$), whereas tightening it by the same quarter costs $0.098$ goal rate with every seed below every nominal seed ($p_{\text{Holm}} = 6 \times 10^{-5}$, $r = -1.00$).

The compound conditions vary two ceilings simultaneously, so a significant effect there cannot be attributed to either bound; they serve as a check on whether the one-at-a-time effects compose. They do not compose additively, and the deviation is informative. Summing the individual deviations from nominal predicts $0.625$ for move $+50\%$, ball $+25\%$, against $0.737$ observed: inflating the ball-to-goal ceiling alongside the move ceiling *recovers* part of what the move perturbation costs on its own ($0.607 \to 0.737$). This is what a weight update driven by relative rather than absolute deficiency predicts, since Eq. $\eqref{eq:dylam-weights}$ normalizes across components: raising both ceilings together leaves the ball-to-goal component competitive for weight, whereas raising the move ceiling alone lets it monopolize. The second compound runs the other way, at $0.670$ against the $0.702$ that linear extrapolation of ball $-25\%$ to $-50\%$ would give, close enough that its deficit is dominated by the ball-to-goal under-estimation rather than by the move relaxation accompanying it.

### Pareto-Oriented Environments {#pareto-oriented-environments}

This section evaluates DyLam on benchmarks whose ground-truth metric is geometric coverage of the Pareto front, a regime for which it was not designed and in which it produces a single adaptive policy trajectory rather than an explicit coverage set. The table under [Metrics](#pareto-metrics) reports hypervolume, cardinality and wall-clock training time on MO-HalfCheetah and MO-Minecart, which we read alongside the qualitative shape of each discovered front in objective space, detailed in [Discovered Fronts and Weight Trajectories](#discovered-fronts-and-weight-trajectories).

On MO-HalfCheetah DyLam attains the best mean on both coverage metrics, but the significance tests give the accurate reading. Its HV advantage over GPI-LS is not significant, so the two are indistinguishable in dominated volume, though DyLam reaches that level with substantially lower across-seed variance; against PGMORL the gap is significant and large, roughly a $0.6$ separation on the $\log_{10}$ scale. Cardinality mirrors this in reverse: DyLam is indistinguishable from the population-based PGMORL and, under the sampling protocol of [Metrics](#pareto-metrics), ahead of GPI-LS. On MO-Minecart, DynMORL's significant HV edge is practically negligible, well under $1\%$ in raw HV, and comes precisely from the high-fuel region DyLam does not enter; on cardinality DyLam returns roughly $1.7\times$ DynMORL's non-dominated set under that same protocol, on the benchmark for which DynMORL was designed and tuned <d-cite key="abels2019dynamicweightsmultiobjectivedeep"></d-cite> and without its weight-conditioned architecture or externally specified schedule.

DyLam reaches these fronts at a substantially lower wall-clock cost than the dedicated methods, every gap significant; the timing is indicative rather than controlled, and [MO-HalfCheetah](#mo-halfcheetah) explains why.

Four concessions bound the reading. DyLam returns no weight-conditioned policy that can be queried for a requested trade-off; it offers no coverage guarantee; it leaves regions of the front unexplored, notably the high-fuel configurations on Minecart and the low-Run extreme on MO-HalfCheetah; and the cardinality comparison is not like-for-like, since the baselines contribute the evaluation front they return at the end of training while DyLam contributes a Pareto filtering of $10^4$ samples of its own training history ([Metrics](#pareto-metrics)). The counts are therefore conditional on that protocol, and we read them as characterizing the density with which DyLam's weight trajectory passes through mutually non-dominated policies rather than as establishing that it returns more policies than a dedicated coverage method. What the stress test establishes is narrower: a mechanism designed for single-policy learning dynamics is, without modification, statistically on par with dedicated methods on dominated volume in both benchmarks, substantially cheaper, and, under the sampling protocol above, dense rather than sparse in the region it does cover, and this follows from its weight trajectory rather than from any coverage objective. We read the outcome as consistent with the mechanism claim of the [Mechanism](#mechanism) section, not as a contribution to multi-objective RL; [Discovered Fronts and Weight Trajectories](#discovered-fronts-and-weight-trajectories) traces the weight path that produces it.

#### Discovered Fronts and Weight Trajectories {#discovered-fronts-and-weight-trajectories}

<figure id="fig-hc-pareto" class="results-figure fig-row">
  <div>
    <img src="{{ 'assets/img/submission/morl_hc_pareto.svg' | relative_url }}" alt="MO-HalfCheetah discovered Pareto fronts">
    <figcaption>(a) Discovered Pareto fronts.</figcaption>
  </div>
  <div>
    <img src="{{ 'assets/img/submission/morl_hc_weights.png' | relative_url }}" alt="MO-HalfCheetah explored weight space">
    <figcaption>(b) Explored weight space.</figcaption>
  </div>
  <figcaption>MO-HalfCheetah: PGMORL (green), GPI-LS (orange), and DyLam (blue). In (a), the dashed lines indicate the per-component $R_{\max}$, with $R_c = 1000 - \text{Control}$. In (b), DyLam traces a near-continuous trajectory along the simplex while the baselines sample it discretely.</figcaption>
</figure>

**Front geometry.** The methods occupy qualitatively different regions in both benchmarks. On MO-HalfCheetah (panel a above), PGMORL spreads its solutions broadly across the trade-off surface, including the low-Run/low-cost region, producing a long but sparsely sampled curve; GPI-LS concentrates near the high-Run extreme; DyLam occupies the intermediate band. On MO-Minecart (figure below, top row), DyLam densely populates the $M_1 + M_2 \approx 1.5$ diagonal, the geometric upper boundary of joint mineral collection under the environment's resource constraint, producing a near-continuous curve along the efficient trade-off, while DynMORL covers the same diagonal sparsely and adds a cluster of solutions well below it and GPI-LS returns a few discrete clusters with no interior coverage. The Fuel projections invert the picture: DyLam concentrates near zero fuel cost across the full Minerium range, whereas DynMORL extends into fuel-inefficient configurations that DyLam never visits. The pattern is consistent with the implicit-curriculum mechanism, since DyLam's policies accumulate where both components are still jointly improving rather than at the extremes where one has already saturated.

<figure id="fig-minecart-pareto-group" class="results-figure">
  <img src="{{ 'assets/img/submission/morl_minecart_pareto_weights.png' | relative_url }}" alt="MO-Minecart pairwise projections of discovered Pareto fronts and explored weight simplex">
  <figcaption>MO-Minecart: pairwise projections of the discovered Pareto fronts (top, $M_1$–$M_2$, $M_1$–Fuel, $M_2$–Fuel) and of the explored weight simplex (bottom) for GPI-LS (orange), DynMORL (green), and DyLam (blue).</figcaption>
</figure>

**The continuous weight trajectory.** The weight-space panels above expose the common mechanism. In both benchmarks DyLam traces a near-continuous interior curve through the simplex, while PGMORL and GPI-LS sample it discretely and DynMORL concentrates on its edges and corners, as its externally scheduled weight protocol implies. The smoothness follows directly from the temporal EMA in the $\lambda$ update: consecutive episodes differ by at most $1 - \tau_\lambda$, producing a path rather than a set of independent samples. This is the mechanism behind the density of DyLam's sampled front, since a continuum of intermediate weight vectors maps to a continuum of fine-grained policies, and it explains why DyLam's Pareto points concentrate where its trajectory dwells longest; the cardinality counts themselves are conditional on the sampling protocol of [Metrics](#pareto-metrics) and are not a like-for-like count against the baselines. The coverage behavior reported here is thus a consequence of the same weight-adaptation rule that produces the curricula of [Learning-Dynamics-Oriented Environments](#learning-dynamics-oriented-environments), not of a separate mechanism.

#### Contemporary Adaptive-Weight Methods {#contemporary-adaptive-weight-methods}

Two recent lines sharpen what varies across responses to this problem. Preference-conditioned methods train a hypernetwork mapping a requested weight vector to policy parameters, so the trade-off remains queryable after training <d-cite key="liu2025pslmorl"></d-cite>, and multi-objective alignment of language models adapts the weights online from hypervolume or gradient signals <d-cite key="lu2025dynamic"></d-cite>. DyLam differs on both axes: it exposes no preference input and returns a single policy, and its adaptation signal is each component's smoothed return relative to a practitioner-supplied bound rather than a coverage or gradient criterion.

#### Pareto-Coverage Comparisons: All Pairwise Tests {#pareto-coverage-comparisons-all-pairwise-tests}

Comparisons behind the [Metrics](#pareto-metrics) table are exact two-sided Mann–Whitney $U$ tests over $10$ seeds per method, Holm–Bonferroni corrected inside each metric's family of four comparisons and reported as $p_{\text{Holm}}$ with the rank-biserial effect size $r$ (positive favours DyLam).

**Hypervolume:** vs. GPI-LS $4.3\times10^{-5}$, $r = +1.00$ on MO-Minecart and $0.089$ n.s., $r = -0.46$ on MO-HalfCheetah; vs. PGMORL $4.3\times10^{-5}$, $r = +1.00$; vs. DynMORL $0.0042$, $r = -0.78$. **Cardinality:** vs. GPI-LS $4.3\times10^{-5}$, $r = +1.00$ (MO-Minecart) and $0.0058$, $r = +0.76$ (MO-HalfCheetah); vs. DynMORL $4.3\times10^{-5}$, $r = +1.00$; vs. PGMORL $0.912$ n.s., $r = +0.04$. **Time:** $4.3\times10^{-5}$, $r = -1.00$ against every baseline, subject to the shared-GPU caveat of [Metrics](#pareto-metrics).

#### Metrics {#pareto-metrics}

<div class="results-table-wrap" markdown="0">
<table>
<caption>Hypervolume ($\log_{10}$), cardinality, and wall-clock training time (minutes) of the approximated Pareto fronts, over $10$ seeds per method. Bold marks the best mean per column (lowest is best for time). Comparisons are exact two-sided Mann–Whitney $U$ tests with Holm–Bonferroni correction inside each metric's family of four, reported with rank-biserial effect sizes; DyLam is statistically indistinguishable from the best baseline on hypervolume in both benchmarks and ahead of every baseline on time, subject to the shared-GPU caveat below. Every pairwise test is listed under Pareto-Coverage Comparisons above.</caption>
<thead><tr><th>Method</th><th colspan="3">MO-HalfCheetah</th><th colspan="3">MO-Minecart</th></tr>
<tr><th></th><th>HV (log10)</th><th>Card.</th><th>Time (min)</th><th>HV (log10)</th><th>Card.</th><th>Time (min)</th></tr></thead>
<tbody>
<tr><td>PGMORL</td><td>5.014 ± 0.111</td><td>21 ± 4</td><td>445 ± 11</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>GPI-LS</td><td>5.631 ± 0.301</td><td>12 ± 11</td><td>6775 ± 3074</td><td>1.776 ± 2.040</td><td>49 ± 23</td><td>436 ± 22</td></tr>
<tr><td>DynMORL</td><td>—</td><td>—</td><td>—</td><td><strong>3.048 ± 0.001</strong></td><td>2949 ± 531</td><td>1285 ± 222</td></tr>
<tr><td>DyLam</td><td><strong>5.644 ± 0.048</strong></td><td><strong>22 ± 9</strong></td><td><strong>148 ± 12</strong></td><td>3.045 ± 0.003</td><td><strong>5090 ± 235</strong></td><td><strong>40 ± 13</strong></td></tr>
</tbody>
</table>
</div>

**Reading the wall-clock column.** The difference in training cost is structural rather than an optimization win: following a single adaptive $\vec{\lambda}$-trajectory is inherently cheaper than sampling the simplex and training toward each target. The column should nonetheless be read as indicative. Seeds shared a GPU, so it measures contention alongside algorithmic cost, and GPI-LS's large coefficient of variation on MO-HalfCheetah is partly attributable to that. The claim above is that DyLam delivers competitive coverage at a much smaller budget, not that the time figures are controlled measurements.

We report three quantities. First, the qualitative shape and coverage of the discovered front, visualized in objective space. Second, the hypervolume (HV, $\log_{10}$) of the approximated Pareto front with respect to a fixed reference point, which quantifies the volume of objective space dominated by each method's solutions. The reference point is $(0, 0, -1000)$ on MO-Minecart and $(-1, -1)$ on MO-HalfCheetah. Each is applied identically to every method's front, after the coordinate shift that places the control-cost axis of MO-HalfCheetah in the positive orthant, so that all methods on a given benchmark are scored against the same point in the same units. Third, the cardinality of the non-dominated set, that is, the number of distinct non-dominated policies a method returns. The three methods return solutions in structurally different forms, so we apply one common protocol. For the coverage-oriented baselines we take the evaluation front each method reports at the end of training, which already consists of one point per returned policy. For DyLam, which produces a single evolving policy rather than a set, we sample the training history of each seed at $10^4$ evenly spaced points and treat the per-component returns logged at each sampled point as one candidate solution. All candidates from a seed are then pooled, duplicates removed, and Pareto-dominated points filtered out, so the reported cardinality counts only mutually non-dominated evaluations; HV is computed on the same filtered set against the same reference point used for the baselines. The comparison is therefore between the non-dominated sets each method makes available at the end of a run, not between the raw numbers of policies evaluated: DyLam's higher count reflects that its continuous weight trajectory passes through many mutually non-dominated policies, and each such policy is a distinct parameterization the practitioner could have checkpointed. This protocol is not symmetric, and we do not claim that it is. The baselines contribute exactly the policies they return, whereas DyLam contributes a Pareto filtering of $10^4$ draws from its own training history, so the candidate pool DyLam is scored on is both larger than and generated differently from the pools the baselines supply. The reported counts are consequently conditional on this protocol. They support the descriptive reading that DyLam's weight trajectory passes densely through mutually non-dominated policies, which is the reading taken above; they do not support a like-for-like claim that DyLam returns more policies than a dedicated coverage method, and the cardinality significance tests reported under Pareto-Coverage Comparisons should be read as valid *given* this protocol rather than as establishing method superiority on the metric. HV captures the *extent* of coverage and cardinality how *densely* the covered region is populated. A method may score well on one and poorly on the other, so the two together give a more complete picture than either alone.

#### MO-HalfCheetah {#mo-halfcheetah}

**Front regions.** In the (Run, Control) objective space of panel (a) above, DyLam's solutions occupy the intermediate band between approximately $(0, 1000)$ and $(400, 700)$, with the per-component $R_{\max}$ lines marking the achievable upper limits. PGMORL's broader spread includes the low-Run/low-cost region that DyLam's trajectory never reaches, since a policy is only produced there when the velocity component is deliberately deprioritized, which the relative-progress rule does not do while velocity remains far from its bound.

**Baseline configuration.** The MORL baselines are run with the `morl-baselines` defaults for each benchmark; we did not tune them. The table below lists the GPI-LS settings, which are the ones the wall-clock discussion above depends on.

<div class="results-table-wrap" markdown="0">
<table>
<caption>GPI-LS configuration as run, taken from the logged run configs.</caption>
<thead><tr><th>Parameter</th><th>MO-HalfCheetah</th><th>MO-Minecart</th></tr></thead>
<tbody>
<tr><td>Variant</td><td>GPI-LS Continuous Action</td><td>GPI-LS</td></tr>
<tr><td>Learning rate</td><td>3×10⁻⁴</td><td>3×10⁻⁴</td></tr>
<tr><td>Batch size</td><td>128</td><td>128</td></tr>
<tr><td>Buffer size</td><td>4×10⁵</td><td>10⁶</td></tr>
<tr><td>Network architecture</td><td>[256, 256]</td><td>[256, 256, 256, 256]</td></tr>
<tr><td>Number of Q-networks</td><td>2</td><td>2</td></tr>
<tr><td>γ</td><td>0.99</td><td>0.98</td></tr>
<tr><td>Target update</td><td>τ = 0.005</td><td>τ = 1, every 200 steps</td></tr>
<tr><td>Gradient updates / step</td><td>20</td><td>10</td></tr>
<tr><td>Prioritized replay</td><td>yes (α = 0.6)</td><td>yes (α = 0.6)</td></tr>
<tr><td>Learning starts</td><td>100</td><td>100</td></tr>
<tr><td>Model-based (dyna)</td><td>disabled</td><td>disabled</td></tr>
<tr><td>Exploration</td><td>—</td><td>ε: 1 → decay over 10⁵ steps</td></tr>
</tbody>
</table>
</div>

GPI-LS exhibits substantially higher across-seed variance than the other methods on every metric, consistent with its generalized-policy-improvement step incurring a variable number of policy evaluations per seed, but it is also inflated by the execution conditions: seeds were run concurrently on the same GPU, so wall-clock times reflect contention as well as algorithmic cost and are indicative rather than controlled measurements. We report them because the order-of-magnitude separation survives any plausible contention correction, not because the individual figures are precise. The practical consequence for the coverage claim is unaffected: GPI-LS's HV parity with DyLam is obtained inconsistently, whereas DyLam attains the same level on every seed.

#### MO-Minecart {#mo-minecart}

**Comparator.** DynMORL is the primary methodological comparator on this benchmark. Like DyLam it operates in a dynamic-weights regime, but it requires an externally specified time-varying weight schedule, a Q-network conditioned on the weight vector, and Diverse Experience Replay to manage the resulting non-stationarity, whereas DyLam derives its weights internally from reward progress using a single unconditioned Q-network and no replay modification. The HV parity reported above is therefore obtained without any of the three mechanisms DynMORL introduces for this purpose.

**Projection-level detail.** The Minecart figure above displays three pairwise projections of the objective space (top row, $M_1$–$M_2$, $M_1$–Fuel, $M_2$–Fuel) and the corresponding three projections of the weight simplex (bottom row). In the $M_1$–$M_2$ projection, DyLam produces a near-continuous curve along the $M_1 + M_2 \approx 1.5$ efficient frontier; DynMORL covers the same diagonal sparsely and additionally produces a cluster of solutions far below it; GPI-LS returns a small number of discrete clusters at distinct $(M_1, M_2)$ ratios with no interior coverage. In the two Fuel projections, DyLam concentrates near Fuel $\approx 0$ across the full Minerium range while DynMORL extends to substantially higher fuel cost, and GPI-LS again samples both axes sparsely. The two methods thus describe partially complementary regions: DyLam recovers the efficient frontier, high collection at minimal fuel cost, while DynMORL additionally explores fuel-inefficient trade-off regions. Those additional points are the source of DynMORL's marginal HV advantage under the chosen reference point.

**Raw-scale conversions.** Because HV is reported on a $\log_{10}$ scale, small separations correspond to large ratios. On MO-HalfCheetah, DyLam's separation over PGMORL is roughly a $4\times$ difference in raw HV. On MO-Minecart, the separation over GPI-LS is over an order of magnitude, while DynMORL's edge over DyLam is under $1\%$.

**Alignment with the benchmark structure.** MO-Minecart is the configuration in our stress test where DyLam's design assumption, that smooth learning-driven weight adaptation produces useful policy diversity, aligns most closely with what a Pareto-coverage benchmark rewards. Unlike MO-HalfCheetah, where HV is contested across the high-Run extreme, the Minecart efficient frontier is exactly the region a smooth interior trajectory populates densely. We attribute the parity with DynMORL specifically to that alignment rather than to a general coverage capability, which is why we present the result as a stress test rather than as a multi-objective contribution.

## Ablation {#ablation}

This section provides a systematic ablation study of the DyLam framework in the Chicken–Banana environment. The goal is to isolate the impact of key hyperparameters on learning dynamics, convergence, and final performance. All experiments are run over 2000 episodes with 10 random seeds, using tabular Q-learning as the base algorithm. Unless stated otherwise, default parameters are: $\tau_\lambda = 0.995$, experience buffer $E = 10$, softmax normalization, and $\epsilon$-greedy decay factor $\epsilon_d = 0.9988$ (decaying from $1.0$ to $0.05$).

Select a sweep and a panel below to inspect any of the $24$ individual per-Component, per-metric curves the four sweeps below produce -- rendered today as four fixed combined figures, this Embed surfaces every panel behind them, including the twenty reachable nowhere else on this page.

<div class="l-page">
  <iframe src="{{ 'assets/html/submission/ablation_curves.html' | relative_url }}" frameborder="0" scrolling="no" height="420" width="100%" style="border: none;"></iframe>
</div>

Each panel plots the interquartile mean across $10$ seeds with a $95\%$ bootstrap confidence band, nominal (grey) against the swept arms (blue/orange/green) -- the same protocol as every learning curve elsewhere on this page. "Reward" panels show cumulative episode reward per Component; "λ weight" panels show the corresponding adaptive weight. Switch the sweep selector between the four studies below; switch panel to compare reward against weight for the Component you are viewing.

### Effect of the DyLam's Update Rate {#effect-of-the-dylams-update-rate}

The smoothing factor $\tau_\lambda$ controls how quickly the smoothed returns $\overline{G}^i$ react to new episodes. Select the "Update rate" sweep above to see the evolution of reward components and adaptive weights for $\tau_\lambda \in \{0.5, 0.7, 0.9\}$.

A smaller $\tau_\lambda$ (e.g., $0.5$) leads to more volatile weights, yet the agent still achieves all objectives in at least half the episodes. This suggests that the quasi-static update assumption ([Proposition 1](#prop-dylam-stability)) is not strictly necessary in low-dimensional tasks, though larger values are recommended for stability in complex environments. Two caveats bound this reading. The sweep is run on Chicken–Banana only, so it does not establish how $\tau_\lambda$ behaves in the continuous-control settings, where the weights are updated at every environment step and the effective horizon $1/(1-\tau_\lambda)$ is therefore counted in steps rather than episodes; the values used there ($\tau_\lambda = 0.9999$) are outside the range swept here for that reason. We also did not run a controlled comparison of the two update cadences, since each implementation uses the cadence natural to its training loop; [Proposition 1](#prop-dylam-stability)(3) bounds the weight displacement over an inter-episode interval independently of how many updates it contains, which is the property that makes the two cadences comparable in principle, but the empirical comparison remains open.

### Effect of Experience Buffer Size {#effect-of-experience-buffer-size}

The buffer size $E$ determines how many episodes are used to compute the smoothed return $\overline{G}^i$ (via moving average). Select the "Replay buffer" sweep above to compare $E = 50, 100, 500$ against the default $E = 10$.

Single-run traces (not reproduced in this Embed, which shows aggregate bands) show smoother weight changes for larger $E$, but the aggregated results across seeds are largely similar. The structural simplicity of the Chicken–Banana task appears to dominate, masking the stabilizing effect of larger buffers.

### Effect of the Deficiency Transform {#effect-of-the-deficiency-transform}

Eq. $\eqref{eq:dylam-weights}$ converts each deficiency $\zeta_i$ into a raw weight through $g(\zeta) = \mathrm{e}^\zeta - 1$ before normalizing. This is one member of the deficiency-routed family the manuscript's implicit-curriculum appendix formalizes, and the choice of $g$ is what sets how sharply the mechanism discriminates between a lagging and a nearly saturated component. We compare the exponential transform against an $\ell_1$ rule that normalizes the deficiencies directly ($\lambda_i = \zeta_i / \sum_j \zeta_j$, the linear $g(\zeta) = \zeta$), and against a min–max rule that rescales by the observed spread, $\lambda_i = (\zeta_i - \min_j \zeta_j)/(\max_j \zeta_j - \min_j \zeta_j)$. We additionally report the *reversed* min–max rule, $(\max_j \zeta_j - \zeta_i)/(\max_j \zeta_j - \min_j \zeta_j)$, which assigns the largest weight to the component with the *smallest* deficiency. It is not a candidate transform but a sign-flip control: it holds every other part of the mechanism fixed and inverts only the direction of routing. Select the "Deficiency transform" sweep above for the reward/weight curves; the table below gives the final-performance comparison.

<div class="results-table-wrap" markdown="0">
<table>
<caption>Deficiency transforms on Chicken–Banana, 10 seeds each (final episode reward, max $200$). Success counts seeds whose final-$10\%$ mean exceeds $160$. $p$ is an exact two-sided Mann–Whitney test against the exponential transform.</caption>
<thead><tr><th>Transform</th><th>Reward</th><th>IQM [95% CI]</th><th>Success</th><th>$p$</th></tr></thead>
<tbody>
<tr><td>Exponential, $g(\zeta) = \mathrm{e}^\zeta - 1$</td><td>178.92 ± 33.76</td><td>188.19 [153.19, 200.00]</td><td>7/10</td><td>—</td></tr>
<tr><td>Linear (ℓ1), $g(\zeta) = \zeta$</td><td>178.90 ± 33.74</td><td>188.16 [153.22, 200.00]</td><td>7/10</td><td>0.912</td></tr>
<tr><td>Min–max</td><td>126.19 ± 95.63</td><td>142.40 [48.60, 200.00]</td><td>6/10</td><td>0.579</td></tr>
<tr><td>Min–max, reversed routing (control)</td><td>127.78 ± 6.96</td><td>130.00 [126.30, 130.00]</td><td>0/10</td><td>0.00288</td></tr>
</tbody>
</table>
</div>

The linear and exponential transforms are indistinguishable on this task, which is what the deficiency-routed family's sharpness property predicts once sharpness is read as relative growth: the two differ in how strongly they amplify a given deficiency gap, not in the ordering they induce, and Chicken–Banana's deficiencies are well separated for most of training. Min–max solves the task in $6$ of $10$ seeds but with far higher across-seed variance, consistent with its rescaling by a running spread that moves whenever the extremes move, even when no component's deficiency has changed. The reversed control is the informative row: inverting the direction of routing, and nothing else, drops the success rate to $0/10$ and pins performance at the level static weighting reaches ($127.78 \pm 6.96$ against UDC's $129.985 \pm 0.047$). The curriculum therefore depends on *which* way effort is routed, not merely on the weights moving; a mechanism that varies the weights with equal magnitude in the wrong direction recovers no benefit at all. We retain the exponential form for its stronger amplification when deficiencies are close together and for the bounded output that [Proposition 1](#prop-dylam-stability) relies on.

### Scaling in the Number of Components {#scaling-in-the-number-of-components}

Our environments decompose into two or three components, which is the range that shaped-reward practice occupies: a shaping vector is designed to be as small as the task allows, and neither our suite nor the MORL/D taxonomy we draw the Pareto benchmarks from <d-cite key="felten2024multi"></d-cite> provides a task with eight or more additive components. We therefore report what the update rule itself implies for larger $n$ rather than an empirical sweep.

Two properties of Eq. $\eqref{eq:dylam-weights}$ are independent of $n$. The ratio between any two weights is

$$
\frac{\lambda_i}{\lambda_j} = \frac{w_i + \epsilon}{w_j + \epsilon} \;\le\; \frac{\mathrm{e} - 1 + \epsilon}{\epsilon},
$$

which caps how far the mechanism can separate a maximally deficient component from a saturated one no matter how many components compete, and [Proposition 1](#prop-dylam-stability)(1) bounds every weight below by $\epsilon / ((n-1)(\mathrm{e}-1) + \epsilon)$, which decays as $1/n$ rather than to zero. Increasing $n$ with the deficiencies held fixed therefore rescales the whole weight vector without changing the ordering it induces: the mechanism dilutes attention across more components, and no subset can capture the budget in the way a softmax with an unbounded logit range could. What growing $n$ does affect is the signal-to-noise ratio of the routing decision, since each component's share of the actor gradient shrinks while the variance of its return estimate does not, so we expect larger $E$ or $\tau_\lambda$ closer to $1$ to be required as $n$ grows. Confirming that expectation requires an environment we do not have; the natural candidate is a multi-agent decomposition in which each agent contributes a component, which we leave to future work.

### Effect of Exploration Rate Decay {#effect-of-exploration-rate-decay}

The exploration–exploitation trade-off is critical in sparse-reward tasks. We test decay factors $\epsilon_d \in \{0.8, 0.9, 0.95\}$, corresponding to increasingly slower exploration decay. Select the "Exploration decay" sweep above for the outcomes.

Only the slowest decay ($0.95$) allows the agent to maintain sufficient exploration (about $50$–$70\%$ exploratory steps) long enough to visit all reward sources. Faster decays cause the agent to settle on suboptimal policies that ignore the hardest component (the Chicken). This highlights that DyLam's success depends on the underlying algorithm's exploration capacity; it facilitates curriculum learning but does not replace the need for adequate exploration.
