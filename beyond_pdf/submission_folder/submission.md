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
  - name: Ablation
---

## Introduction

Reinforcement learning agents are famously easy to reward and famously hard to reward *well*. Give an agent a single scalar signal and, given enough interaction, it will find a way to maximize it <d-cite key="sutton2018reinforcement"></d-cite> — the trouble starts when the thing you actually care about doesn't collapse into one number. Robotic manipulation <d-cite key="intelligence2025pi"></d-cite><d-cite key="todorov2012mujoco"></d-cite><d-cite key="rsoccer"></d-cite>, competitive game-playing <d-cite key="silver2016mastering"></d-cite><d-cite key="silver2018general"></d-cite><d-cite key="berner2019dota"></d-cite><d-cite key="vinyals2019grandmaster"></d-cite>, and aligning language models with human feedback <d-cite key="kaufmann2023survey"></d-cite> all end up juggling several objectives at once, and the standard move is to fold them into one reward with a set of hand-picked weights: $r = \sum_i \lambda_i r_i$.

Those weights are not cosmetic. They define an implicit curriculum <d-cite key="curriculum"></d-cite> — which sub-goal the agent is effectively being pushed toward at any given moment — and getting the curriculum wrong quietly wrecks training. Tilt the weights too far toward whichever component is easiest to farm and the agent will happily overfit to it: a soccer-playing agent rewarded generously for holding onto the ball can become excellent at holding onto the ball and never discover the sparser, harder objective of actually scoring. The usual fix is to sit down and hand-tune $\lambda$ — a slow, brittle, per-environment ritual that has to be redone from scratch the moment the reward structure changes even slightly.

People have tried to get out of this by other doors. Reward shaping <d-cite key="ng1999policy"></d-cite> adds auxiliary feedback to make learning easier, but the priorities it bakes in are static — they don't know or care how far along the agent already is. Curriculum learning <d-cite key="bengio2009curriculum"></d-cite> tackles the staging problem more directly, and automated variants <d-cite key="portelas2020automatic"></d-cite><d-cite key="li2023understanding"></d-cite><d-cite key="graves2017automated"></d-cite> — including self-paced learning, where the model's own performance decides what to emphasize next <d-cite key="kumar2010self"></d-cite> — go a long way toward removing the hand-crafting. But almost all of that machinery is built to sequence *tasks* or *environments* <d-cite key="xiao2025collaborative"></d-cite><d-cite key="lv2026metagrasp"></d-cite><d-cite key="mead2026multi"></d-cite>, not the components sitting inside a single reward. Hand-staged curricula over reward terms specifically remain a bespoke, per-domain exercise <d-cite key="xiao2023flying"></d-cite><d-cite key="lian2026curriculum"></d-cite><d-cite key="efendi2026learning"></d-cite>. Meanwhile, multi-objective RL methods that decompose the reward — Q-decomposition keeps one value function per component and coordinates them through a shared decision rule <d-cite key="russell2003q"></d-cite><d-cite key="vanseijen2017hybrid"></d-cite><d-cite key="fatemi2022orchestrated"></d-cite>, and methods like Envelope Q-learning or GPI-based prioritization <d-cite key="yang-envelope"></d-cite><d-cite key="alegre2023sample"></d-cite> push the idea further — but they're usually aimed at recovering a whole Pareto front of trade-off policies, not at handing back one well-shaped policy for a single deployment.

That's the gap DyLam sits in: nobody was treating the *weights themselves* as the curriculum. The idea is to keep the reward decomposed — one component, one signal — and let a competence-style measure decide, online, how much attention each component deserves right now, borrowing the "responsibility signal" intuition from Multiple Model-Based RL <d-cite key="doya2002multiple"></d-cite>. Concretely, DyLam tracks each component's recent return against a rough estimate of where that component saturates, and reweights toward whichever components are still underperforming while quietly turning down the ones the agent has already mastered. No fixed schedule, no per-environment retuning — just two numbers per component, a floor and a ceiling on its return, and the weights find their own way over the course of training. An interactive walkthrough of this update over one slice of real training data follows next, before the formal statement of the mechanism.

<link rel="stylesheet" href="{{ '/assets/css/mechanism.css' | relative_url }}">

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

<!-- BEGIN #43: lambda-simplex scrubber Embed (results). Provisional
     placement -- issue #39's assembly ticket moves this into a full
     "curriculum results" subsection once that prose exists. -->

## Results

DyLam decomposes a multi-objective reward into per-Component critics and adapts a weight $\lambda(t)$ over training to trade them off. Scrub the slider below to watch $\lambda$ move through training on ChickenBanana and see the greedy policy it induces at each moment. Click anywhere in the weight simplex to try a different, fixed weighting against the *same* learned Component Q-tables at that same moment -- the counterfactual this Embed exists to make checkable.

<div class="l-page">
  <iframe src="{{ 'assets/html/submission/lambda_simplex_scrubber.html' | relative_url }}" frameborder="0" scrolling="no" height="720" width="100%" style="border: none;"></iframe>
</div>

Left: the ChickenBanana grid, with the currently-displayed policy's greedy action drawn in each reachable cell (before either pickup) and its rollout path overlaid statically -- the path is not animated, so two weightings can be compared at a glance by switching between them. Middle: the three-Component weight simplex, coloured by the behaviour class each weighting's policy achieves at the currently selected episode, with DyLam's own $\lambda$ trajectory drawn as a trail up to that episode. Clicking elsewhere fixes that weighting (free mode, amber) so the time slider keeps scrubbing the counterfactual; the button above returns to DyLam's own trajectory (follow mode, green). Right: the rollout's per-Component returns (solid bars) against the method's own r_max ceilings (dashed), with DyLam's actual training-time return at that episode ghosted behind for reference.

<!-- END #43 -->
