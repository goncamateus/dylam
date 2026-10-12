# The primal–dual baseline normalizes its multipliers onto the simplex and never clips them

The reviewer asked for a Lagrangian baseline with DyLam's thresholds to test the
"proportional vs. integral" claim of Sec. III-C. We treat every component as a
constraint G_i ≥ R_max^i (no primary objective), integrate the normalized signed
violation v_i = (R_max^i − Ḡ_i)/(R_max^i − R_min^i) as μ_i ← [μ_i + η v_i]_+, and
feed w = μ/Σμ into the same decomposed SAC / Q-learning update DyLam uses. Only
the weight rule differs; Ḡ comes from the same code path.

We do not clip μ: clipping is anti-windup, the remedy PID-Lagrangian proposes for
exactly this failure, so a clipped baseline would no longer be integral. Leaving
μ unnormalized (Σ μ_i Q_i) was rejected as a strawman: the actor's gradient scale
would grow without bound.

## Consequences

In HalfCheetah and VSS no threshold is ever met, so every μ_i grows forever and
w converges to the running average of past violations — DyLam with unbounded
memory. The divergence the paper describes is real for μ but invisible to the
policy, so Sec. III-C must describe the observable effect (weights tied to the
whole violation history), not divergence. η then only sets how fast the uniform
prior μ_0 = 1 is forgotten, which is why η = 1 − τ_λ needs no tuning.

No environment exercises relaxation. Chicken–Banana's R_max are the algebraic
maxima, so Ḡ ≤ R_max and the violation is never negative: a met component's μ
freezes instead of decaying, and keeps its share of the weight. That frozen share
is the integral term's memory, and it is the sharpest P-vs-I contrast available:
DyLam's weight on a met component falls to the floor, the dual's does not.
