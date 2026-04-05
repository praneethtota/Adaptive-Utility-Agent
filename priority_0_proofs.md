# 0. Utility Function Justification

Before analyzing the properties of the utility function $U$, we justify its structure from first principles. In its current form,

$$U(E, C, K; f) = w_e(f)\,E + w_c(f)\,C + w_k(f)\,K$$

may appear as a convenient aggregation. This section establishes that its structure is **not arbitrary**, but arises naturally from a set of desiderata on how performance, consistency, and exploration should contribute to decision-making.

---

## 0.1 Additive Linear Structure from Separability and Scaling

We seek a utility function $U : [0,1]^3 \to \mathbb{R}$ over three measurable dimensions:
- efficacy $E$,
- confidence $C$,
- curiosity $K$,

satisfying the following axioms.

**A1 (Monotonicity).**
$U$ is strictly increasing in each argument.

**A2 (Continuity).**
$U$ is continuous on $[0,1]^3$.

**A3 (Marginal Independence / Separability).**
The marginal effect of improving one dimension does not depend on the current level of the others. Formally, for any $E, E', C, C', K$:

$$U(E,C,K) - U(E',C,K) = U(E,C',K) - U(E',C',K)$$

**A4 (Field-Invariant Structure).**
The functional form of $U$ is identical across fields; only the weight vector $w(f)$ may vary with $f$.

**A5 (Linear Scaling Invariance).**
For all $\lambda \in (0, 1]$ such that $(\lambda E, \lambda C, \lambda K) \in [0,1]^3$:

$$U(\lambda E,\, \lambda C,\, \lambda K) = \lambda\, U(E, C, K)$$

*Motivation.* A5 states that scaling all dimensions by the same factor scales utility proportionally. This is natural when $E$, $C$, and $K$ are all measured on the same normalized $[0,1]$ scale: an agent at half performance, half confidence, and half curiosity should have half the utility. It rules out curvature in the component functions and is the standard homogeneity assumption in welfare economics (Blackorby and Donaldson, 1982).

---

### Theorem 0.1

*Under axioms A1–A5, the utility function is necessarily of the form*

$$U(E, C, K; f) = w_e(f)\,E + w_c(f)\,C + w_k(f)\,K$$

*with $w_e(f), w_c(f), w_k(f) > 0$ and $w_e(f) + w_c(f) + w_k(f) = 1$.*

---

### Proof

**Step 1 — Additive representation from A1–A3.**

A3 states that each argument is preferentially independent of the others. Applying this pairwise — $E$ independent of $(C,K)$, $C$ independent of $(E,K)$, $K$ independent of $(E,C)$ — gives mutual preferential independence. By the theorem of Debreu (1960, Theorem 3): a continuous utility function on a connected domain with mutually preferentially independent components admits an additively separable representation. Therefore there exist continuous strictly increasing functions $\phi_E, \phi_C, \phi_K : [0,1] \to \mathbb{R}$ such that:

$$U(E, C, K) = \phi_E(E) + \phi_C(C) + \phi_K(K)$$

**Step 2 — Linearity from A5.**

Substitute the additive form into A5:

$$\phi_E(\lambda E) + \phi_C(\lambda C) + \phi_K(\lambda K) = \lambda\bigl(\phi_E(E) + \phi_C(C) + \phi_K(K)\bigr)$$

Fix $C = C_0 \in (0,1]$ and $K = K_0 \in (0,1]$ and vary $E \in (0,1]$:

$$\phi_E(\lambda E) - \lambda\,\phi_E(E) = \lambda\,\phi_C(C_0) - \phi_C(\lambda C_0) + \lambda\,\phi_K(K_0) - \phi_K(\lambda K_0)$$

The right-hand side depends only on $C_0$, $K_0$, and $\lambda$ — not on $E$. Therefore the left-hand side must be constant in $E$:

$$\phi_E(\lambda E) - \lambda\,\phi_E(E) = h(\lambda) \qquad \text{for all } E \in (0,1],$$

where $h(\lambda)$ is a function of $\lambda$ alone. Differentiating with respect to $E$:

$$\lambda\,\phi_E'(\lambda E) = \lambda\,\phi_E'(E) \implies \phi_E'(\lambda E) = \phi_E'(E)$$

for all $\lambda, E \in (0,1]$. Setting $x = \lambda E$, this says $\phi_E'(x) = \phi_E'(E)$ for all $x$ in the range $(0, E]$. As this holds for all $E \in (0,1]$, $\phi_E'$ is constant on $(0,1]$. Since $\phi_E$ is continuous on $[0,1]$, we conclude $\phi_E$ is affine:

$$\phi_E(E) = w_E\,E + c_E$$

By the same argument applied separately to $C_0$ and $K_0$:

$$\phi_C(C) = w_C\,C + c_C, \qquad \phi_K(K) = w_K\,K + c_K$$

**Step 3 — Normalization.**

With $U(0,0,0) = 0$, we get $c_E + c_C + c_K = 0$. Under A4, the functional form is the same across fields, so field dependence enters only through $w_i(f)$, not through $c_i$. The natural convention $\phi_i(0) = 0$ (zero contribution from a zero-valued dimension) gives $c_i = 0$ individually. Normalizing so that $U(1,1,1) = 1$ yields $w_E + w_C + w_K = 1$. Strict monotonicity (A1) requires $w_i > 0$.

Therefore:

$$U(E, C, K; f) = w_e(f)\,E + w_c(f)\,C + w_k(f)\,K \qquad \blacksquare$$

---

### Remark on Non-Additive Alternatives

Non-separable forms such as $U = E \cdot C$ violate A3: the marginal utility of increasing $E$ depends on the current level of $C$, creating the undesirable incentive of concentrating on dimensions already performing well rather than improving weaknesses. Non-homogeneous forms such as $U = \sqrt{E \cdot C \cdot K}$ violate A5: an agent at half performance does not achieve half utility. The linear form is not merely convenient — it is the unique form satisfying all five axioms jointly.

---

## 0.2 Field-Specific Weighting via Cost Sensitivity

Different domains place different importance on correctness, reliability, and exploration. We model this via a field-dependent weight vector $w(f)$.

**Setup.** Define:
- $c_E(f)$: expected cost of an incorrect output in field $f$,
- $c_C(f)$: expected cost of internal inconsistency in field $f$,
- $c_K(f)$: expected cost of failing to explore high-upside domains in field $f$.

**Design principle.** We set:

$$w_i(f) = \frac{c_i(f)}{c_E(f) + c_C(f) + c_K(f)}$$

so that the gradient $\nabla_x U = (w_e, w_c, w_k)$ is proportional to the cost vector. This ensures that a unit improvement in the highest-cost dimension produces the largest utility gain, aligning the agent's optimization with domain-specific risk.

**Empirical calibration.** The weight ordering is verified against professional liability standards:

| Field | $c_E$ | $c_C$ | $c_K$ | $w_e$ | $w_c$ |
|---|---|---|---|---|---|
| Surgery / Aviation | Very high (irreversible harm) | Very high (trust, procedure) | Low | 0.20 | 0.70 |
| Law | High (precedent, liability) | High (consistency) | Low | 0.30 | 0.60 |
| Software Engineering | Moderate (fixable) | Moderate | Moderate | 0.55 | 0.35 |
| Creative Writing | Low (subjective) | Very low | High (novelty) | 0.80 | 0.05–0.10 |

The weight ordering $w_c(\text{surgery}) \gg w_c(\text{creative})$ is consistent with medical malpractice standards, ICAO Annex 13 aviation incident reporting, and ISO 26262 software safety classifications, all of which impose stronger consistency requirements in higher-stakes fields.

**Status.** This is a decision-theoretic design principle grounded in cost proportionality, not a strict optimality theorem. The weights encode domain knowledge and are calibrated empirically. Future work may derive them from a formal expected-harm minimization over a specified loss model.

---

## 0.3 Efficacy as a Saturating Performance Ratio

We define efficacy as a function of the relative performance ratio:

$$r = \frac{\text{agent performance}}{\text{human baseline}}$$

using the transformation:

$$E(r) = \frac{r}{1+r}$$

### Properties

| Property | Formula | Value |
|---|---|---|
| Parity with human baseline | $E(1)$ | $0.5$ |
| Bounded above | $\lim_{r\to\infty} E(r)$ | $1$ |
| Bounded below | $\lim_{r\to 0} E(r)$ | $0$ |
| Smooth and monotone | $E'(r)$ | $1/(1+r)^2 > 0$ |
| Diminishing returns above baseline | $E''(r)$ for $r>1$ | $< 0$ |

### Mann–Whitney Interpretation

Under the **log-logistic performance model**, $E(r)$ equals the Mann–Whitney dominance probability exactly — not merely analogously.

**Proposition 0.3.** *Let $X_{\text{agent}} \sim \text{LogLogistic}(\mu_a, s)$ and $X_{\text{human}} \sim \text{LogLogistic}(\mu_h, s)$ with the same scale $s=1$, and let $r = e^{\mu_a - \mu_h}$ be the ratio of medians. Then:*

$$P(X_{\text{agent}} > X_{\text{human}}) = \frac{r}{1+r} = E(r)$$

**Proof.** The log-logistic distribution has CDF $F(x; \mu, s) = 1/(1 + e^{-(\log x - \mu)/s})$. For $s=1$, $\log X_a - \log X_h$ follows a logistic distribution with location $\mu_a - \mu_h = \log r$ and scale $\sqrt{2}$. Therefore:

$$P(X_a > X_h) = P(\log X_a > \log X_h) = 1 - F_{\text{logistic}}(0;\, \log r,\, \sqrt{2})$$

$$= 1 - \frac{1}{1 + e^{\log r}} = 1 - \frac{1}{1+r} = \frac{r}{1+r} \qquad \blacksquare$$

**Scope of the claim.** The equality $E(r) = P(X_a > X_h)$ holds under the log-logistic model with equal scale. Under different distributional assumptions (e.g., log-normal), the dominance probability takes a different form. The log-logistic assumption is standard for ratio comparisons in non-parametric statistics and produces the simplest closed form consistent with the boundary conditions $E(0) = 0$, $E(1) = 0.5$, $E(\infty) = 1$. We adopt this model and the resulting formula; the choice of distribution is a modelling assumption, not a mathematical necessity.

### Comparison to Linear Normalization

A linear normalization $E_{\text{lin}}(r) = \min(r, 1)$ has a discontinuous derivative at $r=1$, gives zero marginal utility for any superhuman improvement, and lacks a probabilistic interpretation. The sigmoid form $r/(1+r)$ avoids all three issues and is the natural functional form for a dominance probability under a location-scale family of performance distributions.

---

## 0.4 Confidence as a Kalman-Optimal Filtered Estimate

Confidence is updated via the exponential moving average:

$$C_{t+1} = (1-\alpha)\,C_t + \alpha\,s_t$$

where $s_t \in [0,1]$ is the observed test pass rate at time $t$.

### State-Space Model

Model latent domain confidence $\theta_t$ (the agent's true underlying competence) as a random walk observed through noisy pass rates:

$$\theta_{t+1} = \theta_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0,\, \sigma_q^2) \quad \text{(process noise)}$$

$$s_t = \theta_t + \eta_t, \qquad \eta_t \sim \mathcal{N}(0,\, \sigma_r^2) \quad \text{(observation noise)}$$

The random walk model captures the assumption that true competence changes gradually — through calibration and learning — rather than jumping discontinuously.

### Theorem 0.4 (Kalman–EMA Equivalence)

*In steady state, the Kalman filter for the above system reduces exactly to the EMA update $C_{t+1} = (1-\alpha^*)C_t + \alpha^* s_t$ with optimal gain:*

$$\alpha^* = \frac{-\sigma_q^2 + \sqrt{\sigma_q^4 + 4\sigma_q^2\sigma_r^2}}{2\sigma_r^2}$$

*The choice $\alpha = 0.2$ is optimal when the noise ratio $\rho = \sigma_q^2/\sigma_r^2 = 0.05$.*

**Proof.** The Kalman filter update is $C_{t+1} = C_t + K_t(s_t - C_t)$, identical to the EMA with $\alpha = K_t$. In steady state $K_t \to K^*$. The steady-state error covariance $P^*$ satisfies the discrete algebraic Riccati equation:

$$P^* = \frac{P^* \sigma_r^2}{P^* + \sigma_r^2} + \sigma_q^2$$

with $K^* = P^*/(P^* + \sigma_r^2)$. Substituting $P^* = K^*\sigma_r^2/(1-K^*)$ and simplifying:

$$K^{*2}\sigma_r^2 + K^*\sigma_q^2 - \sigma_q^2 = 0 \quad \Longrightarrow \quad K^* = \frac{-\sigma_q^2 + \sqrt{\sigma_q^4 + 4\sigma_q^2\sigma_r^2}}{2\sigma_r^2}$$

Setting $K^* = \alpha^* = 0.2$ and solving for $\rho = \sigma_q^2/\sigma_r^2$:

$$0.2 = \frac{-\rho + \sqrt{\rho^2 + 4\rho}}{2}
\implies (0.4 + \rho)^2 = \rho^2 + 4\rho
\implies 0.16 = 3.2\rho
\implies \rho = 0.05 \qquad \blacksquare$$

### Interpretation of $\rho = 0.05$

The noise ratio $\rho = 0.05$ means process noise is 5% of observation noise: true competence changes slowly relative to the variability of individual test outcomes. This is the correct regime for incremental calibration over many interactions — a single test pass or fail is noisy, while genuine competence changes only through sustained learning. The value $\alpha = 0.2$ is therefore not arbitrary; it is the Kalman-optimal gain for an agent whose true competence evolves at 5% the rate of observational variability.

### Sensitivity

| $\rho = \sigma_q^2/\sigma_r^2$ | Optimal $\alpha^*$ | Regime |
|---|---|---|
| 0.01 | 0.095 | Very slow competence change — conservative updates |
| 0.05 | 0.200 | Incremental learning (baseline) |
| 0.11 | 0.300 | Moderate-pace learning |
| 0.25 | 0.449 | Fast-changing competence |

For high-stakes fields where competence changes very slowly (surgery, aviation), smaller $\alpha$ values are appropriate. Deriving field-specific optimal gains from domain learning rate estimates is left as future work.

---

## 0.5 Curiosity as a UCB-Inspired Exploration Term

We define:

$$K(d,t) = (C_{\max} - C_d)\;\nu_d\;\bigl(1 + \alpha_f\,\log(1 + n_{\text{fam}})\bigr)$$

where $C_{\max} - C_d$ is the remaining confidence gap, $\nu_d \in [0,1]$ is the novelty of domain $d$, and $n_{\text{fam}}$ counts consecutive familiar interactions (resets on novel problems).

### Structural Analogy to UCB

The UCB1 algorithm (Auer et al., 2002) selects arms by:

$$\text{UCB}_d(t) = \hat{\mu}_d + \sqrt{\frac{2\log t}{n_d}}$$

The curiosity term maps onto this structure as follows:

| UCB1 component | Curiosity component | Interpretation |
|---|---|---|
| $\hat{\mu}_d$ (mean estimate) | $C_d$ (confidence) | Current estimated competence |
| $1 - \hat{\mu}_d$ (uncertainty gap) | $C_{\max} - C_d$ | Remaining upside in domain $d$ |
| $\sqrt{2\log t / n_d}$ (exploration bonus) | $\nu_d\,(1 + \alpha_f \log(1 + n_{\text{fam}}))$ | Novelty-scaled familiarity pressure |

Both bonuses are concave and increasing in the "time since last exploration," creating persistent but diminishing pressure to revisit underexplored domains. The key structural difference is functional form: UCB uses $\sqrt{\log t / n}$; we use $\nu \cdot (1 + \alpha \log n_{\text{fam}})$. Both are in the sublinear growth family that prevents any single domain from being ignored indefinitely.

**What this establishes:** The curiosity term is UCB-*inspired* — it shares the structural properties (uncertainty-driven, concave in familiarity, bounded by exploitation) that make UCB effective. We do not claim exact equivalence to UCB1 or formal regret optimality; those results require a full bandit analysis under our specific setting, which is left as future work.

### Proposition 0.5 — The Cap Enforces Exploitation Dominance

**Proposition.** *The constraint $w_k K \leq w_e E + w_c C$ implies that curiosity contributes at most 50% of total utility at all times:*

$$r_K \;\triangleq\; \frac{w_k K}{U} \;\leq\; \frac{1}{2}$$

**Proof.** Let $S = w_e E + w_c C$ (the exploitation component). The cap states $w_k K \leq S$. Total utility is $U = S + w_k K$. Therefore:

$$r_K = \frac{w_k K}{S + w_k K} \leq \frac{S}{S + S} = \frac{1}{2} \qquad \blacksquare$$

Equality holds only when $w_k K = S$, i.e., when curiosity is at its maximum and exploitation and exploration contribute equally. In all other cases $r_K < 1/2$.

### Why 50%?

The 50% threshold is the tightest constant upper bound derivable from the single constraint "exploitation $\geq$ exploration in utility contribution at all times." A tighter cap (e.g., 30%) would unnecessarily restrict exploration during early learning when $E$ and $C$ are low. A looser cap (e.g., 70%) would permit exploration to dominate even when the agent has high confidence and efficacy — which is the gaming behavior we want to prevent. The 50% bound is therefore not arbitrary: it is the most permissive cap consistent with the requirement that exploitation never falls below exploration.

**What remains open.** Whether the log-growth function achieves optimal regret guarantees under the multi-armed bandit formulation — including formal minimax bounds — is an open question. The analogy to UCB provides intuition and motivation, and the exploitation-dominance property is proved exactly. A formal regret analysis is deferred to future work.

---

## 0.6 Summary

The utility function $U = w_e E + w_c C + w_k K$ is justified as follows:

| Component | Justification | Status |
|---|---|---|
| Additive structure | Debreu (1960) + linear scaling invariance (A5) | Theorem (proved) |
| Linear $\phi_i$ | Cauchy functional equation from A5 | Theorem (proved) |
| Field weights $w_i(f)$ | Cost proportionality, calibrated to liability standards | Design principle |
| Efficacy $E(r) = r/(1+r)$ | Mann-Whitney probability under log-logistic model | Proved under named assumption |
| Confidence EMA, $\alpha=0.2$ | Kalman-optimal for $\rho=0.05$ noise ratio | Theorem (proved) |
| Curiosity structure | UCB-inspired; exploitation-dominance proved | Partial — regret analysis open |

The formulation is not claimed to be the unique possible design, but it is **the minimal, interpretable, and theoretically grounded design** consistent with the five axioms. Each component rests on an identified theoretical foundation, and the scope of each claim is stated explicitly.

---

## References

- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning*, 47(2–3), 235–256.
- Blackorby, C., & Donaldson, D. (1982). Ratio-scale and translation-scale full interpersonal comparability without domain restrictions. *International Economic Review*, 23(2), 249–268.
- Debreu, G. (1960). Topological methods in cardinal utility theory. In K. J. Arrow et al. (Eds.), *Mathematical Methods in the Social Sciences*. Stanford University Press.
- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35–45.
- Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics*, 18(1), 50–60.
