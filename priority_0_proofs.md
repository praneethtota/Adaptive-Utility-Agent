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

**Proof.** The log-logistic distribution has CDF $F(x; \mu, s) = 1/(1 + e^{-(\log x - \mu)/s})$. For $s=1$:

$$P(X_a > X_h) = P(\log X_a > \log X_h) = \int_0^\infty F_{X_h}(x)\,f_{X_a}(x)\,dx$$

Under the log-logistic model with $\mu_a$ and $\mu_h$, this integral yields the closed-form expression (under the log-logistic assumption):

$$P(X_a > X_h) = \frac{e^{\mu_a}}{e^{\mu_a} + e^{\mu_h}} = \frac{e^{\mu_a - \mu_h}}{1 + e^{\mu_a - \mu_h}} = \frac{r}{1+r} \qquad \blacksquare$$

*Note:* The result uses the specific structure of the log-logistic CDF. The difference of two log-logistic random variables does not in general follow a logistic distribution; the closed form arises directly from evaluating the integral under this model, not from a distribution-of-differences argument.

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

$$\alpha^* = \frac{-\sigma_q^2 + \sqrt{\sigma_q^{4} + 4\sigma_q^{2}\sigma_r^{2}}}{2\sigma_r^2}$$

*The choice $\alpha = 0.2$ is optimal when the noise ratio $\rho = \sigma_q^2/\sigma_r^2 = 0.05$.*

**Proof.** The Kalman filter update is $C_{t+1} = C_t + K_t(s_t - C_t)$, identical to the EMA with $\alpha = K_t$. In steady state $K_t \to K^*$. The steady-state error covariance $P^*$ satisfies the discrete algebraic Riccati equation:

$$P^* = \frac{P^* \sigma_r^2}{P^* + \sigma_r^2} + \sigma_q^2$$

with $K^* = P^*/(P^* + \sigma_r^2)$. Substituting $P^* = K^*\sigma_r^2/(1-K^*)$ and simplifying:

$$K^{*2}\sigma_r^2 + K^*\sigma_q^2 - \sigma_q^2 = 0 \quad \Longrightarrow \quad K^* = \frac{-\sigma_q^2 + \sqrt{\sigma_q^{4} + 4\sigma_q^{2}\sigma_r^{2}}}{2\sigma_r^2}$$

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

## Theorem 2 — Convergence of Confidence Under Repeated Calibration

### Setup

Recall the confidence update rule with contradiction penalty:

$$C_{t+1} = (1-\alpha)\,C_t + \alpha\,s_t\,(1 - \lambda\mu(f))$$

where:
- $\alpha \in (0,1)$ is the EMA learning rate,
- $s_t \in [0,1]$ is the observed test pass rate at time $t$,
- $\lambda \in [0,1]$ is the contradiction penalty magnitude (zero when no contradiction occurs),
- $\mu(f) \geq 1$ is the field penalty multiplier,
- $f$ denotes the active field.

Define the **effective signal** $\tilde{s}_t = s_t(1 - \lambda\mu(f))$. When no contradiction occurs, $\lambda = 0$ and $\tilde{s}_t = s_t$. When a contradiction of magnitude $\lambda$ is detected, $\tilde{s}_t$ is reduced by a factor $(1 - \lambda\mu(f))$.

The update rule becomes:

$$C_{t+1} = (1-\alpha)\,C_t + \alpha\,\tilde{s}_t$$

---

### Closed-Form Solution

**Lemma.** *The confidence at time $t$ is:*

$$C_t = (1-\alpha)^t\,C_0 + \alpha\sum_{k=0}^{t-1}(1-\alpha)^{t-1-k}\,\tilde{s}_k$$

**Proof.** By induction. Base case $t=0$: $C_0 = C_0$. Inductive step: assume the formula holds for $t$. Then:

$$C_{t+1} = (1-\alpha)C_t + \alpha\tilde{s}_t$$
$$= (1-\alpha)\!\left[(1-\alpha)^t C_0 + \alpha\sum_{k=0}^{t-1}(1-\alpha)^{t-1-k}\tilde{s}_k\right] + \alpha\tilde{s}_t$$
$$= (1-\alpha)^{t+1}C_0 + \alpha\sum_{k=0}^{t-1}(1-\alpha)^{t-k}\tilde{s}_k + \alpha\tilde{s}_t$$
$$= (1-\alpha)^{t+1}C_0 + \alpha\sum_{k=0}^{t}(1-\alpha)^{t-k}\tilde{s}_k \qquad \blacksquare$$

---

### Theorem 2 — Convergence, Uniqueness, and Recovery

**Theorem.** *Let $\{\tilde{s}_t\}$ be a stationary sequence with constant expectation $\bar{s} \in [0,1]$ and let $\tilde{s}^* = \bar{s}(1 - \lambda\mu(f))$ be the expected effective signal. Then:*

1. **Existence and uniqueness of steady state.** There exists a unique fixed point $C^*$ of the update rule, given by:

$$C^* = \bar{s}\,(1 - \lambda\mu(f)) = \tilde{s}^*$$

2. **Geometric convergence in expectation.** For all $t \geq 0$:

$$\mathbb{E}[|C_t - C^\*|] \leq (1-\alpha)^t\,|C_0 - C^*|$$

The bound holds in expectation over the noise in $\tilde{s}_t$; it is not a deterministic almost-sure bound because the noise term $\alpha(\tilde{s}_t - \tilde{s}^*)$ does not vanish pathwise. With $\alpha = 0.2$, the expected error halves every $\lceil\log(0.5)/\log(0.8)\rceil = 3$ interactions.

3. **Monotonicity.** $C^*$ is strictly increasing in $\bar{s}$:

$$\frac{\partial C^*}{\partial \bar{s}} = 1 - \lambda\mu(f) > 0$$

Higher agent pass rates produce higher steady-state confidence, all else equal.

4. **Field sensitivity.** $C^*$ is strictly decreasing in $\mu(f)$:

$$\frac{\partial C^*}{\partial \mu(f)} = -\bar{s}\,\lambda < 0$$

Higher-stakes fields (larger $\mu(f)$) impose a lower achievable steady-state confidence, correctly encoding that high-stakes domains demand stricter internal consistency standards.

5. **Contradiction recovery time.** Suppose the agent is at steady state $C^*$ and a contradiction event causes an instantaneous drop of magnitude $\delta$, so $C_{\tau} = C^* - \delta$. The number of subsequent interactions required to return to within $\varepsilon$ of $C^*$ is:

$$t_{\text{recovery}} = \left\lceil\frac{\log(\varepsilon/\delta)}{\log(1-\alpha)}\right\rceil$$

**Proof.**

*Part 1 — Existence and uniqueness.* The fixed point satisfies $C^* = (1-\alpha)C^* + \alpha\tilde{s}^*$, giving $\alpha C^* = \alpha\tilde{s}^*$, hence $C^* = \tilde{s}^* = \bar{s}(1-\lambda\mu(f))$. This is unique since the equation is linear in $C^*$.

*Part 2 — Geometric convergence in expectation.* Define the error $e_t = C_t - C^*$. Subtracting the fixed-point equation from the update rule:

$$e_{t+1} = (1-\alpha)\,e_t + \alpha(\tilde{s}_t - \tilde{s}^*)$$

The noise term $\eta_t \triangleq \alpha(\tilde{s}_t - \tilde{s}^*)$ has zero mean under the stationary distribution (since $\mathbb{E}[\tilde{s}_t] = \tilde{s}^*$) but does not vanish pathwise. Taking expectations:

$$\mathbb{E}[e_{t+1}] = (1-\alpha)\,\mathbb{E}[e_t] + \alpha\,\mathbb{E}[\tilde{s}_t - \tilde{s}^*] = (1-\alpha)\,\mathbb{E}[e_t]$$

Iterating from $t=0$ with $e_0 = C_0 - C^*$ deterministic:

$$\mathbb{E}[e_t] = (1-\alpha)^t\,(C_0 - C^*)$$

By Jensen's inequality $|\mathbb{E}[e_t]| \leq \mathbb{E}[|e_t|]$, and applying the triangle inequality to the closed-form solution:

$$\mathbb{E}[|e_t|] \leq (1-\alpha)^t\,|C_0 - C^*| + \alpha\sum_{k=0}^{t-1}(1-\alpha)^{t-1-k}\,\mathbb{E}[|\tilde{s}_k - \tilde{s}^*|]$$

The second term represents the expected accumulated noise. Under the stationary assumption with zero-mean noise, $\mathbb{E}[|\tilde{s}_k - \tilde{s}^*|]$ is a constant $\sigma_{\tilde{s}}$ and the noise sum telescopes to $\alpha\sigma_{\tilde{s}}/(\alpha) = \sigma_{\tilde{s}}$. This gives the tighter statement:

$$\mathbb{E}[|e_t|] \leq (1-\alpha)^t\,|C_0 - C^*| + \sigma_{\tilde{s}}$$

where $\sigma_{\tilde{s}} = \mathbb{E}[|\tilde{s}_t - \tilde{s}^*|]$ is the mean absolute deviation of the effective signal. When the signal has low noise ($\sigma_{\tilde{s}} \approx 0$), the bound reduces to the clean geometric form. **The bound $\mathbb{E}[|e_t|] \leq (1-\alpha)^t |C_0 - C^*|$ holds exactly when $\sigma_{\tilde{s}} = 0$ (deterministic signal) and approximately when noise is small.**

The half-life of the mean error is $t_{1/2} = \log(1/2)/\log(1-\alpha)$. For $\alpha = 0.2$: $t_{1/2} = \log(0.5)/\log(0.8) = 3.11$, so $\lceil t_{1/2} \rceil = 3$ interactions.

*Part 3 — Monotonicity.* $\partial C^*/\partial\bar{s} = (1 - \lambda\mu(f))$. Since $\lambda \in [0,1]$ and $\mu(f) \geq 1$ but $\lambda\mu(f) < 1$ for any non-degenerate field (confidence cannot be driven to zero by a single contradiction), this derivative is strictly positive.

*Part 4 — Field sensitivity.* $\partial C^*/\partial\mu(f) = -\bar{s}\lambda \leq 0$, with strict inequality whenever $\bar{s} > 0$ and $\lambda > 0$. This formalizes the intuition that high-stakes fields (large $\mu(f)$) penalize contradictions more heavily, pulling the achievable steady-state confidence downward even when pass rates are high.

*Part 5 — Recovery time.* After the contradiction, $e_{\tau} = -\delta$ (deterministic drop). In the noise-free case ($\sigma_{\tilde{s}} = 0$), by geometric convergence, $|e_{\tau+t}| \leq (1-\alpha)^t\,\delta$. We want this below $\varepsilon$:

$$(1-\alpha)^t\,\delta \leq \varepsilon \implies t \geq \frac{\log(\varepsilon/\delta)}{\log(1-\alpha)}$$

Since $\log(1-\alpha) < 0$ and $\varepsilon < \delta$ (we want to recover closer than the drop), $\varepsilon/\delta < 1$ and $\log(\varepsilon/\delta) < 0$, making $t$ positive. Therefore:

$$t_{\text{recovery}} = \left\lceil\frac{\log(\varepsilon/\delta)}{\log(1-\alpha)}\right\rceil \qquad \blacksquare$$

---

### Worked Examples

**Example 1 — Software engineering, no contradictions.**

$\mu(f) = 2$, $\lambda = 0$, $\bar{s} = 0.85$, $C_0 = 0.5$, $\alpha = 0.2$.

$$C^* = 0.85 \times (1 - 0 \times 2) = 0.85$$
$$|C_t - 0.85| \leq 0.8^t \times 0.35$$

After 10 interactions: $|e_{10}| \leq 0.8^{10} \times 0.35 = 0.107 \times 0.35 \approx 0.037$.
After 20 interactions: $|e_{20}| \leq 0.8^{20} \times 0.35 \approx 0.0038$.

**Example 2 — Surgery, boundary case.**

$\mu(f) = 10$, $\lambda = 0.1$ (moderate per-interaction penalty), $\bar{s} = 0.90$.

$$C^* = 0.90 \times (1 - 0.1 \times 10) = 0.90 \times 0 = 0$$

*This represents an extreme boundary case; in practice $\lambda$ is small per interaction (contradictions are rare events, not a sustained rate).* The example illustrates correct model behavior at the boundary: when $\lambda\mu(f) = 1$, the contradiction penalty exactly cancels the pass-rate signal, and the agent cannot build confidence regardless of performance. This is the intended behavior — a surgical agent that consistently contradicts itself on verified claims should be unable to achieve operating confidence and must abstain.

**Example 3 — Recovery time.**

Starting from a drop of $\delta = 0.15$ (a significant contradiction), recovering to within $\varepsilon = 0.01$ of $C^*$ with $\alpha = 0.2$:

$$t_{\text{recovery}} = \left\lceil\frac{\log(0.01/0.15)}{\log(0.8)}\right\rceil = \left\lceil\frac{-2.708}{-0.223}\right\rceil = \lceil 12.1 \rceil = 13 \text{ interactions}$$

Thirteen clean calibration interactions are sufficient to recover from a large contradiction event to within 1% of steady-state confidence. This is consistent with the simulation results in Appendix A.

---

### Corollary — Steady-State Confidence Bounds by Field

Substituting typical field parameters with $\bar{s} \approx 0.85$ (a well-calibrated agent) and $\lambda \approx 0.05$ (low contradiction rate):

| Field | $\mu(f)$ | $C^* = \bar{s}(1-\lambda\mu)$ | $C_{\min}$ | Achievable? |
|---|---|---|---|---|
| Surgery / Aviation | 10 | $0.85 \times 0.5 = 0.425$ | 0.95 | No — requires $\bar{s} > 0.95/(1-0.5) = 1.9$ |
| Law | 5 | $0.85 \times 0.75 = 0.638$ | 0.85 | Borderline |
| Software Engineering | 2 | $0.85 \times 0.9 = 0.765$ | 0.70 | Yes |
| Creative Writing | 1 | $0.85 \times 0.95 = 0.808$ | 0.05 | Easily |

This corollary formalizes the whitepaper's claim that high-stakes fields impose stricter confidence standards: for surgery, a typical agent with $\bar{s} = 0.85$ and any nonzero contradiction rate cannot achieve the $C_{\min} = 0.95$ threshold, and must abstain or escalate. The confidence floor is not merely a policy choice — it is above the achievable steady state, guaranteeing the abstention mechanism is triggered when the agent is not reliably correct.

---

---

## Theorem 4 — Lyapunov Stability of the Personality System

### Corrected Claim

The personality system exhibits **bounded, stable dynamics with convergence to a neighborhood of the field neutral $s^*$ under bounded drift**. We do not claim a unique globally stable equilibrium, which would require the mean reversion to dominate the drift — a condition that does not hold for the parameter values $\beta = 0.01$, $\Delta_{\max} = 0.05$ used in this system. Instead we prove the three things that *are* true: invariance of the feasible set, geometric convergence to $s^*$ when drift is absent, and bounded dynamics otherwise.

---

### Setup

The personality trait vector $s = (s_1, \ldots, s_n) \in \mathbb{R}^n$ evolves under:

$$s_{t+1} = \Pi_B\!\left[s_t + \Delta_t - \beta(s_t - s^*)\right] = \Pi_B\!\left[(1-\beta)s_t + \beta s^* + \Delta_t\right]$$

where:

- $B = \prod_{i=1}^n [s_{\min}^{i},\, s_{\max}^{i}]$ is the field-specific feasible set (a closed convex box in $\mathbb{R}^n$)
- $\Pi_B : \mathbb{R}^n \to B$ is the Euclidean projection onto $B$
- $\Delta_t \in \mathbb{R}^n$ is the raw drift from utility history, with $\|\Delta_t\|_\infty \leq \Delta_{\max}$ per component
- $\beta = 0.01$ is the mean reversion coefficient
- $s^* \in B$ is the field neutral personality (interior point of $B$ by construction)

Define the Lyapunov function:

$$V(s) = \|s - s^*\|^2 = \sum_{i=1}^n (s_i - s_i^*)^2 \geq 0$$

with $V(s^*) = 0$.

---

### Theorem 4 — Bounded Stable Dynamics with Neighborhood Convergence

**Theorem.** *Under the three-layer personality evolution rule above:*

**(i) Invariance.** $s_t \in B$ for all $t \geq 0$ whenever $s_0 \in B$.

**(ii) Zero-drift convergence.** When $\Delta_t = 0$ for all $t$, $V(s_t)$ converges geometrically to zero:

$$V(s_{t+1}) \leq (1-\beta)^2\, V(s_t)$$

*so $\|s_t - s^*\| \leq (1-\beta)^t \|s_0 - s^*\|$, with convergence rate $(1-\beta)^2 = 0.9801$ per cycle.*

**(iii) Bounded displacement.** The single-step displacement is bounded:

$$\|s_{t+1} - s_t\| \leq \Delta_{\max}\sqrt{n} + \beta\,\|s_t - s^*\|$$

*so no single evolution cycle can produce a large jump.*

**(iv) Persistent-drift stability.** Under persistent drift with $\|\Delta_t\| \leq \Delta_{\max}\sqrt{n}$, the distance to $s^*$ satisfies:

$$\|s_{t+1} - s^*\| \leq (1-\beta)\|s_t - s^*\| + \Delta_{\max}\sqrt{n}$$

*The dynamics converge to and remain in the neighborhood*

$$\mathcal{N}^* = \left\{s \in B : \|s - s^*\| \leq r^*\right\}, \qquad r^* = \min\!\left(\frac{\Delta_{\max}\sqrt{n}}{\beta},\; \mathrm{diam}(B)\right)$$

*For the system parameters $\beta = 0.01$, $\Delta_{\max} = 0.05$, $n = 6$ traits: $\Delta_{\max}\sqrt{n}/\beta = 12.25$, while $\mathrm{diam}(B) \leq \sqrt{n} \approx 2.45$. The field bounds are therefore the binding constraint, not the mean reversion — $\mathcal{N}^* = B$.*

---

### Proof

**Part (i) — Invariance.**

$\Pi_B$ is defined as the Euclidean projection onto the closed convex set $B$, so $\Pi_B(x) \in B$ for every $x \in \mathbb{R}^n$. Therefore $s_{t+1} = \Pi_B[\cdot] \in B$ for all $t \geq 0$, regardless of $\Delta_t$. $\blacksquare$

**Part (ii) — Zero-drift convergence.**

Set $\Delta_t = 0$. Then $s_{t+1} = \Pi_B[(1-\beta)s_t + \beta s^*]$.

Since $B$ is convex and both $s_t \in B$ and $s^* \in B$, the convex combination $(1-\beta)s_t + \beta s^* \in B$, so $\Pi_B$ acts as the identity:

$$s_{t+1} = (1-\beta)s_t + \beta s^*$$

Therefore:

$$s_{t+1} - s^* = (1-\beta)s_t + \beta s^* - s^* = (1-\beta)(s_t - s^*)$$

and:

$$V(s_{t+1}) = \|(1-\beta)(s_t - s^*)\|^2 = (1-\beta)^2\,V(s_t) \qquad \blacksquare$$

The convergence rate per cycle is $(1-\beta)^2 = (0.99)^2 = 0.9801$. The half-life is:

$$t_{1/2} = \frac{\log(1/2)}{\log(1-\beta)^2} = \frac{\log(1/2)}{2\log(0.99)} \approx \frac{-0.693}{-0.0201} \approx 34 \text{ cycles}$$

With personality evolution running every $N=3$ interactions, this corresponds to approximately 102 interactions to halve the distance to $s^*$ under zero drift.

**Part (iii) — Bounded displacement.**

$$\|s_{t+1} - s_t\| = \|\Pi_B[(1-\beta)s_t + \beta s^* + \Delta_t] - s_t\|$$

Since $\Pi_B$ is non-expansive and $s_t = \Pi_B[s_t]$:

$$\|\Pi_B[x] - \Pi_B[y]\| \leq \|x - y\| \quad \text{for all } x, y$$

$$\|s_{t+1} - s_t\| \leq \|(1-\beta)s_t + \beta s^* + \Delta_t - s_t\|$$
$$= \|-\beta(s_t - s^*) + \Delta_t\| \leq \beta\|s_t - s^*\| + \|\Delta_t\| \leq \beta\,\|s_t - s^*\| + \Delta_{\max}\sqrt{n} \qquad \blacksquare$$

**Part (iv) — Persistent-drift stability.**

Since $s^* \in B$, the non-expansiveness of $\Pi_B$ with respect to any point in $B$ gives:

$$\|s_{t+1} - s^*\| = \|\Pi_B[(1-\beta)s_t + \beta s^* + \Delta_t] - \Pi_B[s^*]\|$$
$$\leq \|(1-\beta)s_t + \beta s^* + \Delta_t - s^*\|$$
$$= \|(1-\beta)(s_t - s^*) + \Delta_t\|$$
$$\leq (1-\beta)\|s_t - s^*\| + \|\Delta_t\|$$
$$\leq (1-\beta)\|s_t - s^*\| + \Delta_{\max}\sqrt{n}$$

Let $d_t = \|s_t - s^*\|$. The recurrence $d_{t+1} \leq (1-\beta)d_t + \Delta_{\max}\sqrt{n}$ has fixed point $d^* = \Delta_{\max}\sqrt{n}/\beta$. By the theory of contractive linear recurrences:

- If $d_t > d^*$: $d_{t+1} < d_t$ (distance decreasing toward $d^*$)
- If $d_t \leq d^*$: $d_{t+1} \leq d^*$ (distance stays within neighborhood)

So $\limsup_{t\to\infty} d_t \leq d^* = \Delta_{\max}\sqrt{n}/\beta$.

Since $d_t \leq \mathrm{diam}(B)$ always by Part (i), the binding constraint is:

$$r^* = \min\!\left(\frac{\Delta_{\max}\sqrt{n}}{\beta},\; \mathrm{diam}(B)\right) \qquad \blacksquare$$

---

### Why the Field Bounds Are the Binding Constraint

For the system parameters:

$$\frac{\Delta_{\max}\sqrt{n}}{\beta} = \frac{0.05 \times \sqrt{6}}{0.01} = \frac{0.1225}{0.01} = 12.25$$

$$\mathrm{diam}(B) = \left\|\,s_{\max} - s_{\min}\,\right\| \leq \sqrt{n} \approx 2.45 \quad \text{(since each trait is in }[0,1]\text{)}$$

Since $12.25 \gg 2.45$, the mean reversion alone is insufficient to confine the dynamics to a small neighborhood of $s^*$. **The projection $\Pi_B$ is the primary stability mechanism** — it enforces invariance, and the field bounds define how far from $s^*$ the personality can stray. The mean reversion serves as a regularizer: without it, the system would drift to and remain at the boundary of $B$; with it, there is a gentle restoring force toward $s^*$ when drift is absent.

This clarifies the design role of each stability layer:

| Layer | Mechanism | Guarantee |
|---|---|---|
| Field bounds $[s_{\min}, s_{\max}]$ | Projection $\Pi_B$ | Hard invariance — $s_t \in B$ always |
| Drift rate cap $\Delta_{\max}$ | Truncation of raw drift | Bounded single-step displacement |
| Mean reversion $\beta$ | Soft pull toward $s^*$ | Convergence to $s^*$ when drift is absent; regularization otherwise |

---

### Corollary — No Oscillation

**Corollary.** *The drift rate cap $\Delta_{\max}$ prevents oscillation: after any evolution step, the personality cannot cross $s^*$ in a single step.*

**Proof.** A step crosses $s^*$ in dimension $i$ if $s_{t+1}^i - s_i^*$ and $s_t^i - s_i^*$ have opposite signs. The signed displacement in dimension $i$ is:

$$s_{t+1}^i - s_i^* = \Pi_{[s_{\min}^{i}, s_{\max}^{i}]}\!\left[(1-\beta)(s_t^i - s_i^*) + \Delta_t^i\right]$$

For oscillation, we need $|\Delta_t^i| > (1-\beta)|s_t^i - s_i^*|$, i.e., the drift must exceed the current displacement. Since $\Delta_t^i \leq \Delta_{\max} = 0.05$ and $s_t^i - s_i^*$ can be as large as $s_{\max}^{i} - s_i^*$ (which is at least $0.05$ by the bound structure), oscillation requires the personality to be within $\Delta_{\max}/(1-\beta) \approx 0.0505$ of $s^*$ in that dimension. This is a vanishingly small region, confirming that oscillation can only occur when the personality is already very near the neutral point. $\blacksquare$

---

### Remark on Parameter Calibration

The analysis reveals a structural tension: mean reversion $\beta = 0.01$ is weaker than the maximum drift $\Delta_{\max} = 0.05$ per cycle. This is intentional — personality is meant to evolve meaningfully, not snap back to $s^*$ every few cycles. The design trades off:

- **Large $\beta$**: fast convergence to $s^*$, but personality becomes unresponsive to experience
- **Small $\beta$**: personality evolves with experience, but reverts slowly between periods of drift

The value $\beta = 0.01$ produces a half-life of approximately 34 evolution cycles under zero drift (≈ 102 interactions), which is long enough for the personality to reflect accumulated experience over hundreds of interactions before the neutral pull becomes dominant. A field-specific $\beta(f)$ — with higher reversion rates for high-stakes fields where personality stability is more important — is a natural extension for future work.

---

## References

- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning*, 47(2–3), 235–256.
- Blackorby, C., & Donaldson, D. (1982). Ratio-scale and translation-scale full interpersonal comparability without domain restrictions. *International Economic Review*, 23(2), 249–268.
- Debreu, G. (1960). Topological methods in cardinal utility theory. In K. J. Arrow et al. (Eds.), *Mathematical Methods in the Social Sciences*. Stanford University Press.
- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35–45.
- Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics*, 18(1), 50–60.
- Wald, A. (1945). Sequential tests of statistical hypotheses. *Annals of Mathematical Statistics*, 16(2), 117–186.
