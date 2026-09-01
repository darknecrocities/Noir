# Affective & Cognitive Engine Specification

## 1. Overview

The Project NOIR Affective Engine calculates a real-time mathematical representation of internal cognitive and emotional dynamics. Unlike game heuristics or simulated meters, every affective state update is directly derived from observable training variables: prediction error, loss gradients, Shannon entropy, task reward, and forward-dynamics novelty.

---

## 2. Mathematical Affective State Vector ($E_t$)

The affective state is represented as an 8-dimensional normalized vector:

$$E_t = \begin{bmatrix} C_t \\ F_t \\ A_t \\ S_t \\ U_t \\ X_t \\ Ca_t \\ P_t \end{bmatrix} \in [0.0, 1.0]^8$$

### Dimensions & Mathematical Meanings

| Symbol | Dimension | Operational Definition | Mathematical Trigger |
| :---: | :--- | :--- | :--- |
| **$C_t$** | **Confidence** | Certainty in current policy/model mastery | Exponential moving average of accuracy & low entropy: $C_t \leftarrow \alpha C_{t-1} + \beta \text{Acc} - \gamma U_t$ |
| **$F_t$** | **Frustration** | Response to stagnation or penalty spikes | Error stagnation counter & repeated penalties: $F_t \leftarrow \alpha F_{t-1} + \delta_{\text{stagnant}}$ |
| **$A_t$** | **Anticipation** | Expectancy of goal attainment | Distance reduction & reward gradient: $A_t \leftarrow 0.5 + 0.5(1 - d/d_0)$ |
| **$S_t$** | **Satisfaction** | Fulfillment upon goal completion or breakthrough | Reward peaks & sharp loss decreases: $S_t \leftarrow \alpha S_{t-1} + \beta \Delta \text{Loss}$ |
| **$U_t$** | **Uncertainty** | Ambiguity in predictions | Normalized Shannon Entropy $H(p)$ & variance |
| **$X_t$** | **Curiosity** | Intrinsic drive for exploration & novelty | Forward-dynamics error $\|f(s,a) - s'\|^2$ & surprise shocks |
| **$Ca_t$**| **Caution** | Sensitivity to catastrophic loss or obstacles | Proximity to penalty regions & high gradient variance |
| **$P_t$** | **Persistence** | Resilience in pursuing optimization | Continuity metric weighted inversely against chronic frustration |

---

## 3. Predictive Uncertainty

For classification probability distributions $p = [p_1, p_2, \dots, p_K]$, predictive uncertainty is measured via normalized Shannon Entropy:

$$H(p) = -\frac{1}{\ln K} \sum_{i=1}^K p_i \ln(p_i)$$

Where $H(p) \in [0, 1]$. An entropy close to $1.0$ indicates maximum ambiguity (uniform prediction), while $0.0$ indicates complete certainty.

---

## 4. Perceptual Surprise Detection

Surprise $S_t$ is computed via self-information or squared prediction error:

$$S_t = \|\hat{s}_{t+1} - s_{t+1}\|^2 \quad \text{or} \quad I(\text{event}) = -\log_2 P(\text{event})$$

The `SurpriseDetector` maintains running mean $\mu_S$ and variance $\sigma_S^2$. When the normalized z-score exceeds the threshold ($\tau = 0.70$), the system triggers:
1. `SURPRISE_DETECTED` event emission.
2. Immediate episodic memory logging.
3. Shockwave visualization pulse in the 3D neural viewport.
4. Elevation of curiosity $X_t$.
