# Project NOIR: Cognitive & Affective Mathematical Architecture

## 1. Executive Summary & Core Rationale

### Why Does an AI Model Need Mathematical "Emotions"?
In classical machine learning and deep learning, models are trained as static numerical optimizers minimizing a scalar objective function $\mathcal{L}(\theta)$. While mathematically sound, scalar loss optimization suffers from fundamental limitations:
1. **The Saddle Point & Local Minima Problem**: When gradients become near-zero ($\nabla_\theta \mathcal{L} \approx 0$), standard optimizers (SGD, AdamW) cannot distinguish between true convergence and barren plateaus.
2. **The Exploration-Exploitation Dilemma**: Without intrinsic motivation, an agent or language model repeats known patterns (overfitting) rather than exploring novel information distributions.
3. **Black-Box Epistemic Opacity**: A scalar loss of `2.84` does not tell a researcher *how* the model is learning. Is it confident but slightly misaligned? Is it guessing randomly with maximum entropy? Is it oscillating violently between conflicting samples?

**Project NOIR replaces scalar opacity with an 8-Dimensional Mathematical Affective Vector $\mathbf{e}_t \in [0.0, 1.0]^8$.**

Every emotional dimension in Project NOIR is **not a synthetic gimmick or game heuristic**. It is an exact, deterministic mathematical function computed directly from PyTorch loss tensors, Softmax probability distributions, empirical gradient variances, and Shannon entropy.

---

## 2. The 8-Dimensional Affective State Vector ($\mathbf{e}_t$)

$$\mathbf{e}_t = \begin{bmatrix} K_t \\ U_t \\ C_t \\ S_t \\ F_t \\ Sa_t \\ A_t \\ Ca_t \\ P_t \end{bmatrix} \in [0.0, 1.0]^9$$

```
                           +-------------------------------------+
                           |      PyTorch Forward & Backward     |
                           |   (Logits, Loss, Gradients, PPL)    |
                           +------------------+------------------+
                                              |
                     +------------------------+------------------------+
                     |                                                 |
                     v                                                 v
      +-----------------------------+                   +-----------------------------+
      |      Information Theory     |                   |     Optimization Dynamics   |
      | - Shannon Entropy H(p)      |                   | - Loss Delta (d L / dt)     |
      | - Prediction Certainty max(p)|                  | - Error Variance Var(L)     |
      | - Out-of-Sample Perplexity  |                   | - Stagnation Counter N_stag |
      +--------------+--------------+                   +--------------+--------------+
                     |                                                 |
                     +------------------------+------------------------+
                                              |
                                              v
                           +-------------------------------------+
                           |      Affective Engine Integrator    |
                           |      (Curiosity, Confidence, etc.)  |
                           +------------------+------------------+
                                              |
        +------------------+------------------+------------------+------------------+
        |                  |                  |                  |                  |
        v                  v                  v                  v                  v
  [Confidence K_t]   [Curiosity C_t]   [Uncertainty U_t]  [Frustration F_t]  [Satisfaction S_t]
  (Loss Convergence) (Novelty Search)  (Shannon Entropy)  (Plateau Detector) (Progress Velocity)
```

---

## 3. Mathematical Formulations of Affective Dimensions

### 1. Confidence ($K_t$) — Epistemic Mastery & Certainty
Confidence represents the model's grounded certainty that its learned internal representations correctly predict the target distribution.

$$\begin{aligned}
\text{Loss Factor} &= \exp\left( -\frac{\min(\mathcal{L}_t, 6.0)}{2.5} \right) \in [0, 1] \\
\text{Certainty Target} &= 0.45 \cdot \text{Loss Factor} + 0.35 \cdot (1.0 - U_t) + 0.20 \cdot \max(\text{Acc}_t, \hat{p}_{\max}) \\
K_t &= \alpha K_{t-1} + (1 - \alpha) \cdot \text{Certainty Target} \quad (\alpha = 0.85)
\end{aligned}$$

- **High Loss / High Entropy ($\mathcal{L} \approx 5.5, U_t \approx 0.9$)**: $\text{Loss Factor} \approx 0.11 \implies K_t \approx 0.05$ (Appropriately humble/uncertain).
- **Converging Loss ($\mathcal{L} \to 0.5, U_t \to 0.1$)**: $\text{Loss Factor} \approx 0.82 \implies K_t \to 0.92$ (Progressive, mathematically grounded confidence).

---

### 2. Uncertainty ($U_t$) — Normalized Shannon Entropy
Uncertainty quantifies the dispersion of the model's Softmax probability distribution over the vocabulary or action space of size $K$:

$$U_t = \frac{-\sum_{i=1}^K p_i \ln(p_i)}{\ln(K)}$$

- **$U_t = 1.0$**: Complete randomness (uniform distribution, maximum entropy).
- **$U_t = 0.0$**: Absolute deterministic certainty ($\exists i \text{ s.t. } p_i = 1.0$).

---

### 3. Curiosity ($C_t$) & Intrinsic Motivation
Curiosity is the intrinsic mathematical drive to explore states or token sequences where forward prediction error or entropy is high:

$$\begin{aligned}
r_{\text{intrinsic}} &= \frac{1}{2} \|\hat{s}_{t+1} - s_{t+1}\|_2^2 \\
C_t &= 0.80 \cdot C_{t-1} + 0.20 \cdot \left[ 0.40 \cdot U_t + 0.40 \cdot S_t + 0.20 \cdot (1.0 - K_t) \right]
\end{aligned}$$

When integrated into Reinforcement Learning (PPO) or LLM active token sampling, curiosity adds an intrinsic reward bonus $r_t = r_{\text{ext}} + \eta \cdot r_{\text{int}}$, driving the model out of uninformative loops.

---

### 4. Surprise ($S_t$) — Statistical Shock Detector
Surprise measures the statistical distance between an observed loss $\mathcal{L}_t$ and the expected moving loss distribution:

$$z_t = \frac{\mathcal{L}_t - \mu_{\mathcal{L}, t}}{\sigma_{\mathcal{L}, t} + \epsilon}$$
$$\text{Surprise}_t = \frac{1}{1 + e^{-z_t}}$$

If $\text{Surprise}_t > \tau$ (default $\tau = 0.70$), the system triggers a **Surprise Event**, storing the anomalous sample in episodic memory and pulsing the 3D neural visualization with a shockwave ring.

---

### 5. Frustration ($F_t$) — Stagnation & Plateau Detection
Frustration quantifies consecutive optimization stagnation ($\Delta \mathcal{L} < 10^{-4}$):

$$F_t = \gamma_F \cdot F_{t-1} + 0.08 \cdot \min\left(1.0, \frac{N_{\text{stagnation}}}{5}\right)$$

When $F_t > 0.75$, Project NOIR's **AI Strategist** detects a local minimum and can autonomously propose hyperparameter interventions (e.g., learning rate warmup, weight decay modulation, or branching).

---

### 6. Satisfaction ($Sa_t$) — Progress Velocity
Satisfaction tracks positive learning derivatives and sustained loss reductions:

$$Sa_t = 0.85 \cdot Sa_{t-1} + 0.15 \cdot \left[ 0.50 \cdot \exp(-\mathcal{L}_t / 2.5) + 0.50 \cdot \text{clamp}(4 \cdot (\mathcal{L}_{t-1} - \mathcal{L}_t), 0, 1) \right]$$

---

### 7. Anticipation ($A_t$), Caution ($Ca_t$), and Persistence ($P_t$)
- **Anticipation ($A_t$)**: Tracks learning acceleration ($A_t = 0.80 A_{t-1} + 0.20(0.5 + 0.3 \Delta \text{Acc} + 0.2(1 - F_t))$).
- **Caution ($Ca_t$)**: Scales during high gradient volatility and high uncertainty to signal potential training instability ($Ca_t = 0.80 Ca_{t-1} + 0.20(0.5 U_t + 0.3 F_t + 0.2(1 - K_t))$).
- **Persistence ($P_t$)**: Represents optimization resilience, staying robustly between $[0.80, 1.00]$ ($P_t = 0.85 + 0.15(1.0 - F_t)$).

---

## 4. How the Affective Engine Directly Improves AI Training

| Cognitive State | Classical ML Behavior | Project NOIR Autonomous Behavior |
| :--- | :--- | :--- |
| **High Frustration ($F_t > 0.75$)** | Optimizer stalls in local saddle point indefinitely. | Strategist triggers LR boost or branches experiment. |
| **High Surprise ($S_t > 0.70$)** | Anomaly is treated like normal batch data, potentially causing gradient explosion. | Anomaly is flagged, logged to Episodic Memory, and visualized with 3D shockwave. |
| **High Curiosity ($C_t > 0.80$)** | Model greedily exploits known tokens / trajectories. | Intrinsic reward boosts exploration of unseen token clusters. |
| **Low Confidence ($K_t < 0.20$)** | Black-box output requires human manual inspection. | Telemetry UI explicitly warns researcher that the model is in exploratory phase. |

---

## 5. Implementation Architecture in Code

- **Vector Engine**: [`noir/mind/affective_engine.py`](file:///c:/Users/Arron/Documents/Project%20Noir/noir/mind/affective_engine.py)
- **Shannon Entropy & Variance**: [`noir/mind/uncertainty.py`](file:///c:/Users/Arron/Documents/Project%20Noir/noir/mind/uncertainty.py)
- **Shock & Anomaly Detection**: [`noir/mind/surprise.py`](file:///c:/Users/Arron/Documents/Project%20Noir/noir/mind/surprise.py)
- **Intrinsic Novelty**: [`noir/mind/curiosity.py`](file:///c:/Users/Arron/Documents/Project%20Noir/noir/mind/curiosity.py)
- **Live GUI Telemetry**: [`noir/ui/widgets/emotion_panel.py`](file:///c:/Users/Arron/Documents/Project%20Noir/noir/ui/widgets/emotion_panel.py)
