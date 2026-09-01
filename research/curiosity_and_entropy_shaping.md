# Intrinsic Curiosity and Predictive Entropy Shaping

**Authors**: Project NOIR Research Group  
**Subject**: Reinforcement Learning, Exploration, Information Theory

---

## 1. The Sparse Reward Problem in Spatial Navigation

In discrete grid environments with sparse goal rewards ($R_{\text{goal}} = +10.0$), standard gradient-based policy optimization suffers from exponential sample complexity due to random walk diffusion.

---

## 2. Dual Formulation: Intrinsic Motivation & Predictive Entropy

To resolve sparse rewards without reward hacking, NOIR employs a dual exploration signal:

### 2.1 Forward Dynamics Error

A neural network $f_\phi(s_t, a_t)$ is trained concurrently on the transition dynamics:

$$\mathcal{L}_{\text{dyn}}(\phi) = \frac{1}{2} \|f_\phi(s_t, a_t) - s_{t+1}\|_2^2$$

The intrinsic reward is computed prior to parameter updates:

$$R_{\text{int}}(s_t, a_t) = \eta \cdot \min\left(1.0, \; \|f_\phi(s_t, a_t) - s_{t+1}\|_2^2\right)$$

### 2.2 Policy Entropy Bonus

To prevent premature policy collapse to suboptimal deterministic cycles, the PPO objective incorporates the policy Shannon entropy:

$$\mathcal{H}(\pi_\theta(\cdot | s_t)) = -\sum_{a \in \mathcal{A}} \pi_\theta(a|s_t) \ln \pi_\theta(a|s_t)$$

---

## 3. Empirical Results

| Exploration Regime | Steps to First Goal | Success Rate (100 Ep) | Mean Episode Reward |
| :--- | :--- | :--- | :--- |
| **Extrinsic Only (PPO)** | $14,200 \pm 3,100$ | 42.5% | $3.12 \pm 1.84$ |
| **PPO + Entropy ($\beta=0.01$)** | $8,600 \pm 1,400$ | 78.0% | $7.45 \pm 0.92$ |
| **PPO + Curiosity ($\eta=0.20$)** | $5,100 \pm 850$ | 91.5% | $8.90 \pm 0.44$ |
| **Full NOIR Mind Engine** | **$3,400 \pm 420$** | **98.0%** | **$9.62 \pm 0.18$** |
