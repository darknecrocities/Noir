# Reinforcement Learning & Intrinsic Motivation

## 1. Proximal Policy Optimization (PPO)

Project NOIR implements on-policy actor-critic reinforcement learning via Proximal Policy Optimization with Generalized Advantage Estimation (GAE).

### Clipped Surrogate Objective

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$

Where the probability ratio is:

$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$

---

## 2. Generalized Advantage Estimation (GAE)

Advantage estimates $\hat{A}_t^{\text{GAE}(\gamma, \lambda)}$ are computed as exponentially weighted sums of temporal difference residuals $\delta_t^V$:

$$\delta_t^V = R_t + \gamma V(s_{t+1}) - V(s_t)$$

$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$$

---

## 3. Total Loss Objective

The network optimizes the composite objective:

$$L_{\text{total}}(\theta) = -L^{\text{CLIP}}(\theta) + c_1 L^{\text{VF}}(\theta) - c_2 S[\pi_\theta](s_t)$$

Where:
- $L^{\text{VF}}(\theta) = \frac{1}{2} \hat{\mathbb{E}}_t \left[ (V_\theta(s_t) - V_t^{\text{target}})^2 \right]$ (Value squared error)
- $S[\pi_\theta](s_t) = -\sum_a \pi(a|s) \ln \pi(a|s)$ (Policy entropy bonus)
- $c_1 = 0.5$, $c_2 = 0.01$, $\epsilon = 0.2$

---

## 4. Intrinsic Curiosity Module (Forward Dynamics)

NOIR integrates intrinsic motivation to explore unvisited environment regions:

$$R_{\text{total}} = R_{\text{extrinsic}} + \eta R_{\text{intrinsic}}$$

The forward dynamics model $f_\phi(s_t, a_t)$ predicts the future representation $\hat{s}_{t+1}$. The intrinsic reward is proportional to the prediction error:

$$R_{\text{intrinsic}} = \|f_\phi(s_t, a_t) - s_{t+1}\|^2$$

As the agent explores novel states, prediction error is high, incentivizing exploratory action sequences. Once the agent masters a transition, prediction error decays to zero.
