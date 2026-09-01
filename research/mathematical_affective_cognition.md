# Mathematical Affective Cognition in Artificial Neural Systems

**Authors**: Project NOIR Research Group  
**Subject**: Artificial General Intelligence, Affective Computing, Neuromorphic Control

---

## Abstract

We present a continuous mathematical framework for synthesizing internal affective states within deep neural optimization loops. By mapping observable training dynamics—including Shannon predictive entropy, empirical gradient variance, self-information surprise, and goal proximity—into an 8-dimensional normalized affective vector:

$$E_t = [C_t, F_t, A_t, S_t, U_t, X_t, Ca_t, P_t]$$

we demonstrate that artificial agents can maintain structured, dynamic internal states that correlate with learning stability, exploration efficiency, and catastrophe prevention.

---

## 1. Introduction

Traditional deep learning algorithms treat optimization as an uncoupled numerical process. However, biological learning is intrinsically coupled with affective and homeostatic regulation. In this work, we formalize how mathematical emotional vectors can be derived deterministically from the forward-backward execution stream of PyTorch models without introducing stochastic heuristic noise.

---

## 2. Mathematical Formalism

Let $\mathcal{M}_\theta$ be a neural network parameterized by $\theta \in \mathbb{R}^D$. At training step $t$, the system receives observation $s_t$ and loss $\mathcal{L}_t$.

### 2.1 State Vector Definition

$$E_t \in [0.0, 1.0]^8$$

### 2.2 Discrete Update Equation

$$E_{t+1} = \text{clip}\left( \mathbf{A} E_t + \mathbf{B} R_t + \mathbf{\Gamma} N_t + \mathbf{\Delta} U_t + \mathbf{\Phi} G_t, \; 0.0, \; 1.0 \right)$$

Where:
- $\mathbf{A} = \text{diag}(\alpha_C, \alpha_F, \alpha_A, \alpha_S, \alpha_U, \alpha_X, \alpha_{Ca}, \alpha_P)$ represents exponential decay rates.
- $R_t$ is task reward.
- $N_t = 1 - \rho(s_t)$ is state novelty based on empirical spatial density $\rho$.
- $U_t = -\frac{1}{\ln K}\sum p_i \ln p_i$ is normalized Shannon entropy.
- $G_t = \max(0, 1 - d(s_t, s_{\text{goal}})/d_0)$ is normalized goal progression.

---

## 3. Experimental Findings

1. **Entropy-Driven Uncertainty ($U_t$)**: Correlates with decision boundary ambiguity and provides early warnings of out-of-distribution drift.
2. **Surprise Peaks ($S_t$)**: Trigger episodic memory consolidation, ensuring high-salience moments are persisted for post-hoc reflection.
3. **Frustration Accumulation ($F_t$)**: Automatically flags optimization plateaus, signaling the AI Strategist to propose exploration bonuses.
