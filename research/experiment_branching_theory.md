# Non-Destructive Experiment Branching & Hypothesis Forking

**Authors**: Project NOIR Research Group  
**Subject**: Machine Learning Operations, Experiment Lineage, Automated Research

---

## 1. The Linearity Problem in ML Exploration

Traditional machine learning workflows explore hyperparameters linearly or via black-box grid/Bayesian searches. When a model encounters numerical instability or an optimization plateau at step $T$, re-training from step $0$ with modified hyperparameters incurs severe computational waste.

---

## 2. Hypothesis-Guided State Forking

Project NOIR introduces state-level experiment branching:

$$\mathcal{E}_{\text{branch}} = \text{Fork}\left( \mathcal{E}_{\text{parent}}, \; \tau_{\text{ckpt}}, \; \Delta \mathbf{\Theta}_{\text{hyper}} \right)$$

1. **State Preservation**: At step $\tau$, the model weights $\theta_\tau$, optimizer momentum $m_\tau, v_\tau$, RNG seeds $\xi_\tau$, and affective vector $E_\tau$ are snapshot atomically into a new experiment sandbox.
2. **Immutability of Parent**: The parent experiment's SQLite metrics, checkpoints, and event logs remain untouched.
3. **Differential Comparison**: Researchers can directly compare metric trajectories starting from the exact bifurcation step.
