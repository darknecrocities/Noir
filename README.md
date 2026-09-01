# PROJECT NOIR

<div align="center">

```
██████╗ ██████╗  ██████╗      ██╗███████╗ ██████╗████████╗    ███╗   ██╗ ██████╗ ██╗██████╗ 
██╔══██╗██╔══██╗██╔═══██╗     ██║██╔════╝██╔════╝╚══██╔══╝    ████╗  ██║██╔═══██╗██║██╔══██╗
██████╔╝██████╔╝██║   ██║     ██║█████╗  ██║        ██║       ██╔██╗ ██║██║   ██║██║██████╔╝
██╔═══╝ ██╔══██╗██║   ██║██   ██║██╔══╝  ██║        ██║       ██║╚██╗██║██║   ██║██║██╔══██╗
██║     ██║  ██║╚██████╔╝╚█████╔╝███████╗╚██████╗   ██║       ██║ ╚████║╚██████╔╝██║██║  ██║
╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚════╝ ╚══════╝ ╚═════╝   ╚═╝       ╚═╝  ╚═══╝ ╚═════╝ ╚═╝╚═╝  ╚═╝
```

### *Experimental Real-Time AI & Machine Learning Research Environment*
**Local-First • Pure Python • Real PyTorch Training • Mathematical Affective Engine • Live 3D Neural Visualization**

</div>

---

## Overview

**Project NOIR** is a local-first, native Python experimental research platform designed to study artificial neural learning, reinforcement learning dynamics, cognitive memory, and mathematical affective systems in real time.

> [!IMPORTANT]
> **This is NOT a simulation and contains NO simulated metrics.**
> Every metric, activation map, gradient norm, and loss curve is computed live via genuine PyTorch forward propagation, backpropagation, and tensor optimization.

---

## System Architecture

```
                                  PROJECT NOIR
                               (Pure Python Core)
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
      PySide6 UI               Training Engine            Storage Engine
   (Dashboard, 3D Net,      (Supervised & PPO RL,       (SQLite Database &
    Emotion & Controls)       Affective Mind Engine)     Safetensors Checkpoints)
            │                          │                          │
            └──────────────────────────┼──────────────────────────┘
                                       │
                                 Event Bus Stream
                                       │
                                       ▼
                       LLM Strategist & MCP Server Tools
```

---

## Key Capabilities

1. **Genuine PyTorch Machine Learning on Real Datasets**:
   - Real tensor forward passes, loss calculation, backpropagation, and weight updates.
   - Dual research modes: **Real-World Benchmark Classification** and **PPO Reinforcement Learning**.
   - **Automated Real Dataset Search & Ingestion**:
     - `digits`: Scikit-learn 8x8 optical handwritten digits (1,797 samples, 64 features, 10 classes)
     - `fashion_mnist`: Zalando's real clothing benchmark (70,000 samples, 784 features, 10 classes)
     - `wine`: Real chemical constituent measurements (178 samples, 13 features, 3 classes)
     - `breast_cancer`: Diagnostic FNA biopsy morphometry (569 samples, 30 features, 2 classes)
     - `mnist`: Classic handwritten digits benchmark (70,000 samples, 784 features, 10 classes)
     - `cifar10`: Natural color object images (60,000 samples, 3072 features, 10 classes)
     - `iris`: Fisher's classic botanical morphological measurements (150 samples, 4 features, 3 classes)
   - **Automatic Architecture Adaptation**: The model automatically reconfigures its input dimension and output classification heads to match the exact dimensional reality of the selected or discovered dataset.
   - Live layer introspection using PyTorch execution hooks.

2. **Mathematical Affective / Emotion Engine**:
   - Continuous vector representation:
     $$E_t = [C_t, F_t, A_t, S_t, U_t, X_t, Ca_t, P_t]$$
     *(Confidence, Frustration, Anticipation, Satisfaction, Uncertainty, Curiosity, Caution, Persistence)*.
   - Dynamically balanced by Shannon entropy, self-information surprise, goal progress, and forward-dynamics prediction error.

3. **Real-Time 3D Neural Network Viewport**:
   - Interactive 3D vector-projected canvas with orbital camera controls (Rotate, Pan, Zoom).
   - Node size & glow mapped to activation magnitude.
   - Synaptic connection thickness and color (cyan vs magenta) mapped to weight magnitude and sign.
   - Traveling energy pulses and disturbance shockwaves triggered by gradients and surprises.

4. **Atomic Persistence & Experiment Branching**:
   - Two-phase commit checkpointing (`checkpoint.tmp` -> validate -> rename).
   - Automatic retention management and instant crash recovery on startup.
   - Non-destructive experiment branching from any historical checkpoint.

5. **LLM Strategist & Model Context Protocol (MCP)**:
   - Asynchronous slow loop strategist generating scientific hypotheses and hyperparameter proposals.
   - Native MCP Server exposing 12+ tools and real-time research resources.

---

## Mathematical Formulations

### 1. Affective State Vector (E_t)
$$E_{t+1} = \text{clip}\left(\alpha E_t + \beta R_t + \gamma N_t + \delta U_t + \epsilon G_t, 0, 1\right)$$
- $R_t$: Environment / task reward
- $N_t$: State novelty (spatial hash density)
- $U_t$: Predictive uncertainty
- $G_t$: Goal progression

### 2. Predictive Uncertainty (Entropy)
$$H(p) = -\sum_{i=1}^K p_i \log(p_i)$$

### 3. Perceptual Surprise
$$S_t = \|\hat{s}_{t+1} - s_{t+1}\|^2 \quad \text{or} \quad I(\text{event}) = -\log P(\text{event})$$

### 4. Curiosity-Driven Intrinsic Motivation
$$R_{\text{total}} = R_{\text{extrinsic}} + \eta R_{\text{intrinsic}} \quad \text{where} \quad R_{\text{intrinsic}} = \|f(s_t, a_t) - s_{t+1}\|^2$$

### 5. Proximal Policy Optimization (PPO)
$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$

---

## Installation & Quick Start

### Prerequisites
- **Python 3.10+** (Tested on Python 3.11)
- Windows PowerShell or Linux/macOS Bash

### Automated Setup

#### Windows:
```powershell
# Clone the repository
git clone https://github.com/darknecrocities/Noir.git
cd Noir

# Run automated setup script
.\setup.ps1

# Launch Project NOIR
.\run.ps1
```

#### Linux / macOS:
```bash
# Clone the repository
git clone https://github.com/darknecrocities/Noir.git
cd Noir

# Make scripts executable and run setup
chmod +x setup.sh run.sh
./setup.sh

# Launch Project NOIR
./run.sh
```

---

### Manual Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

# 2. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Launch application
python -m noir.main
```

---

## User Interface Overview

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ PROJECT NOIR                                                        LIVE    │
├─────────────────────────────────────────────────────────────────────────────┤
│ [DATASET: Real Digits ▼] [LR: 0.00100] [START] [PAUSE] [SAVE CHECKPOINT]    │
├──────────────────────────────┬───────────────────────────────┬──────────────┤
│                              │                               │ AFFECTIVE    │
│  3D NEURAL NETWORK VIEWPORT  │     REAL-TIME TELEMETRY       │ Curiosity    │
│                              │     [Loss & Accuracy Plots]   │ Confidence   │
│       ●──●──●──●             │                               │ Uncertainty  │
│      ╱╲╱╲╱╲╱╲                ├───────────────────────────────┤ Frustration  │
│     ●──●──●──●               │     EVENT STREAM TIMELINE     │ Anticipation │
│                              │     14:20:01 WEIGHTS_UPDATED  │ Satisfaction │
│  (Rotate / Pan / Zoom / Probe)│     14:20:05 SURPRISE_DETECTED│ Caution      │
│                              │     14:20:10 CHECKPOINT_SAVED │ Persistence  │
├──────────────────────────────┴───────────────────────────────┴──────────────┤
│ STATUS: RUNNING │ STEP: 12,450 │ EPOCH: 32 │ LOSS: 0.241 │ CPU: 18% │ GPU: CUDA│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Model Context Protocol (MCP) Integration

Project NOIR exposes an embedded MCP Server at `http://127.0.0.1:8765`.

### Supported MCP Tools
- `get_training_status`: Lifecycle state, step, and epoch.
- `get_latest_metrics`: Real-time loss, accuracy, and reward values.
- `get_emotion_state`: 8-dimensional affective vector.
- `inspect_model`: Layer blueprint, neuron count, weight statistics.
- `inspect_layer`: Activations and gradient stats for a specific layer.
- `create_checkpoint`: Immediate atomic checkpoint save.
- `branch_experiment`: Fork experiment with modified hyperparameters.
- `pause_training` / `resume_training` / `stop_training`

### Connecting with LLMs
Configure via `.env`:
```ini
NOIR_LLM_PROVIDER=local
NOIR_LLM_MODEL=llama3
NOIR_LLM_API_BASE=http://localhost:11434/v1
NOIR_MCP_ENABLED=true
NOIR_MCP_PORT=8765
```

---

## Configuration (`config/default.yaml`)

```yaml
project:
  name: Project NOIR
  version: "0.1.0"

training:
  device: auto # auto, cuda, cpu
  learning_rate: 0.0003
  batch_size: 64
  checkpoint_interval_steps: 100
  autosave_interval_seconds: 60

reinforcement_learning:
  algorithm: PPO
  gamma: 0.99
  gae_lambda: 0.95
  clip_eps: 0.2
  env_id: GridWorld-v0

emotion:
  enabled: true
  curiosity_weight: 0.2
  surprise_threshold: 0.70
  frustration_decay: 0.95
  confidence_decay: 0.99
```

---

## Testing

Run the full automated test suite:

```bash
python -m pytest tests/ -v
```

Tests verify:
- Atomic checkpointing and integrity verification.
- EventBus thread-safe dispatching and event ordering.
- Mathematical affective vector bounds ($0 \le E_i \le 1$).
- Shannon entropy and prediction variance calculations.
- PPO policy optimization steps and reward accumulation.
- Non-destructive experiment branching.
- Real dataset loading, normalization, and model dimension adaptation.

---

## Troubleshooting

- **PySide6 / Display on Headless Servers**:
  Run in headless mode using:
  ```bash
  python -m noir.main --headless --mode supervised
  ```
- **CUDA Out of Memory**:
  NOIR automatically detects CPU execution if CUDA is unavailable. To force CPU mode, set `NOIR_DEVICE=cpu` in `.env`.
- **Database Reset**:
  Delete `noir.db` or specify a new database path in `.env`.

---

## License
MIT License. Built for advanced artificial intelligence and cognitive systems research.
