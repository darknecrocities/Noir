# System Architecture & Technical Specification

## 1. High-Level Design

Project NOIR is a pure Python, local-first experimental AI and machine learning research platform. It integrates real numerical neural network training (via PyTorch) with a mathematical affective/cognitive engine, real-time 3D tensor visualization (via PySide6), atomic multi-tier persistence (via SQLite and Safetensors), and Model Context Protocol (MCP) tooling.

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

## 2. Fast Loop vs. Slow Loop Dual Architecture

To maintain smooth 60 FPS user interface responsiveness and prevent blocking numerical optimization, NOIR decouples execution into two operational loops:

```
FAST LOOP (100 Hz - 1000 Hz)             SLOW LOOP (0.01 Hz - 0.1 Hz)
─────────────────────────────            ─────────────────────────────
• Environment Step Interaction           • Moving Average Metrics Analysis
• Forward Tensor Propagation             • Strategic Hypothesis Generation
• Loss Computation & Backprop            • Hyperparameter Branch Proposals
• Intrinsic Curiosity Model Step         • Episodic Memory Consolidation
• Immediate Event Bus Emission           • MCP Tool Interactions
```

---

## 3. Asynchronous Event Bus

The `EventBus` (`noir.events.event_bus`) coordinates all inter-module communication using typed immutable `NoirEvent` payloads:

```python
@dataclass
class NoirEvent:
    event_id: str
    timestamp: float
    event_type: EventType
    experiment_id: str
    training_step: int
    epoch: int
    payload: Dict[str, Any]
```

Subscribers can listen to specific `EventType` instances or receive global event streams. Events are dispatched thread-safely via background worker queues.

---

## 4. Multi-Tier Persistence Model

Persistence in Project NOIR operates across three distinct storage modalities:

1. **Relational Database (`noir.db` - SQLite + SQLAlchemy)**:
   - Experiments catalog and branching hierarchy
   - Metric time-series (`step`, `loss`, `accuracy`, `reward`, `grad_norm`)
   - Affective vector historical logs
   - AI Strategist hypotheses and action plans
2. **Safetensors & PyTorch State Archives (`checkpoints/`)**:
   - Model weights in zero-overhead `.safetensors` format
   - Optimizer, scheduler, and RNG states in `state.pt`
   - Human-readable `meta.json` with step and environment metadata
3. **Cognitive Memory (`memory/`)**:
   - Salient episodic discoveries and semantic concepts
