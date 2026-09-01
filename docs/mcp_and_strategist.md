# Model Context Protocol (MCP) & AI Strategist

## 1. Overview

Project NOIR exposes an embedded Model Context Protocol (MCP) server listening at `http://127.0.0.1:8765`. This enables external AI agents (like Claude, Gemini, Antigravity, or local LLMs) to inspect live research telemetry, extract layer activations, and steer training experiments autonomously.

---

## 2. MCP JSON-RPC Protocol

Requests follow the standard MCP JSON-RPC 2.0 schema:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "get_training_status",
    "arguments": {}
  }
}
```

---

## 3. Tool Reference

| Tool Name | Parameters | Description |
| :--- | :--- | :--- |
| `get_training_status` | None | Returns lifecycle state, experiment ID, global step, and epoch. |
| `get_latest_metrics` | None | Returns latest loss, accuracy, reward, and gradient norm values. |
| `get_emotion_state` | None | Returns 8-dimensional affective vector ($C, F, A, S, U, X, Ca, P$). |
| `inspect_model` | None | Returns layer specification, neuron counts, and weight norms. |
| `inspect_layer` | `{"layer_name": str}` | Returns activation shapes, statistics, and weights for a specific layer. |
| `create_checkpoint` | `{"tag": str}` | Atomically saves a checkpoint immediately. |
| `load_checkpoint` | `{"checkpoint_path": str}` | Restores weights and states from disk. |
| `branch_experiment` | `{"new_name": str, "config_overrides": dict}` | Forks current experiment without modifying the parent. |
| `pause_training` | None | Pauses active training loop. |
| `resume_training` | None | Resumes paused training loop. |
| `stop_training` | None | Terminates active training loop. |

---

## 4. AI Strategist Loop

The Strategist (`noir.strategy.strategist.Strategist`) operates as an asynchronous slow loop:
1. Gathers rolling statistics (loss slope, gradient variance, affective frustration).
2. Formulates scientific hypotheses:
   ```json
   {
     "hypothesis": "Policy is trapped in a local reward plateau. Exploration pressure is insufficient.",
     "confidence": 0.82,
     "proposed_actions": [
       {"parameter": "reinforcement_learning.entropy_coef", "action": "increase", "factor": 2.0},
       {"parameter": "emotion.curiosity_weight", "action": "increase", "factor": 1.5}
     ],
     "explanation": "Boosting curiosity and entropy bonus forces exploration of alternate navigation paths."
   }
   ```
3. Persists hypotheses into the database and notifies the user via the desktop dashboard.
