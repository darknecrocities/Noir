# Storage, Persistence, and Crash Recovery

## 1. Two-Phase Atomic Checkpoint Protocol

To ensure absolute safety against power failures, application termination, or unexpected exceptions, Project NOIR strictly follows an atomic two-phase write pattern:

```
Step 1: Write to staging directory
        checkpoints/<exp_id>/.tmp_step_00010000_epoch_0025/
        ├── model.safetensors
        ├── state.pt
        └── meta.json

Step 2: Validate metadata integrity and file sizes
        (Read back meta.json and verify tensor headers)

Step 3: Atomic filesystem rename
        .tmp_... ──[ rename ]──► checkpoints/<exp_id>/history/step_00010000_epoch_0025/

Step 4: Update 'latest' and 'best' pointers atomically
        Copy final contents to latest/ and best/ pointers.
```

---

## 2. Experiment Branching

Branching allows researchers to fork training from any historical checkpoint while altering hyperparameters (e.g. learning rate, curiosity weight, optimizer):

```
                EXPERIMENT MAIN (Parent)
                          │
                     STEP 10,000
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
        Experiment A              Experiment B
        LR = 0.001                LR = 0.0001
              │                       │
              ▼                       ▼
          Score 0.82              Score 0.94
```

- **Parent Guarantee**: Parent experiment weights and databases remain completely immutable and unaltered.
- **Lineage Tracking**: Each branched experiment preserves its `parent_id` in SQLite.

---

## 3. Crash Recovery

Upon launch, `RecoveryManager` scans `noir.db` and the checkpoint directory hierarchy for the latest valid checkpoint. If a previous run was interrupted:
- The **Recovery Modal Dialog** pops up automatically.
- Offers three actions:
  1. `[Resume Training]`: Restores model weights, optimizer momentum, scheduler states, RNG seeds, and affective vectors, continuing immediately.
  2. `[Load Without Resuming]`: Restores the model to memory in a paused state for manual inspection.
  3. `[Start New Experiment]`: Initializes a fresh experiment from scratch.
