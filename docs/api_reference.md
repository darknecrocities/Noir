# Python API Reference

## Module: `noir.core.engine`

### `class NoirEngine(config_path: Optional[str | Path] = None)`
Master orchestrator managing all research components.

- `start_supervised_experiment(name, dataset_name, hidden_dims, learning_rate, num_epochs, batch_size) -> str`: Starts a real supervised benchmark run.
- `start_rl_experiment(name, env_id, learning_rate, n_steps, max_episodes) -> str`: Starts a PPO RL run.
- `pause_training()`: Pauses numerical optimization thread.
- `resume_training()`: Resumes paused optimization thread.
- `stop_training()`: Gracefully halts training thread.
- `save_checkpoint(tag: Optional[str] = None) -> Path`: Immediately saves an atomic checkpoint.
- `load_checkpoint(checkpoint_path: str | Path) -> dict`: Restores weights and states.
- `branch_experiment(new_name: str, config_overrides: Optional[dict] = None) -> str`: Creates a new experiment branch.
- `shutdown()`: Terminates all worker threads, servers, and persists state.

---

## Module: `noir.datasets.real_datasets`

### `class RealDatasetManager(data_dir: str | Path = 'data')`
- `search_datasets(query: str, domain: Optional[str]) -> List[dict]`: Searches available catalog.
- `load_dataset(dataset_name: str, batch_size: int, val_split: float) -> DatasetBundle`: Loads, caches, standardizes, and yields train/val DataLoaders with metadata.

---

## Module: `noir.mind.affective_engine`

### `class AffectiveEngine(experiment_id, surprise_threshold, curiosity_weight, frustration_decay, confidence_decay)`
- `update_from_supervised_step(loss, accuracy, probabilities, step) -> EmotionState`: Updates affective state from supervised batch metrics.
- `update_from_rl_step(state, action, reward, next_state, done, info, step) -> Tuple[EmotionState, float]`: Updates state from RL interaction and computes intrinsic curiosity reward.

---

## Module: `noir.visualization.visualizer_3d`

### `class NeuralVisualizer3D(parent: Optional[QWidget] = None)`
Interactive 3D vector-projected neural network viewport.
- `set_graph(graph: NeuralGraph)`: Pushes new topology and activation states to canvas.
- `trigger_surprise_shock(intensity: float)`: Visual perturbation wave on surprise.
- `trigger_reward_pulse(amount: float)`: Glowing energy burst on reward.
