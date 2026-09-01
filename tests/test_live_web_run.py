import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import torch
from noir.core.engine import NoirEngine

def run_live_verification():
    print("=== STARTING LIVE OPEN WEB GPU VERIFICATION ===")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Active Device: {device_name}")

    engine = NoirEngine()
    exp_id = engine.start_open_web_llm_experiment(
        name="Live Internet Test",
        max_steps=20,
    )

    print(f"Experiment started with ID: {exp_id}")
    time.sleep(6)

    print("\n=== AUTHENTIC INTERNET RESOURCES INGESTED ===")
    history = engine.trainer.streamer.get_resource_history()
    for idx, r in enumerate(history, 1):
        safe_title = r['title'].encode('ascii', 'ignore').decode('ascii')
        safe_snippet = r['snippet'].encode('ascii', 'ignore').decode('ascii')
        print(f"{idx}. [{r['source_type']}] {safe_title}")
        print(f"   URL: {r['url']}")
        print(f"   Size: {r['character_count']} chars, {r['token_count']} tokens")
        print(f"   Snippet: {safe_snippet}")

    print("\n=== LIVE TRAINING METRICS ON GPU ===")
    metrics = engine.trainer.latest_metrics
    print(f"Training Step: {engine.trainer.global_step}")
    print(f"Train Cross-Entropy Loss: {metrics.get('train_loss', 0.0):.4f}")
    print(f"Train Perplexity: {metrics.get('perplexity', 0.0):.2f}")
    print(f"Validation Perplexity: {metrics.get('val_perplexity', 0.0):.2f}")
    safe_gen = engine.trainer.latest_generated_text.encode('ascii', 'ignore').decode('ascii')
    print(f"Live Sample Completion: '{safe_gen}'")

    engine.stop_training()
    engine.shutdown()
    print("\n=== VERIFICATION COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    run_live_verification()
