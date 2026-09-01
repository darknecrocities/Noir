"""Tests for Open Web Streamer, Causal Transformer, and Live LLM Learning."""

import time
import pytest
import torch

from noir.datasets.open_web import OpenWebStreamer, WebTextTokenizer
from noir.models.transformer import NoirTransformerLM
from noir.training.llm_trainer import OpenWebLLMTrainer


def test_web_text_tokenizer():
    tokenizer = WebTextTokenizer()
    text = "Artificial intelligence on the open internet!"
    tokens = tokenizer.encode(text)
    assert len(tokens) > 0
    decoded = tokenizer.decode(tokens)
    assert decoded == text


def test_open_web_streamer():
    streamer = OpenWebStreamer()
    X, Y, title, url = streamer.create_batch(batch_size=4, block_size=16)

    assert X.shape == (4, 16)
    assert Y.shape == (4, 16)
    assert isinstance(title, str)
    assert isinstance(url, str)


def test_transformer_forward_and_generate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NoirTransformerLM(
        vocab_size=256,
        block_size=32,
        n_layers=2,
        n_heads=2,
        embed_dim=64,
    ).to(device)

    # Test forward pass with targets
    idx = torch.randint(0, 256, (2, 16), dtype=torch.long, device=device)
    targets = torch.randint(0, 256, (2, 16), dtype=torch.long, device=device)

    logits, loss = model(idx, targets=targets)
    assert logits.shape == (2, 16, 256)
    assert loss is not None
    assert loss.item() > 0.0

    # Test autoregressive generation
    prompt = torch.tensor([[65, 66, 67]], dtype=torch.long, device=device)
    out = model.generate(prompt, max_new_tokens=5, temperature=1.0)
    assert out.shape == (1, 8)


def test_open_web_llm_trainer_step():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NoirTransformerLM(
        vocab_size=256,
        block_size=16,
        n_layers=2,
        n_heads=2,
        embed_dim=32,
    )

    trainer = OpenWebLLMTrainer(
        experiment_id="test_open_web_llm",
        model=model,
        learning_rate=0.001,
        batch_size=4,
        block_size=16,
        max_steps=5,
        device=device,
    )

    trainer.start_training()
    time.sleep(1.5)
    trainer.stop_training(wait=True)

    assert trainer.global_step > 0
    assert trainer.latest_metrics.get("train_loss") is not None
    assert trainer.latest_metrics.get("perplexity") is not None
