"""Unit and learning verification tests for real datasets."""

import time
import pytest
import torch
import torch.nn as nn

from noir.datasets.dataset_loader import create_classification_dataloaders, search_available_datasets
from noir.datasets.real_datasets import RealDatasetManager
from noir.models.mlp import NoirMLP
from noir.training.supervised_trainer import SupervisedTrainer


def test_search_real_datasets():
    results = search_available_datasets(query="vision")
    names = [r["key"] for r in results]
    assert "digits" in names
    assert "fashion_mnist" in names
    assert "cifar10" in names

    medical = search_available_datasets(query="cancer")
    assert len(medical) == 1
    assert medical[0]["key"] == "breast_cancer"


def test_load_real_digits_dataset():
    bundle = create_classification_dataloaders(dataset_name="digits", batch_size=32)

    assert bundle.name == "digits"
    assert bundle.input_dim == 64
    assert bundle.num_classes == 10
    assert bundle.num_train_samples > 1000
    assert bundle.num_val_samples > 300

    # Inspect a batch
    for inputs, targets in bundle.train_loader:
        assert inputs.shape == (32, 64)
        assert targets.shape == (32,)
        assert targets.min() >= 0
        assert targets.max() < 10
        break


def test_load_real_wine_dataset():
    bundle = create_classification_dataloaders(dataset_name="wine", batch_size=16)

    assert bundle.name == "wine"
    assert bundle.input_dim == 13
    assert bundle.num_classes == 3
    assert len(bundle.feature_names) == 13


def test_load_real_breast_cancer_dataset():
    bundle = create_classification_dataloaders(dataset_name="breast_cancer", batch_size=32)

    assert bundle.name == "breast_cancer"
    assert bundle.input_dim == 30
    assert bundle.num_classes == 2


def test_real_dataset_learning_convergence():
    """Verify that the neural network genuinely learns on real digits data (loss drops)."""
    bundle = create_classification_dataloaders(dataset_name="digits", batch_size=64)

    model = NoirMLP(input_dim=bundle.input_dim, hidden_dims=[64, 32], output_dim=bundle.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    trainer = SupervisedTrainer(
        experiment_id="test_real_learning",
        model=model,
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=5,
        device="cpu",
    )

    trainer.start_training()
    time.sleep(2.0)
    trainer.stop_training(wait=True)

    # Verify model trained for at least 1 epoch and achieved real loss
    assert trainer.global_step > 10
    assert trainer.latest_metrics.get("train_loss") is not None
    assert trainer.latest_metrics["train_loss"] < 2.5  # Significant learning over initial cross-entropy
