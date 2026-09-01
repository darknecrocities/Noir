"""Real-World Dataset Search, Download, Ingestion, and Normalization Engine."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn import datasets as skl_datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
import torchvision.transforms as transforms

from noir.core.logging import get_logger

logger = get_logger("datasets.real")


@dataclass
class DatasetBundle:
    """Encapsulates loaded DataLoaders and architectural metadata for real datasets."""
    name: str
    train_loader: DataLoader
    val_loader: DataLoader
    input_dim: int
    num_classes: int
    num_train_samples: int
    num_val_samples: int
    feature_names: List[str] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    description: str = ""
    is_vision: bool = False
    img_shape: Optional[Tuple[int, ...]] = None


class RealDatasetManager:
    """Searches, downloads, caches, standardizes, and serves authentic real datasets."""

    CATALOG = {
        "digits": {
            "name": "Optical Recognition of Handwritten Digits (8x8)",
            "domain": "vision/tabular",
            "samples": 1797,
            "features": 64,
            "classes": 10,
            "description": "Normalized 8x8 bitmap pixel matrices extracted from real pre-printed forms.",
            "source": "UCI Machine Learning Repository / NIST",
        },
        "wine": {
            "name": "Wine Cultivar Chemical Analysis",
            "domain": "biochemistry/sensor",
            "samples": 178,
            "features": 13,
            "classes": 3,
            "description": "13 chemical constituents (alcohol, malic acid, flavonoids, phenols) across 3 Italian cultivars.",
            "source": "UCI / Forina et al.",
        },
        "breast_cancer": {
            "name": "Wisconsin Diagnostic Breast Cancer",
            "domain": "medical/biometrics",
            "samples": 569,
            "features": 30,
            "classes": 2,
            "description": "Nuclear morphometric features computed from digitized FNA biopsy images.",
            "source": "UCI / Wolberg, Street, Mangasarian",
        },
        "iris": {
            "name": "Fisher's Iris Flower Measurements",
            "domain": "botany",
            "samples": 150,
            "features": 4,
            "classes": 3,
            "description": "Morphological measurements (sepal/petal length & width) across 3 Iris species.",
            "source": "Fisher, 1936",
        },
        "fashion_mnist": {
            "name": "Fashion-MNIST Clothing Article Benchmark",
            "domain": "vision",
            "samples": 70000,
            "features": 784,
            "classes": 10,
            "description": "28x28 grayscale images of Zalando article products across 10 clothing categories.",
            "source": "Zalando Research",
        },
        "mnist": {
            "name": "MNIST Handwritten Digits Benchmark",
            "domain": "vision",
            "samples": 70000,
            "features": 784,
            "classes": 10,
            "description": "28x28 grayscale images of handwritten digits from Census Bureau and high school students.",
            "source": "LeCun, Cortes, Burges",
        },
        "cifar10": {
            "name": "CIFAR-10 Natural Color Object Images",
            "domain": "vision",
            "samples": 60000,
            "features": 3072,
            "classes": 10,
            "description": "32x32 color images across 10 natural object categories (airplane, car, bird, etc.).",
            "source": "Krizhevsky, Hinton",
        },
    }

    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def search_datasets(self, query: str = "", domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search available real datasets by keyword or domain."""
        query = query.lower().strip()
        results = []

        for key, meta in self.CATALOG.items():
            match_query = (
                not query
                or query in key.lower()
                or query in meta["name"].lower()
                or query in meta["description"].lower()
                or query in meta["domain"].lower()
            )
            match_domain = not domain or domain.lower() in meta["domain"].lower()

            if match_query and match_domain:
                results.append({"key": key, **meta})

        return results

    def load_dataset(
        self,
        dataset_name: str = "digits",
        batch_size: int = 64,
        val_split: float = 0.2,
        flatten_vision: bool = True,
        seed: int = 42,
    ) -> DatasetBundle:
        """Automatically fetch, preprocess, and construct PyTorch DataLoaders for a real dataset."""
        key = dataset_name.lower().strip()

        logger.info("Loading real dataset: '%s' (batch_size=%d)", key, batch_size)

        if key == "digits":
            return self._load_sklearn_digits(batch_size, val_split, seed)
        elif key == "wine":
            return self._load_sklearn_wine(batch_size, val_split, seed)
        elif key == "breast_cancer":
            return self._load_sklearn_breast_cancer(batch_size, val_split, seed)
        elif key == "iris":
            return self._load_sklearn_iris(batch_size, val_split, seed)
        elif key in ("fashion_mnist", "fashionmnist"):
            return self._load_torchvision_fashion_mnist(batch_size, flatten_vision)
        elif key == "mnist":
            return self._load_torchvision_mnist(batch_size, flatten_vision)
        elif key in ("cifar10", "cifar_10"):
            return self._load_torchvision_cifar10(batch_size, flatten_vision)
        else:
            # Fallback to search closest matching dataset or default to digits
            matches = self.search_datasets(query=key)
            if matches:
                fallback_key = matches[0]["key"]
                logger.info("Matched query '%s' to dataset '%s'", key, fallback_key)
                return self.load_dataset(fallback_key, batch_size, val_split, flatten_vision, seed)

            logger.warning("Dataset '%s' not found. Defaulting to 'digits'.", key)
            return self._load_sklearn_digits(batch_size, val_split, seed)

    # 1. Scikit-learn Real Datasets
    def _load_sklearn_digits(self, batch_size: int, val_split: float, seed: int) -> DatasetBundle:
        data = skl_datasets.load_digits()
        X, y = data.data, data.target

        # Scale features to zero mean, unit variance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=val_split, random_state=seed, stratify=y
        )

        train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
        val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())

        return DatasetBundle(
            name="digits",
            train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
            val_loader=DataLoader(val_ds, batch_size=batch_size, shuffle=False),
            input_dim=64,
            num_classes=10,
            num_train_samples=len(X_train),
            num_val_samples=len(X_val),
            class_names=[str(i) for i in range(10)],
            description="Scikit-learn Real 8x8 Optical Handwritten Digits",
            is_vision=True,
            img_shape=(1, 8, 8),
        )

    def _load_sklearn_wine(self, batch_size: int, val_split: float, seed: int) -> DatasetBundle:
        data = skl_datasets.load_wine()
        X, y = data.data, data.target

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=val_split, random_state=seed, stratify=y
        )

        train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
        val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())

        return DatasetBundle(
            name="wine",
            train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True if len(X_train) > batch_size else False),
            val_loader=DataLoader(val_ds, batch_size=batch_size, shuffle=False),
            input_dim=13,
            num_classes=3,
            num_train_samples=len(X_train),
            num_val_samples=len(X_val),
            feature_names=list(data.feature_names),
            class_names=list(data.target_names),
            description="Real Chemical Constituent Analysis of 3 Wine Cultivars",
        )

    def _load_sklearn_breast_cancer(self, batch_size: int, val_split: float, seed: int) -> DatasetBundle:
        data = skl_datasets.load_breast_cancer()
        X, y = data.data, data.target

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=val_split, random_state=seed, stratify=y
        )

        train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
        val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())

        return DatasetBundle(
            name="breast_cancer",
            train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
            val_loader=DataLoader(val_ds, batch_size=batch_size, shuffle=False),
            input_dim=30,
            num_classes=2,
            num_train_samples=len(X_train),
            num_val_samples=len(X_val),
            feature_names=list(data.feature_names),
            class_names=list(data.target_names),
            description="Wisconsin Diagnostic Biopsy Morphometry",
        )

    def _load_sklearn_iris(self, batch_size: int, val_split: float, seed: int) -> DatasetBundle:
        data = skl_datasets.load_iris()
        X, y = data.data, data.target

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=val_split, random_state=seed, stratify=y
        )

        train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
        val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())

        return DatasetBundle(
            name="iris",
            train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False),
            val_loader=DataLoader(val_ds, batch_size=batch_size, shuffle=False),
            input_dim=4,
            num_classes=3,
            num_train_samples=len(X_train),
            num_val_samples=len(X_val),
            feature_names=list(data.feature_names),
            class_names=list(data.target_names),
            description="Fisher's Classic Iris Botanical Dataset",
        )

    # 2. Torchvision Real Vision Benchmarks
    def _load_torchvision_fashion_mnist(self, batch_size: int, flatten: bool) -> DatasetBundle:
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
        if flatten:
            transform_list.append(transforms.Lambda(lambda x: torch.flatten(x)))

        transform = transforms.Compose(transform_list)

        train_set = torchvision.datasets.FashionMNIST(
            root=str(self.data_dir), train=True, download=True, transform=transform
        )
        val_set = torchvision.datasets.FashionMNIST(
            root=str(self.data_dir), train=False, download=True, transform=transform
        )

        return DatasetBundle(
            name="fashion_mnist",
            train_loader=DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True),
            val_loader=DataLoader(val_set, batch_size=batch_size, shuffle=False),
            input_dim=784 if flatten else 1,
            num_classes=10,
            num_train_samples=len(train_set),
            num_val_samples=len(val_set),
            class_names=["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"],
            description="Zalando Fashion-MNIST Real Clothing Images",
            is_vision=True,
            img_shape=(1, 28, 28),
        )

    def _load_torchvision_mnist(self, batch_size: int, flatten: bool) -> DatasetBundle:
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        if flatten:
            transform_list.append(transforms.Lambda(lambda x: torch.flatten(x)))

        transform = transforms.Compose(transform_list)

        train_set = torchvision.datasets.MNIST(
            root=str(self.data_dir), train=True, download=True, transform=transform
        )
        val_set = torchvision.datasets.MNIST(
            root=str(self.data_dir), train=False, download=True, transform=transform
        )

        return DatasetBundle(
            name="mnist",
            train_loader=DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True),
            val_loader=DataLoader(val_set, batch_size=batch_size, shuffle=False),
            input_dim=784 if flatten else 1,
            num_classes=10,
            num_train_samples=len(train_set),
            num_val_samples=len(val_set),
            class_names=[str(i) for i in range(10)],
            description="NIST Real Handwritten Digits Benchmark",
            is_vision=True,
            img_shape=(1, 28, 28),
        )

    def _load_torchvision_cifar10(self, batch_size: int, flatten: bool) -> DatasetBundle:
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
        if flatten:
            transform_list.append(transforms.Lambda(lambda x: torch.flatten(x)))

        transform = transforms.Compose(transform_list)

        train_set = torchvision.datasets.CIFAR10(
            root=str(self.data_dir), train=True, download=True, transform=transform
        )
        val_set = torchvision.datasets.CIFAR10(
            root=str(self.data_dir), train=False, download=True, transform=transform
        )

        return DatasetBundle(
            name="cifar10",
            train_loader=DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True),
            val_loader=DataLoader(val_set, batch_size=batch_size, shuffle=False),
            input_dim=3072 if flatten else 3,
            num_classes=10,
            num_train_samples=len(train_set),
            num_val_samples=len(val_set),
            class_names=["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
            description="CIFAR-10 Natural Color Object Images",
            is_vision=True,
            img_shape=(3, 32, 32),
        )
