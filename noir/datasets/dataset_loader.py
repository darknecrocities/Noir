"""Dataset loading helpers and DataLoader factories for real-world datasets."""

from typing import Optional, Tuple
from torch.utils.data import DataLoader

from noir.datasets.real_datasets import DatasetBundle, RealDatasetManager

_manager = RealDatasetManager()


def create_classification_dataloaders(
    dataset_name: str = "digits",
    batch_size: int = 64,
    val_split: float = 0.2,
    seed: int = 42,
) -> DatasetBundle:
    """Load authentic real-world dataset and return complete DatasetBundle with metadata."""
    return _manager.load_dataset(
        dataset_name=dataset_name,
        batch_size=batch_size,
        val_split=val_split,
        seed=seed,
    )


def search_available_datasets(query: str = "", domain: Optional[str] = None):
    """Search available real datasets."""
    return _manager.search_datasets(query=query, domain=domain)
