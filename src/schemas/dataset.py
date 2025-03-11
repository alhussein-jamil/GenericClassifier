from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ImageInfo:
    """Schema for image information."""

    path: Path  # Path to the image file
    class_name: str  # Class label name
    class_id: int  # Numeric class ID


@dataclass
class ClassInfo:
    """Schema for class information."""

    name: str  # Class name
    id: int  # Class ID
    sample_count: int = 0  # Number of samples in this class


@dataclass
class DatasetMetadata:
    """Schema for dataset metadata."""

    name: str  # Name of the dataset
    num_classes: int  # Number of classes
    total_samples: int  # Total number of samples
    class_distribution: Dict[str, int]  # Distribution of samples per class
    class_mapping: Dict[str, int]  # Mapping from class name to ID
    inv_class_mapping: Dict[int, str]  # Mapping from ID to class name


@dataclass
class Dataset:
    """Schema for the full dataset."""

    train_images: List[ImageInfo] = field(default_factory=list)  # Training images
    val_images: List[ImageInfo] = field(default_factory=list)  # Validation images
    classes: List[ClassInfo] = field(default_factory=list)  # Class information
    metadata: Optional[DatasetMetadata] = None  # Dataset metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset to dictionary."""
        return {
            "train_images": [
                {
                    "path": str(img.path),
                    "class_name": img.class_name,
                    "class_id": img.class_id,
                }
                for img in self.train_images
            ],
            "val_images": [
                {
                    "path": str(img.path),
                    "class_name": img.class_name,
                    "class_id": img.class_id,
                }
                for img in self.val_images
            ],
            "classes": [
                {"name": cls.name, "id": cls.id, "sample_count": cls.sample_count}
                for cls in self.classes
            ],
            "metadata": {
                "name": self.metadata.name,
                "num_classes": self.metadata.num_classes,
                "total_samples": self.metadata.total_samples,
                "class_distribution": self.metadata.class_distribution,
                "class_mapping": self.metadata.class_mapping,
                "inv_class_mapping": {
                    str(k): v for k, v in self.metadata.inv_class_mapping.items()
                },
            }
            if self.metadata
            else None,
        }

    @classmethod
    def from_dict(cls, dataset_dict: Dict[str, Any]) -> "Dataset":
        """Create dataset from dictionary."""
        train_images = [
            ImageInfo(
                path=Path(img["path"]),
                class_name=img["class_name"],
                class_id=img["class_id"],
            )
            for img in dataset_dict.get("train_images", [])
        ]

        val_images = [
            ImageInfo(
                path=Path(img["path"]),
                class_name=img["class_name"],
                class_id=img["class_id"],
            )
            for img in dataset_dict.get("val_images", [])
        ]

        classes = [
            ClassInfo(
                name=cls["name"],
                id=cls["id"],
                sample_count=cls["sample_count"],
            )
            for cls in dataset_dict.get("classes", [])
        ]

        metadata = None
        if dataset_dict.get("metadata"):
            metadata = DatasetMetadata(
                name=dataset_dict["metadata"]["name"],
                num_classes=dataset_dict["metadata"]["num_classes"],
                total_samples=dataset_dict["metadata"]["total_samples"],
                class_distribution=dataset_dict["metadata"]["class_distribution"],
                class_mapping=dataset_dict["metadata"]["class_mapping"],
                inv_class_mapping={
                    int(k): v
                    for k, v in dataset_dict["metadata"]["inv_class_mapping"].items()
                },
            )

        return cls(
            train_images=train_images,
            val_images=val_images,
            classes=classes,
            metadata=metadata,
        )
