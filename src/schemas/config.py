from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DataConfig:
    """Schema for data loading and preprocessing configuration."""

    zip_path: Path  # Path to the zip file containing image dataset
    extract_dir: Path  # Directory to extract the zip contents
    img_size: tuple[int, int] = (224, 224)  # Target image size
    batch_size: int = 32  # Batch size for training
    val_split: float = 0.2  # Validation split ratio
    augmentation: bool = True  # Whether to use data augmentation
    shuffle: bool = True  # Whether to shuffle the training data


@dataclass
class ModelConfig:
    """Schema for model configuration."""

    model_name: str = "resnet50"  # Model architecture name
    num_classes: int = 0  # Number of classes (will be set based on data)
    pretrained: bool = True  # Whether to use pretrained weights
    learning_rate: float = 0.001  # Learning rate for training
    epochs: int = 10  # Number of epochs to train
    optimizer: str = "adam"  # Optimizer name
    loss: str = "categorical_crossentropy"  # Loss function
    metrics: List[str] = None  # Evaluation metrics

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["accuracy"]


@dataclass
class AppConfig:
    """Schema for application configuration."""

    title: str = "Image Classification App"  # App title
    description: str = (
        "Upload an image to classify it using the trained model"  # App description
    )
    examples_dir: Optional[Path] = None  # Directory containing example images
    max_file_size: int = 5  # Max file size in MB
    port: int = 7860  # Port for Gradio app


@dataclass
class Config:
    """Main configuration schema combining all sub-configurations."""

    data: DataConfig
    model: ModelConfig
    app: AppConfig
    checkpoint_dir: Path = Path("checkpoints")  # Directory to save model checkpoints
    output_dir: Path = Path("output")  # Directory to save outputs

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "data": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in vars(self.data).items()
            },
            "model": vars(self.model),
            "app": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in vars(self.app).items()
            },
            "checkpoint_dir": str(self.checkpoint_dir),
            "output_dir": str(self.output_dir),
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        data_config = DataConfig(
            zip_path=Path(config_dict["data"]["zip_path"]),
            extract_dir=Path(config_dict["data"]["extract_dir"]),
            img_size=tuple(config_dict["data"]["img_size"]),
            batch_size=config_dict["data"]["batch_size"],
            val_split=config_dict["data"]["val_split"],
            augmentation=config_dict["data"]["augmentation"],
            shuffle=config_dict["data"]["shuffle"],
        )

        model_config = ModelConfig(
            model_name=config_dict["model"]["model_name"],
            num_classes=config_dict["model"]["num_classes"],
            pretrained=config_dict["model"]["pretrained"],
            learning_rate=config_dict["model"]["learning_rate"],
            epochs=config_dict["model"]["epochs"],
            optimizer=config_dict["model"]["optimizer"],
            loss=config_dict["model"]["loss"],
            metrics=config_dict["model"]["metrics"],
        )

        app_config = AppConfig(
            title=config_dict["app"]["title"],
            description=config_dict["app"]["description"],
            examples_dir=Path(config_dict["app"]["examples_dir"])
            if config_dict["app"].get("examples_dir")
            else None,
            max_file_size=config_dict["app"]["max_file_size"],
            port=config_dict["app"]["port"],
        )

        return cls(
            data=data_config,
            model=model_config,
            app=app_config,
            checkpoint_dir=Path(config_dict.get("checkpoint_dir", "checkpoints")),
            output_dir=Path(config_dict.get("output_dir", "output")),
        )
