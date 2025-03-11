from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrainingHistory:
    """Schema for model training history."""

    accuracy: List[float] = field(default_factory=list)  # Training accuracy per epoch
    val_accuracy: List[float] = field(
        default_factory=list
    )  # Validation accuracy per epoch
    loss: List[float] = field(default_factory=list)  # Training loss per epoch
    val_loss: List[float] = field(default_factory=list)  # Validation loss per epoch
    epochs_trained: int = 0  # Number of epochs trained

    def update(self, epoch_history: Dict[str, float]) -> None:
        """Update history with results from a single epoch."""
        if "accuracy" in epoch_history:
            self.accuracy.append(epoch_history["accuracy"])
        if "val_accuracy" in epoch_history:
            self.val_accuracy.append(epoch_history["val_accuracy"])
        if "loss" in epoch_history:
            self.loss.append(epoch_history["loss"])
        if "val_loss" in epoch_history:
            self.val_loss.append(epoch_history["val_loss"])
        self.epochs_trained += 1


@dataclass
class ModelInfo:
    """Schema for model metadata."""

    name: str  # Model name/architecture
    num_classes: int  # Number of classes
    input_shape: tuple[int, int, int]  # Input shape (height, width, channels)
    class_mapping: Dict[int, str]  # Mapping from class ID to name
    date_trained: str  # Date when model was trained
    params_count: int = 0  # Number of parameters

    def to_dict(self) -> Dict[str, Any]:
        """Convert model info to dictionary."""
        return {
            "name": self.name,
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
            "class_mapping": {str(k): v for k, v in self.class_mapping.items()},
            "date_trained": self.date_trained,
            "params_count": self.params_count,
        }

    @classmethod
    def from_dict(cls, info_dict: Dict[str, Any]) -> "ModelInfo":
        """Create model info from dictionary."""
        return cls(
            name=info_dict["name"],
            num_classes=info_dict["num_classes"],
            input_shape=tuple(info_dict["input_shape"]),
            class_mapping={int(k): v for k, v in info_dict["class_mapping"].items()},
            date_trained=info_dict["date_trained"],
            params_count=info_dict.get("params_count", 0),
        )


@dataclass
class ModelResults:
    """Schema for model evaluation results."""

    accuracy: float  # Overall accuracy
    class_accuracies: Dict[str, float]  # Accuracy per class
    confusion_matrix: List[List[int]]  # Confusion matrix
    f1_score: Optional[float] = None  # F1 score
    precision: Optional[float] = None  # Precision
    recall: Optional[float] = None  # Recall

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "accuracy": self.accuracy,
            "class_accuracies": self.class_accuracies,
            "confusion_matrix": self.confusion_matrix,
            "f1_score": self.f1_score,
            "precision": self.precision,
            "recall": self.recall,
        }

    @classmethod
    def from_dict(cls, results_dict: Dict[str, Any]) -> "ModelResults":
        """Create results from dictionary."""
        return cls(
            accuracy=results_dict["accuracy"],
            class_accuracies=results_dict["class_accuracies"],
            confusion_matrix=results_dict["confusion_matrix"],
            f1_score=results_dict.get("f1_score"),
            precision=results_dict.get("precision"),
            recall=results_dict.get("recall"),
        )


@dataclass
class Model:
    """Schema for the complete model."""

    info: ModelInfo  # Model information
    history: TrainingHistory = field(
        default_factory=TrainingHistory
    )  # Training history
    results: Optional[ModelResults] = None  # Evaluation results
    checkpoint_path: Optional[Path] = None  # Path to model checkpoint

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "info": self.info.to_dict(),
            "history": {
                "accuracy": self.history.accuracy,
                "val_accuracy": self.history.val_accuracy,
                "loss": self.history.loss,
                "val_loss": self.history.val_loss,
                "epochs_trained": self.history.epochs_trained,
            },
            "results": self.results.to_dict() if self.results else None,
            "checkpoint_path": str(self.checkpoint_path)
            if self.checkpoint_path
            else None,
        }

    @classmethod
    def from_dict(cls, model_dict: Dict[str, Any]) -> "Model":
        """Create model from dictionary."""
        info = ModelInfo.from_dict(model_dict["info"])

        history = TrainingHistory(
            accuracy=model_dict["history"]["accuracy"],
            val_accuracy=model_dict["history"]["val_accuracy"],
            loss=model_dict["history"]["loss"],
            val_loss=model_dict["history"]["val_loss"],
            epochs_trained=model_dict["history"]["epochs_trained"],
        )

        results = None
        if model_dict.get("results"):
            results = ModelResults.from_dict(model_dict["results"])

        checkpoint_path = None
        if model_dict.get("checkpoint_path"):
            checkpoint_path = Path(model_dict["checkpoint_path"])

        return cls(
            info=info,
            history=history,
            results=results,
            checkpoint_path=checkpoint_path,
        )
