import datetime
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.nn import functional as F
from tqdm import tqdm

from src.schemas import (
    Dataset,
    Model,
    ModelConfig,
    ModelInfo,
    ModelResults,
    TrainingHistory,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: ModelConfig,
    dataset: Dataset,
    checkpoint_dir: Path,
) -> Tuple[nn.Module, Model]:
    """
    Train the model.

    Args:
        model: PyTorch model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: ModelConfig with training settings.
        dataset: Dataset object with class mapping.
        checkpoint_dir: Directory to save model checkpoints.

    Returns:
        Tuple of (trained_model, model_schema).
    """
    # Create checkpoint directory if it doesn't exist
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Move model to device
    model = model.to(device)

    # Set up criterion
    criterion = _get_criterion(config.loss)

    # Set up optimizer
    optimizer = _get_optimizer(
        config.optimizer, model.parameters(), config.learning_rate
    )

    # Set up learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Training history
    history = TrainingHistory()

    # Best model state
    best_val_accuracy = 0.0
    best_model_state = None

    # Training loop
    logger.info(f"Starting training for {config.epochs} epochs")
    for epoch in range(config.epochs):
        logger.info(f"Epoch {epoch + 1}/{config.epochs}")

        # Train for one epoch
        train_loss, train_acc = _train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        # Evaluate on validation set
        val_loss, val_acc = _evaluate(
            model=model, loader=val_loader, criterion=criterion, device=device
        )

        # Update learning rate
        scheduler.step(val_loss)

        # Save to history
        epoch_history = {
            "accuracy": train_acc,
            "val_accuracy": val_acc,
            "loss": train_loss,
            "val_loss": val_loss,
        }
        history.update(epoch_history)

        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_state = model.state_dict().copy()

            # Save checkpoint
            best_checkpoint_path = checkpoint_dir / f"{config.model_name}_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_accuracy": val_acc,
                    "model_name": config.model_name,
                    "num_classes": config.num_classes,
                },
                best_checkpoint_path,
            )

            logger.info(
                f"Saved best model checkpoint with validation accuracy: {val_acc:.4f}"
            )

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Evaluate the model on validation data
    logger.info("Evaluating model on validation data")
    eval_results = evaluate_model(
        model, val_loader, dataset.metadata.inv_class_mapping, device
    )

    # Create a timestamp for the model
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Get the input shape from the first batch
    sample_batch, _ = next(iter(train_loader))
    input_height, input_width = sample_batch.shape[2], sample_batch.shape[3]

    # Create ModelInfo object
    model_info = ModelInfo(
        name=config.model_name,
        num_classes=config.num_classes,
        input_shape=(input_height, input_width, 3),
        class_mapping=dataset.metadata.inv_class_mapping,
        date_trained=timestamp,
        params_count=sum(p.numel() for p in model.parameters()),
    )

    # Get the best checkpoint path
    best_checkpoint = checkpoint_dir / f"{config.model_name}_best.pt"

    # Create Model schema object
    model_schema = Model(
        info=model_info,
        history=history,
        results=eval_results,
        checkpoint_path=best_checkpoint,
    )

    # Save the model schema as JSON
    model_schema_path = checkpoint_dir / f"{config.model_name}_schema.json"
    with open(model_schema_path, "w") as f:
        json.dump(model_schema.to_dict(), f, indent=2)

    logger.info(f"Model schema saved to {model_schema_path}")

    return model, model_schema


def _train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model to train.
        loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to use.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(loader, desc="Training"):
        # Move data to device
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    # Calculate metrics
    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def _evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model on a dataset.

    Args:
        model: PyTorch model to evaluate.
        loader: Data loader.
        criterion: Loss function.
        device: Device to use.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating"):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    # Calculate metrics
    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def _get_criterion(loss_name: str) -> nn.Module:
    """
    Get a loss function by name.

    Args:
        loss_name: Name of the loss function.

    Returns:
        PyTorch criterion.
    """
    loss_mapping = {
        "cross_entropy": nn.CrossEntropyLoss,
        "categorical_crossentropy": nn.CrossEntropyLoss,
        "bce": nn.BCEWithLogitsLoss,
        "binary_crossentropy": nn.BCEWithLogitsLoss,
        "mse": nn.MSELoss,
        "mean_squared_error": nn.MSELoss,
    }

    if loss_name.lower() in loss_mapping:
        return loss_mapping[loss_name.lower()]()
    else:
        logger.warning(f"Loss {loss_name} not found, using CrossEntropyLoss")
        return nn.CrossEntropyLoss()


def _get_optimizer(
    optimizer_name: str, parameters, learning_rate: float
) -> torch.optim.Optimizer:
    """
    Get an optimizer by name.

    Args:
        optimizer_name: Name of the optimizer.
        parameters: Model parameters to optimize.
        learning_rate: Learning rate for the optimizer.

    Returns:
        PyTorch optimizer.
    """
    optimizer_mapping = {
        "adam": optim.Adam,
        "sgd": optim.SGD,
        "rmsprop": optim.RMSprop,
        "adagrad": optim.Adagrad,
        "adadelta": optim.Adadelta,
        "adamw": optim.AdamW,
    }

    if optimizer_name.lower() in optimizer_mapping:
        if optimizer_name.lower() == "sgd":
            return optimizer_mapping[optimizer_name.lower()](
                parameters, lr=learning_rate, momentum=0.9
            )
        else:
            return optimizer_mapping[optimizer_name.lower()](
                parameters, lr=learning_rate
            )
    else:
        logger.warning(f"Optimizer {optimizer_name} not found, using Adam")
        return optim.Adam(parameters, lr=learning_rate)


def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    class_mapping: Dict[int, str],
    device: torch.device,
) -> ModelResults:
    """
    Evaluate the model on validation data.

    Args:
        model: Trained PyTorch model.
        loader: Validation data loader.
        class_mapping: Mapping from class ID to class name.
        device: Device to use.

    Returns:
        ModelResults object with evaluation metrics.
    """
    model.eval()

    # Get the true labels and predictions
    true_labels = []
    predictions = []

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating model"):
            # Move data to device
            inputs = inputs.to(device)

            # Get predictions
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Add to lists
            true_labels.extend(targets.cpu().numpy())
            predictions.extend(preds.cpu().numpy())

    # Convert to numpy arrays
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)

    # Calculate metrics
    cm = confusion_matrix(true_labels, predictions)
    report = classification_report(true_labels, predictions, output_dict=True)

    # Calculate per-class accuracy
    class_accuracies = {}
    for class_id, class_name in class_mapping.items():
        if str(class_id) in report:
            class_accuracies[class_name] = report[str(class_id)]["precision"]
        else:
            class_accuracies[class_name] = 0.0

    # Create ModelResults object
    results = ModelResults(
        accuracy=report["accuracy"],
        class_accuracies=class_accuracies,
        confusion_matrix=cm.tolist(),
        f1_score=report["macro avg"]["f1-score"],
        precision=report["macro avg"]["precision"],
        recall=report["macro avg"]["recall"],
    )

    logger.info(
        f"Evaluation results: Accuracy={results.accuracy:.4f}, F1={results.f1_score:.4f}"
    )

    return results


def load_model_from_checkpoint(checkpoint_path: Path) -> nn.Module:
    """
    Load a model from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file.

    Returns:
        Loaded PyTorch model.
    """
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")

    try:
        # Load schema file to get model architecture
        schema_path = (
            checkpoint_path.parent
            / f"{checkpoint_path.stem.split('_best')[0]}_schema.json"
        )

        if not schema_path.exists():
            logger.warning(f"Schema file not found: {schema_path}")
            # Try using just the model name
            schema_path = (
                checkpoint_path.parent
                / f"{checkpoint_path.stem.replace('_best', '')}_schema.json"
            )

        if schema_path.exists():
            # Load the model schema to get architecture information
            with open(schema_path, "r") as f:
                model_schema_dict = json.load(f)

            # Get model architecture name
            model_name = model_schema_dict.get("info", {}).get("name", "mobilenetv2")
            num_classes = model_schema_dict.get("info", {}).get("num_classes", 2)
            input_shape = model_schema_dict.get("info", {}).get(
                "input_shape", [224, 224, 3]
            )

            # Create the model with the same architecture
            from src.models.model_factory import create_model
            from src.schemas import ModelConfig

            # Create a model config
            model_config = ModelConfig(
                model_name=model_name,
                num_classes=num_classes,
                pretrained=False,  # Don't use pretrained weights as we'll load from checkpoint
            )

            # Create model with the same architecture
            model = create_model(model_config, tuple(input_shape))

            # Load the weights
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            model.load_state_dict(checkpoint["model_state_dict"])

            logger.info(
                f"Successfully loaded model {model_name} with {num_classes} classes"
            )
            return model
        else:
            logger.warning(
                "Schema file not found, attempting to infer model architecture from checkpoint"
            )

            # Load on CPU to avoid GPU memory issues
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

            # Examine state dict to guess architecture
            state_dict = checkpoint["model_state_dict"]

            # Detect architecture based on keys in state_dict
            if any("features" in key for key in state_dict.keys()):
                # MobileNet, DenseNet, etc.
                from torchvision import models

                if any(
                    "mobilenet" in checkpoint_path.stem.lower()
                    or "mobilenetv2" in checkpoint_path.stem.lower()
                ):
                    model = models.mobilenet_v2()
                    # Adjust the classifier
                    last_channel = model.classifier[1].in_features
                    model.classifier[1] = nn.Linear(
                        last_channel, 2
                    )  # Default to binary classification
                elif any("densenet" in checkpoint_path.stem.lower()):
                    model = models.densenet121()
                    # Adjust the classifier
                    last_channel = model.classifier.in_features
                    model.classifier = nn.Linear(last_channel, 2)
                else:
                    # Default to MobileNetV2
                    model = models.mobilenet_v2()
                    # Adjust the classifier
                    last_channel = model.classifier[1].in_features
                    model.classifier[1] = nn.Linear(last_channel, 2)
            elif any("layer" in key for key in state_dict.keys()):
                # ResNet
                from torchvision import models

                model = models.resnet50()
                # Adjust the fully connected layer
                last_channel = model.fc.in_features
                model.fc = nn.Linear(last_channel, 2)
            else:
                # Fallback to a simple model
                logger.warning(
                    "Could not determine model architecture, using a simple model"
                )
                model = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(128 * 56 * 56, 2),  # Default to binary classification
                )

            # Try to load state dictionary
            try:
                model.load_state_dict(state_dict)
                logger.info("Model loaded successfully with inferred architecture")
                return model
            except Exception as e:
                logger.error(f"Error loading model with inferred architecture: {e}")

                # Last resort: Try to create a model that matches the state dict structure
                from collections import OrderedDict

                # Create a sequential model with layers matching the state dict
                layers = OrderedDict()
                layer_idx = 0
                for key in state_dict.keys():
                    if "weight" in key:
                        # Extract shape information
                        weight = state_dict[key]
                        if len(weight.shape) == 4:  # Conv layer
                            out_channels, in_channels = weight.shape[0], weight.shape[1]
                            layers[f"{layer_idx}"] = nn.Conv2d(
                                in_channels, out_channels, kernel_size=3, padding=1
                            )
                            layer_idx += 1
                        elif len(weight.shape) == 2:  # Linear layer
                            out_features, in_features = weight.shape
                            layers[f"{layer_idx}"] = (
                                nn.Flatten() if layer_idx == 0 else nn.Identity()
                            )
                            layer_idx += 1
                            layers[f"{layer_idx}"] = nn.Linear(
                                in_features, out_features
                            )
                            layer_idx += 1

                # Create model if layers were found
                if layers:
                    model = nn.Sequential(layers)
                    try:
                        model.load_state_dict(state_dict)
                        logger.info("Model loaded with dynamic architecture")
                        return model
                    except Exception as e:
                        logger.error(
                            f"Failed to load model with dynamic architecture: {e}"
                        )

                raise ValueError("Could not load the model with any architecture")

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def predict_image(
    model: nn.Module, image: np.ndarray, class_mapping: Dict[int, str]
) -> Tuple[str, float, Dict[str, float]]:
    """
    Predict the class of an image.

    Args:
        model: Trained PyTorch model.
        image: Preprocessed image as numpy array.
        class_mapping: Mapping from class ID to class name.

    Returns:
        Tuple of (predicted_class, confidence, all_predictions).
    """
    # Set model to evaluation mode
    model.eval()

    # Convert numpy array to tensor
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    # Get the predicted class and confidence
    predicted_class_id = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class_id].item()

    # Get the predicted class name
    predicted_class = class_mapping.get(
        predicted_class_id, f"Unknown ({predicted_class_id})"
    )

    # Get all predictions
    all_predictions = {
        class_mapping.get(i, f"Class {i}"): prob.item()
        for i, prob in enumerate(probabilities)
    }

    return predicted_class, confidence, all_predictions
