import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from src.schemas import ModelConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_model(config: ModelConfig, input_shape: Tuple[int, int, int]) -> nn.Module:
    """
    Create a model based on the specified architecture.

    Args:
        config: ModelConfig object with model configuration.
        input_shape: Input shape for the model (height, width, channels).

    Returns:
        PyTorch model.
    """
    # Get the number of classes
    num_classes = config.num_classes

    logger.info(f"Creating model: {config.model_name} with {num_classes} classes")

    # Create the base model
    model = _get_base_model(config.model_name, input_shape, config.pretrained)

    # Add custom top layers for classification
    model = _add_classification_layers(model, config.model_name, num_classes)

    # Get the optimizer
    _get_optimizer(config.optimizer, model.parameters(), config.learning_rate)

    logger.info(
        f"Model compiled with optimizer: {config.optimizer}, lr: {config.learning_rate}"
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model summary: {total_params:,} total parameters")

    return model


def _get_base_model(
    model_name: str, input_shape: Tuple[int, int, int], pretrained: bool
) -> nn.Module:
    """
    Get a base model from torchvision's pre-built models.

    Args:
        model_name: Name of the model architecture.
        input_shape: Input shape (height, width, channels).
        pretrained: Whether to use pretrained weights.

    Returns:
        Base model.
    """
    weights = "IMAGENET1K_V1" if pretrained else None

    # Map model name to torchvision models
    model_mapping = {
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
        "vgg16": models.vgg16,
        "vgg19": models.vgg19,
        "mobilenet": models.mobilenet_v2,
        "mobilenetv2": models.mobilenet_v2,
        "densenet121": models.densenet121,
        "densenet169": models.densenet169,
        "densenet201": models.densenet201,
        "efficientnetb0": models.efficientnet_b0,
        "efficientnetb1": models.efficientnet_b1,
        "efficientnetb2": models.efficientnet_b2,
        "efficientnetb3": models.efficientnet_b3,
        "efficientnetb4": models.efficientnet_b4,
        "efficientnetb5": models.efficientnet_b5,
        "efficientnetb6": models.efficientnet_b6,
        "efficientnetb7": models.efficientnet_b7,
    }

    # Get the model class
    if model_name.lower() in model_mapping:
        model_class = model_mapping[model_name.lower()]
        if pretrained:
            base_model = model_class(weights=weights)
        else:
            base_model = model_class()
    else:
        # Default to a simple CNN for custom model
        logger.warning(
            f"Model {model_name} not found in predefined models, using a custom CNN"
        )
        base_model = _create_custom_cnn(input_shape)

    return base_model


def _create_custom_cnn(input_shape: Tuple[int, int, int]) -> nn.Module:
    """
    Create a custom CNN model.

    Args:
        input_shape: Input shape (height, width, channels).

    Returns:
        Custom CNN model.
    """

    class CustomCNN(nn.Module):
        def __init__(self, input_shape):
            super(CustomCNN, self).__init__()
            self.conv1 = nn.Conv2d(
                input_shape[2], 32, kernel_size=3, stride=1, padding=1
            )
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2)

            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2)

            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.relu3 = nn.ReLU()
            self.pool3 = nn.MaxPool2d(kernel_size=2)

            self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
            self.relu4 = nn.ReLU()
            self.pool4 = nn.MaxPool2d(kernel_size=2)

            # Calculate flattened size
            h, w = input_shape[0] // 16, input_shape[1] // 16  # after 4 pooling layers
            self.fc_input_size = 128 * h * w

        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.pool3(self.relu3(self.conv3(x)))
            x = self.pool4(self.relu4(self.conv4(x)))
            x = x.view(x.size(0), -1)  # Flatten
            return x

    return CustomCNN(input_shape)


def _add_classification_layers(
    base_model: nn.Module, model_name: str, num_classes: int
) -> nn.Module:
    """
    Add classification layers on top of the base model.

    Args:
        base_model: Base model to add layers to.
        model_name: Model name for determining architecture type.
        num_classes: Number of classes for the classification task.

    Returns:
        Model with classification layers added.
    """
    # For custom CNN, add a classifier head
    if (
        isinstance(base_model, nn.Module)
        and not hasattr(base_model, "fc")
        and not hasattr(base_model, "classifier")
    ):

        class CustomClassifier(nn.Module):
            def __init__(self, base_model, fc_input_size, num_classes):
                super(CustomClassifier, self).__init__()
                self.base = base_model
                self.fc1 = nn.Linear(fc_input_size, 512)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.5)
                self.fc2 = nn.Linear(512, num_classes)

            def forward(self, x):
                x = self.base(x)
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x

        if hasattr(base_model, "fc_input_size"):
            fc_input_size = base_model.fc_input_size
        else:
            fc_input_size = 512  # Default

        model = CustomClassifier(base_model, fc_input_size, num_classes)

    # For standard models, replace the classification head
    else:
        # For models with 'fc' attribute (like ResNet)
        if hasattr(base_model, "fc"):
            in_features = base_model.fc.in_features
            base_model.fc = nn.Linear(in_features, num_classes)
            model = base_model

        # For models with 'classifier' attribute (like VGG, DenseNet)
        elif hasattr(base_model, "classifier"):
            if isinstance(base_model.classifier, nn.Sequential):
                in_features = base_model.classifier[-1].in_features
                base_model.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                in_features = base_model.classifier.in_features
                base_model.classifier = nn.Linear(in_features, num_classes)
            model = base_model

        # For models with special classifier structure like EfficientNet
        elif hasattr(base_model, "classifier") and hasattr(base_model.classifier, "fc"):
            in_features = base_model.classifier.fc.in_features
            base_model.classifier.fc = nn.Linear(in_features, num_classes)
            model = base_model

        else:
            logger.warning(
                f"Model structure not recognized for {model_name}, creating a custom wrapper"
            )
            model = nn.Sequential(
                base_model,
                nn.Linear(1000, num_classes),  # Assuming a standard 1000-class output
            )

    return model


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
        PyTorch optimizer instance.
    """
    # Map optimizer name to PyTorch optimizer
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
