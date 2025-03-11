import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.schemas import AppConfig, Config, DataConfig, ModelConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_default_config(zip_path: Path) -> Config:
    """
    Create a default configuration.

    Args:
        zip_path: Path to the dataset zip file.

    Returns:
        Default Config object.
    """
    # Create extraction directory based on zip name
    zip_name = zip_path.stem
    extract_dir = Path("data") / zip_name

    # Create data config
    data_config = DataConfig(
        zip_path=zip_path,
        extract_dir=extract_dir,
        img_size=(224, 224),
        batch_size=32,
        val_split=0.2,
        augmentation=True,
        shuffle=True,
    )

    # Create model config
    model_config = ModelConfig(
        model_name="mobilenetv2",
        num_classes=0,  # Will be set based on data
        pretrained=True,
        learning_rate=0.001,
        epochs=10,
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Create app config
    app_config = AppConfig(
        title="Image Classification App",
        description="Upload an image to classify it using the trained model",
        examples_dir=None,  # Will be set after training
        max_file_size=5,
        port=7860,
    )

    # Create main config
    config = Config(
        data=data_config,
        model=model_config,
        app=app_config,
        checkpoint_dir=Path("checkpoints"),
        output_dir=Path("output"),
    )

    return config


def save_config(config: Config, config_path: Path) -> None:
    """
    Save a configuration to a JSON file.

    Args:
        config: Config object to save.
        config_path: Path to save the config to.
    """
    # Create parent directories if they don't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert config to dictionary
    config_dict = config.to_dict()

    # Save to JSON
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"Configuration saved to {config_path}")


def load_config(config_path: Path) -> Optional[Config]:
    """
    Load a configuration from a JSON file.

    Args:
        config_path: Path to the config file.

    Returns:
        Config object or None if loading fails.
    """
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return None

    try:
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        config = Config.from_dict(config_dict)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None


def update_config(config: Config, updates: Dict[str, Any]) -> Config:
    """
    Update a configuration with new values.

    Args:
        config: Config object to update.
        updates: Dictionary with updates in the format {'section.key': value}.

    Returns:
        Updated Config object.
    """
    # Convert config to dictionary
    config_dict = config.to_dict()

    # Apply updates
    for key_path, value in updates.items():
        parts = key_path.split(".")

        if len(parts) != 2:
            logger.warning(f"Invalid update key: {key_path}, skipping")
            continue

        section, key = parts

        if section not in config_dict:
            logger.warning(f"Section {section} not found in config, skipping")
            continue

        if key not in config_dict[section]:
            logger.warning(f"Key {key} not found in section {section}, skipping")
            continue

        config_dict[section][key] = value

    # Create updated config
    updated_config = Config.from_dict(config_dict)

    return updated_config
