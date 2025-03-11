import argparse
import logging
import sys
from pathlib import Path

import torch

from src.app import launch_app
from src.data import create_data_generators, load_dataset
from src.models import create_model, train_model
from src.schemas import Config
from src.utils import create_default_config, load_config, save_config, update_config

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Image classification framework")

    # Main operation mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "app", "both"],
        default="app",
        help="Operation mode: train, app, or both (default: app)",
    )

    # Dataset parameters
    parser.add_argument("--zip", type=str, help="Path to dataset zip file")
    parser.add_argument(
        "--extract_dir", type=str, help="Directory to extract the dataset to"
    )

    # Model parameters
    parser.add_argument(
        "--model", type=str, default="mobilenetv2", help="Model architecture to use"
    )
    parser.add_argument("--epochs", type=int, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument(
        "--img_size", type=int, nargs=2, help="Image size (height width)"
    )

    # App parameters
    parser.add_argument(
        "--port", type=int, default=7860, help="Port for the Gradio app"
    )
    parser.add_argument("--title", type=str, help="Title for the app")

    # Config file
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--save_config", type=str, help="Path to save config file")

    # Checkpoint directory
    parser.add_argument(
        "--checkpoint_dir", type=str, help="Directory to save model checkpoints"
    )

    # Device
    parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu"], help="Device to use for training"
    )

    # Use interactive mode (drag & drop interface)
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=True,
        help="Use interactive mode with drag & drop interface (default: True)",
    )

    return parser.parse_args()


def get_config(args) -> Config:
    """Get configuration from command line arguments or config file."""
    config = None

    # Load config from file if provided
    if args.config:
        config_path = Path(args.config)
        config = load_config(config_path)
        if config is None:
            logger.error(f"Failed to load config from {config_path}")
            sys.exit(1)

    # Create default config if no config file provided
    if config is None:
        if args.zip:
            zip_path = Path(args.zip)
            config = create_default_config(zip_path)
        elif not args.interactive:
            # Only show error if not in interactive mode
            logger.error("No config file or zip file provided")
            sys.exit(1)
        else:
            # In interactive mode, create minimal config for the app
            from src.schemas import AppConfig, DataConfig, ModelConfig

            data_config = DataConfig(
                zip_path=Path("placeholder.zip"),  # Will be replaced in UI
                extract_dir=Path("data"),
                img_size=(224, 224),
                batch_size=32,
                val_split=0.2,
                augmentation=True,
                shuffle=True,
            )

            model_config = ModelConfig(
                model_name="mobilenetv2",
                num_classes=0,
                pretrained=True,
                learning_rate=0.001,
                epochs=10,
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )

            app_config = AppConfig(
                title="All-in-One Image Classification",
                description="Upload a dataset, train a model, and classify images - all in one place!",
                examples_dir=None,
                max_file_size=5,
                port=args.port,
            )

            config = Config(
                data=data_config,
                model=model_config,
                app=app_config,
                checkpoint_dir=Path("checkpoints"),
                output_dir=Path("output"),
            )

    # Update config from command line arguments
    updates = {}

    if args.extract_dir:
        updates["data.extract_dir"] = Path(args.extract_dir)

    if args.model:
        updates["model.model_name"] = args.model

    if args.epochs:
        updates["model.epochs"] = args.epochs

    if args.batch_size:
        updates["data.batch_size"] = args.batch_size

    if args.lr:
        updates["model.learning_rate"] = args.lr

    if args.img_size:
        updates["data.img_size"] = tuple(args.img_size)

    if args.port:
        updates["app.port"] = args.port

    if args.title:
        updates["app.title"] = args.title

    if args.checkpoint_dir:
        config.checkpoint_dir = Path(args.checkpoint_dir)

    # Apply updates
    if updates:
        config = update_config(config, updates)

    # Save config if requested
    if args.save_config:
        save_config(config, Path(args.save_config))

    return config


def train(config: Config) -> str:
    """
    Train a model with the given configuration.

    Args:
        config: Configuration object.

    Returns:
        Name of the trained model.
    """
    logger.info("Starting training workflow")

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info(f"Loading dataset from {config.data.zip_path}")
    dataset = load_dataset(config.data)

    # Update model config with number of classes
    config.model.num_classes = dataset.metadata.num_classes

    # Create data loaders
    logger.info("Creating data loaders")
    train_loader, val_loader = create_data_generators(dataset, config.data)

    # Create model
    logger.info(f"Creating model: {config.model.model_name}")
    model = create_model(
        config.model, input_shape=(config.data.img_size[0], config.data.img_size[1], 3)
    )

    # Train model
    logger.info(f"Training model for {config.model.epochs} epochs")
    model, model_schema = train_model(
        model, train_loader, val_loader, config.model, dataset, config.checkpoint_dir
    )

    logger.info(
        f"Training completed with accuracy: {model_schema.results.accuracy:.4f}"
    )

    # Set examples directory for the app
    examples_dir = config.checkpoint_dir / "examples"
    examples_dir.mkdir(exist_ok=True)

    # Save some example images for the app (one per class)
    class_examples = {}
    for img_info in dataset.val_images:
        if img_info.class_name not in class_examples:
            # Copy the image to examples directory
            dest_path = examples_dir / f"{img_info.class_name}{img_info.path.suffix}"
            with open(img_info.path, "rb") as src, open(dest_path, "wb") as dst:
                dst.write(src.read())
            class_examples[img_info.class_name] = dest_path

        # Stop once we have an example for each class
        if len(class_examples) == dataset.metadata.num_classes:
            break

    # Update app config with examples directory
    config.app.examples_dir = examples_dir

    # Save updated config
    save_config(config, config.checkpoint_dir / "config.json")

    return config.model.model_name


def run_app(config: Config, model_name: str = None) -> None:
    """
    Run the Gradio app.

    Args:
        config: Configuration object.
        model_name: Name of the model to use (optional).
    """
    if model_name:
        logger.info(f"Starting app with model: {model_name}")
    else:
        logger.info("Starting app in interactive mode")

    launch_app(config.app, config.checkpoint_dir, model_name)


def main():
    """Main entry point."""
    print("Image Classification Framework")
    print("-----------------------------")

    # Parse command line arguments
    args = parse_args()

    # Get configuration
    config = get_config(args)

    # Create checkpoint directory if it doesn't exist
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Determine the model name (None in interactive mode unless training was done)
    model_name = None

    # Run in the specified mode
    if args.mode in ["train", "both"]:
        model_name = train(config)

    if args.mode in ["app", "both"]:
        run_app(config, model_name)


if __name__ == "__main__":
    main()
