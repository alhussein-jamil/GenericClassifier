import json
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.data import (
    create_data_generators,
    load_dataset,
    preprocess_image,
)
from src.models import create_model, load_model_from_checkpoint, train_model
from src.schemas import AppConfig, Config, DataConfig, Model, ModelConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model_schema(checkpoint_dir: Path, model_name: str) -> Optional[Model]:
    """
    Load the model schema from a JSON file.

    Args:
        checkpoint_dir: Directory containing model files.
        model_name: Name of the model.

    Returns:
        Model object or None if not found.
    """
    schema_path = checkpoint_dir / f"{model_name}_schema.json"

    if not schema_path.exists():
        logger.warning(f"Model schema not found: {schema_path}")
        return None

    try:
        with open(schema_path, "r") as f:
            schema_dict = json.load(f)

        model_schema = Model.from_dict(schema_dict)
        logger.info(
            f"Loaded model schema: {model_schema.info.name}, {model_schema.info.num_classes} classes"
        )
        return model_schema
    except Exception as e:
        logger.error(f"Error loading model schema: {e}")
        return None


def plot_training_history(model_schema: Model) -> np.ndarray:
    """
    Plot the training history.

    Args:
        model_schema: Model object with training history.

    Returns:
        Figure as numpy array.
    """
    history = model_schema.history

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot accuracy
    ax1.plot(history.accuracy, label="Training Accuracy")
    ax1.plot(history.val_accuracy, label="Validation Accuracy")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    ax1.grid(True)

    # Plot loss
    ax2.plot(history.loss, label="Training Loss")
    ax2.plot(history.val_loss, label="Validation Loss")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True)

    fig.tight_layout()

    # Convert figure to numpy array
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)

    return img


def plot_confusion_matrix(model_schema: Model) -> np.ndarray:
    """
    Plot the confusion matrix.

    Args:
        model_schema: Model object with evaluation results.

    Returns:
        Figure as numpy array.
    """
    if not model_schema.results or not model_schema.results.confusion_matrix:
        logger.warning("No confusion matrix available in model schema")
        return np.zeros((10, 10, 3), dtype=np.uint8)

    cm = np.array(model_schema.results.confusion_matrix)
    class_names = [
        model_schema.info.class_mapping.get(str(i), f"Class {i}")
        for i in range(len(model_schema.info.class_mapping))
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot confusion matrix
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Set labels
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    # Set ticks
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # Add values to cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    fig.tight_layout()

    # Convert figure to numpy array
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)

    return img


def handle_zip_upload(zip_file) -> Tuple[str, str, Path, Path]:
    """
    Handle the uploaded zip file containing a dataset.

    Args:
        zip_file: Path to the uploaded zip file.

    Returns:
        Tuple of (status, message, zip_path, extract_dir)
    """
    if zip_file is None:
        return "error", "No zip file provided", None, None

    try:
        zip_path = Path(zip_file.name)

        # Create a temporary directory for extraction
        temp_dir = Path(tempfile.mkdtemp())
        extract_dir = temp_dir / "dataset"
        extract_dir.mkdir(exist_ok=True)

        # Validate zip file
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                # Check if zip contains folders (potential classes)
                has_folders = any("/" in name for name in zip_ref.namelist())
                if not has_folders:
                    return (
                        "error",
                        "Zip file must contain folders (one per class)",
                        None,
                        None,
                    )

                # Extract the zip file
                zip_ref.extractall(extract_dir)
        except zipfile.BadZipFile:
            return "error", "Invalid zip file", None, None

        # Count classes (folders)
        class_dirs = [d for d in extract_dir.iterdir() if d.is_dir()]
        if not class_dirs:
            return "error", "No class folders found in the zip file", None, None

        return (
            "success",
            f"Dataset uploaded with {len(class_dirs)} classes",
            zip_path,
            extract_dir,
        )

    except Exception as e:
        logger.error(f"Error handling zip upload: {e}")
        return "error", f"Error processing zip file: {str(e)}", None, None


def train_model_from_config(config: Config) -> Tuple[str, str, Optional[str]]:
    """
    Train a model with the given configuration.

    Args:
        config: Configuration object.

    Returns:
        Tuple of (status, message, model_name)
    """
    logger.info("Starting training workflow")

    try:
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
            config.model,
            input_shape=(config.data.img_size[0], config.data.img_size[1], 3),
        )

        # Train model
        logger.info(f"Training model for {config.model.epochs} epochs")
        model, model_schema = train_model(
            model,
            train_loader,
            val_loader,
            config.model,
            dataset,
            config.checkpoint_dir,
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
                dest_path = (
                    examples_dir / f"{img_info.class_name}{img_info.path.suffix}"
                )
                with open(img_info.path, "rb") as src, open(dest_path, "wb") as dst:
                    dst.write(src.read())
                class_examples[img_info.class_name] = dest_path

            # Stop once we have an example for each class
            if len(class_examples) == dataset.metadata.num_classes:
                break

        # Update app config with examples directory
        config.app.examples_dir = examples_dir

        # Save updated config
        save_config_path = config.checkpoint_dir / "config.json"
        with open(save_config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        return (
            "success",
            f"Model trained successfully with accuracy: {model_schema.results.accuracy:.4f}",
            config.model.model_name,
        )

    except Exception as e:
        logger.error(f"Error during training: {e}")
        return "error", f"Error during training: {str(e)}", None


def create_all_in_one_app() -> gr.Blocks:
    """
    Create a Gradio app with integrated dataset upload, training, and inference.

    Returns:
        Gradio Blocks app.
    """
    # Create app state variables
    state = {
        "zip_path": None,
        "extract_dir": None,
        "model_name": None,
        "checkpoint_dir": Path("checkpoints"),
        "trained_model": None,
        "model_schema": None,
    }

    def train_from_ui(
        zip_file,
        model_type,
        img_height,
        img_width,
        batch_size,
        epochs,
        learning_rate,
        use_pretrained,
        val_split,
        use_augmentation,
    ):
        # Handle zip upload
        status, message, zip_path, extract_dir = handle_zip_upload(zip_file)
        if status == "error":
            return message, None, None, None, None

        # Update state
        state["zip_path"] = zip_path
        state["extract_dir"] = extract_dir

        # Create configurations
        data_config = DataConfig(
            zip_path=zip_path,
            extract_dir=extract_dir,
            img_size=(img_height, img_width),
            batch_size=batch_size,
            val_split=val_split,
            augmentation=use_augmentation,
            shuffle=True,
        )

        model_config = ModelConfig(
            model_name=model_type,
            num_classes=0,  # Will be set based on dataset
            pretrained=use_pretrained,
            learning_rate=learning_rate,
            epochs=epochs,
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        app_config = AppConfig(
            title="Image Classification App",
            description="Upload an image to classify it using the trained model",
            examples_dir=None,  # Will be set during training
            max_file_size=5,
            port=7860,
        )

        # Create checkpoint directory
        checkpoint_dir = Path("checkpoints") / f"{model_type}_{zip_path.stem}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state["checkpoint_dir"] = checkpoint_dir

        # Create config
        config = Config(
            data=data_config,
            model=model_config,
            app=app_config,
            checkpoint_dir=checkpoint_dir,
            output_dir=Path("output"),
        )

        # Train model
        status, message, model_name = train_model_from_config(config)
        if status == "error":
            return message, None, None, None, None

        # Update state
        state["model_name"] = model_name

        # Load model schema
        model_schema = load_model_schema(checkpoint_dir, model_name)
        if model_schema:
            state["model_schema"] = model_schema

            # Load model
            model = load_model_from_checkpoint(model_schema.checkpoint_path)
            model.eval()
            state["trained_model"] = model

            # Plot training history and confusion matrix
            history_plot = plot_training_history(model_schema)
            confusion_matrix_plot = plot_confusion_matrix(model_schema)

            return (
                message,
                history_plot,
                confusion_matrix_plot,
                True,
                gr.update(visible=True),
            )
        else:
            return f"{message}, but failed to load model schema", None, None, None, None

    def predict_image(image):
        if image is None:
            return "No image provided", None

        # Check if model is loaded
        model = state.get("trained_model")
        model_schema = state.get("model_schema")

        if model is None or model_schema is None:
            return "No trained model available. Please train a model first.", None

        # Get input shape and class mapping
        input_shape = model_schema.info.input_shape
        img_size = (input_shape[0], input_shape[1])
        class_mapping = {int(k): v for k, v in model_schema.info.class_mapping.items()}

        # Set device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        try:
            # Preprocess image - handle both file object or direct path string
            if hasattr(image, "name"):
                # It's a file-like object
                img_path = Path(image.name)
            else:
                # It's a string path
                img_path = Path(image)

            # Preprocess the image
            img_array = preprocess_image(img_path, img_size)

            # Convert to PyTorch tensor
            img_tensor = (
                torch.from_numpy(img_array.transpose(2, 0, 1)).float().unsqueeze(0)
            )
            img_tensor = img_tensor.to(device)

            # Make prediction
            with torch.no_grad():
                outputs = model(img_tensor)
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

            # Sort predictions by confidence
            sorted_preds = sorted(
                all_predictions.items(), key=lambda x: x[1], reverse=True
            )

            # Format the result for Gradio Label component (dictionary format)
            result_dict = {cls: conf for cls, conf in sorted_preds}

            return f"Prediction: {predicted_class} ({confidence:.2%})", result_dict

        except Exception as e:
            logger.error(f"Error during image prediction: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return f"Error during prediction: {str(e)}", None

    # Create Gradio app with Blocks interface
    with gr.Blocks(title="All-in-One Image Classification") as app:
        gr.Markdown("# All-in-One Image Classification Framework")
        gr.Markdown(
            "Upload a dataset, train a model, and classify images - all in one place!"
        )

        with gr.Tabs():
            # Tab 1: Train a Model
            with gr.TabItem("Train a Model"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Step 1: Upload Dataset")
                        gr.Markdown(
                            "Upload a zip file containing your dataset organized in folders (one folder per class)"
                        )

                        # Dataset upload
                        dataset_upload = gr.File(
                            label="Upload Dataset Zip File (drag & drop)",
                            file_types=[".zip"],
                            file_count="single",
                        )

                        gr.Markdown("## Step 2: Configure Training")

                        # Model configuration
                        model_type = gr.Dropdown(
                            label="Model Architecture",
                            choices=[
                                "mobilenetv2",
                                "resnet50",
                                "densenet121",
                                "efficientnetb0",
                                "vgg16",
                            ],
                            value="mobilenetv2",
                        )

                        with gr.Row():
                            img_height = gr.Number(
                                label="Image Height", value=224, precision=0
                            )
                            img_width = gr.Number(
                                label="Image Width", value=224, precision=0
                            )

                        with gr.Row():
                            batch_size = gr.Slider(
                                label="Batch Size",
                                minimum=8,
                                maximum=64,
                                value=32,
                                step=8,
                            )
                            epochs = gr.Slider(
                                label="Epochs", minimum=1, maximum=50, value=10, step=1
                            )

                        with gr.Row():
                            learning_rate = gr.Slider(
                                label="Learning Rate",
                                minimum=0.0001,
                                maximum=0.01,
                                value=0.001,
                                step=0.0001,
                            )
                            val_split = gr.Slider(
                                label="Validation Split",
                                minimum=0.1,
                                maximum=0.5,
                                value=0.2,
                                step=0.05,
                            )

                        with gr.Row():
                            use_pretrained = gr.Checkbox(
                                label="Use Pretrained Weights", value=True
                            )
                            use_augmentation = gr.Checkbox(
                                label="Use Data Augmentation", value=True
                            )

                        # Train button
                        train_button = gr.Button("Train Model", variant="primary")

                    with gr.Column():
                        # Training results
                        training_output = gr.Textbox(label="Training Output")

                        # Training history
                        history_plot = gr.Image(label="Training History", visible=True)

                        # Confusion matrix
                        confusion_matrix = gr.Image(
                            label="Confusion Matrix", visible=True
                        )

                        # Training success indicator (hidden)
                        training_success = gr.Checkbox(
                            label="Training Success", value=False, visible=False
                        )

                # Connect the train button
                train_button.click(
                    fn=train_from_ui,
                    inputs=[
                        dataset_upload,
                        model_type,
                        img_height,
                        img_width,
                        batch_size,
                        epochs,
                        learning_rate,
                        use_pretrained,
                        val_split,
                        use_augmentation,
                    ],
                    outputs=[
                        training_output,
                        history_plot,
                        confusion_matrix,
                        training_success,
                        history_plot,  # Just to update visibility
                    ],
                )

            # Tab 2: Classify Images
            with gr.TabItem("Classify Images"):
                with gr.Row():
                    with gr.Column():
                        # Image upload
                        image_input = gr.Image(
                            type="filepath",
                            label="Upload an image to classify (drag & drop)",
                        )

                        # Classify button
                        classify_button = gr.Button("Classify Image", variant="primary")

                    with gr.Column():
                        # Classification result
                        result_text = gr.Textbox(label="Result")

                        # Class probabilities
                        class_probs = gr.Label(label="Class Probabilities")

                # Connect the classify button
                classify_button.click(
                    fn=predict_image,
                    inputs=[image_input],
                    outputs=[result_text, class_probs],
                )

    return app


def create_app(config: AppConfig, checkpoint_dir: Path, model_name: str) -> gr.Blocks:
    """
    Create a Gradio app for image classification.

    Args:
        config: AppConfig object with app settings.
        checkpoint_dir: Directory containing model files.
        model_name: Name of the model to use.

    Returns:
        Gradio Blocks app.
    """
    # Load model schema
    model_schema = load_model_schema(checkpoint_dir, model_name)

    if not model_schema or not model_schema.checkpoint_path:
        raise ValueError(f"Model schema or checkpoint not found for {model_name}")

    # Load PyTorch model
    model = load_model_from_checkpoint(model_schema.checkpoint_path)

    # Set model to evaluation mode
    model.eval()

    # Get class mapping
    class_mapping = {int(k): v for k, v in model_schema.info.class_mapping.items()}

    # Get input shape
    input_shape = model_schema.info.input_shape
    img_size = (input_shape[0], input_shape[1])

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define prediction function
    def predict(image):
        if image is None:
            return "No image provided", None, None

        try:
            # Preprocess image - handle both file object or direct path string
            if hasattr(image, "name"):
                # It's a file-like object
                img_path = Path(image.name)
            else:
                # It's a string path
                img_path = Path(image)

            # Preprocess the image
            img_array = preprocess_image(img_path, img_size)

            # Convert to PyTorch tensor
            img_tensor = (
                torch.from_numpy(img_array.transpose(2, 0, 1)).float().unsqueeze(0)
            )
            img_tensor = img_tensor.to(device)

            # Make prediction
            with torch.no_grad():
                outputs = model(img_tensor)
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

            # Sort predictions by confidence
            sorted_preds = sorted(
                all_predictions.items(), key=lambda x: x[1], reverse=True
            )

            # Format the result for Gradio Label component (dictionary format)
            result_dict = {cls: conf for cls, conf in sorted_preds}

            return f"Prediction: {predicted_class} ({confidence:.2%})", result_dict

        except Exception as e:
            logger.error(f"Error during image prediction: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return f"Error during prediction: {str(e)}", None, None

    # Define example selection function
    def load_example(example):
        return example

    # Get example images if provided
    examples = []
    if config.examples_dir and Path(config.examples_dir).exists():
        examples = [
            str(f)
            for f in Path(config.examples_dir).glob("*.*")
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

    # Create Gradio app with Blocks interface
    with gr.Blocks(title=config.title) as app:
        gr.Markdown(f"# {config.title}")
        gr.Markdown(config.description)

        with gr.Tabs():
            with gr.TabItem("Classify Images"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            type="filepath", label="Upload an image (drag & drop)"
                        )

                        with gr.Row():
                            submit_btn = gr.Button("Classify", variant="primary")
                            clear_btn = gr.Button("Clear")

                        if examples:
                            gr.Examples(
                                examples=examples,
                                inputs=image_input,
                                fn=load_example,
                                examples_per_page=5,
                            )

                    with gr.Column():
                        result_text = gr.Textbox(label="Result")
                        label_output = gr.Label(label="Class Probabilities")

                submit_btn.click(
                    fn=predict,
                    inputs=[image_input],
                    outputs=[result_text, label_output],
                )

                clear_btn.click(
                    fn=lambda: (None, None, None),
                    inputs=[],
                    outputs=[image_input, result_text, label_output],
                )

            with gr.TabItem("Model Information"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(f"## Model: {model_schema.info.name}")
                        gr.Markdown(f"- **Classes:** {model_schema.info.num_classes}")
                        gr.Markdown(
                            f"- **Input Shape:** {model_schema.info.input_shape}"
                        )
                        gr.Markdown(
                            f"- **Parameters:** {model_schema.info.params_count:,}"
                        )
                        gr.Markdown(
                            f"- **Date Trained:** {model_schema.info.date_trained}"
                        )

                        if model_schema.results:
                            gr.Markdown(
                                f"- **Accuracy:** {model_schema.results.accuracy:.4f}"
                            )
                            gr.Markdown(
                                f"- **F1 Score:** {model_schema.results.f1_score:.4f}"
                            )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Training History")
                        gr.Image(value=plot_training_history(model_schema))

                    with gr.Column():
                        gr.Markdown("## Confusion Matrix")
                        if (
                            model_schema.results
                            and model_schema.results.confusion_matrix
                        ):
                            gr.Image(value=plot_confusion_matrix(model_schema))

    return app


def launch_app(config: AppConfig, checkpoint_dir: Path, model_name: str) -> None:
    """
    Launch the Gradio app.

    Args:
        config: AppConfig object with app settings.
        checkpoint_dir: Directory containing model files.
        model_name: Name of the model to use.
    """
    # If no model name or checkpoint is provided, launch the all-in-one app
    if not model_name or not checkpoint_dir.exists():
        logger.info("No model found, launching all-in-one app")
        app = create_all_in_one_app()
    else:
        logger.info(f"Launching app with model: {model_name}")
        app = create_app(config, checkpoint_dir, model_name)

    app.launch(server_name="0.0.0.0", server_port=config.port, share=False)
