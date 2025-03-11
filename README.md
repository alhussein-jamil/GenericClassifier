# Image Classification Framework

A complete framework for training and deploying image classification models.

## Features

- Import datasets from zip files
- Train models using various architectures
- Interactive Gradio interface for testing
- Configurable settings
- Model evaluation and visualization

## Installation

1. Create a virtual environment (recommended):

```bash
python -m venv venv
```

2. Activate the virtual environment:

On Windows:
```bash
venv\Scripts\activate
```

On macOS/Linux:
```bash
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

The simplest way to use the framework is to provide a zip file containing your dataset:

```bash
python app.py --zip path/to/dataset.zip
```

This will:
1. Extract the dataset from the zip file
2. Train a MobileNetV2 model (default)
3. Launch a Gradio app for testing the model

### Dataset Format

Your dataset zip file should have the following structure:

```
dataset.zip
├── class_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class_2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

Each folder name becomes a class label in the model.

### Command Line Arguments

The framework provides many command line arguments to customize behavior:

```bash
python app.py --help
```

Common options:

- `--mode` - Operation mode: 'train', 'app', or 'both' (default)
- `--zip` - Path to dataset zip file
- `--model` - Model architecture to use (default: mobilenetv2)
- `--epochs` - Number of epochs to train
- `--batch_size` - Batch size for training
- `--lr` - Learning rate
- `--img_size` - Image size (height width)
- `--port` - Port for the Gradio app
- `--checkpoint_dir` - Directory to save model checkpoints

### Configuration File

You can also specify settings in a JSON configuration file:

```bash
python app.py --config path/to/config.json
```

To generate a default config file:

```bash
python app.py --zip path/to/dataset.zip --save_config path/to/config.json --mode train
```

## Available Models

The framework supports various model architectures:

- mobilenetv2 (default, smallest and fastest)
- resnet50
- resnet101
- resnet152
- vgg16
- vgg19
- densenet121
- densenet169
- densenet201
- efficientnetb0 - efficientnetb7

## Examples

### Train a model with custom settings

```bash
python app.py --zip dataset.zip --model resnet50 --epochs 20 --batch_size 16 --img_size 299 299
```

### Launch app only (skips training)

```bash
python app.py --config config.json --mode app
```

### Train only (skips app launch)

```bash
python app.py --zip dataset.zip --mode train
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 