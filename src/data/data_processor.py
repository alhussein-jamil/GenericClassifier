import numpy as np
from pathlib import Path
from typing import Tuple
import logging
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torchvision import transforms
from PIL import Image

from src.schemas import Dataset, DataConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def preprocess_image(img_path: Path, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Load and preprocess a single image.
    
    Args:
        img_path: Path to the image file.
        target_size: Target size (height, width) for resizing.
        
    Returns:
        Preprocessed image as a numpy array.
    """
    # Read the image
    img = Image.open(img_path).convert('RGB')
    
    # Resize the image
    img = img.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(img) / 255.0
    
    return img_array


def create_data_generators(dataset: Dataset, config: DataConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        dataset: Dataset object with train and validation splits.
        config: DataConfig object with batch size and augmentation settings.
        
    Returns:
        Tuple of (train_loader, val_loader).
    """
    # Get class mapping
    num_classes = dataset.metadata.num_classes
    
    logger.info(f"Creating data loaders with {num_classes} classes")
    
    # Create augmentation transformations for training
    if config.augmentation:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(config.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(config.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Create transformations for validation (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(config.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create custom datasets
    train_dataset = ImageDataset(
        images=dataset.train_images,
        num_classes=num_classes,
        transform=train_transform
    )
    
    val_dataset = ImageDataset(
        images=dataset.val_images,
        num_classes=num_classes,
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


class ImageDataset(TorchDataset):
    """Custom dataset for our image data."""
    
    def __init__(
        self,
        images: list,
        num_classes: int,
        transform=None
    ):
        """
        Initialize the dataset.
        
        Args:
            images: List of ImageInfo objects.
            num_classes: Number of classes for one-hot encoding.
            transform: Torchvision transformations to apply.
        """
        self.images = images
        self.num_classes = num_classes
        self.transform = transform
    
    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.images)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        # Get image info
        img_info = self.images[index]
        
        # Read image
        image = Image.open(img_info.path).convert('RGB')
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        
        # Create label tensor
        label = torch.tensor(img_info.class_id, dtype=torch.long)
        
        return image, label 