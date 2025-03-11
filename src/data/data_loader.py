import zipfile
import random
import shutil
from pathlib import Path
import logging

from src.schemas import Dataset, ImageInfo, ClassInfo, DatasetMetadata, DataConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_zip_dataset(config: DataConfig) -> Path:
    """
    Extract the dataset from a zip file.
    
    Args:
        config: DataConfig object with zip_path and extract_dir.
        
    Returns:
        Path to the extracted dataset.
    """
    zip_path = config.zip_path
    extract_dir = config.extract_dir
    
    # Create extraction directory if it doesn't exist
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Clean extraction directory if it exists
        for item in extract_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    
    logger.info(f"Extracting {zip_path} to {extract_dir}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        logger.info("Extraction completed successfully")
    except Exception as e:
        logger.error(f"Error extracting zip file: {e}")
        raise
    
    # Find the root directory of the dataset
    # If the zip has a parent directory, we need to find it
    contents = list(extract_dir.iterdir())
    
    # If there's only one item and it's a directory, use that
    if len(contents) == 1 and contents[0].is_dir():
        dataset_dir = contents[0]
    else:
        dataset_dir = extract_dir
    
    logger.info(f"Dataset directory: {dataset_dir}")
    return dataset_dir


def is_valid_image(filepath: Path) -> bool:
    """
    Check if a file is a valid image based on its extension.
    
    Args:
        filepath: Path to the file.
        
    Returns:
        True if the file is a valid image, False otherwise.
    """
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    return filepath.suffix.lower() in valid_extensions


def load_dataset(config: DataConfig) -> Dataset:
    """
    Load the dataset from the extracted zip file and create a Dataset object.
    
    Args:
        config: DataConfig object with extraction settings.
        
    Returns:
        Dataset object with train and validation splits.
    """
    # Extract the dataset
    dataset_dir = extract_zip_dataset(config)
    
    # Collect all class directories
    class_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")
    
    # Map class names to IDs
    class_mapping = {class_dir.name: idx for idx, class_dir in enumerate(sorted(class_dirs))}
    inv_class_mapping = {idx: name for name, idx in class_mapping.items()}
    
    # Count samples per class
    class_counts = {}
    all_images = []
    
    # Collect all valid images
    for class_dir in class_dirs:
        class_name = class_dir.name
        class_id = class_mapping[class_name]
        
        # Find all image files in the class directory
        image_files = [
            f for f in class_dir.glob("**/*") 
            if f.is_file() and is_valid_image(f)
        ]
        
        class_counts[class_name] = len(image_files)
        
        # Create ImageInfo objects for each image
        for img_path in image_files:
            all_images.append(ImageInfo(
                path=img_path,
                class_name=class_name,
                class_id=class_id
            ))
    
    # Shuffle the images
    if config.shuffle:
        random.shuffle(all_images)
    
    # Split into train and validation sets
    split_idx = int(len(all_images) * (1 - config.val_split))
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    logger.info(f"Split dataset: {len(train_images)} training samples, {len(val_images)} validation samples")
    
    # Create class info objects
    classes = []
    for class_name, count in class_counts.items():
        classes.append(ClassInfo(
            name=class_name,
            id=class_mapping[class_name],
            sample_count=count
        ))
    
    # Create dataset metadata
    metadata = DatasetMetadata(
        name=dataset_dir.name,
        num_classes=len(class_dirs),
        total_samples=len(all_images),
        class_distribution=class_counts,
        class_mapping=class_mapping,
        inv_class_mapping=inv_class_mapping
    )
    
    # Create and return the Dataset object
    return Dataset(
        train_images=train_images,
        val_images=val_images,
        classes=classes,
        metadata=metadata
    ) 