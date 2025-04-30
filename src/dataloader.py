import os
import torch
import random 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import cv2
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.ops import masks_to_boxes

from torchvision.transforms.v2 import functional as F_v2
from skimage import measure
from torchvision.transforms import v2 as T
from torchvision.transforms import functional as TF
import torch.nn.functional as F
from torchvision.transforms.functional import convert_image_dtype
import torch.nn as nn

from dataset_manager import CamVidDatasetManager



#Intialise Dataset
#Inherits from PyTorch's dataset class
#Takes image paths, mask paths, class mapping, target classes and transforms 
#Creates mapping from class names to indices (0 = background)

class CamVidInstanceDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, mask_paths, class_map, target_classes, transforms=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
        self.class_map = class_map  # RGB values for each class
        self.target_classes = target_classes  # List of class names
        
        # Create a mapping from class names to label indices (starting from 1, as 0 is background)
        self.class_to_idx = {cls_name: i+1 for i, cls_name in enumerate(target_classes) if cls_name != 'Void'}
    
    def __getitem__(self, idx):
        """
        Loads image and its corresponding mask at specified index
        Converts image to floating point format
        Converts RGB mask to class indices using helper method
        """
        # Load image and mask
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Read image using OpenCV
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            raise RuntimeError(f"Failed to read image at {img_path}")
        
        # Convert BGR to RGB and rearrange dimensions for PyTorch format
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img_cv.transpose(2, 0, 1))
        
        # Convert to float and normalize
        img = F_v2.convert_image_dtype(img, dtype=torch.float)
        
        # Read mask using OpenCV
        mask_cv = cv2.imread(mask_path)
        if mask_cv is None:
            raise RuntimeError(f"Failed to read mask at {mask_path}")
        
        # Convert BGR to RGB and rearrange dimensions for PyTorch format
        mask_cv = cv2.cvtColor(mask_cv, cv2.COLOR_BGR2RGB)
        mask_rgb = torch.from_numpy(mask_cv.transpose(2, 0, 1))
        
        # Convert RGB mask to class indices
        mask_idx = self._rgb_to_class_idx(mask_rgb)
        
        # Initialize lists to store instance masks, boxes and labels
        instance_masks = []
        boxes = []
        labels = []
        
        # Process each class (except Void)
        for class_name, class_idx in self.class_to_idx.items():
            # Create binary mask for this class
            binary_mask = (mask_idx == class_idx)
            
            if not binary_mask.any():
                continue  # Skip if no pixels of this class
                
            # Find connected components (instances) of this class
            # We need to convert to numpy for connected components analysis
            binary_mask_np = binary_mask.numpy().astype(np.uint8)
            
            # Use scikit-image to identify connected components
            from skimage import measure
            components = measure.label(binary_mask_np, connectivity=2)
            
            # Process each instance
            for instance_id in range(1, components.max() + 1):
                # Create binary mask for this instance
                instance_mask = torch.tensor(components == instance_id, dtype=torch.uint8)
                
                # Skip small instances (optional, helps filter noise)
                if instance_mask.sum() < 100:  # Minimum area threshold
                    continue
                    
                # Get bounding box
                box = masks_to_boxes(instance_mask.unsqueeze(0)).squeeze(0)
                
                # Skip if box is invalid
                if box[0] >= box[2] or box[1] >= box[3]:
                    continue
                    
                # Add to lists
                instance_masks.append(instance_mask)
                boxes.append(box)
                labels.append(torch.tensor(class_idx, dtype=torch.int64))
        
        # If no valid instances found, create a dummy instance
        if len(instance_masks) == 0:
            instance_masks = torch.zeros((1, *binary_mask.shape), dtype=torch.uint8)
            boxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
            labels = torch.tensor([0], dtype=torch.int64)  # Background
        else:
            # Stack all instances
            instance_masks = torch.stack(instance_masks)
            boxes = torch.stack(boxes)
            labels = torch.stack(labels)
        
        # Create target dictionary with all the instance information:
        # Bounding boxes(x_min, y_min, x_max, y_max)
        # Class labels
        # Instance masks
        # Image ID
        # Box areas
        # 'iscrowd' flags
        target = {}

        target["boxes"] = tv_tensors.BoundingBoxes(
            boxes, format="XYXY", canvas_size=F_v2.get_size(img)
        )
        
        target["labels"] = labels
        target["masks"] = tv_tensors.Mask(instance_masks)
        target["image_id"] = torch.tensor(idx)
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros_like(labels)
        
        # Apply transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return img, target
    
    def _rgb_to_class_idx(self, mask_rgb):
        """Convert RGB mask to class indices"""
        # Initialize with zeros (background)
        h, w = mask_rgb.shape[1], mask_rgb.shape[2]
        mask_idx = torch.zeros((h, w), dtype=torch.int64)
        
        # For each class, set corresponding pixels
        for class_name, class_idx in self.class_to_idx.items():
            # Find RGB for this class
            class_rgb = torch.tensor(self.class_map[self.target_classes.index(class_name)])
            
            # Create binary mask for this class
            # Compare each channel
            r_match = mask_rgb[0] == class_rgb[0]
            g_match = mask_rgb[1] == class_rgb[1]
            b_match = mask_rgb[2] == class_rgb[2]
            
            # Combine matches
            match = r_match & g_match & b_match
            
            # Set class index
            mask_idx[match] = class_idx
            
        return mask_idx


    def prepare_coco_format(self):
        
        """Prepare COCO-format annotations for evaluation"""
        
        # Create COCO-style dataset structure
        coco_dataset = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Add categories (classes)
        for idx, class_name in enumerate([cls for cls in self.target_classes if cls != 'Void']):
            coco_dataset["categories"].append({
                "id": idx + 1,  # COCO uses 1-indexed class IDs
                "name": class_name,
                "supercategory": "object"
            })
        
        # Add images and annotations
        ann_id = 1  # Annotation ID counter (COCO uses 1-indexed annotation IDs)
        
        for img_id in range(len(self.image_paths)):
            # Add image info
            img_path = self.image_paths[img_id]
            
            # Debug information
            print(f"Processing image {img_id+1}/{len(self.image_paths)}: {img_path}")
            
            # Read image with OpenCV
            img_cv = cv2.imread(img_path)
            if img_cv is None:
                print(f"Warning: Could not read image at {img_path}")
                continue  # Skip this image
                
            h, w = img_cv.shape[0], img_cv.shape[1]
            
            coco_dataset["images"].append({
                "id": img_id,
                "file_name": os.path.basename(img_path),
                "height": h,
                "width": w
            })
            
            # Get masks and process instances
            mask_path = self.mask_paths[img_id]
            
            # Read mask with OpenCV
            mask_cv = cv2.imread(mask_path)
            if mask_cv is None:
                print(f"Warning: Could not read mask at {mask_path}")
                continue  # Skip this image
                
            # Convert BGR to RGB and rearrange dimensions for PyTorch format
            mask_rgb = torch.from_numpy(cv2.cvtColor(mask_cv, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
            
            # Convert RGB mask to class indices
            mask_idx = self._rgb_to_class_idx(mask_rgb)
            
            # For each class, find instances
            for class_name, class_idx in self.class_to_idx.items():
                binary_mask = (mask_idx == class_idx).numpy().astype(np.uint8)
                
                if not np.any(binary_mask):
                    continue
                
                # Find connected components
                components = measure.label(binary_mask, connectivity=2)
                
                # Process each instance
                for instance_id in range(1, components.max() + 1):
                    instance_mask = (components == instance_id).astype(np.uint8)
                    
                    # Skip small instances
                    if np.sum(instance_mask) < 100:
                        continue
                    
                    # Calculate bounding box [x, y, width, height]
                    pos = np.where(instance_mask)
                    if len(pos[0]) == 0 or len(pos[1]) == 0:  # Safety check
                        continue
                        
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])
                    width = xmax - xmin + 1
                    height = ymax - ymin + 1
                    
                    # Skip invalid boxes
                    if width <= 0 or height <= 0:
                        continue
                    
                    # Create annotation
                    coco_dataset["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": class_idx,
                        "bbox": [float(xmin), float(ymin), float(width), float(height)],
                        "area": float(np.sum(instance_mask)),
                        "segmentation": self._mask_to_polygon(instance_mask),
                        "iscrowd": 0
                    })
                    
                    ann_id += 1
        
        print(f"COCO dataset prepared with {len(coco_dataset['images'])} images and {len(coco_dataset['annotations'])} annotations")
        return coco_dataset
        
    def _mask_to_polygon(self, mask):
        """Convert a binary mask to COCO polygon format"""
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []
        
        for contour in contours:
            # Convert each contour to the format expected by COCO
            if len(contour) >= 6:  # Need at least 3 points (6 coordinates) to make a polygon
                segmentation.append(contour.flatten().tolist())
        
        # If no valid polygons, create a simple dummy polygon to avoid errors
        if not segmentation:
            h, w = mask.shape
            segmentation = [[0, 0, 0, h-1, w-1, h-1, w-1, 0]]
        
        return segmentation
    
    def __len__(self):
        return len(self.image_paths)
    


#Histogram-based Augmnetations 

class HistogramBasedAdaptiveAugmentation:
    def __init__(self, 
                 dark_percentile_threshold=0.6,  # If 60% of pixels are below 0.3, image is dark
                 dark_pixel_threshold=0.3,
                 dark_brightness=1.75, 
                 dark_gamma=0.51,
                 medium_percentile_threshold=0.4, # For medium darkness
                 medium_brightness=1.4,
                 medium_gamma=0.65):
        """
        Applies brightness and gamma adjustments based on histogram analysis of pixel values.
        
        Args:
            dark_percentile_threshold: If the percentage of dark pixels exceeds this, apply strong enhancement
            dark_pixel_threshold: Pixels below this value are considered "dark"
            dark_brightness: Brightness factor for very dark images
            dark_gamma: Gamma value for very dark images
            medium_percentile_threshold: If percentage exceeds this but not dark_threshold, apply medium enhancement
            medium_brightness: Brightness factor for medium dark images
            medium_gamma: Gamma value for medium dark images
        """
        self.dark_percentile_threshold = dark_percentile_threshold
        self.dark_pixel_threshold = dark_pixel_threshold
        self.dark_brightness = dark_brightness
        self.dark_gamma = dark_gamma
        self.medium_percentile_threshold = medium_percentile_threshold
        self.medium_brightness = medium_brightness
        self.medium_gamma = medium_gamma
    
    def __call__(self, img, target=None):
        # Convert to numpy for histogram calculation
        img_np = img.numpy()
        
        # Calculate histogram
        hist, bins = np.histogram(img_np.flatten(), bins=10, range=(0, 1))
        
        # Calculate percentage of dark pixels
        dark_bin_idx = int(self.dark_pixel_threshold * 10)  # Convert threshold to bin index
        dark_pixel_percentage = np.sum(hist[:dark_bin_idx]) / np.sum(hist)
        
        # Apply adjustments based on darkness assessment
        if dark_pixel_percentage > self.dark_percentile_threshold:
            # Very dark image - apply stronger adjustment
            img = TF.adjust_brightness(img, self.dark_brightness)
            img = TF.adjust_gamma(img, self.dark_gamma)
        elif dark_pixel_percentage > self.medium_percentile_threshold:
            # Moderately dark image - apply medium adjustment
            img = TF.adjust_brightness(img, self.medium_brightness)
            img = TF.adjust_gamma(img, self.medium_gamma)
        
        if target is not None:
            return img, target
        return img

def get_transform(train=True):
    """
    Creates a data transformation pipeline for image preprocessing and augmentation.
    
    Args:
        train (bool): Indicates whether the transformation is for training.
                      If True, data augmentation is applied.
    
    Returns:
        T.Compose: A composed transformation pipeline.
    """
    transforms = []
    
    if train:
        # Add horizontal flipping with 50% probability
        transforms.append(T.RandomHorizontalFlip(0.5))
        
        # Add histogram-based adaptive brightness and gamma adjustment
        # This will only enhance images that need it based on their histograms
        transforms.append(HistogramBasedAdaptiveAugmentation(
            dark_percentile_threshold=0.6,  # 60% of pixels need to be dark to trigger strong enhancement
            dark_pixel_threshold=0.3,       # Pixels below 0.3 are considered "dark"
            dark_brightness=1.75,           # Strong brightness enhancement for very dark images
            dark_gamma=0.51,                # Strong gamma enhancement for very dark images
            medium_percentile_threshold=0.4, # Medium threshold
            medium_brightness=1.4,          # Moderate brightness enhancement
            medium_gamma=0.65               # Moderate gamma enhancement
        ))
    
    # Convert to float and normalize
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    
    return T.Compose(transforms)

def visualize_histogram_based_augmentation(dataset_manager, num_samples=6, results_dir="results"):
    """
    Visualizes the histogram-based adaptive augmentation using images from the dataset.
    
    Args:
        dataset_manager: Instance of CamVidDatasetManager
        num_samples (int): Number of images to sample
        results_dir (str): Directory to save results
    """
    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")
    
    # Use the train directory from dataset manager
    image_dir = dataset_manager.train_dir
    
    # Create a figure - now with 4 columns to include histogram
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    
    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Sample images if there are more than requested
    if len(image_files) > num_samples:
        image_files = random.sample(image_files, num_samples)
    
    # Column titles
    col_titles = ['Original Image', 'Brightness Histogram', 'Classification', 'Augmented (if applicable)']
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=14)
    
    # Create the histogram-based augmenter
    augmenter = HistogramBasedAdaptiveAugmentation()
    
    for i, img_file in enumerate(image_files):
        # Read image
        img_path = os.path.join(image_dir, img_file)
        try:
            img = read_image(img_path)
            # Convert to float [0,1]
            img = img.float() / 255.0
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            continue
        
        # Convert to numpy for histogram calculation
        img_np = img.numpy()
        
        # Calculate histogram
        hist, bins = np.histogram(img_np.flatten(), bins=10, range=(0, 1))
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        # Calculate percentage of dark pixels
        dark_bin_idx = int(augmenter.dark_pixel_threshold * 10)
        dark_pixel_percentage = np.sum(hist[:dark_bin_idx]) / np.sum(hist)
        
        # Determine category and apply augmentation if needed
        if dark_pixel_percentage > augmenter.dark_percentile_threshold:
            category = f"Dark Image\n{dark_pixel_percentage:.1%} dark pixels\n(threshold: {augmenter.dark_percentile_threshold:.1%})"
            augmented = TF.adjust_gamma(TF.adjust_brightness(img.clone(), 
                                                            augmenter.dark_brightness), 
                                        augmenter.dark_gamma)
            adjustment_text = f"Applied:\nBrightness: {augmenter.dark_brightness}\nGamma: {augmenter.dark_gamma}"
        elif dark_pixel_percentage > augmenter.medium_percentile_threshold:
            category = f"Medium Dark\n{dark_pixel_percentage:.1%} dark pixels\n(threshold: {augmenter.medium_percentile_threshold:.1%})"
            augmented = TF.adjust_gamma(TF.adjust_brightness(img.clone(), 
                                                            augmenter.medium_brightness), 
                                        augmenter.medium_gamma)
            adjustment_text = f"Applied:\nBrightness: {augmenter.medium_brightness}\nGamma: {augmenter.medium_gamma}"
        else:
            category = f"Bright Image\n{dark_pixel_percentage:.1%} dark pixels\n(below thresholds)"
            augmented = img.clone()  # No augmentation for bright images
            adjustment_text = "No adjustments needed"
        
        # Display original image
        axes[i, 0].imshow(img.permute(1, 2, 0))
        axes[i, 0].set_title(f"Original: {img_file}", fontsize=10)
        axes[i, 0].axis('off')
        
        # Display histogram with dark pixel threshold marked
        axes[i, 1].bar(bin_centers, hist, width=0.08)
        axes[i, 1].axvline(x=augmenter.dark_pixel_threshold, color='r', linestyle='--', 
                          label=f'Dark threshold: {augmenter.dark_pixel_threshold}')
        axes[i, 1].fill_between(bin_centers[:dark_bin_idx], hist[:dark_bin_idx], 
                               alpha=0.3, color='blue', label='Dark pixels')
        axes[i, 1].set_xlabel('Pixel Value')
        axes[i, 1].set_ylabel('Frequency')
        axes[i, 1].legend(fontsize=8)
        
        # Display classification and percentage
        axes[i, 2].text(0.5, 0.5, category, 
                      horizontalalignment='center', 
                      verticalalignment='center',
                      fontsize=12, transform=axes[i, 2].transAxes)
        
        # Add adjustment details at the bottom
        axes[i, 2].text(0.5, 0.2, adjustment_text,
                      horizontalalignment='center',
                      verticalalignment='center',
                      fontsize=10, transform=axes[i, 2].transAxes)
        axes[i, 2].axis('off')
        
        # Display augmented image
        axes[i, 3].imshow(augmented.permute(1, 2, 0))
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    # Create the output file path inside the results directory
    output_path = os.path.join(results_dir, 'histogram_based_augmentation.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved visualization to {output_path}")
    
    plt.show()


# Initialize the dataset manager
dataset_manager = CamVidDatasetManager()

# Call the visualization function with the dataset manager
visualize_histogram_based_augmentation(dataset_manager, num_samples=6, results_dir="results")



#Weighted Loss

def calculate_class_weights(pixel_percentages, method='inverse_square_root'):
    """
    Calculate class weights based on pixel distribution.
    
    Args:
        pixel_percentages: List of pixel percentages for each class
        method: Weighting method ('inverse', 'inverse_square_root', or 'effective_samples')
    
    Returns:
        Tensor of class weights
    """
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-6
    pixel_fractions = np.array(pixel_percentages) + epsilon
    
    if method == 'inverse':
        # Simple inverse frequency weighting
        weights = 1.0 / pixel_fractions
    elif method == 'inverse_square_root':
        # Square root helps moderate extreme weights
        weights = 1.0 / np.sqrt(pixel_fractions)
    elif method == 'effective_samples':
        # Effective number of samples weighting (from "Class-Balanced Loss")
        beta = 0.9999
        weights = (1 - beta) / (1 - beta ** pixel_fractions)
    
    # Normalize weights so they sum to the number of classes
    normalized_weights = weights / weights.sum() * len(weights)
    return torch.tensor(normalized_weights, dtype=torch.float32)

# Your class pixel percentages
pixel_percentages = [0.62, 0.12, 0.09, 0.0001, 0.04]  # MotorcycleScooter value adjusted for calculation
class_names = ['Car', 'Pedestrian', 'Bicyclist', 'MotorcycleScooter', 'Truck_Bus']

# Calculate weights using inverse square root method (handles extreme imbalances better)
class_weights = calculate_class_weights(pixel_percentages, method='inverse_square_root')


class WeightedRCNNLoss(nn.Module):
    def __init__(self, class_weights):
        super(WeightedRCNNLoss, self).__init__()
        self.class_weights = class_weights
        
    def forward(self, classification_logits, true_classes):
        """
        Apply class weights to the classification loss component of Mask R-CNN.
        
        Args:
            classification_logits: Model predictions (N, num_classes)
            true_classes: Ground truth class indices (N)
            
        Returns:
            Weighted cross-entropy loss
        """
        # Create a weight tensor for each example based on its class
        weights = self.class_weights[true_classes]
        
        # Calculate standard cross-entropy loss
        ce_loss = F.cross_entropy(classification_logits, true_classes, reduction='none')
        
        # Apply the weights
        weighted_loss = ce_loss * weights
        
        return weighted_loss.mean()
    
def weighted_mask_loss(pred_masks, true_masks, class_weights):
    """
    Apply class weights to segmentation mask loss.
    
    Args:
        pred_masks: Predicted masks (B, C, H, W) where C is number of classes
        true_masks: Target masks (B, H, W) with class indices
        class_weights: Weights for each class
        
    Returns:
        Weighted segmentation loss
    """
    # Standard cross entropy for segmentation
    loss = F.cross_entropy(pred_masks, true_masks, reduction='none')
    
    # Create a weight map based on the class at each pixel
    pixel_weights = torch.ones_like(true_masks, dtype=torch.float32)
    
    # For each class, set the corresponding weight
    for cls_idx, weight in enumerate(class_weights):
        pixel_weights[true_masks == cls_idx] = weight
        
    # Apply weights to the loss
    weighted_loss = loss * pixel_weights
    
    return weighted_loss.mean()

def get_model_instance_segmentation(num_classes):
    # Calculate class weights
    pixel_percentages = [0.62, 0.12, 0.09, 0.0001, 0.04]  # Adjusted for calculation
    class_weights = calculate_class_weights(pixel_percentages)
    
    # Create the model
    model = CustomMaskRCNN(num_classes=num_classes, class_weights=class_weights)
    
    # Customize ROI heads for class balancing
    model.model.roi_heads.batch_size_per_image = 512  
    model.model.roi_heads.positive_fraction = 0.5  # Increase from default 0.25
    
    return model




#Combined Sampler 
def create_combined_sampler(dataset, epoch, max_epochs):
    """
    Creates a combined sampler that incorporates:
    - Instance-aware weighting
    - Curriculum learning
    - Class distribution awareness
    
    Args:
        dataset: The training dataset
        epoch: Current training epoch
        max_epochs: Maximum number of epochs
        
    Returns:
        A WeightedRandomSampler
    """
    # Initialize weights
    weights = torch.ones(len(dataset))
    
    # Class importance factors (higher for rare classes)
    class_importance = {
        1: 2.0,   # Bicyclist
        2: 1.0,   # Car
        3: 15.0,  # MotorcycleScooter (base weight, will be adjusted by curriculum)
        4: 1.0,   # Pedestrian
        5: 3.0    # Truck_Bus
    }
    
    # Calculate curriculum phase (0 to 1)
    phase = min(1.0, epoch / (max_epochs * 0.7))
    
    # Adjust MotorcycleScooter weight based on curriculum phase
    # Start with a moderate weight and gradually increase
    curriculum_multiplier = 1.0 + 9.0 * phase  # Grows from 1x to 10x
    class_importance[3] *= curriculum_multiplier
    
    print(f"Epoch {epoch}/{max_epochs}: MotorcycleScooter weight = {class_importance[3]:.2f}")
    
    # Calculate instance-based weights
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        
        # Get labels and areas
        labels = target['labels'].cpu()
        areas = target['area'].cpu() if 'area' in target else None
        
        # Calculate image weight based on instances
        img_weight = 1.0
        
        # Record if this image has any rare classes
        has_rare_class = False
        
        for i, label in enumerate(labels):
            label_idx = label.item()
            
            # Skip background
            if label_idx == 0:
                continue
                
            # Add weight from each instance, based on class importance
            instance_weight = class_importance.get(label_idx, 1.0)
            
            # If we have area information, adjust weight based on instance size
            if areas is not None:
                area = areas[i].item()
                size_factor = 1.0
                
                if area < 1000:  # Small instance
                    size_factor = 1.5  # Small instances are harder to detect
                elif area > 10000:  # Very large instance
                    size_factor = 0.8  # Large instances are easier
                    
                instance_weight *= size_factor
            
            # Add to the image weight
            img_weight += instance_weight
            
            # Mark rare classes
            if label_idx == 3:  # MotorcycleScooter
                has_rare_class = True
        
        # Give additional boost to images with rare classes
        # This ensures they're sampled even more frequently
        if has_rare_class:
            img_weight *= 2.0
            
        weights[idx] = img_weight
    
    # Normalize weights
    weights = weights / weights.sum() * len(weights)
    
    return torch.utils.data.WeightedRandomSampler(
        weights.tolist(),
        num_samples=len(dataset),
        replacement=True
    )