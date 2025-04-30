import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import json
import tempfile

from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.io import read_image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_util

from model import CustomMaskRCNN
from dataloader import CamVidInstanceDataset
from dataset_manager import CamVidDatasetManager


# Import model definition and dataset class
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision



dataset_manager = CamVidDatasetManager()

def get_model_instance_segmentation(num_classes):
    """
    Create a Mask R-CNN model with a ResNet-50 backbone.

    Args:
        num_classes (int): Number of classes (including background)

    Returns:
        model: The Mask R-CNN model
    """
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model


def get_transform():
    """
    Creates a transformation pipeline for test images.
    """
    transforms = []
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def load_model(model_path, num_classes):
    """
    Load a trained model from disk.

    Args:
        model_path (str): Path to the saved model
        num_classes (int): Number of classes (including background)

    Returns:
        model: The loaded model
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Create a temporary CustomMaskRCNN to load the state dict
    temp_model = CustomMaskRCNN(num_classes=num_classes, pretrained=True)

    # Load the state dict into our custom model
    state_dict = torch.load(model_path, map_location=device)

    # Check if state dict keys have "model." prefix
    if all(k.startswith("model.") for k in state_dict.keys()):
        # Create a new state dict with modified keys
        new_state_dict = {}
        for k, v in state_dict.items():
            # Extract the part after 'model.'
            name = k[6:]  # Skip first 6 characters ("model.")
            new_state_dict[name] = v
        temp_model.model.load_state_dict(new_state_dict)
    else:
        # If no prefix, load directly
        temp_model.model.load_state_dict(state_dict)

    # Extract the inner model
    model = temp_model.model

    model.to(device)
    model.eval()
    return model, device


def get_colormaps():
    """
    Create color maps for visualization.

    Returns:
        dict: A dictionary mapping class indices to colors
    """
    # Define a color for each class (including background)
    colors = {
        0: (0, 0, 0),        # Background (black)
        1: (0, 255, 0),      # Bicyclist (green)
        2: (255, 0, 0),      # Car (red)
        3: (255, 0, 255),    # MotorcycleScooter (magenta)
        4: (255, 255, 0),    # Pedestrian (yellow)
        5: (0, 255, 255),    # Truck_Bus (cyan)
    }
    return colors


def visualize_prediction(image, output, threshold=0.5, colors=None, class_names=None):
    """
    Visualize predictions on an image.

    Args:
        image (Tensor): The input image
        output (dict): The model output containing boxes, labels, scores, and masks
        threshold (float): Score threshold for showing predictions
        colors (dict): Dictionary mapping class indices to colors
        class_names (list): List of class names

    Returns:
        PIL.Image: The image with visualized predictions
    """
    if colors is None:
        colors = get_colormaps()

    if class_names is None:
        class_names = ['Background', 'Bicyclist', 'Car', 'MotorcycleScooter', 'Pedestrian', 'Truck_Bus']

    # Get predictions
    boxes = output['boxes'].cpu()
    labels = output['labels'].cpu()
    scores = output['scores'].cpu()
    masks = output['masks'].cpu()

    # Keep only predictions with score > threshold
    keep = scores > threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    masks = masks[keep]

    # Convert image to uint8 format for visualization
    image = (image * 255).byte()

    # Draw boxes
    box_colors = [colors[label.item()] for label in labels]
    labeled_boxes = [f"{class_names[label]}: {score:.2f}" for label, score in zip(labels, scores)]

    # Draw bounding boxes
    image_with_boxes = draw_bounding_boxes(
        image,
        boxes=boxes,
        labels=labeled_boxes,
        colors=box_colors,
        width=2
    )

    # Convert masks to boolean format and binary threshold
    if len(masks) > 0:
        # Convert to boolean - mask predictions are typically float values between 0 and 1
        # threshold them at 0.5 to get binary masks
        masks = (masks > 0.5).bool()

        # Draw segmentation masks
        image_with_masks = draw_segmentation_masks(
            image_with_boxes,
            masks=masks.squeeze(1),
            alpha=0.5,
            colors=[color for color in box_colors]
        )
    else:
        image_with_masks = image_with_boxes

    # Convert to PIL Image for display
    pil_image = T.ToPILImage()(image_with_masks)
    return pil_image


def evaluate_model_on_test_set(model, test_dataset, device, num_samples=10, threshold=0.5, save_dir='results'):
    """
    Evaluate model on test set and visualize results.

    Args:
        model: The Mask R-CNN model
        test_dataset: The test dataset
        device: The device to run inference on
        num_samples (int): Number of samples to visualize
        threshold (float): Score threshold for showing predictions
        save_dir (str): Directory to save visualizations
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Get colormap and class names
    colors = get_colormaps()
    class_names = ['Background', 'Bicyclist', 'Car', 'MotorcycleScooter', 'Pedestrian', 'Truck_Bus']

    # Select random samples from test set
    indices = np.random.choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)

    # Create a 3x3 grid for each sample (original, ground truth, prediction)
    for i, idx in enumerate(indices):
        # Get sample from test dataset
        image, target = test_dataset[idx]
        image_orig = image.clone()

        # Move image to device
        image = image.to(device)

        # Perform inference
        with torch.no_grad():
            prediction = model([image])[0]

        # Visualize original image, ground truth, and prediction
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original image
        axes[0].imshow(T.ToPILImage()(image_orig))
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # Ground truth
        # Create a visualization of ground truth
        gt_image = image_orig.clone()
        gt_boxes = target['boxes'].cpu()
        gt_labels = target['labels'].cpu()
        gt_masks = target['masks'].cpu()

        # Convert ground truth to visualization format
        gt_colors = [colors[label.item()] for label in gt_labels]
        gt_labeled_boxes = [f"{class_names[label]}" for label in gt_labels]

        # Draw ground truth boxes
        gt_image_with_boxes = draw_bounding_boxes(
            (gt_image * 255).byte(),
            boxes=gt_boxes,
            labels=gt_labeled_boxes,
            colors=gt_colors,
            width=2
        )

        # Draw ground truth masks
        # Convert uint8 masks to boolean masks
        gt_masks_bool = gt_masks.bool()
        gt_image_with_masks = draw_segmentation_masks(
            gt_image_with_boxes,
            masks=gt_masks_bool,
            alpha=0.5,
            colors=[color for color in gt_colors]
        )

        # Show ground truth visualization
        axes[1].imshow(T.ToPILImage()(gt_image_with_masks))
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')

        # Model prediction
        prediction_vis = visualize_prediction(
            image_orig,
            prediction,
            threshold=threshold,
            colors=colors,
            class_names=class_names
        )

        # Show prediction visualization
        axes[2].imshow(prediction_vis)
        axes[2].set_title(f"Prediction (threshold={threshold})")
        axes[2].axis('off')

        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"sample_{i}.png"), dpi=200, bbox_inches='tight')
        plt.close()

        # Print progress
        print(f"Processed {i+1}/{len(indices)} samples")

    print(f"Visualizations saved to {save_dir}")


def compute_metrics(model, test_loader, device):
    """
    Compute metrics on the test set.

    Args:
        model: The Mask R-CNN model
        test_loader: DataLoader for the test set
        device: The device to run inference on

    Returns:
        dict: Dictionary containing precision and recall metrics
    """
    # Set model to evaluation mode
    model.eval()

    # Initialize counters for precision and recall calculation
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    # IoU threshold for considering a prediction correct
    iou_threshold = 0.5

    # Confidence threshold for considering a prediction
    score_threshold = 0.5

    # Process each batch
    for images, targets in test_loader:
        # Move images to device
        images = [img.to(device) for img in images]

        # Get predictions
        with torch.no_grad():
            outputs = model(images)

        # Evaluate each image in the batch
        for output, target in zip(outputs, targets):
            # Get predictions above threshold
            pred_boxes = output['boxes'][output['scores'] > score_threshold].cpu()
            pred_labels = output['labels'][output['scores'] > score_threshold].cpu()

            # Get ground truth
            gt_boxes = target['boxes'].cpu()
            gt_labels = target['labels'].cpu()

            # Match predictions to ground truth
            matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)

            # Count true positives and false positives
            for pred_idx, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                # Calculate IoU with all ground truth boxes
                ious = box_iou(pred_box.unsqueeze(0), gt_boxes)[0]

                # Find best matching ground truth box
                best_iou, best_gt_idx = ious.max(0)

                # Check if it's a good match (IoU > threshold and same class)
                if (best_iou > iou_threshold and
                    pred_label == gt_labels[best_gt_idx] and
                    not matched_gt[best_gt_idx]):
                    # It's a true positive
                    total_true_positives += 1
                    matched_gt[best_gt_idx] = True
                else:
                    # It's a false positive
                    total_false_positives += 1

            # Count false negatives (ground truth boxes that weren't matched)
            total_false_negatives += (matched_gt == False).sum().item()

    # Calculate precision and recall
    precision = total_true_positives / (total_true_positives + total_false_positives + 1e-6)
    recall = total_true_positives / (total_true_positives + total_false_negatives + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': total_true_positives,
        'false_positives': total_false_positives,
        'false_negatives': total_false_negatives
    }


def box_iou(boxes1, boxes2):
    """
    Compute intersection over union between boxes.

    Args:
        boxes1 (Tensor[N, 4]): First set of boxes in format (x1, y1, x2, y2)
        boxes2 (Tensor[M, 4]): Second set of boxes in format (x1, y1, x2, y2)

    Returns:
        Tensor[N, M]: IoU values for each pair of boxes
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Get coordinates of intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    # Calculate area of intersection
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    intersection = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    # Calculate IoU
    union = area1[:, None] + area2 - intersection
    iou = intersection / (union + 1e-6)

    return iou


def visualize_metrics(metrics, save_dir="evaluation_results"):
    """
    Create visualizations of evaluation metrics.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        save_dir: Directory to save visualization plots
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Plot per-class AP for bbox and segmentation
    class_names = list(metrics['bbox_per_class'].keys())
    bbox_ap = [metrics['bbox_per_class'][cls] for cls in class_names]
    segm_ap = [metrics['segm_per_class'][cls] for cls in class_names]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bbox_bars = ax.bar(x - width/2, bbox_ap, width, label='Detection (bbox) AP@0.5', color='steelblue')
    segm_bars = ax.bar(x + width/2, segm_ap, width, label='Segmentation AP@0.5', color='darkorange')
    
    ax.set_title('Per-Class AP@0.5', fontsize=16)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Average Precision', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
                        
    add_value_labels(bbox_bars)
    add_value_labels(segm_bars)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_class_ap.png'), dpi=300)
    plt.close()
    
    # 2. Plot mAP at different IoU thresholds
    metrics_types = ['bbox', 'segm']
    metrics_labels = ['Detection (bbox)', 'Segmentation (mask)']
    metrics_values = [
        [metrics['bbox']['AP@[.5:.95]'], metrics['bbox']['AP@.5'], metrics['bbox']['AP@.75']],
        [metrics['segm']['AP@[.5:.95]'], metrics['segm']['AP@.5'], metrics['segm']['AP@.75']]
    ]
    
    iou_labels = ['mAP@[.5:.95]', 'mAP@.5', 'mAP@.75']
    x = np.arange(len(iou_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, (values, label, color) in enumerate(zip(metrics_values, metrics_labels, ['steelblue', 'darkorange'])):
        bars = ax.bar(x + (i - 0.5) * width, values, width, label=label, color=color)
        add_value_labels(bars)
    
    ax.set_title('mAP at Different IoU Thresholds', fontsize=16)
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Average Precision', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(iou_labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'map_at_iou_thresholds.png'), dpi=300)
    plt.close()
    
    print(f"Metric visualizations saved to {save_dir}")


def evaluate_coco_metrics(model, test_dataset, device, result_dir='evaluation_results'):
    """
    Evaluate model using COCO metrics (mAP at different IoU thresholds).
    
    Args:
        model: The Mask R-CNN model
        test_dataset: The test dataset
        device: The device to run inference on
        result_dir: Directory to save evaluation results
    
    Returns:
        dict: Dictionary containing COCO evaluation metrics
    """
    # Create save directory if it doesn't exist
    os.makedirs(result_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Step 1: Get the COCO-format ground truth annotations from the dataset
    # This uses the prepare_coco_format method from your CamVidInstanceDataset class
    coco_dataset = test_dataset.prepare_coco_format()
    
    # Save the ground truth annotations to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
        gt_file = f.name
        json.dump(coco_dataset, f)
    
    # Create a COCO object for the ground truth
    coco_gt = COCO(gt_file)
    
    # Step 2: Create a list to store model predictions in COCO format
    coco_predictions = []
    
    # Process each test sample
    print("Generating predictions for COCO evaluation...")
    for idx in range(len(test_dataset)):
        # Get image and target
        image, target = test_dataset[idx]
        image_id = target["image_id"].item()
        
        # Move image to device and make prediction
        image = image.to(device)
        with torch.no_grad():
            output = model([image])[0]
        
        # Convert prediction to COCO format
        for box, label, score, mask in zip(
            output['boxes'].cpu().numpy(), 
            output['labels'].cpu().numpy(), 
            output['scores'].cpu().numpy(),
            output['masks'].cpu().numpy()
        ):
            # Skip low confidence predictions
            if score < 0.05:  # Lower threshold for evaluation
                continue
                
            # Convert box from [x1, y1, x2, y2] to [x, y, width, height]
            coco_box = [
                float(box[0]),
                float(box[1]),
                float(box[2] - box[0]),
                float(box[3] - box[1])
            ]
            
            # Binarise mask
            binary_mask = (mask[0] > 0.5).astype(np.uint8)
            
            # Convert mask to RLE format
            rle = mask_util.encode(np.asfortranarray(binary_mask))
            rle['counts'] = rle['counts'].decode('utf-8')
            
            # Create prediction entry
            coco_predictions.append({
                'image_id': image_id,
                'category_id': int(label),
                'bbox': coco_box,
                'score': float(score),
                'segmentation': rle,
                'area': float(box[2] - box[0]) * float(box[3] - box[1])
            })
        
        # Print progress
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(test_dataset)} images")
    
    # Save predictions to a file
    pred_file = os.path.join(result_dir, "coco_predictions.json")
    with open(pred_file, 'w') as f:
        json.dump(coco_predictions, f)
    
    # Step 3: Evaluate using COCO API
    print("Evaluating predictions with COCO metrics...")
    
    # Load predictions
    coco_dt = coco_gt.loadRes(pred_file)
    
    # Initialize COCO evaluator
    metrics = {}
    
    # Evaluate bounding box detection
    coco_eval_bbox = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval_bbox.evaluate()
    coco_eval_bbox.accumulate()
    coco_eval_bbox.summarize()
    
    # Store bounding box metrics
    metrics['bbox'] = {
        'AP@[.5:.95]': coco_eval_bbox.stats[0],
        'AP@.5': coco_eval_bbox.stats[1],
        'AP@.75': coco_eval_bbox.stats[2]
    }
    
    # Per-class AP@.5 for bbox
    metrics['bbox_per_class'] = {}
    for i, cat_id in enumerate(coco_gt.getCatIds()):
        cat_name = next(cat['name'] for cat in coco_dataset['categories'] if cat['id'] == cat_id)
        metrics['bbox_per_class'][cat_name] = coco_eval_bbox.eval['precision'][0, :, i, 0, 2].mean()
    
    # Evaluate segmentation
    coco_eval_segm = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval_segm.evaluate()
    coco_eval_segm.accumulate()
    coco_eval_segm.summarize()
    
    # Store segmentation metrics
    metrics['segm'] = {
        'AP@[.5:.95]': coco_eval_segm.stats[0],
        'AP@.5': coco_eval_segm.stats[1],
        'AP@.75': coco_eval_segm.stats[2]
    }
    
    # Per-class IoU (AP@.5) for segmentation
    metrics['segm_per_class'] = {}
    for i, cat_id in enumerate(coco_gt.getCatIds()):
        cat_name = next(cat['name'] for cat in coco_dataset['categories'] if cat['id'] == cat_id)
        metrics['segm_per_class'][cat_name] = coco_eval_segm.eval['precision'][0, :, i, 0, 2].mean()
    
    # Save metrics to a file
    with open(os.path.join(result_dir, "coco_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Print metrics
    print("\nCOCO Evaluation Metrics:")
    print("=" * 60)
    print("Detection (Bounding Box):")
    print(f"mAP @[0.5:0.95]: {metrics['bbox']['AP@[.5:.95]']:.4f}")
    print(f"mAP @0.5: {metrics['bbox']['AP@.5']:.4f}")
    print(f"mAP @0.75: {metrics['bbox']['AP@.75']:.4f}")
    
    print("\nSegmentation (Mask):")
    print(f"mAP @[0.5:0.95]: {metrics['segm']['AP@[.5:.95]']:.4f}")
    print(f"mAP @0.5: {metrics['segm']['AP@.5']:.4f}")
    print(f"mAP @0.75: {metrics['segm']['AP@.75']:.4f}")
    
    print("\nPer-class AP@0.5 for Detection:")
    for class_name, ap in metrics['bbox_per_class'].items():
        print(f"{class_name}: {ap:.4f}")
    
    print("\nPer-class AP@0.5 for Segmentation:")
    for class_name, ap in metrics['segm_per_class'].items():
        print(f"{class_name}: {ap:.4f}")
    
    # Clean up temporary files
    os.remove(gt_file)
    
    return metrics


def main():
    """
    Main function to test the model, visualize results, and evaluate with COCO metrics.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Paths and settings
    model_path = "/home/roboticsstudent/Documents/COMP0248_RamneekAhluwalia/src/camvid_maskrcnn_model.pth"  # Path to your trained model
    num_classes = 6  # Background + 5 classes
    result_dir = "results/visualization_results"  # Directory to save visualizations
    eval_dir = "results/evaluation_results"  # Directory to save evaluation results

    # Define the target classes (including void for background)
    target_classes = ['Bicyclist', 'Car', 'MotorcycleScooter', 'Pedestrian', 'Truck_Bus']

    # Class RGB values from your existing class_map
    class_map = [
        [  0, 128, 192],  # Bicyclist
        [ 64,   0, 128],  # Car
        [192,   0, 192],  # MotorcycleScooter
        [ 64,  64,   0],  # Pedestrian
        [192, 128, 192]   # Truck_Bus
    ]

    # Define test image and mask paths
    #test_img_paths = [pair[0] for pair in test_pairs]  # Use from your original code
    #test_mask_paths = [pair[1] for pair in test_pairs]  # Use from your original code

    # Get test image and mask paths using the dataset manager
    test_img_paths = dataset_manager.get_test_image_paths()
    test_mask_paths = dataset_manager.get_test_mask_paths()

    # Create test dataset
    test_dataset = CamVidInstanceDataset(
        test_img_paths,
        test_mask_paths,
        class_map,
        target_classes,
        transforms=get_transform()
    )

    # Create test data loader
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn  # Use the same collate_fn from your training code
    )

    # Load the trained model
    model, device = load_model(model_path, num_classes)
    print(f"Model loaded to {device}")

    # Evaluate and visualize results
    print("Evaluating model on test set...")
    evaluate_model_on_test_set(
        model,
        test_dataset,
        device,
        num_samples=10,  # Number of samples to visualize
        threshold=0.5,   # Score threshold for predictions
        save_dir=result_dir
    )

    # Compute basic metrics on the test set
    print("Computing basic metrics...")
    basic_metrics = compute_metrics(model, test_data_loader, device)

    # Print basic metrics
    print("\nBasic Test Set Metrics:")
    print(f"Precision: {basic_metrics['precision']:.4f}")
    print(f"Recall: {basic_metrics['recall']:.4f}")
    print(f"F1 Score: {basic_metrics['f1_score']:.4f}")
    print(f"True Positives: {basic_metrics['true_positives']}")
    print(f"False Positives: {basic_metrics['false_positives']}")
    print(f"False Negatives: {basic_metrics['false_negatives']}")

    # Save basic metrics to file
    with open(os.path.join(result_dir, "basic_metrics.txt"), "w") as f:
        f.write("Basic Test Set Metrics:\n")
        f.write(f"Precision: {basic_metrics['precision']:.4f}\n")
        f.write(f"Recall: {basic_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {basic_metrics['f1_score']:.4f}\n")
        f.write(f"True Positives: {basic_metrics['true_positives']}\n")
        f.write(f"False Positives: {basic_metrics['false_positives']}\n")
        f.write(f"False Negatives: {basic_metrics['false_negatives']}\n")
    
    # Compute COCO metrics
    print("\nComputing COCO evaluation metrics...")
    coco_metrics = evaluate_coco_metrics(model, test_dataset, device, eval_dir)
    
    # Visualize COCO metrics
    print("\nCreating metric visualizations...")
    visualize_metrics(coco_metrics, eval_dir)
    
    # Generate a single comprehensive report
    print("\nGenerating comprehensive evaluation report...")
    with open(os.path.join(eval_dir, "full_evaluation_report.txt"), "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MASK R-CNN EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Part 1: Basic metrics
        f.write("BASIC METRICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Precision: {basic_metrics['precision']:.4f}\n")
        f.write(f"Recall: {basic_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {basic_metrics['f1_score']:.4f}\n")
        f.write(f"True Positives: {basic_metrics['true_positives']}\n")
        f.write(f"False Positives: {basic_metrics['false_positives']}\n")
        f.write(f"False Negatives: {basic_metrics['false_negatives']}\n\n")
        
        # Part 2: COCO metrics
        f.write("COCO METRICS:\n")
        f.write("-" * 80 + "\n")
        f.write("Detection (Bounding Box):\n")
        f.write(f"mAP @[0.5:0.95]: {coco_metrics['bbox']['AP@[.5:.95]']:.4f}\n")
        f.write(f"mAP @0.5: {coco_metrics['bbox']['AP@.5']:.4f}\n")
        f.write(f"mAP @0.75: {coco_metrics['bbox']['AP@.75']:.4f}\n\n")
        
        f.write("Segmentation (Mask):\n")
        f.write(f"mAP @[0.5:0.95]: {coco_metrics['segm']['AP@[.5:.95]']:.4f}\n")
        f.write(f"mAP @0.5: {coco_metrics['segm']['AP@.5']:.4f}\n")
        f.write(f"mAP @0.75: {coco_metrics['segm']['AP@.75']:.4f}\n\n")
        
        f.write("Per-class AP@0.5 for Detection:\n")
        for class_name, ap in coco_metrics['bbox_per_class'].items():
            f.write(f"{class_name}: {ap:.4f}\n")
        f.write("\n")
        
        f.write("Per-class AP@0.5 for Segmentation:\n")
        for class_name, ap in coco_metrics['segm_per_class'].items():
            f.write(f"{class_name}: {ap:.4f}\n")

    print(f"\nEvaluation complete!")
    print(f"- Basic metrics: {os.path.join(result_dir, 'basic_metrics.txt')}")
    print(f"- COCO metrics: {os.path.join(eval_dir, 'coco_metrics.json')}")
    print(f"- Visualizations: {result_dir}")
    print(f"- Comprehensive report: {os.path.join(eval_dir, 'full_evaluation_report.txt')}")
    print(f"- Metric visualizations: {eval_dir}")


if __name__ == "__main__":
    # Import the collate_fn from utils
    from src.pytorch_helper.utils import collate_fn
    
    # Run the main function
    main()