import os
import json
import torch
import tempfile
from pycocotools.coco import COCO
from src.pytorch_helper.utils import MetricLogger, SmoothedValue, reduce_dict, collate_fn
from dataloader import CamVidInstanceDataset, calculate_class_weights, get_transform, create_combined_sampler
from dataset_manager import CamVidDatasetManager
from src.pytorch_helper.coco_eval import CocoEvaluator
from model import get_model_instance_segmentation


def create_coco_eval_dataset(dataset):
    # Get the COCO-format annotations from the dataset
    coco_dataset = dataset.prepare_coco_format()

    # Save to a temporary file - specify mode='w' for text mode
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
        json_file = f.name
        json.dump(coco_dataset, f)

    # Create a COCO object for evaluation
    coco_gt = COCO(json_file)
    return coco_gt, json_file

def train_one_epoch_with_weighted_loss(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    # Class pixel percentages
    pixel_percentages = [0.62, 0.12, 0.09, 0.0001, 0.04]  # MotorcycleScooter value adjusted for calculation
    class_names = ['Car', 'Pedestrian', 'Bicyclist', 'MotorcycleScooter', 'Truck_Bus']

    # Calculate weights using inverse square root method (handles extreme imbalances better)
    class_weights = calculate_class_weights(pixel_percentages, method='inverse_square_root')    
    class_weights = class_weights.to(device)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass and get losses
        loss_dict = model(images, targets)
        
        # Calculate total loss
        losses = sum(loss for loss in loss_dict.values())

        # Reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def main():

    dataset_manager = CamVidDatasetManager()

    # Get image and mask paths
    train_img_paths = dataset_manager.get_train_image_paths()
    train_mask_paths = dataset_manager.get_train_mask_paths()
    val_img_paths = dataset_manager.get_val_image_paths()
    val_mask_paths = dataset_manager.get_val_mask_paths()
    test_img_paths = dataset_manager.get_test_image_paths()
    test_mask_paths = dataset_manager.get_test_mask_paths()

    # Device configuration
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    # Target classes (including void for background)
    target_classes = ['Bicyclist', 'Car', 'MotorcycleScooter', 'Pedestrian', 'Truck_Bus']
    
    # Class RGB values from your existing class_map
    class_map = [
        [  0, 128, 192],
        [ 64,   0, 128],
        [192,   0, 192],
        [64, 64,  0],
        [192, 128, 192]
    ]
    
    # Create datasets
    train_dataset = CamVidInstanceDataset(
        train_img_paths, 
        train_mask_paths,
        class_map,
        target_classes,
        transforms=get_transform(train=True)
    )
    
    val_dataset = CamVidInstanceDataset(
        val_img_paths, 
        val_mask_paths,
        class_map,
        target_classes,
        transforms=get_transform(train=False)
    )

    # Create COCO ground truth for evaluation
    coco_gt, json_file = create_coco_eval_dataset(val_dataset)
    
    # Create validation data loader (doesn't change during training)
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create model
    num_classes = 1 + 5  # background + 5 classes
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    
    # Set up optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    # Set up learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )
        
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        # Create a new sampler for each epoch - this is key for curriculum learning
        train_sampler = create_combined_sampler(train_dataset, epoch, num_epochs)        

        # Create data loader with current sampler for this epoch
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=2,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=4
        )

        # Train for one epoch
        train_one_epoch_with_weighted_loss(model, optimizer, train_data_loader, device, epoch)

        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate on validation set
        model.eval()
        coco_evaluator = CocoEvaluator(coco_gt, ["bbox", "segm"])
        
        with torch.no_grad():
            for images, targets in val_data_loader:
                images = list(img.to(device) for img in images)
                
                # Get predictions
                outputs = model(images)
                
                # Move outputs to CPU before passing to evaluator
                cpu_outputs = []
                for output in outputs:
                    cpu_output = {k: v.cpu() for k, v in output.items()}
                    cpu_outputs.append(cpu_output)
                
                # Map image_id to output
                results = {}
                for target, output in zip(targets, cpu_outputs):
                    img_id = target["image_id"].item()
                    results[img_id] = output
                
                # Update the evaluator
                coco_evaluator.update(results)
        
        # Print evaluation results for this epoch
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        print(f"Epoch {epoch+1}/{num_epochs} completed")
                
    print("Training completed!")

    # Save the final model
    torch.save(model.state_dict(), "camvid_maskrcnn_model.pth")

if __name__ == "__main__":
    main()