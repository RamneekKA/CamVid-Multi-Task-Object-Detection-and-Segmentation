import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.ops import MultiScaleRoIAlign

class CustomMaskRCNN(nn.Module):
    def __init__(self, num_classes, class_weights=None, pretrained=True):
        super(CustomMaskRCNN, self).__init__()
        # Initialize the base model
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None)
        
        # Get number of input features
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        
        # Replace box predictor
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Replace mask predictor
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
        
        # Store the class weights
        self.class_weights = class_weights
        
    def forward(self, images, targets=None):
        if self.training and targets is not None:
            # Original Mask R-CNN computes the loss internally
            loss_dict = self.model(images, targets)
            
            # Apply class weights to the classification loss
            if 'loss_classifier' in loss_dict and self.class_weights is not None:
                # Increase the weight of classification loss for rare classes
                # For the motorcycle class (index 3), we apply a higher weight
                motorcycle_weight = self.class_weights[3].item() if len(self.class_weights) > 3 else 5.0
                
                # Scale up the classifier loss to emphasize rare classes
                loss_dict['loss_classifier'] = loss_dict['loss_classifier'] * motorcycle_weight
                
                # Optionally scale up the mask loss too
                if 'loss_mask' in loss_dict:
                    loss_dict['loss_mask'] = loss_dict['loss_mask'] * (motorcycle_weight * 0.5)
            
            return loss_dict
        else:
            return self.model(images)
            
    def load_state_dict(self, state_dict, strict=True):
        """Custom load_state_dict to handle the nested model structure"""
        # Check if keys have 'model.' prefix
        if all(k.startswith("model.") for k in state_dict.keys()):
            # Create a new state dict with modified keys
            new_state_dict = {}
            for k, v in state_dict.items():
                # Extract the part after 'model.'
                name = k[6:]  # Skip first 6 characters ("model.")
                new_state_dict[name] = v
            return self.model.load_state_dict(new_state_dict, strict=strict)
        else:
            return self.model.load_state_dict(state_dict, strict=strict)
        

# Replace the get_model_instance_segmentation function with this one
def get_model_instance_segmentation(num_classes):
    """
    Create a Mask R-CNN model with a ResNet-50 backbone.

    Args:
        num_classes (int): Number of classes (including background)

    Returns:
        model: The Mask R-CNN model
    """
    # Create the custom model
    model = CustomMaskRCNN(num_classes=num_classes, pretrained=True)
    return model

