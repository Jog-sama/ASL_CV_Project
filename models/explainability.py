"""
GradCAM for model explainability
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    """Generate GradCAM heatmaps for CNN models"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.device = next(model.parameters()).device  # Get model's device
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save forward pass activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate GradCAM heatmap
        
        Args:
            input_tensor: Input image tensor [1, 3, H, W]
            target_class: Target class index for CAM
        
        Returns:
            cam_overlay: RGB image with heatmap overlay
            cam_heatmap: Heatmap visualization
        """
        # Ensure input is on the same device as model
        input_tensor = input_tensor.to(self.device)
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], device=self.device)  # [H, W] on same device
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Move to CPU for visualization
        cam = cam.cpu().numpy()
        
        # Get original image
        original_img = input_tensor[0].cpu().numpy()
        original_img = np.transpose(original_img, (1, 2, 0))
        
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        original_img = std * original_img + mean
        original_img = np.clip(original_img * 255, 0, 255).astype(np.uint8)
        
        # Resize CAM to match input image size
        cam_resized = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on original image
        alpha = 0.4
        cam_overlay = (alpha * heatmap + (1 - alpha) * original_img).astype(np.uint8)
        
        return cam_overlay, heatmap