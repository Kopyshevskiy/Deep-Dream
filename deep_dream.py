#!/usr/bin/env python3
"""
Deep Dream implementation that processes images from 'imgs' folder.
This script applies the Deep Dream algorithm using pre-trained neural networks
to create psychedelic, dream-like versions of input images.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class DeepDream:
    def __init__(self, model=None, device='cpu'):
        """Initialize Deep Dream processor."""
        self.device = device
        
        # Use pre-trained VGG19 model if none provided
        if model is None:
            self.model = models.vgg19(pretrained=True).features
        else:
            self.model = model
            
        self.model.eval()
        self.model.to(device)
        
        # Normalization for ImageNet pre-trained models
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Denormalization for display
        self.denormalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )

    def preprocess_image(self, image_path, max_size=512):
        """Load and preprocess image."""
        image = Image.open(image_path).convert('RGB')
        
        # Resize if too large
        w, h = image.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        tensor = transform(image).unsqueeze(0)
        return tensor.to(self.device)

    def dream_loss(self, input_tensor, target_layer=28):
        """Calculate loss for deep dream - maximize activation at target layer."""
        x = input_tensor
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i == target_layer:
                # Return mean of all activations at this layer
                return -torch.mean(x)
        return -torch.mean(x)

    def generate_dream(self, image_tensor, iterations=20, lr=0.1, target_layer=28):
        """Generate deep dream version of image."""
        # Clone input to avoid modifying original
        dream_tensor = image_tensor.clone().requires_grad_(True)
        
        # Optimizer
        optimizer = torch.optim.Adam([dream_tensor], lr=lr)
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            # Calculate loss (negative to maximize activations)
            loss = self.dream_loss(dream_tensor, target_layer)
            loss.backward()
            
            # Apply gradient ascent
            optimizer.step()
            
            # Clamp values to valid range
            with torch.no_grad():
                dream_tensor.clamp_(0, 1)
                
            if (i + 1) % 5 == 0:
                print(f"Iteration {i+1}/{iterations}, Loss: {loss.item():.4f}")
        
        return dream_tensor

    def tensor_to_image(self, tensor):
        """Convert tensor back to PIL Image."""
        # Remove batch dimension and move to CPU
        tensor = tensor.squeeze(0).detach().cpu()
        
        # Convert to numpy array
        image_array = tensor.permute(1, 2, 0).numpy()
        
        # Clip values and convert to 0-255 range
        image_array = np.clip(image_array, 0, 1)
        image_array = (image_array * 255).astype(np.uint8)
        
        return Image.fromarray(image_array)

    def process_image(self, input_path, output_path, iterations=20, lr=0.1, target_layer=28):
        """Process a single image with deep dream."""
        print(f"Processing {input_path}...")
        
        # Load and preprocess image
        image_tensor = self.preprocess_image(input_path)
        
        # Generate dream
        dream_tensor = self.generate_dream(
            image_tensor, iterations=iterations, lr=lr, target_layer=target_layer
        )
        
        # Convert back to image and save
        dream_image = self.tensor_to_image(dream_tensor)
        dream_image.save(output_path)
        
        print(f"Saved dream image to {output_path}")
        return dream_image


def process_imgs_folder(imgs_folder='imgs', outputs_folder='outputs', 
                       iterations=20, lr=0.1, target_layer=28):
    """Process all images in the imgs folder."""
    
    if not os.path.exists(imgs_folder):
        print(f"Error: {imgs_folder} folder not found!")
        return
    
    # Create outputs folder if it doesn't exist
    os.makedirs(outputs_folder, exist_ok=True)
    
    # Initialize Deep Dream processor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    deep_dream = DeepDream(device=device)
    
    # Supported image formats
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Process each image in the imgs folder
    processed_count = 0
    for filename in os.listdir(imgs_folder):
        file_path = os.path.join(imgs_folder, filename)
        
        # Check if it's a file and has supported format
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(filename.lower())
            if ext in supported_formats:
                # Generate output filename
                name_without_ext = os.path.splitext(filename)[0]
                output_filename = f"{name_without_ext}_dream.png"
                output_path = os.path.join(outputs_folder, output_filename)
                
                try:
                    deep_dream.process_image(
                        file_path, output_path, 
                        iterations=iterations, lr=lr, target_layer=target_layer
                    )
                    processed_count += 1
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
            else:
                print(f"Skipping {filename} - unsupported format")
    
    print(f"\nCompleted! Processed {processed_count} images.")
    print(f"Dream images saved in '{outputs_folder}' folder.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Deep Dream image processor')
    parser.add_argument('--imgs-folder', default='imgs', 
                       help='Input images folder (default: imgs)')
    parser.add_argument('--outputs-folder', default='outputs', 
                       help='Output folder for dream images (default: outputs)')
    parser.add_argument('--iterations', type=int, default=20,
                       help='Number of iterations (default: 20)')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate (default: 0.1)')
    parser.add_argument('--target-layer', type=int, default=28,
                       help='Target layer for dreaming (default: 28)')
    
    args = parser.parse_args()
    
    process_imgs_folder(
        imgs_folder=args.imgs_folder,
        outputs_folder=args.outputs_folder,
        iterations=args.iterations,
        lr=args.lr,
        target_layer=args.target_layer
    )