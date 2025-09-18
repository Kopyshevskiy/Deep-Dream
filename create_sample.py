#!/usr/bin/env python3
"""
Create a simple test image for the Deep Dream processor.
"""

from PIL import Image, ImageDraw
import os

def create_sample_image():
    """Create a simple geometric pattern for testing."""
    # Create a 400x400 image
    width, height = 400, 400
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Draw some geometric shapes
    # Circles
    for i in range(5):
        x = 50 + i * 70
        y = 50 + i * 30
        draw.ellipse([x, y, x+60, y+60], fill=f'rgb({200-i*40}, {100+i*30}, {150+i*20})')
    
    # Rectangles
    for i in range(4):
        x = 30 + i * 80
        y = 200
        draw.rectangle([x, y, x+50, y+80], fill=f'rgb({150+i*25}, {50+i*40}, {100+i*35})')
    
    # Lines
    for i in range(10):
        x = i * 40
        draw.line([x, 320, x+100, 380], fill=f'rgb({100+i*15}, {200-i*10}, {80+i*15})', width=3)
    
    return image

if __name__ == "__main__":
    # Create imgs folder if it doesn't exist
    os.makedirs('imgs', exist_ok=True)
    
    # Create and save sample image
    sample_image = create_sample_image()
    sample_image.save('imgs/sample.png')
    print("Created sample image: imgs/sample.png")