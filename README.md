# Deep-Dream

A Python implementation of the Deep Dream algorithm that processes images from the 'imgs' folder.

## Features

- Process images from the 'imgs' folder using Deep Dream algorithm
- Uses pre-trained VGG19 neural network for feature extraction
- Configurable iterations, learning rate, and target layers
- Outputs dream images to the 'outputs' folder
- Supports common image formats (JPG, PNG, BMP, TIFF)

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your input images in the `imgs` folder
2. Run the Deep Dream processor:
```bash
python deep_dream.py
```

### Command Line Options

```bash
python deep_dream.py --help
```

Options:
- `--imgs-folder`: Input images folder (default: imgs)
- `--outputs-folder`: Output folder for dream images (default: outputs)
- `--iterations`: Number of iterations (default: 20)
- `--lr`: Learning rate (default: 0.1)
- `--target-layer`: Target layer for dreaming (default: 28)

### Example

```bash
# Process images with custom settings
python deep_dream.py --iterations 30 --lr 0.05 --target-layer 25
```

## Creating Sample Images

To create a sample test image:
```bash
python create_sample.py
```

This will create a sample geometric image in the `imgs` folder that you can use to test the Deep Dream algorithm.

## Output

Processed images are saved in the `outputs` folder with "_dream" suffix added to the original filename.