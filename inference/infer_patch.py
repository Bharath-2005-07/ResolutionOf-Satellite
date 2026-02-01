"""
Single Patch Inference for Satellite Super-Resolution
Quick inference for small patches without tiling
"""
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models import EDSR, ESRGANLite, get_model


def load_model(model_path: str, model_type: str = 'esrgan_lite', 
               scale_factor: int = 4, device: str = 'cuda') -> torch.nn.Module:
    """
    Load a trained super-resolution model
    
    Args:
        model_path: Path to model checkpoint
        model_type: 'edsr', 'esrgan', or 'esrgan_lite'
        scale_factor: Upscaling factor (4 or 8)
        device: 'cuda' or 'cpu'
    
    Returns:
        Loaded model ready for inference
    """
    # Create model
    if model_type == 'edsr':
        model = EDSR(scale_factor=scale_factor)
    else:
        model = get_model(model_type, scale_factor=scale_factor)
    
    # Load weights
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model path {model_path} not found. Using random weights.")
    
    model.to(device)
    model.eval()
    return model


def preprocess_image(image: np.ndarray, device: str = 'cuda') -> torch.Tensor:
    """
    Preprocess image for model inference
    
    Args:
        image: Input image (H, W, C) with values in [0, 255] or [0, 1]
        device: Target device
    
    Returns:
        Preprocessed tensor (1, C, H, W)
    """
    # Ensure float32 and [0, 1] range
    if image.max() > 1.0:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
    
    # Convert to tensor (C, H, W)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    return tensor.to(device)


def postprocess_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert model output back to image
    
    Args:
        tensor: Model output (1, C, H, W)
    
    Returns:
        Image array (H, W, C) with values in [0, 255]
    """
    image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image = np.clip(image, 0, 1) * 255
    return image.astype(np.uint8)


@torch.no_grad()
def infer_patch(model: torch.nn.Module, 
                image: np.ndarray,
                device: str = 'cuda') -> np.ndarray:
    """
    Run inference on a single patch
    
    Args:
        model: Loaded SR model
        image: Input LR image (H, W, C)
        device: Device to use
    
    Returns:
        Super-resolved image (H*scale, W*scale, C)
    """
    # Preprocess
    input_tensor = preprocess_image(image, device)
    
    # Inference
    output_tensor = model(input_tensor)
    
    # Postprocess
    return postprocess_image(output_tensor)


def infer_file(model: torch.nn.Module,
               input_path: str,
               output_path: str = None,
               device: str = 'cuda') -> np.ndarray:
    """
    Run inference on an image file
    
    Args:
        model: Loaded SR model
        input_path: Path to input image
        output_path: Optional path to save output
        device: Device to use
    
    Returns:
        Super-resolved image
    """
    # Load image
    image = np.array(Image.open(input_path).convert('RGB'))
    
    # Run inference
    sr_image = infer_patch(model, image, device)
    
    # Save if output path provided
    if output_path:
        Image.fromarray(sr_image).save(output_path)
        print(f"Saved to {output_path}")
    
    return sr_image


def bicubic_upsample(image: np.ndarray, scale_factor: int = 4) -> np.ndarray:
    """
    Bicubic upsampling baseline for comparison
    
    Args:
        image: Input image (H, W, C)
        scale_factor: Upscaling factor
    
    Returns:
        Bicubic upsampled image
    """
    h, w = image.shape[:2]
    return cv2.resize(image, (w * scale_factor, h * scale_factor), 
                     interpolation=cv2.INTER_CUBIC)


def compare_with_bicubic(model: torch.nn.Module,
                        lr_image: np.ndarray,
                        hr_image: np.ndarray = None,
                        scale_factor: int = 4,
                        device: str = 'cuda') -> dict:
    """
    Compare model output with bicubic baseline
    
    Args:
        model: Loaded SR model
        lr_image: Low-resolution input
        hr_image: Optional ground truth HR image
        scale_factor: Upscaling factor
        device: Device to use
    
    Returns:
        Dictionary with comparison results and metrics
    """
    from training.metrics import calculate_psnr, calculate_ssim
    
    # Get model output
    sr_image = infer_patch(model, lr_image, device)
    
    # Get bicubic baseline
    bicubic_image = bicubic_upsample(lr_image, scale_factor)
    
    results = {
        'lr': lr_image,
        'sr': sr_image,
        'bicubic': bicubic_image
    }
    
    # Calculate metrics if HR ground truth is available
    if hr_image is not None:
        results['hr'] = hr_image
        
        # Convert to tensors for metrics
        sr_tensor = torch.from_numpy(sr_image / 255.0).permute(2, 0, 1).unsqueeze(0)
        hr_tensor = torch.from_numpy(hr_image / 255.0).permute(2, 0, 1).unsqueeze(0)
        bicubic_tensor = torch.from_numpy(bicubic_image / 255.0).permute(2, 0, 1).unsqueeze(0)
        
        results['metrics'] = {
            'sr_psnr': calculate_psnr(sr_tensor, hr_tensor),
            'sr_ssim': calculate_ssim(sr_tensor, hr_tensor),
            'bicubic_psnr': calculate_psnr(bicubic_tensor, hr_tensor),
            'bicubic_ssim': calculate_ssim(bicubic_tensor, hr_tensor),
        }
        
    return results


def batch_inference(model: torch.nn.Module,
                   images: list,
                   batch_size: int = 8,
                   device: str = 'cuda') -> list:
    """
    Run inference on multiple images efficiently
    
    Args:
        model: Loaded SR model
        images: List of input images (H, W, C)
        batch_size: Batch size for processing
        device: Device to use
    
    Returns:
        List of super-resolved images
    """
    results = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        
        # Preprocess batch
        tensors = [preprocess_image(img, device).squeeze(0) for img in batch]
        batch_tensor = torch.stack(tensors)
        
        # Inference
        with torch.no_grad():
            sr_batch = model(batch_tensor)
        
        # Postprocess
        for j in range(sr_batch.shape[0]):
            sr_image = postprocess_image(sr_batch[j:j+1])
            results.append(sr_image)
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Single patch inference')
    parser.add_argument('--input', '-i', required=True, help='Input image path')
    parser.add_argument('--output', '-o', default='output_sr.png', help='Output path')
    parser.add_argument('--model', '-m', default='checkpoints/best_model.pth', help='Model path')
    parser.add_argument('--model-type', default='esrgan_lite', help='Model type')
    parser.add_argument('--scale', type=int, default=4, help='Scale factor')
    parser.add_argument('--device', default='cuda', help='Device')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model, args.model_type, args.scale, args.device)
    
    # Run inference
    sr_image = infer_file(model, args.input, args.output, args.device)
    
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Output shape: {sr_image.shape}")
