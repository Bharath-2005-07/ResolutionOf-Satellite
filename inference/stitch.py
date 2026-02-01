"""
Full Image Inference with Tiling and Stitching
For processing large satellite images without running out of memory
"""
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.tiling import TiledProcessor, process_large_image, extract_tiles, stitch_tiles
from inference.infer_patch import load_model, preprocess_image, postprocess_image


class SatelliteSRInference:
    """
    Complete inference pipeline for satellite super-resolution
    Handles large images via tiling
    """
    def __init__(self, 
                 model_path: str = None,
                 model_type: str = 'esrgan_lite',
                 scale_factor: int = 4,
                 tile_size: int = 64,
                 overlap: int = 16,
                 device: str = 'cuda'):
        """
        Args:
            model_path: Path to model checkpoint
            model_type: Type of model to load
            scale_factor: Upscaling factor (4 or 8)
            tile_size: Size of tiles for processing
            overlap: Overlap between tiles
            device: 'cuda' or 'cpu'
        """
        self.scale_factor = scale_factor
        self.tile_size = tile_size
        self.overlap = overlap
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load model
        if model_path:
            self.model = load_model(model_path, model_type, scale_factor, self.device)
        else:
            from models import get_model
            self.model = get_model(model_type, scale_factor).to(self.device)
            self.model.eval()
            print("Using untrained model (for demo purposes)")
        
        self.tiled_processor = TiledProcessor(
            tile_size=tile_size,
            overlap=overlap,
            scale_factor=scale_factor,
            device=self.device
        )
    
    @torch.no_grad()
    def process(self, image: np.ndarray, batch_size: int = 4) -> np.ndarray:
        """
        Process a large image using tiling
        
        Args:
            image: Input image (H, W, C) with values in [0, 255] or [0, 1]
            batch_size: Number of tiles to process at once
        
        Returns:
            Super-resolved image
        """
        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)
        
        # Process with tiling
        sr_image = self.tiled_processor.process(self.model, image, batch_size)
        
        # Convert back to [0, 255]
        sr_image = (np.clip(sr_image, 0, 1) * 255).astype(np.uint8)
        
        return sr_image
    
    def process_file(self, input_path: str, output_path: str = None,
                    batch_size: int = 4) -> np.ndarray:
        """
        Process an image file
        
        Args:
            input_path: Path to input image
            output_path: Optional path to save output
            batch_size: Batch size for tiled processing
        
        Returns:
            Super-resolved image
        """
        # Load image
        image = np.array(Image.open(input_path).convert('RGB'))
        
        print(f"Processing {input_path}")
        print(f"Input size: {image.shape}")
        
        # Process
        sr_image = self.process(image, batch_size)
        
        print(f"Output size: {sr_image.shape}")
        
        # Save if output path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(sr_image).save(output_path)
            print(f"Saved to {output_path}")
        
        return sr_image
    
    def process_folder(self, input_dir: str, output_dir: str,
                      extensions: list = ['.png', '.jpg', '.tif'],
                      batch_size: int = 4):
        """
        Process all images in a folder
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            extensions: File extensions to process
            batch_size: Batch size for processing
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = []
        for ext in extensions:
            files.extend(input_dir.glob(f'*{ext}'))
        
        print(f"Found {len(files)} images to process")
        
        for i, file_path in enumerate(files):
            output_path = output_dir / f"{file_path.stem}_sr{file_path.suffix}"
            print(f"\n[{i+1}/{len(files)}] ", end='')
            self.process_file(str(file_path), str(output_path), batch_size)
    
    @torch.no_grad()
    def process_quick(self, image: np.ndarray) -> np.ndarray:
        """
        Quick processing for small images (no tiling)
        Use this for patches smaller than tile_size
        
        Args:
            image: Input image (H, W, C)
        
        Returns:
            Super-resolved image
        """
        # Normalize
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # To tensor
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Inference
        sr_tensor = self.model(tensor)
        
        # To numpy
        sr_image = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        sr_image = (np.clip(sr_image, 0, 1) * 255).astype(np.uint8)
        
        return sr_image


def create_comparison_grid(lr: np.ndarray, sr: np.ndarray, 
                          bicubic: np.ndarray, hr: np.ndarray = None) -> np.ndarray:
    """
    Create a comparison grid for visualization
    
    Args:
        lr: Low-resolution input
        sr: Super-resolved output
        bicubic: Bicubic upsampled baseline
        hr: Optional ground truth
    
    Returns:
        Grid image
    """
    import cv2
    
    # Upsample LR to match SR size for display
    scale = sr.shape[0] // lr.shape[0]
    lr_display = cv2.resize(lr, (sr.shape[1], sr.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    if hr is not None:
        # 4-image grid: LR, Bicubic, SR, HR
        top = np.hstack([lr_display, bicubic])
        bottom = np.hstack([sr, hr])
        grid = np.vstack([top, bottom])
    else:
        # 3-image grid: LR, Bicubic, SR
        grid = np.hstack([lr_display, bicubic, sr])
    
    return grid


def quick_inference(input_path: str, output_path: str = None,
                   model_path: str = None, scale: int = 4) -> np.ndarray:
    """
    Quick one-line inference function
    
    Args:
        input_path: Path to input image
        output_path: Optional output path
        model_path: Optional model checkpoint path
        scale: Scale factor
    
    Returns:
        Super-resolved image
    """
    inferencer = SatelliteSRInference(
        model_path=model_path,
        scale_factor=scale
    )
    return inferencer.process_file(input_path, output_path)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Satellite image super-resolution with tiling')
    parser.add_argument('--input', '-i', required=True, help='Input image or folder')
    parser.add_argument('--output', '-o', default='output', help='Output path or folder')
    parser.add_argument('--model', '-m', default=None, help='Model checkpoint path')
    parser.add_argument('--model-type', default='esrgan_lite', help='Model type')
    parser.add_argument('--scale', type=int, default=4, help='Scale factor (4 or 8)')
    parser.add_argument('--tile-size', type=int, default=64, help='Tile size')
    parser.add_argument('--overlap', type=int, default=16, help='Tile overlap')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--device', default='cuda', help='Device')
    
    args = parser.parse_args()
    
    # Create inferencer
    inferencer = SatelliteSRInference(
        model_path=args.model,
        model_type=args.model_type,
        scale_factor=args.scale,
        tile_size=args.tile_size,
        overlap=args.overlap,
        device=args.device
    )
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_dir():
        inferencer.process_folder(str(input_path), args.output, batch_size=args.batch_size)
    else:
        output_path = args.output if not Path(args.output).is_dir() else str(Path(args.output) / f"{input_path.stem}_sr.png")
        inferencer.process_file(str(input_path), output_path, args.batch_size)
