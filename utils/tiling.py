"""
Tiling Utilities for Memory-Efficient Satellite Image Processing
Critical for handling large satellite images on free-tier GPUs (Colab T4)
"""
import numpy as np
import torch
from typing import Tuple, List, Optional


def calculate_tile_positions(image_size: Tuple[int, int], 
                            tile_size: int, 
                            overlap: int = 16) -> List[Tuple[int, int, int, int]]:
    """
    Calculate tile positions with overlap for seamless stitching
    
    Args:
        image_size: (height, width) of input image
        tile_size: Size of each tile (square)
        overlap: Overlap between tiles to avoid seam artifacts
    
    Returns:
        List of (y_start, x_start, y_end, x_end) tuples
    """
    h, w = image_size
    stride = tile_size - overlap
    
    positions = []
    
    y = 0
    while y < h:
        x = 0
        while x < w:
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            
            # Adjust start if we're at the edge
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)
            
            positions.append((y_start, x_start, y_end, x_end))
            
            if x_end >= w:
                break
            x += stride
            
        if y_end >= h:
            break
        y += stride
    
    return positions


def extract_tiles(image: np.ndarray, 
                  tile_size: int = 64, 
                  overlap: int = 16) -> Tuple[List[np.ndarray], List[Tuple]]:
    """
    Extract overlapping tiles from an image
    
    Args:
        image: Input image (H, W, C) or (H, W)
        tile_size: Size of each tile
        overlap: Overlap between tiles
    
    Returns:
        tiles: List of tile arrays
        positions: List of (y_start, x_start, y_end, x_end) positions
    """
    h, w = image.shape[:2]
    positions = calculate_tile_positions((h, w), tile_size, overlap)
    
    tiles = []
    for y_start, x_start, y_end, x_end in positions:
        tile = image[y_start:y_end, x_start:x_end]
        
        # Pad if tile is smaller than tile_size
        if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
            if len(image.shape) == 3:
                padded = np.zeros((tile_size, tile_size, image.shape[2]), dtype=image.dtype)
            else:
                padded = np.zeros((tile_size, tile_size), dtype=image.dtype)
            padded[:tile.shape[0], :tile.shape[1]] = tile
            tile = padded
            
        tiles.append(tile)
    
    return tiles, positions


def stitch_tiles(tiles: List[np.ndarray], 
                 positions: List[Tuple], 
                 output_size: Tuple[int, int],
                 scale_factor: int = 4,
                 blend_overlap: bool = True) -> np.ndarray:
    """
    Stitch tiles back together with optional blending
    
    Args:
        tiles: List of processed tiles (at HR resolution)
        positions: Original tile positions (at LR resolution)
        output_size: (height, width) of output image at HR resolution
        scale_factor: Upscaling factor
        blend_overlap: Use weighted blending in overlap regions
    
    Returns:
        Stitched image at HR resolution
    """
    h, w = output_size
    
    # Determine number of channels
    if len(tiles[0].shape) == 3:
        channels = tiles[0].shape[2]
        output = np.zeros((h, w, channels), dtype=np.float32)
        weights = np.zeros((h, w, channels), dtype=np.float32)
    else:
        output = np.zeros((h, w), dtype=np.float32)
        weights = np.zeros((h, w), dtype=np.float32)
    
    for tile, (y_start, x_start, y_end, x_end) in zip(tiles, positions):
        # Scale positions to HR resolution
        hr_y_start = y_start * scale_factor
        hr_x_start = x_start * scale_factor
        hr_y_end = y_end * scale_factor
        hr_x_end = x_end * scale_factor
        
        # Crop tile to actual size needed
        tile_h = hr_y_end - hr_y_start
        tile_w = hr_x_end - hr_x_start
        tile = tile[:tile_h, :tile_w]
        
        if blend_overlap:
            # Create weight mask for blending (higher weight in center)
            weight = create_blend_weight(tile.shape[:2])
            if len(tile.shape) == 3:
                weight = weight[:, :, np.newaxis]
        else:
            weight = np.ones_like(tile)
        
        output[hr_y_start:hr_y_end, hr_x_start:hr_x_end] += tile * weight
        weights[hr_y_start:hr_y_end, hr_x_start:hr_x_end] += weight
    
    # Normalize by weights
    weights = np.maximum(weights, 1e-8)  # Avoid division by zero
    output = output / weights
    
    return output


def create_blend_weight(size: Tuple[int, int], 
                        edge_width: int = 16) -> np.ndarray:
    """
    Create a weight mask for smooth blending
    Higher weight in center, lower at edges
    """
    h, w = size
    weight = np.ones((h, w), dtype=np.float32)
    
    # Create ramps for edges
    for i in range(min(edge_width, h // 2)):
        factor = (i + 1) / edge_width
        weight[i, :] *= factor
        weight[h - 1 - i, :] *= factor
        
    for j in range(min(edge_width, w // 2)):
        factor = (j + 1) / edge_width
        weight[:, j] *= factor
        weight[:, w - 1 - j] *= factor
    
    return weight


class TiledProcessor:
    """
    Memory-efficient tiled processing for large satellite images
    """
    def __init__(self, tile_size: int = 64, overlap: int = 16, 
                 scale_factor: int = 4, device: str = 'cuda'):
        """
        Args:
            tile_size: LR tile size
            overlap: Overlap between tiles (LR pixels)
            scale_factor: Upscaling factor
            device: 'cuda' or 'cpu'
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.scale_factor = scale_factor
        self.device = device
        
    @torch.no_grad()
    def process(self, model: torch.nn.Module, 
                image: np.ndarray,
                batch_size: int = 4) -> np.ndarray:
        """
        Process large image through model using tiling
        
        Args:
            model: Super-resolution model
            image: Input LR image (H, W, C), values in [0, 1]
            batch_size: Number of tiles to process at once
        
        Returns:
            Super-resolved image at HR resolution
        """
        model.eval()
        model.to(self.device)
        
        h, w = image.shape[:2]
        hr_h, hr_w = h * self.scale_factor, w * self.scale_factor
        
        # Extract tiles
        tiles, positions = extract_tiles(image, self.tile_size, self.overlap)
        
        # Process tiles in batches
        sr_tiles = []
        for i in range(0, len(tiles), batch_size):
            batch = tiles[i:i + batch_size]
            
            # Convert to tensor batch
            batch_tensor = torch.stack([
                torch.from_numpy(t).permute(2, 0, 1).float() 
                for t in batch
            ]).to(self.device)
            
            # Run through model
            sr_batch = model(batch_tensor)
            
            # Convert back to numpy
            sr_batch = sr_batch.permute(0, 2, 3, 1).cpu().numpy()
            sr_tiles.extend([sr_batch[j] for j in range(len(batch))])
            
        # Stitch tiles
        output = stitch_tiles(sr_tiles, positions, (hr_h, hr_w), 
                             self.scale_factor, blend_overlap=True)
        
        return np.clip(output, 0, 1)


def process_large_image(model: torch.nn.Module,
                       image: np.ndarray,
                       tile_size: int = 64,
                       overlap: int = 16,
                       scale_factor: int = 4,
                       batch_size: int = 4,
                       device: str = 'cuda') -> np.ndarray:
    """
    Convenience function for tiled processing
    
    Args:
        model: Super-resolution model
        image: Input image (H, W, C) with values in [0, 1]
        tile_size: Size of tiles for processing
        overlap: Overlap between tiles
        scale_factor: Upscaling factor
        batch_size: Tiles per batch
        device: 'cuda' or 'cpu'
    
    Returns:
        Super-resolved image
    """
    processor = TiledProcessor(tile_size, overlap, scale_factor, device)
    return processor.process(model, image, batch_size)


if __name__ == '__main__':
    # Test tiling
    print("Testing tiling utilities...")
    
    # Create test image
    test_img = np.random.rand(256, 256, 3).astype(np.float32)
    
    # Extract tiles
    tiles, positions = extract_tiles(test_img, tile_size=64, overlap=16)
    print(f"Extracted {len(tiles)} tiles from 256x256 image")
    
    # Simulate SR (just upsample for testing)
    scale = 4
    sr_tiles = [np.kron(t, np.ones((scale, scale, 1))) for t in tiles]
    
    # Stitch back
    output = stitch_tiles(sr_tiles, positions, (256*scale, 256*scale), scale)
    print(f"Output shape: {output.shape}")
