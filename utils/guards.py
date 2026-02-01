"""
Hallucination Guardrails for Satellite Super-Resolution
Prevents the model from inventing features that don't exist

Key principle: The model should RECOVER details, not INVENT them.
Placing a building where there's a forest = CRITICAL FAILURE
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import cv2


class HallucinationGuard:
    """
    Detects potential hallucinations in super-resolved satellite images
    
    Strategy:
    1. Semantic consistency check (land cover should match)
    2. Edge preservation check (edges should align with LR)
    3. Color distribution check (no extreme color shifts)
    4. Structural integrity check (no phantom features)
    """
    
    def __init__(self, threshold_semantic: float = 0.15,
                 threshold_edge: float = 0.3,
                 threshold_color: float = 0.2):
        """
        Args:
            threshold_semantic: Max allowed semantic divergence
            threshold_edge: Max allowed edge mismatch
            threshold_color: Max allowed color distribution shift
        """
        self.threshold_semantic = threshold_semantic
        self.threshold_edge = threshold_edge
        self.threshold_color = threshold_color
        
    def check(self, lr_image: np.ndarray, sr_image: np.ndarray,
              scale_factor: int = 4) -> Dict:
        """
        Run all hallucination checks
        
        Args:
            lr_image: Low-resolution input (H, W, C)
            sr_image: Super-resolved output (H*scale, W*scale, C)
            scale_factor: Upscaling factor
        
        Returns:
            Dictionary with check results and confidence scores
        """
        results = {
            'passed': True,
            'confidence': 1.0,
            'checks': {}
        }
        
        # 1. Semantic consistency
        semantic_score = self._check_semantic_consistency(lr_image, sr_image, scale_factor)
        results['checks']['semantic'] = {
            'score': semantic_score,
            'passed': semantic_score > (1 - self.threshold_semantic)
        }
        
        # 2. Edge preservation
        edge_score = self._check_edge_preservation(lr_image, sr_image, scale_factor)
        results['checks']['edge'] = {
            'score': edge_score,
            'passed': edge_score > (1 - self.threshold_edge)
        }
        
        # 3. Color consistency
        color_score = self._check_color_consistency(lr_image, sr_image)
        results['checks']['color'] = {
            'score': color_score,
            'passed': color_score > (1 - self.threshold_color)
        }
        
        # 4. Structure check
        structure_score = self._check_structure_integrity(lr_image, sr_image, scale_factor)
        results['checks']['structure'] = {
            'score': structure_score,
            'passed': structure_score > 0.7
        }
        
        # Overall assessment
        scores = [semantic_score, edge_score, color_score, structure_score]
        results['confidence'] = np.mean(scores)
        results['passed'] = all(check['passed'] for check in results['checks'].values())
        
        return results
    
    def _check_semantic_consistency(self, lr: np.ndarray, sr: np.ndarray,
                                   scale: int) -> float:
        """
        Check if SR maintains semantic content of LR
        Downscale SR and compare with LR
        """
        # Downscale SR to LR resolution
        sr_down = cv2.resize(sr, (lr.shape[1], lr.shape[0]), interpolation=cv2.INTER_AREA)
        
        # Normalize images
        lr_norm = lr.astype(np.float32) / 255.0 if lr.max() > 1 else lr
        sr_down_norm = sr_down.astype(np.float32) / 255.0 if sr_down.max() > 1 else sr_down
        
        # Calculate MSE
        mse = np.mean((lr_norm - sr_down_norm) ** 2)
        
        # Convert to similarity score (1 = perfect match)
        score = np.exp(-mse * 10)  # Scale factor for reasonable range
        
        return float(score)
    
    def _check_edge_preservation(self, lr: np.ndarray, sr: np.ndarray,
                                scale: int) -> float:
        """
        Check if SR edges align with LR edges
        """
        # Extract edges from LR
        lr_gray = cv2.cvtColor(lr, cv2.COLOR_RGB2GRAY) if len(lr.shape) == 3 else lr
        lr_edges = cv2.Canny(lr_gray.astype(np.uint8), 50, 150)
        
        # Upscale LR edges
        lr_edges_up = cv2.resize(lr_edges, (sr.shape[1], sr.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
        
        # Extract edges from SR
        sr_gray = cv2.cvtColor(sr, cv2.COLOR_RGB2GRAY) if len(sr.shape) == 3 else sr
        sr_edges = cv2.Canny(sr_gray.astype(np.uint8), 50, 150)
        
        # Dilate LR edges to allow some tolerance
        kernel = np.ones((3, 3), np.uint8)
        lr_edges_dilated = cv2.dilate(lr_edges_up, kernel, iterations=2)
        
        # Check how many SR edges are near LR edges
        sr_edge_pixels = sr_edges > 0
        lr_edge_region = lr_edges_dilated > 0
        
        if np.sum(sr_edge_pixels) == 0:
            return 1.0  # No edges = no hallucinated edges
        
        # Ratio of SR edges that are near LR edges
        aligned_edges = np.sum(sr_edge_pixels & lr_edge_region)
        total_sr_edges = np.sum(sr_edge_pixels)
        
        score = aligned_edges / total_sr_edges
        
        return float(score)
    
    def _check_color_consistency(self, lr: np.ndarray, sr: np.ndarray) -> float:
        """
        Check if color distribution is preserved
        """
        # Calculate histograms for each channel
        lr_norm = lr.astype(np.float32) / 255.0 if lr.max() > 1 else lr
        sr_norm = sr.astype(np.float32) / 255.0 if sr.max() > 1 else sr
        
        scores = []
        for c in range(min(3, lr_norm.shape[-1])):
            lr_hist, _ = np.histogram(lr_norm[..., c].flatten(), bins=50, range=(0, 1))
            sr_hist, _ = np.histogram(sr_norm[..., c].flatten(), bins=50, range=(0, 1))
            
            # Normalize histograms
            lr_hist = lr_hist / (lr_hist.sum() + 1e-8)
            sr_hist = sr_hist / (sr_hist.sum() + 1e-8)
            
            # Calculate histogram intersection (similarity)
            intersection = np.minimum(lr_hist, sr_hist).sum()
            scores.append(intersection)
        
        return float(np.mean(scores))
    
    def _check_structure_integrity(self, lr: np.ndarray, sr: np.ndarray,
                                  scale: int) -> float:
        """
        Check for phantom features (structures that appear in SR but not in LR)
        Uses local variance analysis
        """
        # Calculate local variance in LR
        lr_gray = cv2.cvtColor(lr, cv2.COLOR_RGB2GRAY) if len(lr.shape) == 3 else lr
        lr_gray = lr_gray.astype(np.float32)
        
        kernel_size = 5
        lr_mean = cv2.blur(lr_gray, (kernel_size, kernel_size))
        lr_var = cv2.blur(lr_gray ** 2, (kernel_size, kernel_size)) - lr_mean ** 2
        
        # Upscale variance map
        lr_var_up = cv2.resize(lr_var, (sr.shape[1], sr.shape[0]), 
                               interpolation=cv2.INTER_LINEAR)
        
        # Calculate local variance in SR
        sr_gray = cv2.cvtColor(sr, cv2.COLOR_RGB2GRAY) if len(sr.shape) == 3 else sr
        sr_gray = sr_gray.astype(np.float32)
        
        sr_mean = cv2.blur(sr_gray, (kernel_size, kernel_size))
        sr_var = cv2.blur(sr_gray ** 2, (kernel_size, kernel_size)) - sr_mean ** 2
        
        # Normalize
        lr_var_norm = lr_var_up / (np.max(lr_var_up) + 1e-8)
        sr_var_norm = sr_var / (np.max(sr_var) + 1e-8)
        
        # Find regions where SR has high variance but LR doesn't
        # These could be hallucinated structures
        suspicious = (sr_var_norm > 0.3) & (lr_var_norm < 0.1)
        
        # Ratio of non-suspicious pixels
        score = 1.0 - (np.sum(suspicious) / suspicious.size)
        
        return float(score)


class ContentAwareGuard:
    """
    Content-aware hallucination detection
    Classifies regions by land cover type and checks consistency
    """
    
    def __init__(self):
        # NDVI-like indices for vegetation detection (simplified)
        self.vegetation_threshold = 0.2
        self.water_threshold = -0.1
        
    def detect_land_cover(self, image: np.ndarray) -> np.ndarray:
        """
        Simple land cover classification based on color
        
        Returns:
            Mask with labels: 0=unknown, 1=vegetation, 2=water, 3=urban, 4=bare
        """
        # Normalize to [0, 1]
        img = image.astype(np.float32)
        if img.max() > 1:
            img = img / 255.0
        
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        
        # Simple indices
        # Green excess for vegetation
        green_excess = (g - r) + (g - b)
        
        # Blue excess for water
        blue_excess = b - (r + g) / 2
        
        # Create mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        # Vegetation (green dominant)
        mask[green_excess > self.vegetation_threshold] = 1
        
        # Water (blue dominant, dark)
        water_cond = (blue_excess > self.water_threshold) & (np.mean(img, axis=-1) < 0.3)
        mask[water_cond] = 2
        
        # Urban (gray, high variance)
        gray = np.mean(img, axis=-1)
        color_var = np.var(img, axis=-1)
        urban_cond = (color_var < 0.01) & (gray > 0.3) & (gray < 0.7)
        mask[urban_cond] = 3
        
        return mask
    
    def check_land_cover_consistency(self, lr: np.ndarray, sr: np.ndarray,
                                    scale: int) -> Dict:
        """
        Check if land cover is preserved between LR and SR
        
        Returns:
            Dictionary with consistency scores per land cover type
        """
        lr_cover = self.detect_land_cover(lr)
        sr_cover = self.detect_land_cover(sr)
        
        # Upscale LR cover to SR resolution
        lr_cover_up = cv2.resize(lr_cover, (sr.shape[1], sr.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        
        results = {
            'overall': 0.0,
            'vegetation': 0.0,
            'water': 0.0,
            'urban': 0.0
        }
        
        # Overall agreement
        results['overall'] = float(np.mean(lr_cover_up == sr_cover))
        
        # Per-class agreement
        for label, name in [(1, 'vegetation'), (2, 'water'), (3, 'urban')]:
            lr_mask = lr_cover_up == label
            sr_mask = sr_cover == label
            
            if np.sum(lr_mask) > 0:
                # How much of LR class is preserved in SR
                preserved = np.sum(lr_mask & sr_mask) / np.sum(lr_mask)
                results[name] = float(preserved)
            else:
                results[name] = 1.0  # No such class in LR
        
        return results


def apply_guardrail(lr_image: np.ndarray, sr_image: np.ndarray,
                   scale_factor: int = 4) -> Tuple[np.ndarray, Dict]:
    """
    Apply hallucination guardrail to SR image
    
    If hallucination is detected, blend with bicubic to reduce artifacts
    
    Args:
        lr_image: Low-resolution input
        sr_image: Super-resolved output
        scale_factor: Upscaling factor
    
    Returns:
        Tuple of (corrected_image, guard_results)
    """
    guard = HallucinationGuard()
    results = guard.check(lr_image, sr_image, scale_factor)
    
    if results['passed']:
        return sr_image, results
    
    # If failed, blend with bicubic based on confidence
    confidence = results['confidence']
    
    # Create bicubic baseline
    bicubic = cv2.resize(lr_image, (sr_image.shape[1], sr_image.shape[0]),
                        interpolation=cv2.INTER_CUBIC)
    
    # Blend: more bicubic if confidence is low
    alpha = confidence  # Use SR
    corrected = (alpha * sr_image + (1 - alpha) * bicubic).astype(sr_image.dtype)
    
    results['correction_applied'] = True
    results['blend_alpha'] = alpha
    
    return corrected, results


def get_confidence_map(lr_image: np.ndarray, sr_image: np.ndarray,
                      scale_factor: int = 4) -> np.ndarray:
    """
    Generate a spatial confidence map showing where hallucinations might occur
    
    Args:
        lr_image: Low-resolution input
        sr_image: Super-resolved output
        scale_factor: Upscaling factor
    
    Returns:
        Confidence map (H, W) with values in [0, 1]
    """
    # Downscale SR and compare with LR
    sr_down = cv2.resize(sr_image, (lr_image.shape[1], lr_image.shape[0]),
                        interpolation=cv2.INTER_AREA)
    
    # Normalize
    lr_norm = lr_image.astype(np.float32) / 255.0 if lr_image.max() > 1 else lr_image.astype(np.float32)
    sr_down_norm = sr_down.astype(np.float32) / 255.0 if sr_down.max() > 1 else sr_down.astype(np.float32)
    
    # Per-pixel difference
    diff = np.mean(np.abs(lr_norm - sr_down_norm), axis=-1)
    
    # Convert to confidence (low diff = high confidence)
    confidence_lr = np.exp(-diff * 5)
    
    # Upscale to SR resolution
    confidence = cv2.resize(confidence_lr, (sr_image.shape[1], sr_image.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
    
    return confidence


if __name__ == '__main__':
    # Test guardrails
    print("Testing hallucination guardrails...")
    
    # Create synthetic test images
    lr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    sr = cv2.resize(lr, (256, 256), interpolation=cv2.INTER_CUBIC)
    
    # Add some "hallucinated" features to SR
    sr[100:150, 100:150] = [255, 0, 0]  # Red square that doesn't exist in LR
    
    guard = HallucinationGuard()
    results = guard.check(lr, sr, scale_factor=4)
    
    print(f"Passed: {results['passed']}")
    print(f"Confidence: {results['confidence']:.3f}")
    for name, check in results['checks'].items():
        print(f"  {name}: {check['score']:.3f} (passed: {check['passed']})")
