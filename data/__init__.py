"""
Data module for satellite super-resolution

Quick Start:
    from data import get_satellite_dataset
    
    # Option 1: Real paired LR/HR data
    dataset = get_satellite_dataset('real', lr_dir='path/lr', hr_dir='path/hr')
    
    # Option 2: Create LR from HR images
    dataset = get_satellite_dataset('synthetic', hr_dir='path/to/satellite_images')
    
    # Option 3: WorldStrat dataset
    dataset = get_satellite_dataset('worldstrat', root_dir='path/to/worldstrat')
"""

from .dataset import (
    SatelliteSRDataset,
    WorldStratDataset,
    SyntheticSRDataset,
    GEEDataset,
    DemoDataset,
    get_dataloader
)
from .gee_fetch import initialize_gee, fetch_patch, fetch_patches_batch

def get_satellite_dataset(dataset_type='real', **kwargs):
    """
    Quick helper to get the right dataset
    
    Args:
        dataset_type: 'real', 'synthetic', 'worldstrat', 'gee', or 'demo'
        **kwargs: Arguments for the specific dataset
    
    Returns:
        Dataset instance
    
    Examples:
        # Real paired data
        dataset = get_satellite_dataset('real', 
                                       lr_dir='satellite_data/lr',
                                       hr_dir='satellite_data/hr')
        
        # Generate LR from HR
        dataset = get_satellite_dataset('synthetic',
                                       hr_dir='satellite_images/')
        
        # WorldStrat
        dataset = get_satellite_dataset('worldstrat',
                                       root_dir='worldstrat/',
                                       split='train')
    """
    if dataset_type == 'real':
        return SatelliteSRDataset(**kwargs)
    elif dataset_type == 'synthetic':
        return SyntheticSRDataset(**kwargs)
    elif dataset_type == 'worldstrat':
        return WorldStratDataset(**kwargs)
    elif dataset_type == 'gee':
        return GEEDataset(**kwargs)
    elif dataset_type == 'demo':
        return DemoDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

__all__ = [
    'SatelliteSRDataset',
    'WorldStratDataset', 
    'SyntheticSRDataset',
    'GEEDataset',
    'DemoDataset',
    'get_dataloader',
    'initialize_gee',
    'fetch_patch',
    'fetch_patches_batch',
    'get_satellite_dataset'
]
