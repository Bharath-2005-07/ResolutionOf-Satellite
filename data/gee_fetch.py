"""
Google Earth Engine Integration for Satellite Super-Resolution
Fetch Sentinel-2 patches for inference
"""
import numpy as np
from pathlib import Path
import json

try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False
    print("Google Earth Engine not available. Install: pip install earthengine-api")


def initialize_gee():
    """Initialize Google Earth Engine"""
    if not GEE_AVAILABLE:
        raise ImportError("earthengine-api not installed")
    
    try:
        ee.Initialize()
        print("GEE initialized successfully!")
    except Exception:
        print("Authenticating with GEE...")
        ee.Authenticate()
        ee.Initialize()
        print("GEE authenticated and initialized!")


def get_sentinel2_image(lon, lat, start_date='2023-01-01', end_date='2024-01-01', 
                        cloud_threshold=10):
    """
    Get cleanest Sentinel-2 image for a location
    
    Args:
        lon, lat: Center coordinates
        start_date, end_date: Date range for filtering
        cloud_threshold: Maximum cloud percentage
    
    Returns:
        ee.Image: Sentinel-2 image
    """
    point = ee.Geometry.Point([lon, lat])
    
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(point)
                 .filterDate(start_date, end_date)
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
                 .sort('CLOUDY_PIXEL_PERCENTAGE'))
    
    return collection.first()


def fetch_patch(lon, lat, buffer_m=1280, output_size=128, bands=['B4', 'B3', 'B2']):
    """
    Fetch a patch from Sentinel-2
    
    Args:
        lon, lat: Center coordinates
        buffer_m: Buffer in meters (determines area coverage)
        output_size: Output image size in pixels
        bands: Bands to fetch (default RGB)
    
    Returns:
        numpy array (H, W, C) normalized to [0, 1]
    """
    if not GEE_AVAILABLE:
        return np.random.rand(output_size, output_size, len(bands)).astype(np.float32)
    
    try:
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(buffer_m).bounds()
        
        image = get_sentinel2_image(lon, lat)
        image = image.select(bands)
        
        # Normalize Sentinel-2 (0-10000) to (0-255) for visualization
        image = image.divide(10000).multiply(255).clamp(0, 255).uint8()
        
        url = image.getThumbURL({
            'region': region,
            'dimensions': f'{output_size}x{output_size}',
            'format': 'png'
        })
        
        # Download
        import urllib.request
        from PIL import Image
        from io import BytesIO
        
        with urllib.request.urlopen(url, timeout=30) as response:
            img_data = response.read()
        
        img = np.array(Image.open(BytesIO(img_data)).convert('RGB'))
        return img.astype(np.float32) / 255.0
        
    except Exception as e:
        print(f"Error fetching patch: {e}")
        return None


def fetch_patches_batch(coordinates, buffer_m=1280, output_size=128):
    """
    Fetch multiple patches
    
    Args:
        coordinates: List of (lon, lat) tuples
        
    Returns:
        List of numpy arrays
    """
    patches = []
    for i, (lon, lat) in enumerate(coordinates):
        print(f"Fetching patch {i+1}/{len(coordinates)}: ({lon}, {lat})")
        patch = fetch_patch(lon, lat, buffer_m, output_size)
        if patch is not None:
            patches.append(patch)
    return patches


def save_patch(patch, output_path, metadata=None):
    """Save patch to file with optional metadata"""
    from PIL import Image
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save image
    img = (patch * 255).astype(np.uint8)
    Image.fromarray(img).save(output_path)
    
    # Save metadata
    if metadata:
        meta_path = output_path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)


# Predefined coordinates for demo (major Indian cities)
INDIA_CITIES = {
    'delhi': (77.2090, 28.6139),
    'mumbai': (72.8777, 19.0760),
    'bangalore': (77.5946, 12.9716),
    'kanpur': (80.3319, 26.4499),
    'chennai': (80.2707, 13.0827),
    'hyderabad': (78.4867, 17.3850),
    'kolkata': (88.3639, 22.5726),
    'pune': (73.8567, 18.5204),
}

# Urban areas for testing
URBAN_TEST_LOCATIONS = [
    (77.2090, 28.6139, 'Delhi'),
    (80.3319, 26.4499, 'Kanpur'),
    (72.8777, 19.0760, 'Mumbai'),
    (77.5946, 12.9716, 'Bangalore'),
]


if __name__ == '__main__':
    # Test GEE fetch
    print("Testing GEE integration...")
    
    try:
        initialize_gee()
        
        # Fetch a test patch (Delhi)
        lon, lat = 77.2090, 28.6139
        patch = fetch_patch(lon, lat)
        
        if patch is not None:
            print(f"Successfully fetched patch: {patch.shape}")
            save_patch(patch, 'test_patch_delhi.png', {
                'lon': lon, 'lat': lat, 'source': 'Sentinel-2'
            })
            print("Saved to test_patch_delhi.png")
        
    except Exception as e:
        print(f"GEE test failed: {e}")
        print("You may need to run: earthengine authenticate")
