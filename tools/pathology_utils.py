import os
import numpy as np
import tiffslide
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from PIL import Image

def get_wsi_thumbnail(wsi_path: str, size: tuple = (1024, 1024)) -> Image.Image:
    """
    Get a thumbnail of the WSI using TiffSlide.
    """
    if not os.path.exists(wsi_path):
        # Return a dummy image if path doesn't exist (for MVP testing without large files)
        return Image.new("RGB", size, (240, 240, 240))
        
    try:
        slide = tiffslide.TiffSlide(wsi_path)
        thumbnail = slide.get_thumbnail(size)
        return thumbnail
    except Exception as e:
        print(f"Error reading slide {wsi_path}: {e}")
        return Image.new("RGB", size, (240, 240, 240))

def extract_tile(wsi_path: str, location: tuple, level: int = 0, size: tuple = (256, 256)) -> np.ndarray:
    """
    Extract a specific tile from the WSI.
    location: (x, y) coordinates at level 0.
    """
    if not os.path.exists(wsi_path):
        # Dummy tile
        return np.ones((size[1], size[0], 3), dtype=np.uint8) * 200
        
    try:
        slide = tiffslide.TiffSlide(wsi_path)
        tile = slide.read_region(location, level, size).convert("RGB")
        return np.array(tile)
    except Exception as e:
        print(f"Error extracting tile: {e}")
        return np.ones((size[1], size[0], 3), dtype=np.uint8) * 200

def segment_nuclei(image_array: np.ndarray) -> np.ndarray:
    """
    Lightweight zero-shot segmentation placeholder mimicking SAM for nuclei/cells.
    Uses basic HSV thresholding (hematoxylin stains blue/purple).
    Returns a boolean mask of the segmented regions.
    """
    if image_array.ndim != 3:
        return np.zeros_like(image_array, dtype=bool)
        
    # Convert RGB to HSV
    hsv = rgb2hsv(image_array)
    # Saturation channel is good for separating tissue from background
    s_channel = hsv[:, :, 1]
    
    try:
        # Otsu thresholding on saturation channel
        thresh = threshold_otsu(s_channel)
        mask = s_channel > thresh
        # Morphological closing to fill gaps
        mask = closing(mask, square(3))
        return mask
    except Exception:
        return np.zeros(image_array.shape[:2], dtype=bool)

def generate_saliency_heatmap(image_array: np.ndarray) -> np.ndarray:
    """
    Generates a visual heatmap overlay for the segmentation mask.
    """
    mask = segment_nuclei(image_array)
    heatmap = np.zeros_like(image_array)
    # Highlight segmented areas in red
    heatmap[mask] = [255, 0, 0]
    
    # Blend original image with heatmap (alpha = 0.4)
    alpha = 0.4
    blended = (image_array * (1 - alpha) + heatmap * alpha).astype(np.uint8)
    # Keep background intact where mask is false
    blended[~mask] = image_array[~mask]
    return blended
