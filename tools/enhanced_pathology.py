"""
Enhanced Pathology Utilities for DERMA-Agent
Advanced image processing, segmentation, and feature extraction for histopathology.
"""

import os
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageEnhance
from skimage import measure, morphology, feature
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage.filters import threshold_otsu, threshold_multiotsu, gaussian
from skimage.segmentation import slic, watershed
from skimage.morphology import remove_small_objects, disk, closing
from skimage.measure import regionprops, label
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import tiffslide
import warnings

warnings.filterwarnings('ignore')


@dataclass
class PathologyFeatures:
    """Container for extracted pathology features."""
    nuclei_count: int
    nuclei_density: float
    avg_nuclei_size: float
    tissue_area_ratio: float
    cellularity: float
    texture_features: Dict[str, float]
    stroma_ratio: Optional[float] = None
    lymphocyte_estimate: Optional[float] = None


class EnhancedWSIProcessor:
    """Enhanced Whole Slide Image processor with advanced features."""
    
    def __init__(self, tile_size: Tuple[int, int] = (512, 512)):
        self.tile_size = tile_size
        
    def open_slide(self, wsi_path: str) -> Optional[tiffslide.TiffSlide]:
        """Open a whole slide image."""
        if not os.path.exists(wsi_path):
            return None
        try:
            return tiffslide.TiffSlide(wsi_path)
        except Exception as e:
            print(f"Error opening slide: {e}")
            return None
    
    def get_thumbnail(self, slide: tiffslide.TiffSlide, 
                     max_size: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
        """Get a thumbnail of the slide."""
        try:
            thumb = slide.get_thumbnail(max_size)
            return np.array(thumb)
        except Exception as e:
            print(f"Error getting thumbnail: {e}")
            return np.ones((512, 512, 3), dtype=np.uint8) * 240
    
    def extract_tile(self, slide: tiffslide.TiffSlide, 
                    location: Tuple[int, int],
                    level: int = 0,
                    size: Tuple[int, int] = None) -> np.ndarray:
        """Extract a specific tile from the slide."""
        size = size or self.tile_size
        try:
            tile = slide.read_region(location, level, size)
            return np.array(tile.convert('RGB'))
        except Exception as e:
            print(f"Error extracting tile: {e}")
            return np.ones((size[1], size[0], 3), dtype=np.uint8) * 200
    
    def get_tissue_mask(self, thumbnail: np.ndarray, 
                       threshold: float = 0.8) -> np.ndarray:
        """Generate a tissue mask from thumbnail."""
        # Convert to grayscale
        gray = rgb2gray(thumbnail)
        
        # Apply Gaussian blur
        blurred = gaussian(gray, sigma=2)
        
        # Threshold to separate tissue from background
        try:
            thresh = threshold_otsu(blurred)
            mask = blurred < thresh
        except:
            mask = blurred < threshold
        
        # Clean up mask
        mask = remove_small_objects(mask, min_size=100)
        mask = closing(mask, disk(5))
        
        return mask


class NucleiSegmenter:
    """Advanced nuclei segmentation for histopathology images."""
    
    def __init__(self, min_nuclei_size: int = 50, max_nuclei_size: int = 5000):
        self.min_nuclei_size = min_nuclei_size
        self.max_nuclei_size = max_nuclei_size
        
    def segment_nuclei(self, image: np.ndarray, method: str = 'hsv') -> np.ndarray:
        """
        Segment nuclei from histopathology image.
        
        Args:
            image: RGB image array
            method: 'hsv', 'lab', or 'adaptive'
            
        Returns:
            Binary mask of segmented nuclei
        """
        if method == 'hsv':
            return self._segment_hsv(image)
        elif method == 'lab':
            return self._segment_lab(image)
        else:
            return self._segment_adaptive(image)
    
    def _segment_hsv(self, image: np.ndarray) -> np.ndarray:
        """Segment using HSV color space (good for H&E stains)."""
        hsv = rgb2hsv(image)
        
        # Use saturation and value channels
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        # Threshold on saturation (nuclei are typically more saturated)
        try:
            s_thresh = threshold_otsu(saturation)
            s_mask = saturation > s_thresh
        except:
            s_mask = saturation > 0.2
        
        # Value threshold to exclude very bright areas
        v_mask = value < 0.95
        
        # Combine masks
        mask = s_mask & v_mask
        
        # Morphological operations
        mask = closing(mask, disk(2))
        mask = remove_small_objects(mask, min_size=self.min_nuclei_size)
        
        return mask
    
    def _segment_lab(self, image: np.ndarray) -> np.ndarray:
        """Segment using LAB color space."""
        lab = rgb2lab(image)
        
        # Use L channel (lightness)
        l_channel = lab[:, :, 0]
        
        # Darker areas are likely nuclei
        try:
            thresh = threshold_otsu(l_channel)
            mask = l_channel < thresh
        except:
            mask = l_channel < 50
        
        mask = closing(mask, disk(2))
        mask = remove_small_objects(mask, min_size=self.min_nuclei_size)
        
        return mask
    
    def _segment_adaptive(self, image: np.ndarray) -> np.ndarray:
        """Adaptive segmentation using multiple methods."""
        # Try both methods and combine
        hsv_mask = self._segment_hsv(image)
        lab_mask = self._segment_lab(image)
        
        # Union of both masks
        combined = hsv_mask | lab_mask
        
        return combined
    
    def segment_individual_nuclei(self, mask: np.ndarray) -> np.ndarray:
        """Separate touching nuclei using watershed."""
        # Distance transform
        from scipy import ndimage
        distance = ndimage.distance_transform_edt(mask)
        
        # Find peaks
        from skimage.feature import peak_local_max
        local_max = peak_local_max(distance, min_distance=10, labels=mask)
        
        # Create markers
        markers = np.zeros_like(mask, dtype=int)
        markers[tuple(local_max.T)] = 1
        markers = label(markers)
        
        # Watershed
        ws_mask = watershed(-distance, markers, mask=mask)
        
        return ws_mask
    
    def extract_nuclei_features(self, image: np.ndarray, 
                               mask: np.ndarray) -> List[Dict]:
        """Extract features for each segmented nucleus."""
        labeled_mask = label(mask)
        regions = regionprops(labeled_mask, intensity_image=rgb2gray(image))
        
        features = []
        for region in regions:
            if self.min_nuclei_size < region.area < self.max_nuclei_size:
                feat = {
                    'area': region.area,
                    'perimeter': region.perimeter,
                    'eccentricity': region.eccentricity,
                    'solidity': region.solidity,
                    'mean_intensity': region.mean_intensity,
                    'centroid': region.centroid,
                    'bbox': region.bbox,
                    'circularity': 4 * np.pi * region.area / (region.perimeter ** 2) if region.perimeter > 0 else 0
                }
                features.append(feat)
        
        return features


class TissueAnalyzer:
    """Analyze tissue morphology and extract quantitative features."""
    
    def __init__(self):
        self.segmenter = NucleiSegmenter()
        
    def analyze_tile(self, image: np.ndarray) -> PathologyFeatures:
        """
        Comprehensive analysis of a pathology tile.
        
        Args:
            image: RGB image array
            
        Returns:
            PathologyFeatures with extracted metrics
        """
        # Segment nuclei
        nuclei_mask = self.segmenter.segment_nuclei(image)
        
        # Get tissue mask
        gray = rgb2gray(image)
        try:
            tissue_thresh = threshold_otsu(gray)
            tissue_mask = gray < tissue_thresh
        except:
            tissue_mask = gray < 0.8
        
        tissue_mask = remove_small_objects(tissue_mask, min_size=500)
        
        # Calculate areas
        total_pixels = image.shape[0] * image.shape[1]
        tissue_pixels = np.sum(tissue_mask)
        nuclei_pixels = np.sum(nuclei_mask)
        
        # Extract individual nuclei features
        nuclei_features = self.segmenter.extract_nuclei_features(image, nuclei_mask)
        
        # Calculate metrics
        nuclei_count = len(nuclei_features)
        nuclei_density = nuclei_count / (tissue_pixels / 1000000) if tissue_pixels > 0 else 0
        avg_nuclei_size = np.mean([f['area'] for f in nuclei_features]) if nuclei_features else 0
        tissue_area_ratio = tissue_pixels / total_pixels
        cellularity = nuclei_pixels / tissue_pixels if tissue_pixels > 0 else 0
        
        # Texture features
        texture = self._extract_texture_features(gray)
        
        # Estimate lymphocytes (small, round nuclei)
        if nuclei_features:
            small_round = [f for f in nuclei_features 
                          if f['area'] < 200 and f['circularity'] > 0.7]
            lymphocyte_estimate = len(small_round) / nuclei_count if nuclei_count > 0 else 0
        else:
            lymphocyte_estimate = None
        
        return PathologyFeatures(
            nuclei_count=nuclei_count,
            nuclei_density=nuclei_density,
            avg_nuclei_size=avg_nuclei_size,
            tissue_area_ratio=tissue_area_ratio,
            cellularity=cellularity,
            texture_features=texture,
            lymphocyte_estimate=lymphocyte_estimate
        )
    
    def _extract_texture_features(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Extract texture features using GLCM and LBP."""
        features = {}
        
        # GLCM features
        try:
            # Downsample for speed
            if gray_image.shape[0] > 256:
                from skimage.transform import resize
                gray_small = resize(gray_image, (256, 256), anti_aliasing=True)
            else:
                gray_small = gray_image
            
            # Quantize to 32 levels
            gray_quant = (gray_small * 31).astype(np.uint8)
            
            glcm = graycomatrix(gray_quant, distances=[1], 
                               angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                               levels=32, symmetric=True, normed=True)
            
            features['contrast'] = np.mean(graycoprops(glcm, 'contrast'))
            features['dissimilarity'] = np.mean(graycoprops(glcm, 'dissimilarity'))
            features['homogeneity'] = np.mean(graycoprops(glcm, 'homogeneity'))
            features['energy'] = np.mean(graycoprops(glcm, 'energy'))
            features['correlation'] = np.mean(graycoprops(glcm, 'correlation'))
            
        except Exception as e:
            features['contrast'] = 0
            features['homogeneity'] = 0
            features['energy'] = 0
        
        # LBP features
        try:
            lbp = local_binary_pattern(gray_small, P=8, R=1, method='uniform')
            lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
            lbp_hist = lbp_hist / np.sum(lbp_hist)
            features['lbp_entropy'] = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))
        except:
            features['lbp_entropy'] = 0
        
        return features
    
    def generate_heatmap(self, image: np.ndarray, 
                        nuclei_mask: np.ndarray,
                        feature_type: str = 'density') -> np.ndarray:
        """Generate a heatmap overlay showing pathological features."""
        if feature_type == 'density':
            # Create density heatmap using convolution
            from scipy import ndimage
            density = ndimage.gaussian_filter(nuclei_mask.astype(float), sigma=20)
            
            # Normalize
            density = (density - density.min()) / (density.max() - density.min() + 1e-10)
            
            # Create colored heatmap
            heatmap = np.zeros_like(image)
            heatmap[:, :, 0] = (density * 255).astype(np.uint8)  # Red channel
            heatmap[:, :, 2] = ((1 - density) * 255).astype(np.uint8)  # Blue channel
            
        elif feature_type == 'nuclei':
            # Highlight nuclei
            heatmap = image.copy()
            heatmap[nuclei_mask] = [255, 100, 100]
            
        else:
            heatmap = image.copy()
        
        # Blend with original
        alpha = 0.5
        blended = (image * (1 - alpha) + heatmap * alpha).astype(np.uint8)
        
        return blended


def analyze_wsi_path(wsi_path: str, max_tiles: int = 10) -> Dict[str, Any]:
    """
    Analyze a whole slide image and return comprehensive statistics.
    
    Args:
        wsi_path: Path to WSI file
        max_tiles: Maximum number of tiles to analyze
        
    Returns:
        Dictionary with analysis results
    """
    processor = EnhancedWSIProcessor()
    analyzer = TissueAnalyzer()
    
    # Open slide
    slide = processor.open_slide(wsi_path)
    if slide is None:
        return {'error': 'Could not open slide', 'path': wsi_path}
    
    # Get thumbnail for overview
    thumbnail = processor.get_thumbnail(slide, max_size=(1024, 1024))
    
    # Get tissue mask
    tissue_mask = processor.get_tissue_mask(thumbnail)
    
    # Sample tiles from tissue regions
    tile_results = []
    
    # Find tissue regions
    from skimage.measure import find_contours
    contours = find_contours(tissue_mask, 0.5)
    
    if contours:
        # Sample points from contours
        n_samples = min(max_tiles, len(contours))
        sampled_contours = np.random.choice(len(contours), n_samples, replace=False)
        
        for idx in sampled_contours:
            contour = contours[idx]
            if len(contour) > 0:
                # Sample a point from contour
                point = contour[len(contour)//2].astype(int)
                
                # Scale to level 0 coordinates
                scale_factor = slide.level_dimensions[0][0] / thumbnail.shape[1]
                location = (int(point[1] * scale_factor), int(point[0] * scale_factor))
                
                # Extract tile
                tile = processor.extract_tile(slide, location)
                
                # Analyze
                features = analyzer.analyze_tile(tile)
                tile_results.append({
                    'location': location,
                    'features': features
                })
    
    # Aggregate statistics
    if tile_results:
        all_features = [t['features'] for t in tile_results]
        
        summary = {
            'n_tiles_analyzed': len(tile_results),
            'avg_nuclei_count': np.mean([f.nuclei_count for f in all_features]),
            'avg_nuclei_density': np.mean([f.nuclei_density for f in all_features]),
            'avg_cellularity': np.mean([f.cellularity for f in all_features]),
            'tissue_area_ratio': np.sum(tissue_mask) / tissue_mask.size,
            'tiles': tile_results
        }
    else:
        summary = {
            'n_tiles_analyzed': 0,
            'error': 'No tissue regions found'
        }
    
    return summary


def create_synthetic_pathology_image(size: Tuple[int, int] = (512, 512),
                                     pattern: str = 'mixed') -> np.ndarray:
    """Create a synthetic pathology image for testing."""
    image = np.ones((*size, 3), dtype=np.uint8) * 240
    
    if pattern == 'mixed':
        # Add some cell-like structures
        n_cells = np.random.randint(20, 50)
        for _ in range(n_cells):
            x = np.random.randint(50, size[1]-50)
            y = np.random.randint(50, size[0]-50)
            radius = np.random.randint(10, 30)
            color = np.random.randint(100, 200, size=3)
            
            cv2.circle(image, (x, y), radius, color.tolist(), -1)
            # Darker center (nucleus)
            cv2.circle(image, (x, y), radius//2, (50, 50, 100), -1)
    
    elif pattern == 'high_cellularity':
        n_cells = np.random.randint(80, 150)
        for _ in range(n_cells):
            x = np.random.randint(0, size[1])
            y = np.random.randint(0, size[0])
            radius = np.random.randint(5, 20)
            cv2.circle(image, (x, y), radius, (150, 100, 200), -1)
    
    return image


if __name__ == "__main__":
    # Test the enhanced pathology utils
    print("Testing Enhanced Pathology Utilities...")
    
    # Create synthetic image
    image = create_synthetic_pathology_image((512, 512), 'mixed')
    
    print("\n1. Testing Nuclei Segmentation:")
    segmenter = NucleiSegmenter()
    mask = segmenter.segment_nuclei(image)
    print(f"   Segmented nuclei pixels: {np.sum(mask)}")
    
    nuclei_features = segmenter.extract_nuclei_features(image, mask)
    print(f"   Individual nuclei detected: {len(nuclei_features)}")
    
    print("\n2. Testing Tissue Analysis:")
    analyzer = TissueAnalyzer()
    features = analyzer.analyze_tile(image)
    print(f"   Nuclei count: {features.nuclei_count}")
    print(f"   Nuclei density: {features.nuclei_density:.2f}")
    print(f"   Cellularity: {features.cellularity:.3f}")
    print(f"   Texture features: {list(features.texture_features.keys())}")
    
    print("\n3. Testing Heatmap Generation:")
    heatmap = analyzer.generate_heatmap(image, mask, 'density')
    print(f"   Heatmap shape: {heatmap.shape}")
    
    print("\n4. Testing Different Patterns:")
    for pattern in ['mixed', 'high_cellularity']:
        img = create_synthetic_pathology_image((256, 256), pattern)
        feats = analyzer.analyze_tile(img)
        print(f"   {pattern}: {feats.nuclei_count} nuclei, {feats.cellularity:.3f} cellularity")
    
    print("\nAll tests complete!")
