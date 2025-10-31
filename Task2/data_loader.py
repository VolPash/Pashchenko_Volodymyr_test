# data_loader.py

import os
import glob
import re
import cv2
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union
import pandas as pd

# --- CONFIGURATION FOR LOCAL PATHS ---
# THE PATH MUST MATCH THE RESULT OF UNPACKING FROM download_kaggle_data.py
BASE_DATASET_PATH = "./data/deforestation-in-ukraine/" 
LABELS_PATH = os.path.join(BASE_DATASET_PATH, 'deforestation_labels.geojson')
MAX_DIM = 512 # Max dimension for image scaling

# --- GIS AND TRANSFORMATION UTILITIES ---

def poly_from_utm(polygon, transform):
    """Transforms a polygon from world coordinates to image pixel coordinates."""
    poly = unary_union(polygon)
    if poly.exterior is None:
        return Polygon()
    # Apply inverse transform to convert world coords to pixel coords
    poly_pts = [~transform * (pt[0], pt[1]) for pt in poly.exterior.coords]
    return Polygon(poly_pts)

def create_deforestation_mask(df_labels, raster_meta):
    """Creates a binary mask of stable areas (1) based on GeoJSON deforestation polygons."""
    df = df_labels.copy()
    df.crs = 'epsg:4326'
    try:
        if raster_meta['crs']:
            df = df.to_crs(raster_meta['crs']) # Reproject labels to raster CRS
    except Exception:
        pass
        
    poly_shp = []
    im_size = (raster_meta['height'], raster_meta['width'])

    for _, row in df.iterrows():
        geometry = row['geometry']
        if geometry is None or geometry.is_empty:
            continue
        
        polygons = geometry.geoms if geometry.geom_type == 'MultiPolygon' else [geometry]
        for p in polygons:
             if p is not None and not p.is_empty:
                poly_shp.append(poly_from_utm(p, raster_meta['transform']))
    
    if not poly_shp:
        return np.ones(im_size, dtype=np.uint8) # Full mask = 1 (no changes/deforestation)
    
    # Rasterize shapes to 1 (deforested areas)
    mask = rasterize(shapes=poly_shp, out_shape=im_size, fill=0, all_touched=True, dtype=np.uint8)
    return 1 - mask # Stable mask (1 outside deforestation, 0 inside)


def load_sentinel_with_meta(path, max_dim=MAX_DIM):
    """Loads JP2 (TCI), metadata, scales, and converts to monochrome."""
    try:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return None, None, None

        with rasterio.open(path, "r", driver='JP2OpenJPEG') as src:
            img_rgb = src.read([1, 2, 3]) 
            meta = src.meta
    except rasterio.errors.RasterioIOError as e:
        print(f"Rasterio error loading {path}: {e}")
        return None, None, None

    img_rgb = reshape_as_image(img_rgb) # Convert (C, H, W) to (H, W, C)
    h, w = img_rgb.shape[:2]
    scale = max_dim / max(h, w)
    new_size = (int(w * scale), int(h * scale))
    
    img_rgb_resized = cv2.resize(img_rgb, new_size, interpolation=cv2.INTER_AREA)
    img_mono = cv2.cvtColor(img_rgb_resized, cv2.COLOR_RGB2GRAY)
    
    return img_mono, img_rgb_resized, meta


def load_process_and_mask_image(path, df_labels):
    """Loads, masks, and prepares data dictionary for a single image."""
    img_mono_raw, img_vis_raw, meta = load_sentinel_with_meta(path, max_dim=MAX_DIM)
    
    if img_mono_raw is None:
        return None
        
    img_mono = img_mono_raw
    img_vis = img_vis_raw
    
    # Apply masking if labels and metadata are available
    if not df_labels.empty and meta:
        try:
            mask_stable = create_deforestation_mask(df_labels, meta)
            H, W = img_mono_raw.shape[:2]
            
            # Resize mask to the downscaled image dimensions
            mask_resized = cv2.resize(mask_stable, (W, H), interpolation=cv2.INTER_NEAREST)

            # Apply mask: zeroes out areas where deforestation occurred 
            img_mono = img_mono_raw * mask_resized
            img_vis = img_vis_raw * np.expand_dims(mask_resized, axis=-1)
        except Exception as e:
            # print(f"WARNING: Masking error: {e}")
            pass
    
    return {
        'path': path,
        'img_mono': img_mono,
        'img_vis': img_vis,
        'meta': meta
    }

# --- PAIR CREATION LOGIC ---

def extract_date_from_path(path):
    """Extracts the YYYYMMDD acquisition date from the Sentinel-2 path."""
    match = re.search(r'MSIL[12]C_(\d{8})T', path)
    if match: return match.group(1)
    match_general = re.search(r'(\d{8})T\d{6}', path)
    if match_general: return match_general.group(1)
    return None

def find_image_pairs():
    """Finds all JP2 files, extracts dates, sorts, and creates sequential pairs."""
    
    # 1. Find all TCI.jp2 files recursively
    # Assumes the JP2 files are under subdirectories within BASE_DATASET_PATH
    search_path = os.path.join(BASE_DATASET_PATH, "**", "*TCI.jp2")
    all_file_paths = sorted(glob.glob(search_path, recursive=True))

    if len(all_file_paths) < 2:
        return pd.DataFrame()

    # 2. Collect metadata and sort
    image_metadata = []
    for path in all_file_paths:
        date_str = extract_date_from_path(path)
        if date_str:
            image_metadata.append({'path': path, 'date': date_str})

    df_meta = pd.DataFrame(image_metadata).sort_values(by='date').reset_index(drop=True)
    
    # 3. Create sequential pairs (Src -> Ref)
    image_pairs = []
    for i in range(len(df_meta) - 1):
        image_pairs.append({
            'path_src': df_meta.iloc[i]['path'],
            'date_src': df_meta.iloc[i]['date'],
            'path_ref': df_meta.iloc[i+1]['path'],
            'date_ref': df_meta.iloc[i+1]['date']
        })

    return pd.DataFrame(image_pairs)


def load_all_data():
    """Main function to load labels and create image pairs."""
    
    # 1. Load labels
    try:
        df_labels = gpd.read_file(LABELS_PATH)
    except Exception:
        print(f"Warning: Failed to load GeoJSON labels from {LABELS_PATH}. Proceeding without mask.")
        df_labels = gpd.GeoDataFrame() 

    # 2. Find and create pairs
    df_pairs = find_image_pairs()
    
    print(f"Loaded {len(df_pairs)} image pairs.")
    return df_pairs, df_labels

if __name__ == '__main__':
    # Test block: Check if paths and loading work
    pairs, labels = load_all_data()
    if not pairs.empty:
        print("Test: First pair found:")
        print(pairs.head(1))
        
        test_path = pairs.iloc[0]['path_src']
        data = load_process_and_mask_image(test_path, labels)
        if data:
            print(f"Test image loaded and masked. Mono shape: {data['img_mono'].shape}")
    else:
        print("Test failed: No image pairs found. Check BASE_DATASET_PATH.")