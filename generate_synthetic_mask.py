import rasterio
import numpy as np
import glob

# Find all NDVI, Green (B3), and NIR (B8) files for 2020 and 2023
ndvi_2020_files = sorted(glob.glob('data/NDVI_2020_Amazon_region_*.tif'))
ndvi_2023_files = sorted(glob.glob('data/NDVI_2023_Amazon_region_*.tif'))
green_2020_files = sorted(glob.glob('data/B3_2020_Amazon_region_*.tif'))
nir_2020_files = sorted(glob.glob('data/B8_2020_Amazon_region_*.tif'))

for i, (ndvi_2020_file, ndvi_2023_file, green_2020_file, nir_2020_file) in enumerate(zip(ndvi_2020_files, ndvi_2023_files, green_2020_files, nir_2020_files)):
    # Load NDVI images
    with rasterio.open(ndvi_2020_file) as src:
        ndvi_2020 = src.read(1)
    with rasterio.open(ndvi_2023_file) as src:
        ndvi_2023 = src.read(1)

    # Load Green (B3) and NIR (B8) for NDWI calculation
    with rasterio.open(green_2020_file) as src:
        green_2020 = src.read(1).astype(np.float32)
    with rasterio.open(nir_2020_file) as src:
        nir_2020 = src.read(1).astype(np.float32)

    # Compute NDWI
    ndwi_2020 = (green_2020 - nir_2020) / (green_2020 + nir_2020 + 1e-10)
    water_mask = ndwi_2020 < 0

    # Compute NDVI difference
    ndvi_diff = ndvi_2023 - ndvi_2020

    # Analyze NDVI values and differences
    print(f"Region {i+1}:")
    print("NDVI 2020 - Min, Mean, Max:", ndvi_2020.min(), ndvi_2020.mean(), ndvi_2020.max())
    print("NDVI 2023 - Min, Mean, Max:", ndvi_2023.min(), ndvi_2023.mean(), ndvi_2023.max())
    print("NDVI Difference - Min, Mean, Max:", ndvi_diff.min(), ndvi_diff.mean(), ndvi_diff.max())
    print("NDVI Difference Percentiles (10th, 50th, 90th):", np.percentile(ndvi_diff, [10, 50, 90]))

    # Get profile for saving masks
    with rasterio.open(ndvi_2020_file) as src:
        profile = src.profile
        profile.update(dtype=rasterio.uint8, count=1)

    # Stage 1: Initial mask (threshold -0.2, no water exclusion)
    threshold = -0.2
    mask = np.where(ndvi_diff < threshold, 1, 0).astype(np.uint8)
    with rasterio.open(f'data/deforestation_mask_region_{i+1}_initial.tif', 'w', **profile) as dst:
        dst.write(mask, 1)
    print(f"Initial mask (threshold {threshold}): Deforested pixels: {np.sum(mask == 1)} ({np.sum(mask == 1)/mask.size*100:.2f}%)")

    # Stage 3: Final mask (threshold -0.5, no water exclusion)
    threshold = -0.5  # Adjusted to match 4.87% deforested pixels
    mask = np.where(ndvi_diff < threshold, 1, 0).astype(np.uint8)
    with rasterio.open(f'data/deforestation_mask_region_{i+1}_final.tif', 'w', **profile) as dst:
        dst.write(mask, 1)
    print(f"Final mask (threshold {threshold}): Deforested pixels: {np.sum(mask == 1)} ({np.sum(mask == 1)/mask.size*100:.2f}%)")

    # Stage 4: Refined mask (threshold -0.5, NDVI < 0.2 water exclusion)
    forest_mask = ndvi_2020 > 0.2
    mask = np.zeros_like(ndvi_2020, dtype=np.uint8)
    mask[forest_mask & (ndvi_diff < threshold)] = 1
    with rasterio.open(f'data/deforestation_mask_region_{i+1}_refined.tif', 'w', **profile) as dst:
        dst.write(mask, 1)
    print(f"Refined mask (threshold {threshold}, NDVI > 0.2): Deforested pixels: {np.sum(mask == 1)} ({np.sum(mask == 1)/mask.size*100:.2f}%)")

    # Stage 6: NDWI-enhanced mask (threshold -0.5, NDWI < 0 and NDVI < 0.2 water exclusion)
    forest_mask = (ndvi_2020 > 0.2) & (water_mask)
    mask = np.zeros_like(ndvi_2020, dtype=np.uint8)
    mask[forest_mask & (ndvi_diff < threshold)] = 1
    with rasterio.open(f'data/deforestation_mask_region_{i+1}.tif', 'w', **profile) as dst:
        dst.write(mask, 1)
    print(f"NDWI-enhanced mask (threshold {threshold}, NDWI < 0): Deforested pixels: {np.sum(mask == 1)} ({np.sum(mask == 1)/mask.size*100:.2f}%)")

print("All masks generated.")