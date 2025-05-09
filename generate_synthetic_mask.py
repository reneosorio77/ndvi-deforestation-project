import rasterio
import numpy as np

# Load NDVI images
with rasterio.open('data/NDVI_2020_Amazon.tif') as src:
    ndvi_2020 = src.read(1)
with rasterio.open('data/NDVI_2023_Amazon.tif') as src:
    ndvi_2023 = src.read(1)

# Compute NDVI difference
ndvi_diff = ndvi_2023 - ndvi_2020

# Analyze NDVI values and differences
print("NDVI 2020 - Min, Mean, Max:", ndvi_2020.min(), ndvi_2020.mean(), ndvi_2020.max())
print("NDVI 2023 - Min, Mean, Max:", ndvi_2023.min(), ndvi_2023.mean(), ndvi_2023.max())
print("NDVI Difference - Min, Mean, Max:", ndvi_diff.min(), ndvi_diff.mean(), ndvi_diff.max())
print("NDVI Difference Percentiles (10th, 50th, 90th):", np.percentile(ndvi_diff, [10, 50, 90]))

# Mask out water bodies (NDVI < 0.2 in 2020)
forest_mask = ndvi_2020 > 0.2  # Only consider pixels that were forested in 2020

# Create a synthetic mask: 1 for deforestation (significant NDVI decrease), 0 otherwise
threshold = -0.35
mask = np.zeros_like(ndvi_2020, dtype=np.uint8)
mask[forest_mask & (ndvi_diff < threshold)] = 1  # Label as deforested only if forested in 2020 and NDVI dropped

# Check class distribution
total_pixels = mask.size
deforested_pixels = np.sum(mask == 1)
non_deforested_pixels = np.sum(mask == 0)
print(f"Deforested pixels (1): {deforested_pixels} ({deforested_pixels/total_pixels*100:.2f}%)")
print(f"Non-deforested pixels (0): {non_deforested_pixels} ({non_deforested_pixels/total_pixels*100:.2f}%)")

# Save the mask as a GeoTIFF
with rasterio.open('data/NDVI_2020_Amazon.tif') as src:
    profile = src.profile
    profile.update(dtype=rasterio.uint8, count=1)

with rasterio.open('data/deforestation_mask.tif', 'w', **profile) as dst:
    dst.write(mask, 1)

print("Synthetic deforestation mask created: data/deforestation_mask.tif")