import ee
import geemap

# Initialize Google Earth Engine
ee.Initialize(project='Replace with your actual Project ID')  # Replace with your actual Project ID

# Define multiple regions of interest (ROIs) in the Amazon
rois = [
    ee.Geometry.Rectangle([-62.0, -3.0, -61.8, -2.8]),  # Region 1
    ee.Geometry.Rectangle([-61.8, -3.0, -61.6, -2.8]),  # Region 2
    ee.Geometry.Rectangle([-61.6, -3.0, -61.4, -2.8]),  # Region 3
]

# Function to calculate NDVI
def calculate_ndvi(image):
    nir = image.select('B8')
    red = image.select('B4')
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    return image.addBands(ndvi)

# Function to mask clouds using SCL
def mask_clouds(image):
    scl = image.select('SCL')
    # Keep only vegetation, water, and soil pixels (exclude clouds, cloud shadows)
    cloud_free = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6))
    return image.updateMask(cloud_free)

# Export NDVI, Green (B3), and NIR (B8) for each region
for i, roi in enumerate(rois):
    # Load Sentinel-2 Level-2A imagery for 2020 and 2023 (dry season: June to August)
    sentinel_2020 = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(roi) \
        .filterDate('2020-06-01', '2020-08-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
        .map(mask_clouds) \
        .map(calculate_ndvi)

    sentinel_2023 = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(roi) \
        .filterDate('2023-06-01', '2023-08-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
        .map(mask_clouds) \
        .map(calculate_ndvi)

    # Export NDVI
    ndvi_2020 = sentinel_2020.select('NDVI').median()
    ndvi_2023 = sentinel_2023.select('NDVI').median()
    geemap.ee_export_image(ndvi_2020, filename=f'data/NDVI_2020_Amazon_region_{i+1}.tif', scale=10, region=roi)
    geemap.ee_export_image(ndvi_2023, filename=f'data/NDVI_2023_Amazon_region_{i+1}.tif', scale=10, region=roi)

    # Export Green (B3) and NIR (B8) for 2020
    green_2020 = sentinel_2020.select('B3').median()
    nir_2020 = sentinel_2020.select('B8').median()
    geemap.ee_export_image(green_2020, filename=f'data/B3_2020_Amazon_region_{i+1}.tif', scale=10, region=roi)
    geemap.ee_export_image(nir_2020, filename=f'data/B8_2020_Amazon_region_{i+1}.tif', scale=10, region=roi)

print("NDVI, Green (B3), and NIR (B8) images exported for multiple regions.")