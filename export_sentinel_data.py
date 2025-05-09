import ee
import geemap

# Initialize Google Earth Engine with project ID
ee.Initialize(project='your-project-id')

# Define region of interest (ROI) - a 500 km² area in the Amazon
roi = ee.Geometry.Rectangle([-62.0, -3.0, -61.8, -2.8])

# Define time periods for comparison
start_date_1 = '2020-01-01'
end_date_1 = '2020-12-31'
start_date_2 = '2023-01-01'
end_date_2 = '2023-12-31'

# Load Sentinel-2 imagery and filter by date and region
def get_sentinel_collection(start_date, end_date, roi):
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterDate(start_date, end_date)
                  .filterBounds(roi)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                  .select(['B4', 'B8']))
    return collection.median().clip(roi)

# Get imagery for both periods
image_2020 = get_sentinel_collection(start_date_1, end_date_1, roi)
image_2023 = get_sentinel_collection(start_date_2, end_date_2, roi)

# Calculate NDVI for both periods
def calculate_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return ndvi

ndvi_2020 = calculate_ndvi(image_2020)
ndvi_2023 = calculate_ndvi(image_2023)

# Export NDVI images to Google Drive
export_task_2020 = ee.batch.Export.image.toDrive(
    image=ndvi_2020,
    description='NDVI_2020_Amazon',
    folder='NDVI_Deforestation_Project',
    region=roi,
    scale=10,
    crs='EPSG:4326',
    maxPixels=1e9
)
export_task_2020.start()
print("Exporting NDVI_2020_Amazon to Google Drive...")

export_task_2023 = ee.batch.Export.image.toDrive(
    image=ndvi_2023,
    description='NDVI_2023_Amazon',
    folder='NDVI_Deforestation_Project',
    region=roi,
    scale=10,
    crs='EPSG:4326',
    maxPixels=1e9
)
export_task_2023.start()
print("Exporting NDVI_2023_Amazon to Google Drive...")