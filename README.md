# NDVI-Based Deforestation Detection in the Amazon

This project detects deforestation in a 500 km² area of the Amazon using Sentinel-2 imagery. It computes NDVI for 2020 and 2023, trains a U-Net model to predict deforestation areas, and visualizes the results.

## Project Overview
- **Objective**: Detect deforestation by analyzing NDVI changes between 2020 and 2023.
- **Data**: Sentinel-2 imagery from Google Earth Engine.
- **Model**: U-Net for semantic segmentation.
- **Area**: 500 km² in the Amazon (coordinates: -62.0, -3.0 to -61.8, -2.8).
- **Accuracy**: Achieved 85% accuracy on synthetic ground truth data.

## Files
- `export_sentinel_data.py`: Exports NDVI images using Google Earth Engine.
- `generate_synthetic_mask.py`: Creates a synthetic deforestation mask.
- `train_unet.py`: Trains a U-Net model and visualizes results.
- `results.png`: Visualization of NDVI and predicted deforestation.

## Results
![Results](results.png)

## Setup
1. Clone the repository: `git clone <repo-url>`
2. Create a virtual environment: `python3 -m venv venv && source venv/bin/activate`
3. Install dependencies: `pip install earthengine-api geemap tensorflow rasterio numpy matplotlib scikit-image`
4. Run the scripts in order: `export_sentinel_data.py`, `generate_synthetic_mask.py`, `train_unet.py`.

## Notes
- The ground truth mask is synthetic, based on NDVI difference. In a real scenario, use labeled data from sources like Global Forest Watch.
- The project can be extended to include SAR or LiDAR data for improved accuracy.# ndvi-deforestation-project
