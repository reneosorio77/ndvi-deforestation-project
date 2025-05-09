# NDVI Deforestation Detection Project

## Overview
This project detects deforestation in the Amazon using Sentinel-2 imagery, NDVI (Normalized Difference Vegetation Index), and a U-Net model. The project compares NDVI data from 2020 and 2023 to identify areas of deforestation.

## Workflow
1. **Data Acquisition**: Sentinel-2 imagery is exported using Google Earth Engine (`export_sentinel_data.py`).
2. **Synthetic Mask Generation**: A synthetic deforestation mask is created based on NDVI differences (`generate_synthetic_mask.py`).
3. **Model Training**: A U-Net model is trained to predict deforestation areas (`train_unet.py`).

## Results
The U-Net model achieved an accuracy of 97.12% after 5 epochs. Below are the NDVI images and predicted deforestation mask:

![Results](results.png)

## Files
- `export_sentinel_data.py`: Exports NDVI data from Google Earth Engine.
- `generate_synthetic_mask.py`: Creates a synthetic deforestation mask.
- `train_unet.py`: Trains the U-Net model and visualizes results.
- `results.png`: Visualization of NDVI 2020, NDVI 2023, and predicted deforestation.

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/reneosorio77/ndvi-deforestation-project.git
   cd ndvi_deforestation_project