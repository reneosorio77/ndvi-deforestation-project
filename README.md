# NDVI Deforestation Detection Project

## Overview
This project detects deforestation in the Amazon using Sentinel-2 imagery, NDVI (Normalized Difference Vegetation Index), and a U-Net model. The project compares NDVI data from 2020 and 2023 to identify areas of deforestation.

## Workflow
1. **Data Acquisition**: Sentinel-2 imagery is exported using Google Earth Engine (`export_sentinel_data.py`).
2. **Synthetic Mask Generation**: A synthetic deforestation mask is created based on NDVI differences (`generate_synthetic_mask.py`).
3. **Model Training**: A U-Net model is trained to predict deforestation areas (`train_unet.py`).

## Results
### Initial Model
The initial model labeled 98.31% of pixels as deforested due to an incorrect threshold, achieving 97.12% accuracy but failing to detect meaningful patterns.

![Initial Results](results.png)

### Improved Model (Weighted Loss)
After applying a weighted loss function and F1 score metric, the model improved but still struggled due to the imbalanced mask.

![Weighted Results](results_weighted.png)

### Final Model (Adjusted Mask)
Adjusted the mask threshold to -0.35, labeling only 4.87% of pixels as deforested, but the prediction highlighted a river due to water body mislabeling (F1 score: 0.146).

![Final Results](results_final.png)

### Refined Model (Excluded Water Bodies)
Refined the synthetic mask to exclude water bodies (NDVI < 0.2), added early stopping, and reduced the learning rate, improving deforestation detection (F1 score: 0.30).

![Refined Results](results_refined.png)

## Files
- `export_sentinel_data.py`: Exports NDVI data from Google Earth Engine.
- `generate_synthetic_mask.py`: Creates a synthetic deforestation mask.
- `train_unet.py`: Trains the U-Net model and visualizes results.
- `results_refined.png`: Final visualization of NDVI 2020, NDVI 2023, and predicted deforestation.

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/reneosorio77/ndvi-deforestation-project.git
   cd ndvi_deforestation_project