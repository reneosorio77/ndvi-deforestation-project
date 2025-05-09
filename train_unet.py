import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import rasterio
import matplotlib.pyplot as plt
from skimage.transform import resize
import glob

# Define custom F1 score metric
def f1_score(y_true, y_pred):
    y_pred = tf.round(y_pred)
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32))
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.float32))
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

# Define U-Net model
def unet_model(input_shape=(256, 256, 2)):
    inputs = layers.Input(input_shape)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    u4 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(c3)
    u4 = layers.concatenate([u4, c2])
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(u4)
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(c4)
    u5 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c1])
    c5 = layers.Conv2D(64, 3, activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(64, 3, activation='relu', padding='same')(c5)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c5)
    return models.Model(inputs, outputs)

# Load and preprocess data
def load_ndvi(file_path):
    with rasterio.open(file_path) as src:
        ndvi = src.read(1)
        ndvi = (ndvi - np.min(ndvi)) / (np.max(ndvi) - np.min(ndvi))
    return ndvi

def load_mask(file_path):
    with rasterio.open(file_path) as src:
        mask = src.read(1)
    return mask

# Function to visualize and save results
def save_results(ndvi_2020, ndvi_2023, pred_mask, filename):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title('NDVI 2020')
    plt.imshow(ndvi_2020, cmap='Greens')
    plt.subplot(1, 3, 2)
    plt.title('NDVI 2023')
    plt.imshow(ndvi_2023, cmap='Greens')
    plt.subplot(1, 3, 3)
    plt.title('Predicted Deforestation')
    plt.imshow(pred_mask, cmap='Reds')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Stage 1: Initial Model (results.png)
print("Stage 1: Initial Model")
# Load data for the first region only
ndvi_2020 = load_ndvi('data/NDVI_2020_Amazon_region_1.tif')
ndvi_2023 = load_ndvi('data/NDVI_2023_Amazon_region_1.tif')
mask = load_mask('data/deforestation_mask_region_1_initial.tif')  # Generated with threshold -0.2

# Resize images
ndvi_2020_resized = resize(ndvi_2020, (256, 256), anti_aliasing=True)
ndvi_2023_resized = resize(ndvi_2023, (256, 256), anti_aliasing=True)
mask = resize(mask, (256, 256), anti_aliasing=False, preserve_range=True).astype(np.uint8)

# Prepare data
images = np.stack([ndvi_2020_resized, ndvi_2023_resized], axis=-1)[np.newaxis, ...]
masks = mask[np.newaxis, ..., np.newaxis]

# Train U-Net
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(images, masks, epochs=5, batch_size=1)

# Predict and save
pred_mask = model.predict(images)[0, ..., 0]
save_results(ndvi_2020_resized, ndvi_2023_resized, pred_mask, 'results.png')

# Stage 2: Improved Model (Weighted Loss) (results_weighted.png)
print("Stage 2: Improved Model (Weighted Loss)")
# Compute class weights
total_pixels = mask.size
deforested_pixels = np.sum(mask == 1)
non_deforested_pixels = np.sum(mask == 0)
weight_for_0 = 1.0
weight_for_1 = (non_deforested_pixels / deforested_pixels) if deforested_pixels > 0 else 10.0
class_weights = {0: weight_for_0, 1: weight_for_1}
print(f"Class weights: {class_weights}")

# Prepare sample weights
sample_weights = np.where(masks == 1, class_weights[1], class_weights[0])

# Train U-Net
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1_score])
model.fit(images, masks, epochs=10, batch_size=1, sample_weight=sample_weights)

# Predict and save
pred_mask = model.predict(images)[0, ..., 0]
save_results(ndvi_2020_resized, ndvi_2023_resized, pred_mask, 'results_weighted.png')

# Stage 3: Final Model (Adjusted Mask) (results_final.png)
print("Stage 3: Final Model (Adjusted Mask)")
# Load mask with threshold -0.35
mask = load_mask('data/deforestation_mask_region_1_final.tif')  # Generated with threshold -0.35
mask = resize(mask, (256, 256), anti_aliasing=False, preserve_range=True).astype(np.uint8)
masks = mask[np.newaxis, ..., np.newaxis]

# Compute class weights
total_pixels = mask.size
deforested_pixels = np.sum(mask == 1)
non_deforested_pixels = np.sum(mask == 0)
weight_for_0 = 1.0
weight_for_1 = (non_deforested_pixels / deforested_pixels) if deforested_pixels > 0 else 10.0
class_weights = {0: weight_for_0, 1: weight_for_1}
print(f"Class weights: {class_weights}")

# Prepare sample weights
sample_weights = np.where(masks == 1, class_weights[1], class_weights[0])

# Train U-Net
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1_score])
model.fit(images, masks, epochs=10, batch_size=1, sample_weight=sample_weights)

# Predict and save
pred_mask = model.predict(images)[0, ..., 0]
save_results(ndvi_2020_resized, ndvi_2023_resized, pred_mask, 'results_final.png')

# Stage 4: Refined Model (Excluded Water Bodies) (results_refined.png)
print("Stage 4: Refined Model (Excluded Water Bodies)")
# Load mask with threshold -0.35 and NDVI < 0.2 water exclusion
mask = load_mask('data/deforestation_mask_region_1_refined.tif')  # Generated with threshold -0.35 and NDVI < 0.2
mask = resize(mask, (256, 256), anti_aliasing=False, preserve_range=True).astype(np.uint8)
masks = mask[np.newaxis, ..., np.newaxis]

# Compute class weights
total_pixels = mask.size
deforested_pixels = np.sum(mask == 1)
non_deforested_pixels = np.sum(mask == 0)
weight_for_0 = 1.0
weight_for_1 = (non_deforested_pixels / deforested_pixels) if deforested_pixels > 0 else 10.0
class_weights = {0: weight_for_0, 1: weight_for_1}
print(f"Class weights: {class_weights}")

# Prepare sample weights
sample_weights = np.where(masks == 1, class_weights[1], class_weights[0])

# Train U-Net with early stopping and reduced learning rate
model = unet_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy', f1_score])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='f1_score', mode='max', patience=3, restore_best_weights=True)
model.fit(images, masks, epochs=20, batch_size=1, sample_weight=sample_weights, callbacks=[early_stopping])

# Predict and save
pred_mask = model.predict(images)[0, ..., 0]
save_results(ndvi_2020_resized, ndvi_2023_resized, pred_mask, 'results_refined.png')

# Stage 5: Multi-Region Model (results_multi_region.png)
print("Stage 5: Multi-Region Model")
# Load data for multiple regions
ndvi_2020_files = sorted(glob.glob('data/NDVI_2020_Amazon_region_*.tif'))
ndvi_2023_files = sorted(glob.glob('data/NDVI_2023_Amazon_region_*.tif'))
mask_files = sorted(glob.glob('data/deforestation_mask_region_*_refined.tif'))  # Use refined masks (NDVI < 0.2)

images_list = []
masks_list = []
sample_weights_list = []

for ndvi_2020_file, ndvi_2023_file, mask_file in zip(ndvi_2020_files, ndvi_2023_files, mask_files):
    ndvi_2020 = load_ndvi(ndvi_2020_file)
    ndvi_2023 = load_ndvi(ndvi_2023_file)
    mask = load_mask(mask_file)

    ndvi_2020 = resize(ndvi_2020, (256, 256), anti_aliasing=True)
    ndvi_2023 = resize(ndvi_2023, (256, 256), anti_aliasing=True)
    mask = resize(mask, (256, 256), anti_aliasing=False, preserve_range=True).astype(np.uint8)

    total_pixels = mask.size
    deforested_pixels = np.sum(mask == 1)
    non_deforested_pixels = np.sum(mask == 0)
    weight_for_0 = 1.0
    weight_for_1 = (non_deforested_pixels / deforested_pixels) if deforested_pixels > 0 else 10.0
    class_weights = {0: weight_for_0, 1: weight_for_1}
    print(f"Region {ndvi_2020_file}: Class weights: {class_weights}")

    image = np.stack([ndvi_2020, ndvi_2023], axis=-1)
    images_list.append(image)
    masks_list.append(mask)

    sample_weights = np.where(mask == 1, class_weights[1], class_weights[0])
    sample_weights_list.append(sample_weights)

images = np.stack(images_list, axis=0)
masks = np.stack(masks_list, axis=0)[..., np.newaxis]
sample_weights = np.stack(sample_weights_list, axis=0)

# Train U-Net
model = unet_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy', f1_score])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='f1_score', mode='max', patience=3, restore_best_weights=True)
model.fit(images, masks, epochs=20, batch_size=1, sample_weight=sample_weights, callbacks=[early_stopping])

# Predict and save for the first region
pred_mask = model.predict(images)[0, ..., 0]
ndvi_2020_resized = resize(load_ndvi(ndvi_2020_files[0]), (256, 256), anti_aliasing=True)
ndvi_2023_resized = resize(load_ndvi(ndvi_2023_files[0]), (256, 256), anti_aliasing=True)
save_results(ndvi_2020_resized, ndvi_2023_resized, pred_mask, 'results_multi_region.png')

# Stage 6: NDWI-Enhanced Model (results_multi_region.png)
print("Stage 6: NDWI-Enhanced Model")
# Load masks with NDWI-based water exclusion
mask_files = sorted(glob.glob('data/deforestation_mask_region_*.tif'))  # Latest masks with NDWI

images_list = []
masks_list = []
sample_weights_list = []

for ndvi_2020_file, ndvi_2023_file, mask_file in zip(ndvi_2020_files, ndvi_2023_files, mask_files):
    ndvi_2020 = load_ndvi(ndvi_2020_file)
    ndvi_2023 = load_ndvi(ndvi_2023_file)
    mask = load_mask(mask_file)

    ndvi_2020 = resize(ndvi_2020, (256, 256), anti_aliasing=True)
    ndvi_2023 = resize(ndvi_2023, (256, 256), anti_aliasing=True)
    mask = resize(mask, (256, 256), anti_aliasing=False, preserve_range=True).astype(np.uint8)

    total_pixels = mask.size
    deforested_pixels = np.sum(mask == 1)
    non_deforested_pixels = np.sum(mask == 0)
    weight_for_0 = 1.0
    weight_for_1 = (non_deforested_pixels / deforested_pixels) if deforested_pixels > 0 else 10.0
    class_weights = {0: weight_for_0, 1: weight_for_1}
    print(f"Region {ndvi_2020_file}: Class weights: {class_weights}")

    image = np.stack([ndvi_2020, ndvi_2023], axis=-1)
    images_list.append(image)
    masks_list.append(mask)

    sample_weights = np.where(mask == 1, class_weights[1], class_weights[0])
    sample_weights_list.append(sample_weights)

images = np.stack(images_list, axis=0)
masks = np.stack(masks_list, axis=0)[..., np.newaxis]
sample_weights = np.stack(sample_weights_list, axis=0)

# Train U-Net
model = unet_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy', f1_score])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='f1_score', mode='max', patience=3, restore_best_weights=True)
model.fit(images, masks, epochs=20, batch_size=1, sample_weight=sample_weights, callbacks=[early_stopping])

# Predict and save for the first region
pred_mask = model.predict(images)[0, ..., 0]
ndvi_2020_resized = resize(load_ndvi(ndvi_2020_files[0]), (256, 256), anti_aliasing=True)
ndvi_2023_resized = resize(load_ndvi(ndvi_2023_files[0]), (256, 256), anti_aliasing=True)
save_results(ndvi_2020_resized, ndvi_2023_resized, pred_mask, 'results_multi_region.png')

print("All stages completed and results saved.")