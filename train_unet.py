import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import rasterio
import matplotlib.pyplot as plt
from skimage.transform import resize

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

# Load NDVI images and mask
ndvi_2020 = load_ndvi('data/NDVI_2020_Amazon.tif')
ndvi_2023 = load_ndvi('data/NDVI_2023_Amazon.tif')
mask = load_mask('data/deforestation_mask.tif')

# Resize images to 256x256 for U-Net
ndvi_2020 = resize(ndvi_2020, (256, 256), anti_aliasing=True)
ndvi_2023 = resize(ndvi_2023, (256, 256), anti_aliasing=True)
mask = resize(mask, (256, 256), anti_aliasing=False, preserve_range=True).astype(np.uint8)

# Compute class weights
total_pixels = mask.size
deforested_pixels = np.sum(mask == 1)
non_deforested_pixels = np.sum(mask == 0)
weight_for_0 = 1.0
weight_for_1 = (non_deforested_pixels / deforested_pixels) if deforested_pixels > 0 else 10.0
class_weights = {0: weight_for_0, 1: weight_for_1}
print(f"Class weights: {class_weights}")

# Stack NDVI images
images = np.stack([ndvi_2020, ndvi_2023], axis=-1)[np.newaxis, ...]
masks = mask[np.newaxis, ..., np.newaxis]

# Build and train U-Net
model = unet_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Reduced learning rate
              loss='binary_crossentropy',
              metrics=['accuracy', f1_score])

# Use sample weights
sample_weights = np.where(masks == 1, class_weights[1], class_weights[0])

# Train with sample weights and early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='f1_score', mode='max', patience=3, restore_best_weights=True)
model.fit(images, masks, epochs=20, batch_size=1, sample_weight=sample_weights, callbacks=[early_stopping])

# Predict deforestation
pred_mask = model.predict(images)[0, ..., 0]

# Visualize results
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
plt.savefig('results_refined.png')
plt.show()

# Save the model
model.save('unet_deforestation_model_refined.h5')
print("Model and results saved.")