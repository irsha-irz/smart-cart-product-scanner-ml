import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split

# Paths to dataset
processed_dir = "processed_dataset"
image_size = (128, 128)  # MobileNetV2 input size (minimum 96x96)
batch_size = 32
epochs = 10  # Faster convergence with transfer learning

# Load dataset
def load_data():
    images, labels = [], []
    product_list = os.listdir(processed_dir)
    product_to_label = {product: idx for idx, product in enumerate(product_list)}

    for product in product_list:
        product_path = os.path.join(processed_dir, product)
        if not os.path.isdir(product_path):
            continue
        
        for file_name in os.listdir(product_path):
            file_path = os.path.join(product_path, file_name)
            image = tf.keras.utils.load_img(file_path, target_size=image_size)
            image = tf.keras.utils.img_to_array(image) / 255.0
            images.append(image)
            labels.append(product_to_label[product])

    return np.array(images), np.array(labels), product_to_label

# Load and split data
images, labels, product_to_label = load_data()
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data augmentation
data_gen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    zoom_range=0.2, horizontal_flip=True
)

# Load MobileNetV2 (pre-trained)
base_model = MobileNetV2(input_shape=(image_size[0], image_size[1], 3), include_top=False, weights="imagenet")

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Replaces Flatten()
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output_layer = Dense(len(product_to_label), activation="softmax")(x)  # Multi-class classification

# Compile final model
model = Model(inputs=base_model.input, outputs=output_layer)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train model
model.fit(data_gen.flow(X_train, y_train, batch_size=batch_size), validation_data=(X_val, y_val), epochs=epochs)

# Save model and label mapping
model.save("product_classifier_transfer.h5")
np.save("product_labels.npy", product_to_label)

print("✅ Model training complete!")
