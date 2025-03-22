from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.utils import img_to_array, load_img
import os

# Define Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=30,  # Rotate images randomly
    width_shift_range=0.2,  # Shift horizontally
    height_shift_range=0.2,  # Shift vertically
    zoom_range=0.2,  # Zoom-in/out
    horizontal_flip=True,  # Flip images
    brightness_range=[0.5, 1.5],  # Adjust brightness
    fill_mode="nearest"

)

# Define directories
input_base_dir = "C:/Users/91974/Projects/smart-checkout-system/v2/dataset"  # Change this to your base directory containing subdirectories of images
output_base_dir = "C:/Users/91974/Projects/smart-checkout-system/v2/dataset-aug"  # Change this to where augmented images should be saved

os.makedirs(output_base_dir, exist_ok=True)

# Iterate over each directory in input_base_dir
for subdir, _, files in os.walk(input_base_dir):
    if not files:  # Skip empty directories
        continue

    # Define output directory for augmented images
    relative_path = os.path.relpath(subdir, input_base_dir)  # Preserve folder structure
    output_dir = os.path.join(output_base_dir, relative_path)
    os.makedirs(output_dir, exist_ok=True)

    # Process each image in the directory
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Ensure only image files are processed
            img_path = os.path.join(subdir, file)
            image = load_img(img_path, target_size=(128, 128))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            # Generate augmented images
            i = 0
            for batch in datagen.flow(image, batch_size=1, save_to_dir=output_dir, save_prefix=f'aug_{os.path.splitext(file)[0]}', save_format='jpg'):
                i += 1
                if i > 10:  # Generate 10 variations per image
                    break

print("Data augmentation completed successfully!")
