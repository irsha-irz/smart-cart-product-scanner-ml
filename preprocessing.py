import os
import cv2
import numpy as np

# Path to dataset directory
dataset_dir = "dataset-aug"
processed_dir = "processed_dataset"

# Parameters
image_size = (128, 128)  # Resize all images to 128x128

def preprocess_images():
    # Create a directory to store processed images
    os.makedirs(processed_dir, exist_ok=True)

    for category in os.listdir(dataset_dir):
        category_path = os.path.join(dataset_dir, category)
        if not os.path.isdir(category_path):
            continue

        # Create folder for processed category images
        processed_category_path = os.path.join(processed_dir, category)
        os.makedirs(processed_category_path, exist_ok=True)

        print(f"Processing category: {category}")
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)

            # Read the image
            image = cv2.imread(file_path)
            if image is None:
                print(f"Could not read {file_path}. Skipping...")
                continue

            # Resize the image
            image_resized = cv2.resize(image, image_size)

            # Normalize pixel values to [0, 1]
            image_normalized = image_resized / 255.0

            # Convert to grayscale (optional, uncomment if needed)
            # image_gray = cv2.cvtColor( image_normalized, cv2.COLOR_BGR2GRAY)

            # Save the processed image
            processed_file_path = os.path.join(processed_category_path, file_name)
            cv2.imwrite(processed_file_path, (image_normalized * 255).astype(np.uint8))  # Convert back to uint8
            print(f"Processed and saved: {processed_file_path}")

if __name__ == "__main__":
    preprocess_images()