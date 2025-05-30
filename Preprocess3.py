import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import time



def preprocess_images_to_dataset(real_path, fake_path, real_output, fake_output, image_size=(128, 128), save_npz=True):
    # Create output folders if they don’t exist
    os.makedirs(real_output, exist_ok=True)
    os.makedirs(fake_output, exist_ok=True)

    images = []
    labels = []
    
    # Process Real Images
    real_counter = 0
    for file in os.listdir(real_path):
        file_path = os.path.join(real_path, file)
        try:
            img = load_img(file_path, target_size=image_size)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(0)  # Label for real
            # Save processed image
            img_uint8 = (img_array * 255).astype(np.uint8)
            Image.fromarray(img_uint8).save(os.path.join(real_output, f"real_{real_counter:04d}.jpg"))
            real_counter += 1
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
        

    # Process Fake Images
    fake_counter = 0
    for file in os.listdir(fake_path):
        file_path = os.path.join(fake_path, file)
        try:
            img = load_img(file_path, target_size=image_size)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(1)  # Label for fake
            # Save processed image
            img_uint8 = (img_array * 255).astype(np.uint8)
            Image.fromarray(img_uint8).save(os.path.join(fake_output, f"fake_{fake_counter:04d}.jpg"))
            fake_counter += 1
        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    X = np.array(images)
    y = np.array(labels)
    

    print(f"✅ Processed {real_counter} real images and {fake_counter} fake images.")
    
    if save_npz:
        dataset_path = os.path.join(os.path.dirname(real_output), "profile_dataset.npz")
        np.savez_compressed(dataset_path, X=X, y=y)
        print(f"✅ Dataset saved to: {dataset_path}")

    return X, y

# Set your specific paths here
real_input_path = r"D:\fake-profile-detector\data\face_detection\training_real"
fake_input_path = r"D:\fake-profile-detector\data\face_detection\training_fake"
real_output_path = r"D:\fake-profile-detector\data\face_detection\processedtraining_real"
fake_output_path = r"D:\fake-profile-detector\data\face_detection\processedtraining_fake"
# Run the preprocessing
X, y = preprocess_images_to_dataset(real_input_path, fake_input_path, real_output_path, fake_output_path)
