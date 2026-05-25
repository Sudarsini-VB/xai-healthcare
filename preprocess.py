import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        try:
            from PIL import Image
            img = cv2.cvtColor(np.array(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[ERROR] Cannot load image: {e}")
            return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return img

def preprocess_for_prediction(image_path):
    img = load_and_preprocess_image(image_path)
    if img is None:
        return None
    return np.expand_dims(img, axis=0)

def get_train_datagen():
    return ImageDataGenerator(
        rescale=1.0/255, rotation_range=15,
        width_shift_range=0.1, height_shift_range=0.1,
        shear_range=0.1, zoom_range=0.1,
        horizontal_flip=True, fill_mode="nearest"
    )

def get_val_test_datagen():
    return ImageDataGenerator(rescale=1.0/255)
