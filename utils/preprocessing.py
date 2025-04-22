import cv2
import numpy as np

def preprocess_image(image):
    """Convert uploaded image to model input format"""
    img = np.array(image)
    
    # Convert to RGB if needed
    if len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Resize and normalize
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    
    return img
