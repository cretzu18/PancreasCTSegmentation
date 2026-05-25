import os
import cv2
import numpy as np
import joblib
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


# Configuration
CONFIG = {
    "pos_img": "./data/raw/images/positive",
    "pos_mask": "./data/raw/masks/positive",
    "neg_img": "./data/raw/images/negative",
    "neg_mask": "./data/raw/masks/negative",
    "model_path": "./src/models/svm_model.pkl",
    "scaler_path": "./src/models/svm_scaler.pkl",
    "img_size": (128, 128),
    "num_pos": 100,
    "num_neg": 25
}

def extract_features(img, size):
    """
    Extract features from the image for each pixel:
    - Intensity
    - Coordinates (x, y)
    - Local mean (texture)
    - Local standard deviation (texture)
    """
    img_res = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    h, w = img_res.shape
    
    # Intensity + Coords
    coords = np.indices((h, w)).reshape(2, -1).T
    intensity = img_res.flatten().reshape(-1, 1)
    
    # Local mean (texture)
    local_mean = cv2.blur(img_res, (3, 3)).flatten().reshape(-1, 1)
    
    # Local std (texture)
    img_float = img_res.astype(np.float32)
    mean_sq = cv2.blur(img_float**2, (3, 3))
    sq_mean = cv2.blur(img_float, (3, 3))**2
    local_std = np.sqrt(np.maximum(0, mean_sq - sq_mean)).flatten().reshape(-1, 1)
    
    return np.hstack((intensity, coords, local_mean, local_std))


def load_and_process(img_dir, mask_dir, num_samples):
    X, y = [], []
    files = os.listdir(img_dir)[:num_samples]
    
    for filename in files:
        img_path = os.path.join(img_dir, filename)
        mask_path = os.path.join(mask_dir, filename.replace("img", "mask"))
        
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue
            
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        features = extract_features(img, CONFIG["img_size"])
        mask_res = cv2.resize(mask, CONFIG["img_size"], interpolation=cv2.INTER_NEAREST)
        labels = (mask_res.flatten() > 0).astype(int)
        
        X.append(features)
        y.append(labels)
        
    return X, y


def train():
    print("Loading and processing images...")
    
    # Separate positive and negative samples
    X_pos, y_pos = load_and_process(CONFIG["pos_img"], CONFIG["pos_mask"], CONFIG["num_pos"])
    X_neg, y_neg = load_and_process(CONFIG["neg_img"], CONFIG["neg_mask"], CONFIG["num_neg"])
    
    # Combine the data
    X = np.vstack(X_pos + X_neg)
    y = np.concatenate(y_pos + y_neg)
    
    # Subsample the negative class to balance the dataset
    X_pos = X[y == 1]
    X_neg = X[y == 0]

    X_neg_upsampled = resample(X_neg, replace=False, n_samples=len(X_pos), random_state=42)
    
    X_balanced = np.vstack((X_pos, X_neg_upsampled))
    y_balanced = np.hstack((np.ones(len(X_pos)), np.zeros(len(X_neg_upsampled))))
    
    # Normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_balanced)

    print("Training SVM...")
    svm = LinearSVC(C=0.01, max_iter=10000, random_state=42)
    svm.fit(X_scaled, y_balanced)

    # Save model and scaler
    os.makedirs("./src/models", exist_ok=True)
    joblib.dump(svm, CONFIG["model_path"])
    joblib.dump(scaler, CONFIG["scaler_path"])
    print(f"Done! Model saved to {CONFIG['model_path']}")


if __name__ == "__main__":
    train()