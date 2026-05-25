import os
import cv2
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# Configuration
CONFIG = {
    "pos_img": "./data/raw/images/positive",
    "pos_mask": "./data/raw/masks/positive",
    "neg_img": "./data/raw/images/negative",
    "neg_mask": "./data/raw/masks/negative",
    "model_path": "./src/models/knn_model.pkl",
    "scaler_path": "./src/models/scaler.pkl",
    "img_size": (128, 128),
    "num_pos": 100,
    "num_neg": 25
}


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
        
        img_res = cv2.resize(img, CONFIG["img_size"], interpolation=cv2.INTER_AREA)
        mask_res = cv2.resize(mask, CONFIG["img_size"], interpolation=cv2.INTER_NEAREST)
        
        h, w = img_res.shape
        coords = np.indices((h, w)).reshape(2, -1).T
        features = np.hstack((img_res.flatten().reshape(-1, 1), coords))
        
        X.append(features)
        y.append((mask_res.flatten() > 0).astype(int))
        
    return X, y


def train():
    print("Loading and processing images...")
    
    # Separate positive and negative samples
    X_pos, y_pos = load_and_process(CONFIG["pos_img"], CONFIG["pos_mask"], CONFIG["num_pos"])
    X_neg, y_neg = load_and_process(CONFIG["neg_img"], CONFIG["neg_mask"], CONFIG["num_neg"])
    
    # Combine the data
    X = np.vstack(X_pos + X_neg)
    y = np.concatenate(y_pos + y_neg)
    
    print(f"Total pixels: {X.shape[0]}. Beginning scaling...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Training k-NN (Neighbors=5)...")
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_scaled, y)
    
    # Save model and scaler
    os.makedirs("./src/models", exist_ok=True)
    joblib.dump(knn, CONFIG["model_path"])
    joblib.dump(scaler, CONFIG["scaler_path"])
    print(f"Done! Model saved to {CONFIG['model_path']}")


if __name__ == "__main__":
    train()