import torch
import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from src.models.unet import UNet

def visualize(image_name, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Images and model paths
    IMG_PATH = f"data/raw/images/positive/{image_name}"
    if not os.path.exists(IMG_PATH):
        IMG_PATH = f"data/raw/images/negative/{image_name}"
        if not os.path.exists(IMG_PATH):
            print(f"Error: {image_name} has not been found.")
            return
        
    MASK_PATH = IMG_PATH.replace("images", "masks").replace("img", "mask")
    MODEL_PATH = f"src/models/{model_name}"

    # Load the model
    model = UNet(in_channels=1,out_channels=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Process the image
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    mask_real = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    mask_real = cv2.resize(mask_real, (256, 256))

    # Normalization and prepare Tensor
    img_input = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_input).unsqueeze(0).unsqueeze(0).to(device)


    # Prediction
    with torch.no_grad():
        output = torch.sigmoid(model(img_tensor))
        output = output.squeeze().cpu().numpy()
        mask_pred = (output > 0.5).astype(np.uint8)

    mask_real_bin = (mask_real > 127).astype(np.uint8)

    # Vizualization
    plt.figure(figsize=(20,5))

    # Original image
    plt.subplot(1, 4, 1)
    plt.title(f"{image_name}")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    # Real mask
    plt.subplot(1, 4, 2)
    plt.title("Real mask")
    plt.imshow(mask_real, cmap='Greens')
    plt.axis('off')

    # Predicted mask
    plt.subplot(1, 4, 3)
    plt.title("Predicted mask")
    plt.imshow(mask_pred, cmap='Reds')
    #plt.imshow(output, cmap='hot') 
    plt.axis('off')

    # Overlay
    # Green = Real, Rosu = Predicted, Yellow = Intersection
    overlay = np.zeros((256,256,3), dtype=np.uint8)
    overlay[..., 1] = mask_real_bin * 255
    overlay[..., 0] = mask_pred * 255

    plt.subplot(1, 4, 4)
    plt.title("Overlay (Yellow = Match)")
    plt.imshow(overlay)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python visualize_unet_prediction.py image_name.png model_name")
    else:
        visualize(sys.argv[1], sys.argv[2])
