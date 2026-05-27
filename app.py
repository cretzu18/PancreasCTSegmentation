import streamlit as st
import cv2 
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from src.models.unet import UNet

st.set_page_config(page_title="Pancreas CT Segmentation", layout="wide")

MODELS = {
    "UNet": {
        "model": "./src/models/unet_model.pth",
    },
    "Unet-optuna": {
        "model": "./src/models/unet_model_best_params.pth"
    }
}

# Load model
@st.cache_resource
def load_unet_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

st.sidebar.title("Model Selection")
selected_model = st.sidebar.selectbox("Choose a model", list(MODELS.keys()))

try:
    unet_model, device = load_unet_model(MODELS[selected_model]["model"])
    st.sidebar.success(f"{selected_model} model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# User interface
st.title("Pancreas CT Segmentation")
st.markdown("A simple application to detect the pancreas from segmented CT scans using UNet based models.")

col_up1, col_up2 = st.columns(2)
with col_up1:
    uploaded_img = st.file_uploader("Upload a CT scan image (png)", type=["png"])
with col_up2:
    uploaded_mask = st.file_uploader("Upload the corresponding mask (png)", type=["png"])

if uploaded_img is not None:
    # Convert the uploaded file to a format suitable for OpenCV
    files_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    img = cv2.imdecode(files_bytes, cv2.IMREAD_GRAYSCALE)

    # Preprocess the image
    IMG_SIZE = (256, 256)
    img_res = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # Normalization and converison to Tensor
    img_input = img_res.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_input).unsqueeze(0).unsqueeze(0).float().to(device)

    with st.spinner("Model is segmenting..."):
        with torch.no_grad():
            logits = unet_model(img_tensor)
            probs = torch.sigmoid(logits)
            pred_mask = (probs > 0.5).float()

            mask_np = pred_mask.squeeze().cpu().numpy()

    mask_pred_full = cv2.resize(mask_np.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Upload the true mask if available
    mask_true_full = None

    if uploaded_mask is not None:
        mask_bytes = np.asarray(bytearray(uploaded_mask.read()), dtype=np.uint8)
        mask_true_full = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)
        mask_true_full = (mask_true_full > 0).astype(np.uint8)

    # Display the results
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Image")
        st.image(img, width='stretch')

    with col2:
        st.subheader("Predicted Mask")
        # Multiply by 255 to visualize the binary mask
        st.image(mask_pred_full * 255, width='stretch')

    with col3:
        if mask_true_full is not None:
            st.subheader("True Mask")
            st.image(mask_true_full * 255, width='stretch')
        else:
            st.info("Upload a true mask to compare with the prediction.")

    # Overlay visualization
    if mask_true_full is not None:
        st.divider()
        st.header("Overlay Visualization")

        # Calculate Dice Score (F1 Score)
        dice = f1_score(mask_true_full.flatten(), mask_pred_full.flatten())

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.metric(label="Dice Coefficient (F1 Score)", value = f"{dice:.4f}")
        
        with col_m2:
            # Create an overlay of the predicted mask on the original image
            # Green for true positives, Red for false positives, Blue for false negatives

            comparison_overlay = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            comparison_overlay[(mask_true_full == 1) & (mask_pred_full == 1)] = [0, 255, 0]  # True Positives - Green
            comparison_overlay[(mask_true_full == 0) & (mask_pred_full == 1)] = [255, 0, 0]  # False Positives - Red
            comparison_overlay[(mask_true_full == 1) & (mask_pred_full == 0)] = [0, 0, 255]  # False Negatives - Blue   

            st.subheader("Error Overlay (Green=True Positives, Red=False Positives, Blue=False Negatives)")
            st.image(comparison_overlay, width='stretch')