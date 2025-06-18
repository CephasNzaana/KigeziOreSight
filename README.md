# Kigezi OreSight: An Open-Source AI Exploration

![A photo of four iron ore hemitite specimens from Buhara, Kabale, arranged on a green background.](https://drive.google.com/file/d/1CSB6j02jy4nUxvcnaaMG5tIkkctgu06w/view?usp=drivesdk)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tech Stack: PyTorch](https://img.shields.io/badge/Tech-PyTorch-orange.svg)](https://pytorch.org/)
[![Status: Prototyping](https://img.shields.io/badge/status-prototyping-brightgreen.svg)](https://github.com/your-username/kigezi-oresight)

**An open-source exploration into using computer vision to identify potential iron ore deposits in my home region of Kabale, Western Uganda.**

---

### 🌱 The Spark

The story of this project started with these rocks right here. With the growing momentum around developing the iron ore industry in my home region, I became fascinated by a simple question: can modern AI help us understand our own land better? Instead of just talking about it, I went out and got my first data points. This project is my journey to answer that question, and I'm building it entirely in the open.

### 💡 The Idea

My goal is to build **Kigezi OreSight**, a deep learning system that will learn to analyze publicly available satellite imagery and create a predictive heat map of potential iron ore deposits. By training a model to recognize the unique patterns of mineral-rich land, I hope to create an open-source tool that can help guide and de-risk initial exploration efforts for our local community. This isn't about creating a commercial product; it's about exploring the art of the possible and sharing the knowledge with everyone.

---

### ✨ The Vision for the Live Demo

The ultimate deliverable will be a simple, interactive web application. Imagine this: you'll be able to select an area on a map of Kabale, and the AI model will overlay a beautiful heatmap in real-time.

![A user selects a region on a map, and a dynamic, animated red/yellow heatmap appears over the satellite image, with Koko the Pangolin pointing to the areas of highest probability.](https://placehold.co/800x450/343A40/FFFFFF?text=Live+Demo+Animation+Vision)

*(This will be a screen recording of the final Streamlit application.)*

---

### ⚙️ How It Will Work: The Technical Journey

1.  **Data Collection:** I will gather multispectral satellite images from the European Space Agency's Sentinel-2 satellite via the Sentinel Hub API.
2.  **Data Labeling:** I'll use existing geological survey maps of the Kigezi region to create labeled training data, marking areas of known iron ore presence.
3.  **Model Training:** A custom U-Net segmentation model, built with **PyTorch**, will be trained on this labeled data. The model will learn to segment any given satellite image into areas of high, medium, and low probability for iron ore deposits.
4.  **Inference & API:** The trained model will be served via a lightweight **FastAPI** application, which will expose an endpoint to process new satellite images.
5.  **Visualization:** A simple web interface built with **Streamlit** will allow users to interact with the model, select an area, and view the resulting heat map visualization.

---

### 📂 Repository Structure

This project will follow a standard structure for a production-ready data science project.

```
kigezi-oresight/
│
├── app/                      # The Streamlit web application
│   └── app.py
│
├── data/
│   ├── raw/                  # Will contain raw satellite image downloads
│   │   └── sample_kabale_image.png # A sample image for testing
│   └── processed/            # Labeled and processed images for training
│
├── notebooks/                # Jupyter notebooks for exploration and analysis
│   └── 01_initial_data_exploration.ipynb
│
├── src/                      # Source code for the project
│   ├── config.py             # Project configuration variables
│   ├── data_loader.py        # Scripts to download and process data
│   ├── model.py              # The PyTorch model architecture (U-Net)
│   └── train.py              # The main script for training the model
│
├── tests/                    # Unit and integration tests
│   └── test_model.py
│
├── .gitignore
├── Dockerfile                # To containerize the FastAPI/Streamlit app
├── LICENSE
├── README.md                 # You are here!
└── requirements.txt
```

---

### 🚀 Getting Started

To get a local copy up and running, follow these steps.

**Prerequisites:**
* Python 3.10+
* Poetry (for dependency management) or pip

**Installation & Training:**
```sh
# Clone the repository
git clone [https://github.com/your-username/kigezi-oresight.git](https://github.com/your-username/kigezi-oresight.git)
cd kigezi-oresight

# Install dependencies
pip install -r requirements.txt

# Run the training script (will use sample data in `data/raw`)
python src/train.py
```

**Running the Demo App:**
```sh
# Run the Streamlit web application
streamlit run app/app.py
```

---

### 🤝 How to Contribute

This is an open project, and collaboration is highly encouraged. Please see `CONTRIBUTING.md` for guidelines on how to get involved.

---
```python
# ========================================================================
# File: kigezi-oresight/app/app.py
# Purpose: The main file for the Streamlit web demo application.
# ========================================================================
import streamlit as st
import numpy as np
from PIL import Image
import os

# Placeholder for a function that would load our trained PyTorch model
def load_model():
    # In a real app, this would load a .pt file
    st.session_state['model_loaded'] = True
    return "Dummy Model"

# Placeholder for a function that processes an image and returns a heatmap
def predict(model, image):
    # Simulate a model prediction
    # In a real app, this would run the image through the U-Net model
    st.session_state['prediction_run'] = True
    heatmap = np.random.rand(*image.shape[:2])
    return (heatmap * 255).astype(np.uint8)

st.set_page_config(layout="wide", page_title="Kigezi OreSight")

# --- Page Header ---
st.title("🛰️ Kigezi OreSight")
st.markdown("An open-source exploration into using AI to identify potential iron ore deposits from satellite imagery.")

# --- Main Interface ---
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload a satellite image of the Kigezi region", type=["jpg", "png"])

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Image")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Satellite Image", use_column_width=True)
    else:
        st.info("Please upload an image to begin analysis.")
        # Display a sample image if it exists
        sample_image_path = os.path.join('data', 'raw', 'sample_kabale_image.png')
        if os.path.exists(sample_image_path):
            sample_image = Image.open(sample_image_path)
            st.image(sample_image, caption="Sample Satellite Image of Kabale", use_column_width=True)
        else:
            st.warning("Sample image not found. Please create 'data/raw/sample_kabale_image.png'")


with col2:
    st.subheader("AI Prediction Heatmap")
    if uploaded_file is not None:
        with st.spinner('Koko the Pangolin is analyzing the terrain... Please wait.'):
            model = load_model()
            pil_image = Image.open(uploaded_file).convert('RGB')
            np_image = np.array(pil_image)
            
            heatmap_data = predict(model, np_image)
            
            # Create a red overlay effect
            heatmap_colored = Image.new("RGBA", pil_image.size)
            overlay_color = (255, 0, 0, 100) # Red with some transparency
            
            for x in range(pil_image.width):
                for y in range(pil_image.height):
                    if heatmap_data[y, x] > 180: # Higher threshold for a less dense map
                        heatmap_colored.putpixel((x, y), overlay_color)
            
            # Composite the original image with the heatmap
            final_image = Image.alpha_composite(pil_image.convert("RGBA"), heatmap_colored)

            st.image(final_image, caption="Predicted Iron Ore Probability (Red = High)", use_column_width=True)
            st.success("Analysis complete!")
    else:
        st.info("Prediction will appear here once you upload an image.")


# --- Footer ---
st.markdown("---")
st.markdown("Built with ❤️ in Kabale. An open-source project by Cephas Nzaana.")

```python
# ========================================================================
# File: kigezi-oresight/src/train.py
# Purpose: A placeholder for the main model training script.
# ========================================================================

import torch
# from model import OreSightUNet # Assuming model architecture is in model.py
# from data_loader import get_dataloader # Assuming data loader is set up

def train_model():
    print("Starting model training process...")

    # 1. Load configuration
    # config = load_config('config.yaml')
    print("Configuration loaded.")

    # 2. Get data loaders
    # train_loader = get_dataloader(config.data.train_path)
    # val_loader = get_dataloader(config.data.val_path)
    print("Data loaders created.")

    # 3. Initialize model, optimizer, and loss function
    # model = OreSightUNet(in_channels=3, out_channels=1)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    # criterion = torch.nn.BCEWithLogitsLoss()
    print("Model and optimizer initialized.")

    # 4. Training loop
    print("Entering training loop...")
    # for epoch in range(config.training.num_epochs):
        # --- Training Step ---
        # for images, masks in train_loader:
            # model.train()
            # ... training logic ...

        # --- Validation Step ---
        # for images, masks in val_loader:
            # model.eval()
            # ... validation logic ...
        
        # print(f"Epoch {epoch+1} completed.")

    print("Model training complete!")
    
    # 5. Save the trained model
    # torch.save(model.state_dict(), 'kigezi_oresight_model.pt')
    print("Trained model saved to 'kigezi_oresight_model.pt'")


if __name__ == "__main__":
    train_model()
```python
# ========================================================================
# File: kigezi-oresight/requirements.txt
# Purpose: Lists the Python dependencies for the project.
# ========================================================================
streamlit
torch
torchvision
numpy
Pillow
# For data handling (add as needed)
# pandas
# geopandas
# rasterio
