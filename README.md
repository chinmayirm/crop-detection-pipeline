# Crop Detection using Spectral Signatures and Deep Learning

This project is an *informal* and exploratory implementation for detecting crop types using satellite imagery and vegetation indices. It uses **Google Earth Engine** (GEE) exports, unsupervised clustering, and neural network models to experiment with spectral signatures for various crops.

This is part of an informal, curiosity-driven attempt by our team to test whether meaningful deep learning models can be trained for crop detection using minimal supervision and synthetic labels. While not production-grade, it captures the potential and process of building such a pipeline from scratch.

This was an exploration project done for my internship at ISRO. This work can form the basis for various applications like yield precition of different crops in an area, growth analysis, yearly output comparision etc. benefition for Government use. 

---

## Project Overview

The pipeline performs the following:

- Loads satellite time-series data and metadata from Google Earth Engine
- Computes vegetation indices like NDVI, SAVI, GNDVI, NDWI, etc.
- Clusters image pixels into crop types using unsupervised learning
- Builds a **spectral signature library** per crop
- Trains deep learning classifiers using both TensorFlow and PyTorch
- Visualizes spectral signatures and exports all model components

---

## File Structure

| File | Description |
|------|-------------|
| `training.py` | Main training pipeline |
| `models/` | Output directory for models, scalers, encoders, configs |
| `S2_Complete_TimeSeries_Nov_Mar_2022_2024_AllBands.tif` | Input image exported from Google Earth Engine |
| `S2_TimeSeries_Metadata.csv` | Metadata for image dates |

---

## Datasets Used

- **Imagery**: Sentinel-2 time-series exported from **Google Earth Engine**
- **Metadata**: CSV with timestamps aligned to bands in the TIFF
- **Labels**: Auto-labeled using unsupervised clustering and manually assigned crop names (e.g., maize, ragi, sugarcane, etc.)

---

## Models Trained

Two deep learning classifiers are trained on extracted spectral features:

### 1. TensorFlow-based Classifier

- 4-layer feedforward neural network
- Dense layers + BatchNorm + Dropout
- Softmax output
- Optimized with Adam, EarlyStopping, and ReduceLROnPlateau

### 2. PyTorch-based Classifier

- Mirror architecture of TensorFlow model
- Uses CrossEntropy loss
- Includes early stopping and learning rate scheduler
- Saved as `.pth` file with preprocessor states

---

## Vegetation Indices Computed

- NDVI (Normalized Difference Vegetation Index)
- NDRE (Red Edge NDVI)
- EVI (Enhanced Vegetation Index)
- SAVI (Soil-Adjusted Vegetation Index)
- GNDVI (Green NDVI)
- NDWI (Water Index)
- NDMI (Moisture Index)
- Red Edge Position Estimator

---

## Tech Stack

- Python 3.x
- NumPy, Pandas
- Rasterio
- Matplotlib, Seaborn
- scikit-learn
- TensorFlow / Keras
- PyTorch
- Google Earth Engine (for data export only)

---

## Example Output
<img width="1989" height="1589" alt="image" src="https://github.com/user-attachments/assets/55c8587a-6848-4554-a04a-0035b896ecbb" />


<img width="1787" height="1189" alt="image" src="https://github.com/user-attachments/assets/33e0e317-7d15-4174-867d-77cc9c5d5de3" />

<img width="1989" height="1475" alt="image" src="https://github.com/user-attachments/assets/b5510399-96a2-4737-94ac-de045f97a56e" />

<img width="1973" height="1590" alt="image" src="https://github.com/user-attachments/assets/7725daf5-cde8-42a6-abe5-e234952d6df2" />


---

##  Outputs

| File | Description |
|------|-------------|
| `crop_detection_tensorflow_model.h5` | Trained TensorFlow model |
| `crop_detection_pytorch_model.pth` | Trained PyTorch model |
| `scaler.pkl`, `label_encoder.pkl` | Preprocessing components |
| `spectral_signature_library.csv` | Signature statistics by crop |
| `training_config.json` | Band names, mapping, and metadata |
| `training_data.npy` | Raw feature matrix |
| `training_labels.npy` | Label vector |

---

## Introspective Value

This project offers a way to understand:

- How spectral features evolve over time across crop types
- How data from **Google Earth Engine** can be translated into meaningful inputs for ML models
- How spectral data can be used to train DL models with minimal labels
- Challenges in crop labeling without ground truth
- The value of **spectral signature visualization** in model validation


---

## Disclaimer

This project was created for *experimental and educational purposes* only. No claims are made about its performance in real-world agricultural settings without proper validation using ground truth data. However, the model provides an accuracy of 95%.

