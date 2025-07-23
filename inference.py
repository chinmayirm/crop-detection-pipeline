#!/usr/bin/env python3
"""
Complete Crop Detection Inference System
Handles both single timestep (12/13 bands) and multi-temporal data
"""

import numpy as np
import pandas as pd
import rasterio
import os
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available")

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️ PyTorch not available")


class CropDetectionInference:
    """Base inference class for crop detection"""
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = models_dir
        self.config = None
        self.signature_library = None
        
        # Model components
        self.tf_model = None
        self.tf_scaler = None
        self.tf_label_encoder = None
        
        self.pytorch_model = None
        self.pytorch_scaler = None
        self.pytorch_label_encoder = None
        
        # Load configuration and models
        self.load_configuration()
        self.load_models()
    
    def load_configuration(self):
        """Load training configuration"""
        config_path = os.path.join(self.models_dir, 'training_config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print(f"✅ Configuration loaded from {config_path}")
        else:
            # Default configuration
            self.config = {
                'band_mapping': {
                    'B1': 0, 'B2': 1, 'B3': 2, 'B4': 3, 'B5': 4, 'B6': 5,
                    'B7': 6, 'B8': 7, 'B8A': 8, 'B9': 9, 'B11': 10, 'B12': 11
                },
                'wavelengths': [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190],
                'band_names': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'],
                'crop_types': ['maize', 'ragi', 'vegetables', 'sugarcane', 'fallow', 'other_crops']
            }
            print("⚠️ Using default configuration")
    
    def load_models(self):
        """Load trained models and preprocessors"""
        
        # Load TensorFlow model
        tf_model_path = os.path.join(self.models_dir, 'crop_detection_tensorflow_model.h5')
        tf_scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        tf_encoder_path = os.path.join(self.models_dir, 'label_encoder.pkl')
        
        if TENSORFLOW_AVAILABLE and os.path.exists(tf_model_path):
            try:
                self.tf_model = keras.models.load_model(tf_model_path)
                print("✅ TensorFlow model loaded")
                
                if os.path.exists(tf_scaler_path):
                    with open(tf_scaler_path, 'rb') as f:
                        self.tf_scaler = pickle.load(f)
                    print("✅ TensorFlow scaler loaded")
                
                if os.path.exists(tf_encoder_path):
                    with open(tf_encoder_path, 'rb') as f:
                        self.tf_label_encoder = pickle.load(f)
                    print("✅ TensorFlow label encoder loaded")
                    
            except Exception as e:
                print(f"⚠️ Error loading TensorFlow model: {e}")
        
        # Load PyTorch model
        pytorch_model_path = os.path.join(self.models_dir, 'crop_detection_pytorch_model.pth')
        pytorch_scaler_path = os.path.join(self.models_dir, 'pytorch_scaler.pkl')
        pytorch_encoder_path = os.path.join(self.models_dir, 'pytorch_label_encoder.pkl')
        
        if PYTORCH_AVAILABLE and os.path.exists(pytorch_model_path):
            try:
                # Load PyTorch model
                checkpoint = torch.load(pytorch_model_path, map_location='cpu')
                
                if 'model_config' in checkpoint:
                    model_config = checkpoint['model_config']
                    self.pytorch_model = self.create_pytorch_model(
                        model_config['input_dim'], 
                        model_config['num_classes']
                    )
                    self.pytorch_model.load_state_dict(checkpoint['model_state_dict'])
                    self.pytorch_model.eval()
                    print("✅ PyTorch model loaded")
                
                if os.path.exists(pytorch_scaler_path):
                    with open(pytorch_scaler_path, 'rb') as f:
                        self.pytorch_scaler = pickle.load(f)
                    print("✅ PyTorch scaler loaded")
                
                if os.path.exists(pytorch_encoder_path):
                    with open(pytorch_encoder_path, 'rb') as f:
                        self.pytorch_label_encoder = pickle.load(f)
                    print("✅ PyTorch label encoder loaded")
                    
            except Exception as e:
                print(f"⚠️ Error loading PyTorch model: {e}")
    
    def create_pytorch_model(self, input_dim, num_classes):
        """Create PyTorch model architecture"""
        class CropNet(nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.3),
                    
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.3),
                    
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.2),
                    
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    
                    nn.Linear(64, num_classes)
                )
            
            def forward(self, x):
                return self.network(x)
        
        return CropNet(input_dim, num_classes)
    
    def calculate_vegetation_indices_for_image(self, image_data: np.ndarray, 
                                             metadata: pd.DataFrame) -> dict:
        """Calculate vegetation indices for the entire image"""
        height, width = image_data.shape[1], image_data.shape[2]
        band_mapping = self.config['band_mapping']
        n_dates = len(metadata)
        
        indices = {
            'NDVI': np.zeros((n_dates, height, width)),
            'NDRE': np.zeros((n_dates, height, width)),
            'EVI': np.zeros((n_dates, height, width)),
            'SAVI': np.zeros((n_dates, height, width)),
            'GNDVI': np.zeros((n_dates, height, width)),
            'NDWI': np.zeros((n_dates, height, width)),
            'NDMI': np.zeros((n_dates, height, width)),
            'RED_EDGE_POSITION': np.zeros((n_dates, height, width))
        }
        
        bands_per_date = 12
        
        for i in range(min(n_dates, image_data.shape[0] // bands_per_date)):
            start_band = i * bands_per_date
            
            if start_band + 11 < image_data.shape[0]:
                # Extract bands
                blue = image_data[start_band + band_mapping['B2']].astype(np.float32)
                green = image_data[start_band + band_mapping['B3']].astype(np.float32)
                red = image_data[start_band + band_mapping['B4']].astype(np.float32)
                red_edge1 = image_data[start_band + band_mapping['B5']].astype(np.float32)
                red_edge2 = image_data[start_band + band_mapping['B6']].astype(np.float32)
                red_edge3 = image_data[start_band + band_mapping['B7']].astype(np.float32)
                nir = image_data[start_band + band_mapping['B8']].astype(np.float32)
                swir1 = image_data[start_band + band_mapping['B11']].astype(np.float32)
                
                # Calculate indices
                valid_mask = (red > 0) & (nir > 0) & (red < 1) & (nir < 1)
                
                # NDVI
                indices['NDVI'][i] = np.where(
                    valid_mask, (nir - red) / (nir + red + 1e-8), np.nan)
                
                # NDRE
                indices['NDRE'][i] = np.where(
                    valid_mask, (nir - red_edge1) / (nir + red_edge1 + 1e-8), np.nan)
                
                # EVI
                indices['EVI'][i] = np.where(
                    valid_mask, 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1), np.nan)
                
                # SAVI
                L = 0.5
                indices['SAVI'][i] = np.where(
                    valid_mask, (nir - red) * (1 + L) / (nir + red + L + 1e-8), np.nan)
                
                # GNDVI
                indices['GNDVI'][i] = np.where(
                    valid_mask, (nir - green) / (nir + green + 1e-8), np.nan)
                
                # NDWI
                indices['NDWI'][i] = np.where(
                    valid_mask, (green - nir) / (green + nir + 1e-8), np.nan)
                
                # NDMI
                indices['NDMI'][i] = np.where(
                    valid_mask, (nir - swir1) / (nir + swir1 + 1e-8), np.nan)
                
                # Red Edge Position
                indices['RED_EDGE_POSITION'][i] = np.where(
                    valid_mask, 705 + 35 * ((red_edge2 - red_edge1) / (red_edge3 - red_edge1 + 1e-8)), np.nan)
        
        return indices
    
    def predict_with_tensorflow(self, image_data, metadata, indices, height, width):
        """Generate predictions using TensorFlow model"""
        if self.tf_model is None or self.tf_scaler is None or self.tf_label_encoder is None:
            print("⚠️ TensorFlow model components not loaded")
            return None
        
        print("🔮 Running TensorFlow inference...")
        
        # Extract features for all pixels
        all_features = []
        valid_pixels = []
        
        for y in range(height):
            for x in range(width):
                features = self.extract_pixel_features_for_inference(
                    image_data, metadata, indices, y, x
                )
                if features is not None:
                    all_features.append(features)
                    valid_pixels.append((y, x))
        
        if len(all_features) == 0:
            print("⚠️ No valid pixels found")
            return None
        
        # Convert to numpy array and scale
        features_array = np.array(all_features)
        features_scaled = self.tf_scaler.transform(features_array)
        
        # Make predictions
        predictions_proba = self.tf_model.predict(features_scaled, verbose=0)
        predictions = np.argmax(predictions_proba, axis=1)
        
        # Create prediction map
        pred_map = np.full((height, width), -1, dtype=int)
        confidence_map = np.zeros((height, width))
        
        for i, (y, x) in enumerate(valid_pixels):
            pred_map[y, x] = predictions[i]
            confidence_map[y, x] = np.max(predictions_proba[i])
        
        # Convert predictions to crop names
        crop_names = self.tf_label_encoder.classes_
        
        return {
            'predictions': pred_map,
            'confidence': confidence_map,
            'crop_names': crop_names,
            'prediction_summary': self.calculate_prediction_summary(pred_map, crop_names)
        }
    
    def predict_with_pytorch(self, image_data, metadata, indices, height, width):
        """Generate predictions using PyTorch model"""
        if self.pytorch_model is None or self.pytorch_scaler is None or self.pytorch_label_encoder is None:
            print("⚠️ PyTorch model components not loaded")
            return None
        
        print("🔮 Running PyTorch inference...")
        
        # Extract features for all pixels
        all_features = []
        valid_pixels = []
        
        for y in range(height):
            for x in range(width):
                features = self.extract_pixel_features_for_inference(
                    image_data, metadata, indices, y, x
                )
                if features is not None:
                    all_features.append(features)
                    valid_pixels.append((y, x))
        
        if len(all_features) == 0:
            print("⚠️ No valid pixels found")
            return None
        
        # Convert to numpy array and scale
        features_array = np.array(all_features)
        features_scaled = self.pytorch_scaler.transform(features_array)
        
        # Convert to torch tensor
        features_tensor = torch.FloatTensor(features_scaled)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.pytorch_model(features_tensor)
            predictions_proba = torch.softmax(outputs, dim=1).numpy()
            predictions = torch.argmax(outputs, dim=1).numpy()
        
        # Create prediction map
        pred_map = np.full((height, width), -1, dtype=int)
        confidence_map = np.zeros((height, width))
        
        for i, (y, x) in enumerate(valid_pixels):
            pred_map[y, x] = predictions[i]
            confidence_map[y, x] = np.max(predictions_proba[i])
        
        # Convert predictions to crop names
        crop_names = self.pytorch_label_encoder.classes_
        
        return {
            'predictions': pred_map,
            'confidence': confidence_map,
            'crop_names': crop_names,
            'prediction_summary': self.calculate_prediction_summary(pred_map, crop_names)
        }
    
    def calculate_prediction_summary(self, pred_map, crop_names):
        """Calculate summary statistics from prediction map"""
        unique, counts = np.unique(pred_map[pred_map >= 0], return_counts=True)
        total_pixels = np.sum(counts)
        
        summary = {}
        for crop_idx, count in zip(unique, counts):
            if crop_idx < len(crop_names):
                crop_name = crop_names[crop_idx]
                percentage = (count / total_pixels) * 100
                summary[crop_name] = percentage
        
        return summary
    
    def extract_pixel_features_for_inference(self, image_data: np.ndarray, 
                                           metadata: pd.DataFrame, 
                                           indices: dict, 
                                           y: int, x: int) -> np.ndarray:
        """Extract features for a single pixel during inference"""
        try:
            features = []
            bands_per_date = 12
            
            # Extract features for each date
            for date_idx in range(len(metadata)):
                start_band = date_idx * bands_per_date
                
                if start_band + 11 < image_data.shape[0]:
                    # Raw spectral values (12 bands)
                    for band_idx in range(12):
                        band_value = image_data[start_band + band_idx, y, x]
                        features.append(band_value)
                    
                    # Vegetation indices
                    for index_name in indices:
                        if date_idx < indices[index_name].shape[0]:
                            index_value = indices[index_name][date_idx, y, x]
                            features.append(index_value if not np.isnan(index_value) else 0)
                        else:
                            features.append(0)
            
            # Temporal statistics
            temporal_ndvi = []
            for date_idx in range(len(metadata)):
                if date_idx < indices['NDVI'].shape[0]:
                    ndvi_val = indices['NDVI'][date_idx, y, x]
                    if not np.isnan(ndvi_val):
                        temporal_ndvi.append(ndvi_val)
            
            if len(temporal_ndvi) > 1:
                features.extend([
                    np.mean(temporal_ndvi),
                    np.std(temporal_ndvi),
                    np.max(temporal_ndvi),
                    np.min(temporal_ndvi),
                    np.max(temporal_ndvi) - np.min(temporal_ndvi),
                ])
            elif len(temporal_ndvi) == 1:
                val = temporal_ndvi[0]
                features.extend([val, 0, val, val, 0])
            else:
                features.extend([0, 0, 0, 0, 0])
            
            # Replace NaN values with 0
            features = np.array(features)
            features[np.isnan(features)] = 0
            
            return features
            
        except Exception as e:
            print(f"Error extracting features for pixel ({y}, {x}): {e}")
            return None
    
    def save_prediction_results(self, results, output_dir, profile):
        """Save prediction results as GeoTIFF files"""
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, result in results.items():
            if result is not None:
                # Save prediction map
                pred_path = os.path.join(output_dir, f'{model_name}_predictions.tif')
                
                with rasterio.open(pred_path, 'w', **profile, count=1, dtype='int16') as dst:
                    dst.write(result['predictions'].astype('int16'), 1)
                
                # Save confidence map
                conf_path = os.path.join(output_dir, f'{model_name}_confidence.tif')
                
                with rasterio.open(conf_path, 'w', **profile, count=1, dtype='float32') as dst:
                    dst.write(result['confidence'].astype('float32'), 1)
                
                print(f"✅ {model_name} results saved to {output_dir}")
    
    def visualize_predictions(self, results, image_data, output_dir):
        """Create visualization plots of predictions"""
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, result in results.items():
            if result is not None:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # Original image (RGB composite)
                if image_data.shape[0] >= 4:
                    rgb = np.stack([
                        image_data[3],  # Red (B4)
                        image_data[2],  # Green (B3)
                        image_data[1]   # Blue (B2)
                    ], axis=0)
                    rgb = np.moveaxis(rgb, 0, -1)
                    rgb = np.clip(rgb * 3, 0, 1)  # Enhance brightness
                    axes[0].imshow(rgb)
                    axes[0].set_title('RGB Composite')
                    axes[0].axis('off')
                
                # Prediction map
                pred_map = result['predictions']
                im1 = axes[1].imshow(pred_map, cmap='tab10', vmin=0, vmax=len(result['crop_names'])-1)
                axes[1].set_title(f'{model_name} - Crop Predictions')
                axes[1].axis('off')
                
                # Add colorbar for predictions
                cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
                cbar1.set_ticks(range(len(result['crop_names'])))
                cbar1.set_ticklabels(result['crop_names'])
                
                # Confidence map
                im2 = axes[2].imshow(result['confidence'], cmap='viridis', vmin=0, vmax=1)
                axes[2].set_title(f'{model_name} - Confidence')
                axes[2].axis('off')
                
                # Add colorbar for confidence
                cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
                cbar2.set_label('Confidence')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{model_name}_visualization.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"✅ {model_name} visualization saved")


class SingleTimestepInference(CropDetectionInference):
    """Extended inference class that handles single timestep data"""
    
    def __init__(self, models_dir: str = 'models'):
        super().__init__(models_dir)
        self.single_timestep_mode = False
    
    def detect_data_format(self, image_data: np.ndarray, metadata: pd.DataFrame = None) -> str:
        """
        Detect if input is single timestep or multi-temporal
        
        Returns:
            'single_timestep': 12 or 13 bands (single date)
            'multi_temporal': Multiple sets of 12 bands
        """
        n_bands = image_data.shape[0]
        
        if n_bands in [12, 13]:  # Sentinel-2 bands (with/without additional band)
            print(f"📅 Detected single timestep data: {n_bands} bands")
            return 'single_timestep'
        elif n_bands % 12 == 0:  # Multiple dates
            n_dates = n_bands // 12
            print(f"📅 Detected multi-temporal data: {n_dates} dates, {n_bands} total bands")
            return 'multi_temporal'
        else:
            print(f"⚠️ Unusual band count: {n_bands}. Assuming single timestep.")
            return 'single_timestep'
    
    def process_new_image(self, image_path: str, metadata_path: str = None, 
                         output_dir: str = 'inference_results') -> dict:
        """
        Modified process_new_image that handles both single and multi-temporal data
        """
        print(f"🔍 Processing image: {image_path}")
        
        # Load image
        with rasterio.open(image_path) as src:
            image_data = src.read()
            profile = src.profile
            height, width = src.height, src.width
        
        # Detect data format
        data_format = self.detect_data_format(image_data)
        self.single_timestep_mode = (data_format == 'single_timestep')
        
        # Handle metadata
        if self.single_timestep_mode:
            # Create minimal metadata for single timestep
            metadata = pd.DataFrame({
                'date': [pd.Timestamp.now()]  # Use current date as placeholder
            })
            print("📝 Using single timestep mode - created minimal metadata")
        else:
            # Multi-temporal mode - load or create metadata
            if metadata_path and os.path.exists(metadata_path):
                metadata = pd.read_csv(metadata_path)
                metadata['date'] = pd.to_datetime(metadata['date'])
            else:
                n_dates = image_data.shape[0] // 12
                dates = pd.date_range('2024-01-01', periods=n_dates, freq='10D')
                metadata = pd.DataFrame({'date': dates})
        
        # Calculate vegetation indices
        indices = self.calculate_vegetation_indices_for_image(image_data, metadata)
        
        # Generate predictions
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        if self.tf_model is not None:
            print("🔮 Generating TensorFlow predictions...")
            tf_predictions = self.predict_with_tensorflow(
                image_data, metadata, indices, height, width
            )
            results['tensorflow'] = tf_predictions
        
        if self.pytorch_model is not None:
            print("🔮 Generating PyTorch predictions...")
            pytorch_predictions = self.predict_with_pytorch(
                image_data, metadata, indices, height, width
            )
            results['pytorch'] = pytorch_predictions
        
        # Save and visualize results
        self.save_prediction_results(results, output_dir, profile)
        self.visualize_predictions(results, image_data, output_dir)
        
        return results
    
    def extract_pixel_features_for_inference(self, image_data: np.ndarray, 
                                           metadata: pd.DataFrame, 
                                           indices: dict, 
                                           y: int, x: int) -> np.ndarray:
        """
        Modified feature extraction for single timestep data
        """
        try:
            features = []
            
            if self.single_timestep_mode:
                # Single timestep mode - extract features differently
                features = self.extract_single_timestep_features(image_data, indices, y, x)
            else:
                # Multi-temporal mode - use original method
                features = self.extract_multitemporal_features(image_data, metadata, indices, y, x)
            
            # Ensure consistent feature length with training data
            expected_length = self.get_expected_feature_length()
            features = self.pad_or_truncate_features(features, expected_length)
            
            return features
            
        except Exception as e:
            print(f"Error extracting features for pixel ({y}, {x}): {e}")
            return None
    
    def extract_single_timestep_features(self, image_data: np.ndarray, 
                                       indices: dict, y: int, x: int) -> np.ndarray:
        """
        Extract features from single timestep data (12/13 bands)
        """
        features = []
        
        # Extract raw spectral bands (first 12 bands)
        n_bands = min(12, image_data.shape[0])
        for band_idx in range(n_bands):
            band_value = image_data[band_idx, y, x]
            features.append(band_value)
        
        # Pad to 12 bands if fewer available
        while len(features) < 12:
            features.append(0)
        
        # Extract vegetation indices (single values)
        for index_name in indices:
            if indices[index_name].shape[0] > 0:
                index_value = indices[index_name][0, y, x]
                features.append(index_value if not np.isnan(index_value) else 0)
            else:
                features.append(0)
        
        # For single timestep, use the current NDVI as temporal statistics
        ndvi_value = indices.get('NDVI', np.array([[0]]))[0, y, x]
        if np.isnan(ndvi_value):
            ndvi_value = 0
        
        # Add temporal statistics (all same value for single timestep)
        temporal_stats = [ndvi_value] * 5  # mean, std, max, min, range
        features.extend(temporal_stats)
        
        return np.array(features)
    
    def extract_multitemporal_features(self, image_data: np.ndarray, 
                                     metadata: pd.DataFrame, 
                                     indices: dict, y: int, x: int) -> np.ndarray:
        """
        Extract features from multi-temporal data (original method)
        """
        features = []
        bands_per_date = 12
        
        # Extract features for each date
        for date_idx in range(len(metadata)):
            start_band = date_idx * bands_per_date
            
            if start_band + 11 < image_data.shape[0]:
                # Raw spectral values (12 bands)
                for band_idx in range(12):
                    band_value = image_data[start_band + band_idx, y, x]
                    features.append(band_value)
                
                # Vegetation indices
                for index_name in indices:
                    if date_idx < indices[index_name].shape[0]:
                        index_value = indices[index_name][date_idx, y, x]
                        features.append(index_value if not np.isnan(index_value) else 0)
                    else:
                        features.append(0)
        
        # Temporal statistics
        temporal_ndvi = []
        for date_idx in range(len(metadata)):
            if date_idx < indices['NDVI'].shape[0]:
                ndvi_val = indices['NDVI'][date_idx, y, x]
                if not np.isnan(ndvi_val):
                    temporal_ndvi.append(ndvi_val)
        
        if len(temporal_ndvi) > 1:
            features.extend([
                np.mean(temporal_ndvi),
                np.std(temporal_ndvi),
                np.max(temporal_ndvi),
                np.min(temporal_ndvi),
                np.max(temporal_ndvi) - np.min(temporal_ndvi),
            ])
        elif len(temporal_ndvi) == 1:
            # Single value - use it for all stats
            val = temporal_ndvi[0]
            features.extend([val, 0, val, val, 0])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        return np.array(features)
    
    def get_expected_feature_length(self) -> int:
        """
        Get expected feature length based on training configuration
        """
        if hasattr(self, 'config') and self.config:
            # Try to get from configuration
            if 'feature_count' in self.config:
                return self.config['feature_count']
        
        # Default calculation based on single timestep
        # 12 bands + 8 indices + 5 temporal stats = 25 features
        return 25
    
    def pad_or_truncate_features(self, features: np.ndarray, expected_length: int) -> np.ndarray:
        """
        Ensure features match expected length
        """
        current_length = len(features)
        
        if current_length < expected_length:
            # Pad with zeros
            padded = np.zeros(expected_length)
            padded[:current_length] = features
            return padded
        elif current_length > expected_length:
            # Truncate
            return features[:expected_length]
        else:
            return features
    
    def calculate_vegetation_indices_for_image(self, image_data: np.ndarray, 
                                             metadata: pd.DataFrame) -> dict:
        """
        Modified vegetation index calculation for single timestep
        """
        height, width = image_data.shape[1], image_data.shape[2]
        band_mapping = self.config['band_mapping']
        
        if self.single_timestep_mode:
            # Single timestep calculation
            n_dates = 1
            indices = {
                'NDVI': np.zeros((1, height, width)),
                'NDRE': np.zeros((1, height, width)),
                'EVI': np.zeros((1, height, width)),
                'SAVI': np.zeros((1, height, width)),
                'GNDVI': np.zeros((1, height, width)),
                'NDWI': np.zeros((1, height, width)),
                'NDMI': np.zeros((1, height, width)),
                'RED_EDGE_POSITION': np.zeros((1, height, width))
            }
            
            # Extract bands (assume they are the first 12 bands)
            blue = image_data[band_mapping['B2']].astype(np.float32)
            green = image_data[band_mapping['B3']].astype(np.float32)
            red = image_data[band_mapping['B4']].astype(np.float32)
            red_edge1 = image_data[band_mapping['B5']].astype(np.float32)
            red_edge2 = image_data[band_mapping['B6']].astype(np.float32)
            red_edge3 = image_data[band_mapping['B7']].astype(np.float32)
            nir = image_data[band_mapping['B8']].astype(np.float32)
            swir1 = image_data[band_mapping['B11']].astype(np.float32)
            
            # Calculate indices
            valid_mask = (red > 0) & (nir > 0) & (red < 1) & (nir < 1)
            
            indices['NDVI'][0] = np.where(valid_mask, (nir - red) / (nir + red + 1e-8), np.nan)
            indices['NDRE'][0] = np.where(valid_mask, (nir - red_edge1) / (nir + red_edge1 + 1e-8), np.nan)
            indices['EVI'][0] = np.where(valid_mask, 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1), np.nan)
            
            L = 0.5
            indices['SAVI'][0] = np.where(valid_mask, (nir - red) * (1 + L) / (nir + red + L + 1e-8), np.nan)
            indices['GNDVI'][0] = np.where(valid_mask, (nir - green) / (nir + green + 1e-8), np.nan)
            indices['NDWI'][0] = np.where(valid_mask, (green - nir) / (green + nir + 1e-8), np.nan)
            indices['NDMI'][0] = np.where(valid_mask, (nir - swir1) / (nir + swir1 + 1e-8), np.nan)
            indices['RED_EDGE_POSITION'][0] = np.where(valid_mask, 705 + 35 * ((red_edge2 - red_edge1) / (red_edge3 - red_edge1 + 1e-8)), np.nan)
            
        else:
            # Multi-temporal calculation (original method)
            indices = super().calculate_vegetation_indices_for_image(image_data, metadata)
        
        return indices


# Usage examples
def example_single_timestep_usage():
    """Example of how to use single timestep inference"""
    
    print("🌾 SINGLE TIMESTEP CROP DETECTION EXAMPLE")
    print("="*50)
    
    # Initialize inference system
    inference = SingleTimestepInference(models_dir='models')
    
    # Process single timestep image (12-13 bands)
    results = inference.process_new_image(
        image_path='single_date_sentinel2.tif',  # 12 or 13 bands
        output_dir='single_timestep_results'
    )
    
    print("✅ Single timestep processing complete!")
    return results


def example_flexible_usage():
    """Example showing automatic detection of data format"""
    
    inference = SingleTimestepInference(models_dir='models')
    
    # The system automatically detects whether it's single or multi-temporal
    results1 = inference.process_new_image('single_date_image.tif')  # 12 bands
    results2 = inference.process_new_image('timeseries_image.tif')   # 60 bands (5 dates)
    
    return results1, results2


def example_batch_processing():
    """Example of processing multiple images"""
    
    inference = SingleTimestepInference(models_dir='models')
    
    # List of images to process
    image_files = [
        'image1.tif',
        'image2.tif', 
        'image3.tif'
    ]
    
    results_collection = {}
    
    for image_file in image_files:
        if os.path.exists(image_file):
            print(f"\n🔍 Processing {image_file}...")
            
            # Create output directory for this image
            output_dir = f"results_{os.path.splitext(image_file)[0]}"
            
            # Process the image
            results = inference.process_new_image(
                image_path=image_file,
                output_dir=output_dir
            )
            
            results_collection[image_file] = results
            print(f"✅ Completed processing {image_file}")
        else:
            print(f"⚠️ File not found: {image_file}")
    
    return results_collection


def example_custom_output():
    """Example with custom output directory and metadata"""
    
    inference = SingleTimestepInference(models_dir='models')
    
    # Process with custom settings
    results = inference.process_new_image(
        image_path='my_satellite_image.tif',
        metadata_path='my_metadata.csv',  # Optional metadata file
        output_dir='custom_results_2024'
    )
    
    # Print summary
    if results:
        for model_name, result in results.items():
            if result and 'prediction_summary' in result:
                print(f"\n📊 {model_name.upper()} RESULTS:")
                for crop, percentage in result['prediction_summary'].items():
                    print(f"  {crop}: {percentage:.1f}%")
    
    return results


if __name__ == "__main__":
    print("🔧 FLEXIBLE CROP DETECTION INFERENCE")
    print("Supports both single timestep and multi-temporal data!")
    print("="*60)
    
    # Check if models exist
    if os.path.exists('models'):
        print("📁 Models directory found")
        
        # List available model files
        model_files = os.listdir('models')
        if model_files:
            print("🔍 Available model files:")
            for file in model_files:
                print(f"  • {file}")
        else:
            print("⚠️ No model files found in models directory")
            print("Please train models first using the training script")
    else:
        print("❌ Models directory not found")
        print("Please create models directory and train models first")
    
    print("\n🚀 USAGE EXAMPLES:")
    print("="*40)
    print("1. Single timestep: example_single_timestep_usage()")
    print("2. Flexible usage: example_flexible_usage()")
    print("3. Batch processing: example_batch_processing()")
    print("4. Custom output: example_custom_output()")
    
    # Uncomment to run examples
    # example_single_timestep_usage()
    # example_flexible_usage()
    # example_batch_processing()
    # example_custom_output()