#!/usr/bin/env python3
"""
Deep Learning Crop Detection - Training System
Creates spectral signature libraries and trains neural networks for crop detection
"""

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
import warnings
import json
import pickle
warnings.filterwarnings('ignore')

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    print("✅ TensorFlow available")
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow not available. Install with: pip install tensorflow")
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    print("✅ PyTorch available")
    PYTORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch not available. Install with: pip install torch")
    PYTORCH_AVAILABLE = False


class SpectralSignatureLibrary:
    """Class to create and manage spectral signature libraries"""

    def __init__(self, image_path, metadata_path):
        self.image_path = image_path
        self.metadata_path = metadata_path

        # Band information
        self.band_mapping = {
            'B1': 0, 'B2': 1, 'B3': 2, 'B4': 3, 'B5': 4, 'B6': 5,
            'B7': 6, 'B8': 7, 'B8A': 8, 'B9': 9, 'B11': 10, 'B12': 11
        }

        self.wavelengths = [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190]
        self.band_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']

        # Initialize signature library
        self.signature_library = {}
        self.training_data = []
        self.labels = []

    def load_data(self):
        """Load satellite data and metadata"""
        print("📥 Loading data for spectral signature creation...")

        with rasterio.open(self.image_path) as src:
            self.data = src.read()
            self.profile = src.profile
            self.transform = src.transform
            self.height, self.width = src.height, src.width

        self.metadata = pd.read_csv(self.metadata_path)
        self.metadata['date'] = pd.to_datetime(self.metadata['date'])
        self.metadata = self.metadata.sort_values('date').reset_index(drop=True)

        print(f"Data shape: {self.data.shape}")
        print(f"Number of dates: {len(self.metadata)}")

        return self

    def calculate_vegetation_indices(self):
        """Calculate vegetation indices for spectral analysis"""
        print("🌱 Calculating vegetation indices...")

        n_dates = len(self.metadata)
        bands_per_date = 12

        self.indices = {
            'NDVI': np.zeros((n_dates, self.height, self.width)),
            'NDRE': np.zeros((n_dates, self.height, self.width)),
            'EVI': np.zeros((n_dates, self.height, self.width)),
            'SAVI': np.zeros((n_dates, self.height, self.width)),
            'GNDVI': np.zeros((n_dates, self.height, self.width)),
            'NDWI': np.zeros((n_dates, self.height, self.width)),
            'NDMI': np.zeros((n_dates, self.height, self.width)),
            'RED_EDGE_POSITION': np.zeros((n_dates, self.height, self.width))
        }

        for i in range(min(n_dates, self.data.shape[0] // bands_per_date)):
            start_band = i * bands_per_date

            if start_band + 11 < self.data.shape[0]:
                # Extract bands
                blue = self.data[start_band + self.band_mapping['B2']].astype(np.float32)
                green = self.data[start_band + self.band_mapping['B3']].astype(np.float32)
                red = self.data[start_band + self.band_mapping['B4']].astype(np.float32)
                red_edge1 = self.data[start_band + self.band_mapping['B5']].astype(np.float32)
                red_edge2 = self.data[start_band + self.band_mapping['B6']].astype(np.float32)
                red_edge3 = self.data[start_band + self.band_mapping['B7']].astype(np.float32)
                nir = self.data[start_band + self.band_mapping['B8']].astype(np.float32)
                red_edge4 = self.data[start_band + self.band_mapping['B8A']].astype(np.float32)
                swir1 = self.data[start_band + self.band_mapping['B11']].astype(np.float32)

                # Calculate indices
                valid_mask = (red > 0) & (nir > 0) & (red < 1) & (nir < 1)

                # NDVI
                self.indices['NDVI'][i] = np.where(
                    valid_mask, (nir - red) / (nir + red + 1e-8), np.nan)

                # NDRE
                self.indices['NDRE'][i] = np.where(
                    valid_mask, (nir - red_edge1) / (nir + red_edge1 + 1e-8), np.nan)

                # EVI
                self.indices['EVI'][i] = np.where(
                    valid_mask, 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1), np.nan)

                # SAVI
                L = 0.5
                self.indices['SAVI'][i] = np.where(
                    valid_mask, (nir - red) * (1 + L) / (nir + red + L + 1e-8), np.nan)

                # GNDVI
                self.indices['GNDVI'][i] = np.where(
                    valid_mask, (nir - green) / (nir + green + 1e-8), np.nan)

                # NDWI
                self.indices['NDWI'][i] = np.where(
                    valid_mask, (green - nir) / (green + nir + 1e-8), np.nan)

                # NDMI
                self.indices['NDMI'][i] = np.where(
                    valid_mask, (nir - swir1) / (nir + swir1 + 1e-8), np.nan)

                # Red Edge Position (simplified)
                self.indices['RED_EDGE_POSITION'][i] = np.where(
                    valid_mask, 705 + 35 * ((red_edge2 - red_edge1) / (red_edge3 - red_edge1 + 1e-8)), np.nan)

        print("✅ Vegetation indices calculated")
        return self

    def perform_unsupervised_clustering(self, n_clusters=6):
        """Perform unsupervised clustering to identify potential crop regions"""
        print("🔍 Performing unsupervised clustering for crop identification...")

        # Create feature matrix for clustering
        sample_pixels = []
        pixel_coords = []

        # Sample pixels across the image
        step_size = max(1, min(self.height, self.width) // 50)  # Sample ~2500 pixels

        for y in range(0, self.height, step_size):
            for x in range(0, self.width, step_size):
                features = self.extract_pixel_features(y, x)
                if features is not None and not np.all(features == 0):
                    sample_pixels.append(features)
                    pixel_coords.append((y, x))

        if len(sample_pixels) == 0:
            print("❌ No valid pixels found for clustering")
            return self

        sample_pixels = np.array(sample_pixels)

        # Standardize features
        scaler = StandardScaler()
        sample_pixels_scaled = scaler.fit_transform(sample_pixels)

        # Perform clustering
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = clusterer.fit_predict(sample_pixels_scaled)

        # Create crop regions based on clusters
        self.cluster_regions = {}
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_pixels = [pixel_coords[i] for i in range(len(pixel_coords)) if cluster_mask[i]]
            self.cluster_regions[f'cluster_{cluster_id}'] = cluster_pixels

        print(f"✅ Identified {n_clusters} potential crop clusters")
        for cluster_id, pixels in self.cluster_regions.items():
            print(f"  {cluster_id}: {len(pixels)} pixels")

        return self

    def extract_pixel_features(self, y, x):
        """Extract comprehensive spectral features for a single pixel"""
        features = []

        try:
            # Extract raw spectral bands for all dates
            for date_idx in range(len(self.metadata)):
                start_band = date_idx * 12

                if start_band + 11 < self.data.shape[0]:
                    # Raw spectral values (12 bands)
                    for band_idx in range(12):
                        band_value = self.data[start_band + band_idx, y, x]
                        features.append(band_value)

                    # Vegetation indices
                    for index_name in self.indices:
                        if date_idx < self.indices[index_name].shape[0]:
                            index_value = self.indices[index_name][date_idx, y, x]
                            features.append(index_value)

            # Statistical features across time
            temporal_ndvi = []
            for date_idx in range(len(self.metadata)):
                if date_idx < self.indices['NDVI'].shape[0]:
                    ndvi_val = self.indices['NDVI'][date_idx, y, x]
                    if not np.isnan(ndvi_val):
                        temporal_ndvi.append(ndvi_val)

            if len(temporal_ndvi) > 0:
                # Temporal statistics
                features.extend([
                    np.mean(temporal_ndvi),      # Mean NDVI
                    np.std(temporal_ndvi),       # NDVI variability
                    np.max(temporal_ndvi),       # Peak NDVI
                    np.min(temporal_ndvi),       # Minimum NDVI
                    np.max(temporal_ndvi) - np.min(temporal_ndvi),  # NDVI range
                ])
            else:
                features.extend([0, 0, 0, 0, 0])

            # Replace NaN values with 0
            features = np.array(features)
            features[np.isnan(features)] = 0

            return features

        except Exception as e:
            print(f"Error extracting features for pixel ({y}, {x}): {e}")
            return None

    def create_training_samples(self, crop_regions):
        """
        Create training samples from defined crop regions

        Args:
            crop_regions: Dictionary with crop types and their pixel coordinates
                         {'maize': [(y1,x1), (y2,x2), ...], 'ragi': [...], ...}
        """
        print("📝 Creating training samples from crop regions...")

        self.training_data = []
        self.labels = []

        for crop_type, pixel_coords in crop_regions.items():
            print(f"Processing {crop_type}: {len(pixel_coords)} pixels")

            for y, x in pixel_coords:
                if 0 <= y < self.height and 0 <= x < self.width:
                    # Extract spectral signature for this pixel
                    pixel_features = self.extract_pixel_features(y, x)

                    if pixel_features is not None:
                        self.training_data.append(pixel_features)
                        self.labels.append(crop_type)

        self.training_data = np.array(self.training_data)
        self.labels = np.array(self.labels)

        print(f"✅ Created {len(self.training_data)} training samples")
        print(f"Feature vector length: {self.training_data.shape[1]}")
        print(f"Crop types: {np.unique(self.labels)}")

        return self

    def build_signature_library(self):
        """Build comprehensive spectral signature library"""
        print("📚 Building spectral signature library...")

        if len(self.training_data) == 0:
            print("❌ No training data available. Run create_training_samples() first.")
            return self

        unique_crops = np.unique(self.labels)

        for crop_type in unique_crops:
            crop_mask = self.labels == crop_type
            crop_samples = self.training_data[crop_mask]

            # Calculate signature statistics
            signature_stats = {
                'mean_signature': np.mean(crop_samples, axis=0),
                'std_signature': np.std(crop_samples, axis=0),
                'median_signature': np.median(crop_samples, axis=0),
                'min_signature': np.min(crop_samples, axis=0),
                'max_signature': np.max(crop_samples, axis=0),
                'sample_count': len(crop_samples),
                'feature_names': self.get_feature_names()
            }

            self.signature_library[crop_type] = signature_stats

            print(f"  {crop_type}: {len(crop_samples)} samples")

        print("✅ Spectral signature library created")
        return self

    def get_feature_names(self):
        """Get feature names for the spectral signature"""
        feature_names = []

        for date_idx in range(len(self.metadata)):
            date_str = self.metadata.iloc[date_idx]['date'].strftime('%Y%m%d')

            # Raw spectral bands
            for band_name in self.band_names:
                feature_names.append(f'{band_name}_{date_str}')

            # Vegetation indices
            for index_name in self.indices.keys():
                feature_names.append(f'{index_name}_{date_str}')

        # Temporal statistics
        feature_names.extend(['NDVI_mean', 'NDVI_std', 'NDVI_max', 'NDVI_min', 'NDVI_range'])

        return feature_names

    def save_training_components(self, output_dir='models'):
        """Save all training components for later use"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save signature library
        with open(f'{output_dir}/signature_library.pkl', 'wb') as f:
            pickle.dump(self.signature_library, f)
        
        # Save training data and labels
        np.save(f'{output_dir}/training_data.npy', self.training_data)
        np.save(f'{output_dir}/training_labels.npy', self.labels)
        
        # Save metadata and configuration
        config = {
            'band_mapping': self.band_mapping,
            'wavelengths': self.wavelengths,
            'band_names': self.band_names,
            'feature_names': self.get_feature_names(),
            'crop_types': list(self.signature_library.keys()) if self.signature_library else [],
            'image_shape': (self.height, self.width) if hasattr(self, 'height') else None
        }
        
        with open(f'{output_dir}/training_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Training components saved to {output_dir}/")


class TensorFlowCropDetector:
    """TensorFlow/Keras-based crop detection neural network"""

    def __init__(self, signature_library):
        self.signature_library = signature_library
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def prepare_data(self):
        """Prepare data for TensorFlow training"""
        if len(self.signature_library.training_data) == 0:
            raise ValueError("No training data available")

        # Prepare features and labels
        X = self.signature_library.training_data
        y = self.signature_library.labels

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # Convert to categorical
        self.num_classes = len(np.unique(y_encoded))
        self.y_train_cat = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test_cat = keras.utils.to_categorical(self.y_test, self.num_classes)

        print(f"Training data: {self.X_train_scaled.shape}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Classes: {self.label_encoder.classes_}")

        return self

    def build_model(self):
        """Build TensorFlow model for crop detection"""
        input_dim = self.X_train_scaled.shape[1]

        self.model = keras.Sequential([
            # Input layer
            layers.Dense(512, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Hidden layers
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),

            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])

        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        print("✅ TensorFlow model built")
        return self

    def train_model(self, epochs=100, batch_size=32):
        """Train the TensorFlow model"""
        print("🚀 Training TensorFlow model...")

        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=15, restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=10, min_lr=0.0001
        )

        # Train model
        self.history = self.model.fit(
            self.X_train_scaled, self.y_train_cat,
            validation_data=(self.X_test_scaled, self.y_test_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        return self

    def evaluate_model(self):
        """Evaluate the trained model"""
        print("📊 Evaluating TensorFlow model...")

        # Predictions
        y_pred_proba = self.model.predict(self.X_test_scaled)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            self.y_test, y_pred,
            target_names=self.label_encoder.classes_
        ))

        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix - TensorFlow Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        return self

    def save_model_and_preprocessors(self, output_dir='models'):
        """Save model and preprocessing components"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save TensorFlow model
        self.model.save(f'{output_dir}/crop_detection_tensorflow_model.h5')
        
        # Save preprocessors
        with open(f'{output_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(f'{output_dir}/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"✅ TensorFlow model and preprocessors saved to {output_dir}/")


class PyTorchCropDetector:
    """PyTorch-based crop detection neural network"""

    def __init__(self, signature_library):
        self.signature_library = signature_library
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    class CropDataset(Dataset):
        """PyTorch Dataset for crop data"""

        def __init__(self, X, y):
            self.X = torch.FloatTensor(X)
            self.y = torch.LongTensor(y)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    class CropNet(nn.Module):
        """PyTorch neural network for crop classification"""

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

    def prepare_data(self):
        """Prepare data for PyTorch training"""
        X = self.signature_library.training_data
        y = self.signature_library.labels

        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Create datasets
        self.train_dataset = self.CropDataset(X_train_scaled, y_train)
        self.test_dataset = self.CropDataset(X_test_scaled, y_test)

        self.num_classes = len(np.unique(y_encoded))
        self.input_dim = X_train_scaled.shape[1]

        print(f"PyTorch training data: {X_train_scaled.shape}")
        print(f"Number of classes: {self.num_classes}")

        return self

    def build_model(self):
        """Build PyTorch model"""
        self.model = self.CropNet(self.input_dim, self.num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.2
        )

        print("✅ PyTorch model built")
        return self

    def train_model(self, epochs=100, batch_size=32):
        """Train PyTorch model"""
        print("🚀 Training PyTorch model...")

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        best_acc = 0
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_correct += (outputs.argmax(1) == y_batch).sum().item()

            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0

            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)

                    val_loss += loss.item()
                    val_correct += (outputs.argmax(1) == y_batch).sum().item()

            train_acc = train_correct / len(self.train_dataset)
            val_acc = val_correct / len(self.test_dataset)

            self.scheduler.step(val_loss)

            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

            # Early stopping
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'models/best_crop_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Load best model
        self.model.load_state_dict(torch.load('models/best_crop_model.pth'))
        print(f"✅ Training completed. Best validation accuracy: {best_acc:.4f}")

        return self

    def save_model_and_preprocessors(self, output_dir='models'):
        """Save PyTorch model and preprocessing components"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save PyTorch model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.input_dim,
                'num_classes': self.num_classes
            }
        }, f'{output_dir}/crop_detection_pytorch_model.pth')
        
        # Save preprocessors
        with open(f'{output_dir}/pytorch_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(f'{output_dir}/pytorch_label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"✅ PyTorch model and preprocessors saved to {output_dir}/")


def visualize_spectral_signatures(signature_lib):
    """Create comprehensive visualizations of spectral signatures"""
    print("📊 Creating spectral signature visualizations...")

    if not signature_lib.signature_library:
        print("❌ No signature library available")
        return

    # Create visualization plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Spectral Signature Analysis', fontsize=16, fontweight='bold')

    # 1. Mean spectral signatures by crop type
    ax1 = axes[0, 0]
    colors = plt.cm.Set3(np.linspace(0, 1, len(signature_lib.signature_library)))

    for i, (crop_type, signature_data) in enumerate(signature_lib.signature_library.items()):
        # Extract raw spectral bands from mean signature
        mean_sig = signature_data['mean_signature']
        spectral_bands = mean_sig[:12]  # First 12 values are spectral bands

        ax1.plot(signature_lib.wavelengths, spectral_bands, 'o-',
                color=colors[i], label=crop_type.title(), linewidth=2, markersize=6)

    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Reflectance')
    ax1.set_title('Mean Spectral Signatures by Crop Type')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2-6. Additional plots (NDVI patterns, Red Edge, etc.)
    # ... [Additional visualization code would go here]

    plt.tight_layout()
    plt.show()
    plt.savefig('spectral_signature_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Spectral signature plots saved")


def export_signature_library(signature_lib, output_dir='models'):
    """Export spectral signature library for future use"""
    print("💾 Exporting spectral signature library...")
    
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Export signature library as CSV
    export_data = []

    for crop_type, signature_data in signature_lib.signature_library.items():
        for i, feature_name in enumerate(signature_data['feature_names']):
            export_data.append({
                'crop_type': crop_type,
                'feature_name': feature_name,
                'mean_value': signature_data['mean_signature'][i],
                'std_value': signature_data['std_signature'][i],
                'median_value': signature_data['median_signature'][i],
                'min_value': signature_data['min_signature'][i],
                'max_value': signature_data['max_signature'][i],
                'sample_count': signature_data['sample_count']
            })

    signature_df = pd.DataFrame(export_data)
    signature_df.to_csv(f'{output_dir}/spectral_signature_library.csv', index=False)

    # Export configuration
    config = {
        'wavelengths': signature_lib.wavelengths,
        'band_names': signature_lib.band_names,
        'band_mapping': signature_lib.band_mapping,
        'crop_types': list(signature_lib.signature_library.keys()),
        'feature_count': len(signature_lib.signature_library[list(signature_lib.signature_library.keys())[0]]['feature_names']),
        'total_samples': len(signature_lib.training_data)
    }

    with open(f'{output_dir}/crop_detection_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✅ Signature library exported to {output_dir}/")


def main():
    """Main training function"""

    print("🤖 DEEP LEARNING CROP DETECTION - TRAINING SYSTEM")
    print("="*60)

    # Initialize spectral signature library
    signature_lib = SpectralSignatureLibrary(
        image_path='S2_Complete_TimeSeries_Nov_Mar_2022_2024_AllBands.tif',
        metadata_path='S2_TimeSeries_Metadata.csv'
    )

    # Build signature library
    print("📚 Building spectral signature library...")
    signature_lib.load_data()
    signature_lib.calculate_vegetation_indices()

    # Perform unsupervised clustering to identify crop regions
    signature_lib.perform_unsupervised_clustering(n_clusters=6)

    # Create training samples from clustered regions
    crop_regions = {}

    print("\n🎯 Assigning crop types to clusters...")
    print("(In practice, you would manually label these or use ground truth data)")

    # Example assignment based on cluster analysis
    cluster_crop_mapping = {
        'cluster_0': 'maize',
        'cluster_1': 'ragi',
        'cluster_2': 'vegetables',
        'cluster_3': 'sugarcane',
        'cluster_4': 'fallow',
        'cluster_5': 'other_crops'
    }

    # Convert cluster regions to crop regions
    for cluster_name, crop_type in cluster_crop_mapping.items():
        if cluster_name in signature_lib.cluster_regions:
            if crop_type not in crop_regions:
                crop_regions[crop_type] = []
            crop_regions[crop_type].extend(signature_lib.cluster_regions[cluster_name])

    # Create training samples
    signature_lib.create_training_samples(crop_regions)
    signature_lib.build_signature_library()

    # Save signature library and training components
    signature_lib.save_training_components()

    # Train models
    print("\n🧠 Training Deep Learning Models...")

    # TensorFlow Model
    if TENSORFLOW_AVAILABLE:
        print("\n🔥 Training TensorFlow Model...")
        tf_detector = TensorFlowCropDetector(signature_lib)
        tf_detector.prepare_data()
        tf_detector.build_model()
        tf_detector.train_model(epochs=50, batch_size=32)
        tf_detector.evaluate_model()
        tf_detector.save_model_and_preprocessors()

    # PyTorch Model
    if PYTORCH_AVAILABLE:
        print("\n🔥 Training PyTorch Model...")
        torch_detector = PyTorchCropDetector(signature_lib)
        torch_detector.prepare_data()
        torch_detector.build_model()
        torch_detector.train_model(epochs=50, batch_size=32)
        torch_detector.save_model_and_preprocessors()

    # Visualize spectral signatures
    visualize_spectral_signatures(signature_lib)

    # Export signature library
    export_signature_library(signature_lib)

    print("\n✅ Training System Complete!")
    print("📁 Generated files in 'models/' directory:")
    print("  • crop_detection_tensorflow_model.h5 (TensorFlow model)")
    print("  • crop_detection_pytorch_model.pth (PyTorch model)")
    print("  • spectral_signature_library.csv (Signature library)")
    print("  • training_config.json (Configuration)")
    print("  • scaler.pkl & label_encoder.pkl (Preprocessors)")

    return signature_lib


if __name__ == "__main__":
    # Run the training system
    signature_lib = main()

    print("\n🎯 NEXT STEPS:")
    print("="*40)
    print("1. Review the spectral signature visualizations")
    print("2. Validate the cluster assignments and crop mappings")
    print("3. Use the trained models for inference on new images")
    print("4. Run the inference system: python crop_detection_inference.py")
    
    print("\n📖 TRAINING COMPLETE - Ready for Inference!")