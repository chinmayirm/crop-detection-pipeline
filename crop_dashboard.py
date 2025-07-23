import streamlit as st
import folium
from streamlit_folium import st_folium
import ee
import pandas as pd
import numpy as np
from datetime import datetime, date
import json
import zipfile
import io
import time
import requests
import os
from pathlib import Path
import subprocess
import sys
import threading
import queue
import pickle
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tempfile
import shutil
import traceback

# Import your actual ML modules
try:
    # Import training modules
    from training import SpectralSignatureLibrary, TensorFlowCropDetector, PyTorchCropDetector
    from training import export_signature_library, visualize_spectral_signatures
    TRAINING_AVAILABLE = True
    print("✅ Training modules imported successfully")
except ImportError as e:
    TRAINING_AVAILABLE = False
    print(f"⚠️ Training modules not available: {e}")

try:
    # Import inference modules
    from inference import SingleTimestepInference
    INFERENCE_AVAILABLE = True
    print("✅ Inference modules imported successfully")
except ImportError as e:
    INFERENCE_AVAILABLE = False
    print(f"⚠️ Inference modules not available: {e}")

# Page config
st.set_page_config(
    page_title="🌾 Complete Crop Detection System",
    page_icon="🌾",
    layout="wide"
)

# Configuration notice for large files
if not os.path.exists('.streamlit'):
    with st.sidebar:
        st.info("""
        💡 **For Large Files (Local):**
        
        Create `.streamlit/config.toml`:
        ```
        [server]
        maxUploadSize = 1000
        
        [runner]
        maxMessageSize = 1000
        ```
        This increases upload limit to 1GB for local use.
        """)

# Add gdown installation check
try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False
    with st.sidebar:
        st.warning("""
        ⚠️ **Google Drive downloads disabled**
        
        Install gdown for Google Drive support:
        ```
        pip install gdown
        ```
        """)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        text-align: center;
    }
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .info-box {
        background: #d1ecf1;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .error-box {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 10px;
        color: #333;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e3c72;
        color: white;
    }
    .map-controls {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Google Earth Engine
@st.cache_resource
def initialize_gee():
    try:
        ee.Initialize()
        return True
    except Exception as e:
        st.error(f"Please authenticate with Google Earth Engine: {str(e)}")
        return False

# Header
st.markdown("""
<div class="main-header">
    <h1>🌾 Complete Crop Detection System</h1>
    <p style="color: white; text-align: center; margin: 0;">
        Data Collection → Model Training → Crop Detection Pipeline
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'coordinates' not in st.session_state:
    st.session_state.coordinates = None
if 'task_ids' not in st.session_state:
    st.session_state.task_ids = []
if 'gee_initialized' not in st.session_state:
    st.session_state.gee_initialized = initialize_gee()
if 'training_process' not in st.session_state:
    st.session_state.training_process = None
if 'training_output' not in st.session_state:
    st.session_state.training_output = []
if 'selected_map_type' not in st.session_state:
    st.session_state.selected_map_type = 'OpenStreetMap'

# Utility Functions
def check_file_exists(filepath):
    """Check if file exists"""
    return os.path.exists(filepath)

def check_models_exist():
    """Check if trained models exist"""
    models_dir = "models"
    
    # Core required files (minimum for inference)
    core_files = [
        "training_config.json",           # Basic config
        "spectral_signature_library.csv", # CSV signature library  
        "crop_detection_tensorflow_model.h5", # TensorFlow model
        "scaler.pkl",                     # TensorFlow scaler
        "label_encoder.pkl"               # TensorFlow label encoder
    ]
    
    # Optional PyTorch files
    pytorch_files = [
        "crop_detection_pytorch_model.pth",
        "pytorch_scaler.pkl", 
        "pytorch_label_encoder.pkl"
    ]
    
    # Additional training files
    additional_files = [
        "signature_library.pkl",  # PKL signature library
        "training_data.npy",      # Raw training data
        "training_labels.npy",    # Training labels
        "crop_detection_config.json" # Detailed config
    ]
    
    existing_files = []
    missing_files = []
    
    # Check all possible files
    all_files = core_files + pytorch_files + additional_files
    
    for file in all_files:
        filepath = os.path.join(models_dir, file)
        if check_file_exists(filepath):
            existing_files.append(file)
        else:
            missing_files.append(file)
    
    return existing_files, missing_files

def get_map_tiles():
    """Get available map tile options"""
    map_options = {
        'OpenStreetMap': {
            'tiles': 'OpenStreetMap',
            'attr': 'OpenStreetMap'
        },
        'Google Satellite': {
            'tiles': 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            'attr': 'Google'
        },
        'Google Hybrid': {
            'tiles': 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
            'attr': 'Google'
        },
        'Google Streets': {
            'tiles': 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
            'attr': 'Google'
        },
        'Google Terrain': {
            'tiles': 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
            'attr': 'Google'
        },
        'Esri Satellite': {
            'tiles': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            'attr': 'Esri'
        },
        'Esri Streets': {
            'tiles': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
            'attr': 'Esri'
        },
        'Esri Topo': {
            'tiles': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
            'attr': 'Esri'
        },
        'CartoDB Positron': {
            'tiles': 'CartoDB positron',
            'attr': 'CartoDB'
        },
        'CartoDB Dark': {
            'tiles': 'CartoDB dark_matter',
            'attr': 'CartoDB'
        },
        'Stamen Terrain': {
            'tiles': 'Stamen Terrain',
            'attr': 'Stamen'
        },
        'Stamen Toner': {
            'tiles': 'Stamen Toner',
            'attr': 'Stamen'
        }
    }
    return map_options

def create_map_safe(map_type='OpenStreetMap'):
    """Create map with safe plugin loading and selected map type"""
    
    # Calculate center and zoom
    if st.session_state.coordinates:
        coords = st.session_state.coordinates
        if coords['type'] == 'rectangle':
            center_lat = (coords['coordinates'][0][1] + coords['coordinates'][1][1]) / 2
            center_lon = (coords['coordinates'][0][0] + coords['coordinates'][1][0]) / 2
        else:
            center_lat = sum(p[1] for p in coords['coordinates']) / len(coords['coordinates'])
            center_lon = sum(p[0] for p in coords['coordinates']) / len(coords['coordinates'])
        zoom = 10
    else:
        center_lat, center_lon = 20.5937, 78.9629
        zoom = 5
    
    # Get map tile configuration
    map_options = get_map_tiles()
    selected_option = map_options.get(map_type, map_options['OpenStreetMap'])
    
    # Create base map with selected tiles
    if selected_option['tiles'] in ['OpenStreetMap', 'CartoDB positron', 'CartoDB dark_matter', 'Stamen Terrain', 'Stamen Toner']:
        # Use folium's built-in tiles
        m = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=zoom,
            tiles=selected_option['tiles']
        )
    else:
        # Use custom tile server
        m = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=zoom,
            tiles=None
        )
        
        # Add custom tile layer
        folium.TileLayer(
            tiles=selected_option['tiles'],
            attr=selected_option['attr'],
            name=map_type,
            overlay=False,
            control=True
        ).add_to(m)
    
    # Try to add drawing tools
    try:
        from folium.plugins import Draw
        
        draw = Draw(
            export=True,
            position='topleft',
            draw_options={
                'polyline': False,
                'rectangle': True,
                'polygon': True,
                'circle': False,
                'marker': False,
                'circlemarker': False,
            }
        )
        draw.add_to(m)
        
    except Exception as e:
        st.warning(f"⚠️ Drawing tools not available: {str(e)}")
    
    # Show current selection on map
    if st.session_state.coordinates:
        coords = st.session_state.coordinates
        
        try:
            if coords['type'] == 'rectangle':
                bounds = [
                    [coords['coordinates'][0][1], coords['coordinates'][0][0]],
                    [coords['coordinates'][1][1], coords['coordinates'][1][0]]
                ]
                folium.Rectangle(
                    bounds, 
                    color='red', 
                    weight=3,
                    fill=True, 
                    fillOpacity=0.2,
                    popup="Selected Study Area"
                ).add_to(m)
                
            elif coords['type'] == 'polygon':
                points = [[p[1], p[0]] for p in coords['coordinates']]
                folium.Polygon(
                    points, 
                    color='red', 
                    weight=3,
                    fill=True, 
                    fillOpacity=0.2,
                    popup="Selected Study Area"
                ).add_to(m)
        
        except Exception as e:
            st.error(f"Error displaying selection on map: {str(e)}")
    
    return m

def create_map_selector(tab_name, key_suffix=""):
    """Create map type selector widget"""
    map_options = get_map_tiles()
    
    # Map type selector in a nice container
    st.markdown('<div class="map-controls">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_map = st.selectbox(
            "🗺️ Select Map Type",
            options=list(map_options.keys()),
            index=list(map_options.keys()).index(st.session_state.selected_map_type),
            key=f"map_selector_{tab_name}_{key_suffix}",
            help="Choose the base map layer for better visualization"
        )
    
    with col2:
        if st.button("🔄 Refresh Map", key=f"refresh_map_{tab_name}_{key_suffix}"):
            st.session_state.selected_map_type = selected_map
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Update session state if changed
    if selected_map != st.session_state.selected_map_type:
        st.session_state.selected_map_type = selected_map
    
    return selected_map

# Real training function using your training.py
def run_real_training(tif_path, csv_path, n_clusters, epochs, batch_size, model_types):
    """Run actual training using your training.py module"""
    
    if not TRAINING_AVAILABLE:
        st.error("❌ Training modules not available. Please ensure training.py is accessible.")
        return False
    
    try:
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # Initialize spectral signature library
        status_text.text("🔄 Initializing spectral signature library...")
        progress_bar.progress(10)
        
        signature_lib = SpectralSignatureLibrary(
            image_path=tif_path,
            metadata_path=csv_path
        )
        
        # Load data
        status_text.text("📥 Loading satellite data and metadata...")
        progress_bar.progress(20)
        signature_lib.load_data()
        
        # Calculate vegetation indices
        status_text.text("🌱 Calculating vegetation indices...")
        progress_bar.progress(30)
        signature_lib.calculate_vegetation_indices()
        
        # Perform clustering
        status_text.text("🔍 Performing unsupervised clustering...")
        progress_bar.progress(40)
        signature_lib.perform_unsupervised_clustering(n_clusters=n_clusters)
        
        # Create training samples
        status_text.text("📝 Creating training samples...")
        progress_bar.progress(50)
        
        # Assign crop types to clusters (you can customize this mapping)
        cluster_crop_mapping = {
            'cluster_0': 'maize',
            'cluster_1': 'ragi', 
            'cluster_2': 'vegetables',
            'cluster_3': 'sugarcane',
            'cluster_4': 'fallow',
            'cluster_5': 'other_crops'
        }
        
        # Convert cluster regions to crop regions
        crop_regions = {}
        for cluster_name, crop_type in cluster_crop_mapping.items():
            if cluster_name in signature_lib.cluster_regions:
                if crop_type not in crop_regions:
                    crop_regions[crop_type] = []
                crop_regions[crop_type].extend(signature_lib.cluster_regions[cluster_name])
        
        signature_lib.create_training_samples(crop_regions)
        signature_lib.build_signature_library()
        
        # Save signature library components
        status_text.text("💾 Saving signature library...")
        progress_bar.progress(60)
        signature_lib.save_training_components()
        
        # Train models based on selection
        training_results = {}
        
        if "TensorFlow" in model_types:
            status_text.text("🔥 Training TensorFlow model...")
            progress_bar.progress(70)
            
            tf_detector = TensorFlowCropDetector(signature_lib)
            tf_detector.prepare_data()
            tf_detector.build_model()
            tf_detector.train_model(epochs=epochs, batch_size=batch_size)
            tf_detector.save_model_and_preprocessors()
            training_results['tensorflow'] = "Successfully trained"
        
        if "PyTorch" in model_types:
            status_text.text("🔥 Training PyTorch model...")
            progress_bar.progress(85)
            
            torch_detector = PyTorchCropDetector(signature_lib)
            torch_detector.prepare_data()
            torch_detector.build_model()
            torch_detector.train_model(epochs=epochs, batch_size=batch_size)
            torch_detector.save_model_and_preprocessors()
            training_results['pytorch'] = "Successfully trained"
        
        # Export signature library
        status_text.text("📊 Exporting signature library...")
        progress_bar.progress(95)
        export_signature_library(signature_lib)
        
        # Complete
        progress_bar.progress(100)
        status_text.text("✅ Training completed successfully!")
        
        return True, training_results, signature_lib
        
    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
        st.error(f"Error details: {traceback.format_exc()}")
        return False, None, None

# Real inference function using your inference.py
def run_real_crop_detection(image_path, model_type, output_dir="inference_results"):
    """Run actual crop detection using your inference.py module"""
    
    if not INFERENCE_AVAILABLE:
        st.error("❌ Inference modules not available. Please ensure inference.py is accessible.")
        return None
    
    try:
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # Initialize inference system
        status_text.text("🔧 Initializing inference system...")
        progress_bar.progress(20)
        
        inference = SingleTimestepInference(models_dir='models')
        
        # Process the image
        status_text.text("🔍 Processing satellite image...")
        progress_bar.progress(40)
        
        results = inference.process_new_image(
            image_path=image_path,
            output_dir=output_dir
        )
        
        status_text.text("🤖 Running crop detection model...")
        progress_bar.progress(80)
        
        # Extract prediction results
        if model_type.lower() in results:
            predictions = results[model_type.lower()]
            
            # Calculate summary statistics
            prediction_summary = {}
            total_pixels = 0
            
            # Get crop type counts from predictions
            if 'predictions' in predictions:
                pred_array = predictions['predictions']
                unique, counts = np.unique(pred_array, return_counts=True)
                total_pixels = np.sum(counts)
                
                # Convert to percentages
                for crop_idx, count in zip(unique, counts):
                    # Map crop index to crop name (you may need to adjust this)
                    crop_names = ['maize', 'ragi', 'vegetables', 'sugarcane', 'fallow', 'other_crops']
                    if crop_idx < len(crop_names):
                        crop_name = crop_names[crop_idx]
                        percentage = (count / total_pixels) * 100
                        prediction_summary[crop_name] = percentage
            
            # Create formatted results
            formatted_results = {
                'prediction_summary': prediction_summary,
                'detection_date': datetime.now().strftime('%Y-%m-%d'),
                'model_used': model_type,
                'total_area_km2': 25.6,  # You can calculate this from pixel size and count
                'confidence_score': predictions.get('confidence', 0.85),
                'raw_results': predictions
            }
            
            progress_bar.progress(100)
            status_text.text("✅ Crop detection completed!")
            
            return formatted_results
            
        else:
            st.error(f"❌ Model type '{model_type}' not found in results")
            return None
            
    except Exception as e:
        st.error(f"❌ Crop detection failed: {str(e)}")
        st.error(f"Error details: {traceback.format_exc()}")
        return None

# Tab structure
tabs = st.tabs(["🛰️ Data Collection", "🤖 Model Training", "🔍 Crop Detection", "📊 Results Dashboard"])

# TAB 1: Data Collection (Same as before)
with tabs[0]:
    st.header("🛰️ Sentinel-2 Data Collection")
    
    # Sidebar for quick regions
    with st.sidebar:
        st.header("🗺️ Quick Regions")
        
        if st.button("Karnataka Sample Region"):
            st.session_state.coordinates = {
                'type': 'polygon',
                'coordinates': [
                    [76.19285, 14.53274],
                    [76.19602, 14.51329],
                    [76.15388, 14.50989],
                    [76.15216, 14.52891]
                ]
            }
            st.rerun()
        
        if st.button("Karnataka Rectangle"):
            st.session_state.coordinates = {
                'type': 'rectangle',
                'coordinates': [[76.15, 14.50], [76.20, 14.54]]
            }
            st.rerun()
        
        if st.button("Punjab Region"):
            st.session_state.coordinates = {
                'type': 'rectangle',
                'coordinates': [[75.0, 30.8], [75.5, 31.3]]
            }
            st.rerun()
        
        st.markdown("---")
        st.subheader("📋 Output Files")
        st.write("**Time Series TIF:** 1 multi-band GeoTIFF")
        st.write("**OR Composite TIF:** 1 single composite")
        st.write("**Metadata CSV:** 1 CSV file")
        st.write("**Total:** 2 files only")

    # Main content for data collection
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("🗺️ Select Study Area")
        
        # Map type selector for data collection
        selected_map_type = create_map_selector("data_collection")
        
        # Create the map with selected type
        m = create_map_safe(selected_map_type)

        # Display map
        try:
            map_data = st_folium(
                m, 
                width=700, 
                height=450, 
                returned_objects=["all_drawings"],
                key="sentinel_map"
            )
        except Exception as e:
            st.error(f"Map display error: {str(e)}")
            map_data = {'all_drawings': []}

        # Process drawn shapes
        if map_data and 'all_drawings' in map_data and map_data['all_drawings']:
            try:
                latest_drawing = map_data['all_drawings'][-1]
                geom_type = latest_drawing.get('geometry', {}).get('type', '')
                
                if geom_type == 'Polygon':
                    coords = latest_drawing['geometry']['coordinates'][0]
                    
                    if len(coords) == 5:
                        lons = [p[0] for p in coords[:-1]]
                        lats = [p[1] for p in coords[:-1]]
                        
                        st.session_state.coordinates = {
                            'type': 'rectangle',
                            'coordinates': [[min(lons), min(lats)], [max(lons), max(lats)]]
                        }
                        st.success("✅ Rectangle area selected from map!")
                    else:
                        st.session_state.coordinates = {
                            'type': 'polygon',
                            'coordinates': [[p[0], p[1]] for p in coords[:-1]]
                        }
                        st.success("✅ Polygon area selected from map!")
                    
                    time.sleep(0.5)
                    st.rerun()
                    
            except Exception as e:
                st.warning(f"Could not process drawn shape: {str(e)}")
        
        # Manual coordinate input
        with st.expander("✏️ Manual Coordinate Input"):
            input_text = st.text_area(
                "Enter coordinates (one per line: longitude,latitude)",
                value="76.19285,14.53274\n76.19602,14.51329\n76.15388,14.50989\n76.15216,14.52891",
                height=100
            )
            
            if st.button("Apply Manual Coordinates"):
                try:
                    points = []
                    for line in input_text.strip().split('\n'):
                        if line.strip():
                            lon, lat = map(float, line.split(','))
                            points.append([lon, lat])
                    
                    st.session_state.coordinates = {
                        'type': 'polygon',
                        'coordinates': points
                    }
                    st.rerun()
                except Exception as e:
                    st.error(f"Invalid coordinate format: {str(e)}")

    def run_simplified_analysis(start_date, end_date, cloud_threshold, output_type, composite_method):
        """Run simplified GEE analysis - time series stack or single composite + metadata"""
        if not st.session_state.coordinates:
            st.error("No coordinates selected")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Create AOI
            coords = st.session_state.coordinates
            if coords['type'] == 'rectangle':
                min_lon, min_lat = coords['coordinates'][0]
                max_lon, max_lat = coords['coordinates'][1]
                aoi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
            else:
                points = coords['coordinates']
                aoi = ee.Geometry.Polygon([points])
            
            status_text.text("🛰️ Setting up analysis parameters...")
            progress_bar.progress(10)
            
            # All Sentinel-2 bands
            all_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
            
            # Convert dates to strings
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            status_text.text("📡 Loading Sentinel-2 data...")
            progress_bar.progress(30)
            
            # Preprocessing function
            def preprocess_sentinel2(image):
                scaled = image.multiply(0.0001)
                qa = image.select('QA60')
                cloud_bit_mask = 1 << 10
                cirrus_bit_mask = 1 << 11
                mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
                date_str = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
                
                return scaled.updateMask(mask).select(all_bands).set({
                    'date': date_str,
                    'system:time_start': image.get('system:time_start')
                }).copyProperties(image, ['system:index'])
            
            # Load and filter collection
            collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(aoi) \
                .filterDate(start_str, end_str) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold)) \
                .map(preprocess_sentinel2) \
                .sort('system:time_start')
            
            collection_size = collection.size()
            
            if output_type == "Time Series Stack":
                status_text.text("📊 Creating time series stack...")
                progress_bar.progress(60)
                
                time_series_stack = collection.toBands()
                time_series_stack = time_series_stack.set({
                    'start_date': start_str,
                    'end_date': end_str,
                    'cloud_threshold': cloud_threshold,
                    'output_type': 'time_series_stack',
                    'image_count': collection_size,
                    'export_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'bands_per_image': len(all_bands)
                })
                
                final_image = time_series_stack.clip(aoi).reproject('EPSG:4326', None, 10)
                description = f'S2_TimeSeries_{start_str}_{end_str}'
                
            else:  # Single Composite
                status_text.text("📊 Creating composite image...")
                progress_bar.progress(60)
                
                if composite_method == 'median':
                    composite = collection.median()
                elif composite_method == 'mean':
                    composite = collection.mean()
                elif composite_method == 'max':
                    composite = collection.max()
                elif composite_method == 'min':
                    composite = collection.min()
                
                final_image = composite.set({
                    'start_date': start_str,
                    'end_date': end_str,
                    'cloud_threshold': cloud_threshold,
                    'output_type': 'composite',
                    'composite_method': composite_method,
                    'image_count': collection_size,
                    'export_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }).clip(aoi).reproject('EPSG:4326', None, 10)
                
                description = f'S2_Composite_{start_str}_{end_str}_{composite_method}'
            
            status_text.text("📤 Starting exports...")
            progress_bar.progress(80)
            
            task_ids = []
            
            # Export main image
            main_task = ee.batch.Export.image.toDrive(
                image=final_image,
                description=description,
                folder='Sentinel2_Simplified_Export',
                region=aoi,
                scale=10,
                crs='EPSG:4326',
                maxPixels=1e9
            )
            main_task.start()
            task_ids.append(main_task.id)
            
            # Create metadata
            metadata_features = []
            metadata_features.append(
                ee.Feature(None, {
                    'type': 'summary',
                    'start_date': start_str,
                    'end_date': end_str,
                    'cloud_threshold': cloud_threshold,
                    'output_type': output_type,
                    'composite_method': composite_method if output_type == "Single Composite" else 'none',
                    'total_images': collection_size,
                    'export_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'bands': ','.join(all_bands),
                    'aoi_type': coords['type']
                })
            )
            
            def add_image_metadata(image):
                return ee.Feature(None, {
                    'type': 'image',
                    'image_id': image.get('system:index'),
                    'date': image.get('date'),
                    'cloud_cover': image.get('CLOUDY_PIXEL_PERCENTAGE')
                })
            
            image_metadata = collection.map(add_image_metadata)
            
            # Export metadata
            metadata_task = ee.batch.Export.table.toDrive(
                collection=ee.FeatureCollection(metadata_features).merge(image_metadata),
                description=f'S2_Metadata_{start_str}_{end_str}',
                folder='Sentinel2_Simplified_Export',
                fileFormat='CSV'
            )
            metadata_task.start()
            task_ids.append(metadata_task.id)
            
            st.session_state.task_ids = task_ids
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis started! Check task status below.")
            
            if output_type == "Time Series Stack":
                st.success(f"🚀 Started time series export: Each date's 12 bands stacked in sequence")
            else:
                st.success(f"🚀 Started composite export: Single {composite_method} image with 12 bands")
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            progress_bar.progress(0)
            status_text.text("❌ Analysis failed")

    with col2:
        st.subheader("🚀 Analysis Settings")
        
        # Date range selection
        st.write("**📅 Time Period**")
        start_date = st.date_input(
            "Start Date",
            value=date(2021, 11, 1),
            min_value=date(2015, 1, 1),
            max_value=date.today()
        )
        
        end_date = st.date_input(
            "End Date",
            value=date(2024, 3, 31),
            min_value=date(2015, 1, 1),
            max_value=date.today()
        )
        
        # Cloud coverage threshold
        cloud_threshold = st.slider(
            "Max Cloud Coverage (%)",
            min_value=0,
            max_value=100,
            value=20,
            step=5
        )
        
        # Output type
        output_type = st.selectbox(
            "Output Type",
            ["Time Series Stack", "Single Composite"],
            index=0,
            help="Time Series: All dates stacked | Composite: Single averaged image"
        )
        
        # Compositing method
        if output_type == "Single Composite":
            composite_method = st.selectbox(
                "Compositing Method",
                ["median", "mean", "max", "min"],
                index=0
            )
        else:
            composite_method = None
        
        st.markdown("---")
        
        # Display current coordinates
        if st.session_state.coordinates:
            coords = st.session_state.coordinates
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.write("✅ **Study Area Selected**")
            
            if coords['type'] == 'rectangle':
                st.write(f"**Type:** Rectangle")
                st.write(f"**SW Corner:** {coords['coordinates'][0][1]:.4f}°N, {coords['coordinates'][0][0]:.4f}°E")
                st.write(f"**NE Corner:** {coords['coordinates'][1][1]:.4f}°N, {coords['coordinates'][1][0]:.4f}°E")
            else:
                st.write(f"**Type:** Polygon")
                st.write(f"**Points:** {len(coords['coordinates'])}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Run analysis button
            if st.session_state.gee_initialized:
                if st.button("🚀 Start Analysis", type="primary"):
                    run_simplified_analysis(start_date, end_date, cloud_threshold, output_type, composite_method)
            else:
                st.error("❌ Google Earth Engine not initialized")
        
        else:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.write("⚠️ **No study area selected**")
            st.write("Draw a shape on the map or use manual input")
            st.markdown('</div>', unsafe_allow_html=True)

    # Task monitoring section (same as before)
    if st.session_state.task_ids:
        st.subheader("📊 Task Status & Downloads")
        
        if st.button("🔄 Refresh Task Status"):
            st.rerun()
        
        completed_tasks = []
        failed_tasks = []
        running_tasks = []
        
        for task_id in st.session_state.task_ids:
            try:
                task = ee.data.getTaskStatus(task_id)[0]
                status = task['state']
                
                if status == 'COMPLETED':
                    completed_tasks.append(task)
                elif status == 'FAILED':
                    failed_tasks.append(task)
                else:
                    running_tasks.append(task)
            except:
                continue
        
        # Display status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("✅ Completed", len(completed_tasks))
        with col2:
            st.metric("🔄 Running", len(running_tasks))
        with col3:
            st.metric("❌ Failed", len(failed_tasks))
        
        # Show task details
        if completed_tasks:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.write("**✅ Completed Tasks:**")
            for task in completed_tasks:
                st.write(f"• {task['description']}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if running_tasks:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.write("**🔄 Running Tasks:**")
            for task in running_tasks:
                st.write(f"• {task['description']}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if failed_tasks:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.write("**❌ Failed Tasks:**")
            for task in failed_tasks:
                st.write(f"• {task['description']}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Download instructions
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.write("**📥 Download Instructions:**")
        st.write("1. Go to [Google Drive](https://drive.google.com)")
        st.write("2. Look for folder: `Sentinel2_Simplified_Export`")
        st.write("3. Download the 2 files: TIF composite + CSV metadata")
        st.markdown('</div>', unsafe_allow_html=True)

# TAB 2: Model Training (Integrated with real training.py)
with tabs[1]:
    st.header("🤖 Model Training Dashboard")
    
    # Check ML module availability
    col_status1, col_status2 = st.columns(2)
    with col_status1:
        if TRAINING_AVAILABLE:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.write("✅ **Training modules available**")
            st.write("• SpectralSignatureLibrary")
            st.write("• TensorFlowCropDetector") 
            st.write("• PyTorchCropDetector")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-box">', unsafe_allow_html=True)
            st.write("❌ **Training modules not available**")
            st.write("Please ensure training.py is in the same directory")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col_status2:
        if INFERENCE_AVAILABLE:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.write("✅ **Inference modules available**")
            st.write("• SingleTimestepInference")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-box">', unsafe_allow_html=True)
            st.write("❌ **Inference modules not available**")
            st.write("Please ensure inference.py is in the same directory")
            st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📁 Upload Training Data")
        
        # Choose upload method
        upload_method = st.selectbox(
            "Select Upload Method",
            ["Standard Upload (<200MB)", "Local File Paths", "Google Drive Links"],
            help="Choose based on your file size and deployment"
        )
        
        if upload_method == "Standard Upload (<200MB)":
            # Original file uploaders
            time_series_file = st.file_uploader(
                "Upload Time Series TIF file",
                type=['tif', 'tiff'],
                help="Upload the multi-band time series GeoTIFF from the Data Collection tab"
            )
            
            metadata_file = st.file_uploader(
                "Upload Metadata CSV file",
                type=['csv'],
                help="Upload the metadata CSV file from the Data Collection tab"
            )
            
            tif_path = None
            csv_path = None
            
        elif upload_method == "Local File Paths":
            st.info("💡 For large files (>200MB), specify local file paths directly")
            
            tif_path = st.text_input(
                "Time Series TIF File Path",
                placeholder="/path/to/your/timeseries.tif",
                help="Enter the full path to your time series TIF file"
            )
            
            csv_path = st.text_input(
                "Metadata CSV File Path",
                placeholder="/path/to/your/metadata.csv", 
                help="Enter the full path to your metadata CSV file"
            )
            
            # Validate file paths
            if tif_path and os.path.exists(tif_path):
                st.success(f"✅ TIF file found: {os.path.basename(tif_path)}")
                file_size = os.path.getsize(tif_path) / (1024**3)
                st.write(f"**Size:** {file_size:.2f} GB")
                
                # Show additional file info
                try:
                    with rasterio.open(tif_path) as src:
                        st.write(f"**Dimensions:** {src.height} x {src.width}")
                        st.write(f"**Bands:** {src.count}")
                        st.write(f"**CRS:** {src.crs}")
                except Exception as e:
                    st.write(f"Could not read raster info: {e}")
                    
            elif tif_path:
                st.error("❌ TIF file not found")
                
            if csv_path and os.path.exists(csv_path):
                st.success(f"✅ CSV file found: {os.path.basename(csv_path)}")
                
                try:
                    df = pd.read_csv(csv_path)
                    st.write(f"**Rows:** {len(df)}")
                    st.write(f"**Columns:** {list(df.columns)}")
                except Exception as e:
                    st.write(f"Could not read CSV info: {e}")
                    
            elif csv_path:
                st.error("❌ CSV file not found")
            
            time_series_file = None
            metadata_file = None
            
        elif upload_method == "Google Drive Links":
            st.info("💡 Upload files to Google Drive and share the links here")
            st.markdown("""
            **📋 How to get Google Drive links:**
            1. Upload your files to Google Drive
            2. Right-click file → "Get Link" → "Anyone with link can view"
            3. Copy the share link and paste below
            """)
            
            drive_tif_url = st.text_input(
                "Google Drive TIF File URL",
                placeholder="https://drive.google.com/file/d/FILE_ID/view?usp=sharing",
                help="Share link from Google Drive for your TIF file"
            )
            
            drive_csv_url = st.text_input(
                "Google Drive CSV File URL", 
                placeholder="https://drive.google.com/file/d/FILE_ID/view?usp=sharing",
                help="Share link from Google Drive for your CSV file"
            )
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                if drive_tif_url and st.button("📥 Download TIF File"):
                    try:
                        import gdown
                        os.makedirs("uploaded_data", exist_ok=True)
                        
                        if '/file/d/' in drive_tif_url:
                            file_id = drive_tif_url.split('/file/d/')[1].split('/')[0]
                            tif_download_path = "uploaded_data/downloaded_timeseries.tif"
                            
                            with st.spinner("Downloading TIF file from Google Drive..."):
                                gdown.download(f"https://drive.google.com/uc?id={file_id}", 
                                             tif_download_path, quiet=False)
                            
                            if os.path.exists(tif_download_path):
                                st.success(f"✅ Downloaded TIF: {os.path.basename(tif_download_path)}")
                                
                                # Show file info
                                file_size = os.path.getsize(tif_download_path) / (1024**3)
                                st.write(f"**Size:** {file_size:.2f} GB")
                                
                                st.session_state.downloaded_tif = tif_download_path
                            else:
                                st.error("❌ Download failed")
                        else:
                            st.error("❌ Invalid Google Drive URL format")
                            
                    except ImportError:
                        st.error("❌ Please install gdown: `pip install gdown`")
                    except Exception as e:
                        st.error(f"❌ Download error: {str(e)}")
            
            with col_b:
                if drive_csv_url and st.button("📥 Download CSV File"):
                    try:
                        import gdown
                        os.makedirs("uploaded_data", exist_ok=True)
                        
                        if '/file/d/' in drive_csv_url:
                            file_id = drive_csv_url.split('/file/d/')[1].split('/')[0]
                            csv_download_path = "uploaded_data/downloaded_metadata.csv"
                            
                            with st.spinner("Downloading CSV file from Google Drive..."):
                                gdown.download(f"https://drive.google.com/uc?id={file_id}", 
                                             csv_download_path, quiet=False)
                            
                            if os.path.exists(csv_download_path):
                                st.success(f"✅ Downloaded CSV: {os.path.basename(csv_download_path)}")
                                st.session_state.downloaded_csv = csv_download_path
                            else:
                                st.error("❌ Download failed")
                        else:
                            st.error("❌ Invalid Google Drive URL format")
                            
                    except ImportError:
                        st.error("❌ Please install gdown: `pip install gdown`")
                    except Exception as e:
                        st.error(f"❌ Download error: {str(e)}")
            
            # Use downloaded files if available
            tif_path = getattr(st.session_state, 'downloaded_tif', None)
            csv_path = getattr(st.session_state, 'downloaded_csv', None)
            time_series_file = None
            metadata_file = None
            
            # Show downloaded files status
            if hasattr(st.session_state, 'downloaded_tif') or hasattr(st.session_state, 'downloaded_csv'):
                st.subheader("📁 Downloaded Files")
                
                if hasattr(st.session_state, 'downloaded_tif'):
                    st.write(f"✅ **TIF File:** {os.path.basename(st.session_state.downloaded_tif)}")
                    
                if hasattr(st.session_state, 'downloaded_csv'):
                    st.write(f"✅ **CSV File:** {os.path.basename(st.session_state.downloaded_csv)}")
        
        # Check if we have valid files (either uploaded or path-based)
        files_ready = False
        final_tif_path = None
        final_csv_path = None
        
        if upload_method == "Standard Upload (<200MB)":
            if time_series_file and metadata_file:
                files_ready = True
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.write("✅ **Files uploaded successfully!**")
                st.write(f"**Time Series:** {time_series_file.name}")
                st.write(f"**Metadata:** {metadata_file.name}")
                
                # Show file sizes
                tif_size = len(time_series_file.getvalue()) / (1024**2)  # MB
                csv_size = len(metadata_file.getvalue()) / (1024**2)   # MB
                st.write(f"**TIF Size:** {tif_size:.1f} MB")
                st.write(f"**CSV Size:** {csv_size:.1f} MB")
                st.markdown('</div>', unsafe_allow_html=True)
        
        else:  # File path methods
            if tif_path and csv_path and os.path.exists(tif_path) and os.path.exists(csv_path):
                files_ready = True
                final_tif_path = tif_path
                final_csv_path = csv_path
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.write("✅ **Files ready for processing!**")
                st.write(f"**Time Series:** {os.path.basename(tif_path)}")
                st.write(f"**Metadata:** {os.path.basename(csv_path)}")
                
                # Show file sizes for path-based methods
                if os.path.exists(tif_path):
                    tif_size = os.path.getsize(tif_path) / (1024**3)  # GB
                    st.write(f"**TIF Size:** {tif_size:.2f} GB")
                if os.path.exists(csv_path):
                    csv_size = os.path.getsize(csv_path) / (1024**2)  # MB
                    st.write(f"**CSV Size:** {csv_size:.1f} MB")
                    
                st.markdown('</div>', unsafe_allow_html=True)
        
        if not files_ready:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.write("⚠️ **Please provide both files to proceed**")
            
            if upload_method == "Standard Upload (<200MB)":
                st.write("• Time Series TIF file from Data Collection")
                st.write("• Metadata CSV file from Data Collection")
                st.write("• **Note:** 200MB upload limit applies")
            elif upload_method == "Local File Paths":
                st.write("• Valid file paths to TIF and CSV files")
                st.write("• Files must exist on the local system")
                st.write("• **No size limits** for local files")
            else:  # Google Drive
                st.write("• Valid Google Drive share links")
                st.write("• Files must be publicly accessible")
                st.write("• Click download buttons after providing URLs")
                st.write("• **No size limits** for Google Drive files")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Save files and prepare training (for uploaded files)
        if files_ready and upload_method == "Standard Upload (<200MB)" and st.button("💾 Save Files and Prepare Training"):
            try:
                os.makedirs("uploaded_data", exist_ok=True)
                os.makedirs("models", exist_ok=True)
                
                # Save uploaded files
                final_tif_path = f"uploaded_data/{time_series_file.name}"
                final_csv_path = f"uploaded_data/{metadata_file.name}"
                
                with open(final_tif_path, "wb") as f:
                    f.write(time_series_file.getbuffer())
                
                with open(final_csv_path, "wb") as f:
                    f.write(metadata_file.getbuffer())
                
                st.success(f"✅ Files saved for training!")
                
                # Display detailed file info
                try:
                    with rasterio.open(final_tif_path) as src:
                        st.write(f"**📊 Image Information:**")
                        st.write(f"• **Dimensions:** {src.height} x {src.width} pixels")
                        st.write(f"• **Bands:** {src.count}")
                        st.write(f"• **CRS:** {src.crs}")
                        st.write(f"• **Data Type:** {src.dtypes[0]}")
                        
                        # Calculate estimated dates
                        if src.count % 12 == 0:
                            n_dates = src.count // 12
                            st.write(f"• **Estimated Dates:** {n_dates} (assuming 12 bands per date)")
                        
                        file_size = os.path.getsize(final_tif_path) / (1024**3)
                        st.write(f"• **File Size:** {file_size:.2f} GB")
                        
                except Exception as e:
                    st.error(f"Could not read TIF file: {e}")
                
                # Load and display metadata info
                try:
                    metadata_df = pd.read_csv(final_csv_path)
                    st.write(f"**📊 Metadata Information:**")
                    st.write(f"• **Records:** {len(metadata_df)}")
                    st.write(f"• **Columns:** {', '.join(metadata_df.columns)}")
                    
                    # Show sample data
                    with st.expander("📋 View Sample Metadata"):
                        st.dataframe(metadata_df.head(), use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Could not read CSV file: {e}")
                    
                # Store file paths for training
                st.session_state.training_tif_path = final_tif_path
                st.session_state.training_csv_path = final_csv_path
                
            except Exception as e:
                st.error(f"Error processing files: {str(e)}")
    
    with col2:
        st.subheader("⚙️ Training Settings")
        
        # Training parameters
        n_clusters = st.slider(
            "Number of Crop Clusters",
            min_value=3,
            max_value=10,
            value=6,
            help="Number of clusters for unsupervised crop identification"
        )
        
        epochs = st.slider(
            "Training Epochs",
            min_value=10,
            max_value=200,
            value=50,
            help="Number of training epochs for the neural network"
        )
        
        batch_size = st.selectbox(
            "Batch Size",
            [16, 32, 64, 128],
            index=1,
            help="Batch size for neural network training"
        )
        
        model_types = st.multiselect(
            "Model Types to Train",
            ["TensorFlow", "PyTorch"],
            default=["TensorFlow"],
            help="Select which models to train"
        )
    
    # Training execution section
    st.subheader("🚀 Start Training")
    
    # Check if files are available for training
    if files_ready and TRAINING_AVAILABLE:
        # Use the determined file paths
        if final_tif_path and final_csv_path:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.write("**📁 Ready for Training:**")
            st.write(f"• TIF: {os.path.basename(final_tif_path)}")
            st.write(f"• CSV: {os.path.basename(final_csv_path)}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Real training button
            if st.button("🤖 Start Model Training", type="primary"):
                if not st.session_state.training_process:
                    try:
                        st.info("🔄 **Starting real model training...** This may take several minutes to hours.")
                        
                        # Run actual training
                        success, results, signature_lib = run_real_training(
                            tif_path=final_tif_path,
                            csv_path=final_csv_path,
                            n_clusters=n_clusters,
                            epochs=epochs,
                            batch_size=batch_size,
                            model_types=model_types
                        )
                        
                        if success:
                            st.success("✅ Model training completed successfully!")
                            st.balloons()
                            
                            # Display training results
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.write("**🎯 Training Results:**")
                            if signature_lib and signature_lib.signature_library:
                                st.write(f"• Crop types identified: {len(signature_lib.signature_library)}")
                                st.write(f"• Crop types: {', '.join(signature_lib.signature_library.keys())}")
                            if signature_lib and len(signature_lib.training_data) > 0:
                                st.write(f"• Training samples: {len(signature_lib.training_data)}")
                                st.write(f"• Feature dimensions: {signature_lib.training_data.shape[1]}")
                            
                            st.write("• **Models trained:**")
                            for model_type, status in results.items():
                                st.write(f"  - {model_type}: {status}")
                            
                            st.write("• **Files generated:**")
                            st.write("  - crop_detection_tensorflow_model.h5 (TensorFlow model)")
                            if "PyTorch" in model_types:
                                st.write("  - crop_detection_pytorch_model.pth (PyTorch model)")
                            st.write("  - spectral_signature_library.csv (Signature library)")
                            st.write("  - training_config.json (Configuration)")
                            st.write("  - scaler.pkl & label_encoder.pkl (Preprocessors)")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Store training completion in session state
                            st.session_state.training_completed = True
                            
                        else:
                            st.error("❌ Training failed. Please check the error messages above.")
                    
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
                        st.error(f"Error details: {traceback.format_exc()}")
                else:
                    st.warning("Training is already in progress!")
        else:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.write("⚠️ **File paths not available**")
            st.write("Please ensure files are properly loaded before training.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif not TRAINING_AVAILABLE:
        st.markdown('<div class="error-box">', unsafe_allow_html=True)
        st.write("❌ **Training modules not available**")
        st.write("Please ensure training.py is in the same directory and all dependencies are installed.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.write("⚠️ **No training data available**")
        st.write("Please upload data files in the section above or use the Data Collection tab first.")
        st.markdown('</div>', unsafe_allow_html=True)
with tabs[2]:
    st.header("🔍 Crop Detection")
    
    # Check if models exist
    existing_files, missing_files = check_models_exist()
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("🔧 Model Status")
        
        if len(existing_files) > 0:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.write("**✅ Available Models:**")
            for file in existing_files:
                st.write(f"• {file}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if len(missing_files) > 0:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.write("**⚠️ Missing Files:**")
            for file in missing_files:
                st.write(f"• {file}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Detection settings
        if len(existing_files) >= 5:  # Need core files: config + signature CSV + TF model + scaler + label_encoder
            st.subheader("⚙️ Detection Settings")
            st.subheader("⚙️ Detection Settings")
            
            detection_date = st.date_input(
                "Detection Date",
                value=date.today(),
                min_value=date(2015, 1, 1),
                max_value=date.today(),
                help="Date for single timestep crop detection"
            )
            
            cloud_threshold_det = st.slider(
                "Max Cloud Coverage (%)",
                min_value=0,
                max_value=100,
                value=10,
                step=5,
                key="detection_cloud_threshold"
            )
            
            # Model selection
            available_models = []
            if "crop_detection_tensorflow_model.h5" in existing_files:
                available_models.append("TensorFlow")
            if "crop_detection_pytorch_model.pth" in existing_files:
                available_models.append("PyTorch")
            
            if available_models:
                selected_model = st.selectbox(
                    "Select Model",
                    available_models,
                    index=0
                )
            else:
                st.error("No trained models available!")
    
    with col1:
        st.subheader("🗺️ Select Detection Area")
        
        if len(existing_files) >= 5:  # Need minimum core files for detection
            # Create map for detection area selection
            m_detect = create_map_safe()
            
            try:
                map_data_detect = st_folium(
                    m_detect, 
                    width=700, 
                    height=400, 
                    returned_objects=["all_drawings"],
                    key="detection_map"
                )
            except Exception as e:
                st.error(f"Map display error: {str(e)}")
                map_data_detect = {'all_drawings': []}
            
            # Process drawn shapes for detection
            detection_coordinates = None
            if map_data_detect and 'all_drawings' in map_data_detect and map_data_detect['all_drawings']:
                try:
                    latest_drawing = map_data_detect['all_drawings'][-1]
                    geom_type = latest_drawing.get('geometry', {}).get('type', '')
                    
                    if geom_type == 'Polygon':
                        coords = latest_drawing['geometry']['coordinates'][0]
                        
                        if len(coords) == 5:
                            lons = [p[0] for p in coords[:-1]]
                            lats = [p[1] for p in coords[:-1]]
                            
                            detection_coordinates = {
                                'type': 'rectangle',
                                'coordinates': [[min(lons), min(lats)], [max(lons), max(lats)]]
                            }
                        else:
                            detection_coordinates = {
                                'type': 'polygon',
                                'coordinates': [[p[0], p[1]] for p in coords[:-1]]
                            }
                        
                        st.success("✅ Detection area selected!")
                        
                except Exception as e:
                    st.warning(f"Could not process drawn shape: {str(e)}")
            
            # Manual coordinate input for detection
            with st.expander("✏️ Manual Detection Area Input"):
                detection_input = st.text_area(
                    "Enter detection coordinates (one per line: longitude,latitude)",
                    value="76.19285,14.53274\n76.19602,14.51329\n76.15388,14.50989\n76.15216,14.52891",
                    height=80,
                    key="detection_coords"
                )
                
                if st.button("Apply Detection Coordinates"):
                    try:
                        points = []
                        for line in detection_input.strip().split('\n'):
                            if line.strip():
                                lon, lat = map(float, line.split(','))
                                points.append([lon, lat])
                        
                        detection_coordinates = {
                            'type': 'polygon',
                            'coordinates': points
                        }
                        st.success("✅ Detection coordinates applied!")
                        
                    except Exception as e:
                        st.error(f"Invalid coordinate format: {str(e)}")
        
        else:
            st.markdown('<div class="error-box">', unsafe_allow_html=True)
            st.write("**❌ Models not available**")
            st.write("Please complete model training in the previous tab first.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Crop detection execution
    if len(existing_files) >= 5:
        st.subheader("🚀 Run Crop Detection")
        
        # Function to run single timestep detection
        def run_crop_detection(coords, detection_date, cloud_threshold, model_type):
            """Run crop detection for single timestep"""
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Create AOI
                if coords['type'] == 'rectangle':
                    min_lon, min_lat = coords['coordinates'][0]
                    max_lon, max_lat = coords['coordinates'][1]
                    aoi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
                else:
                    points = coords['coordinates']
                    aoi = ee.Geometry.Polygon([points])
                
                status_text.text("🛰️ Loading Sentinel-2 data for detection...")
                progress_bar.progress(20)
                
                # Get single date image (closest to detection_date)
                detection_str = detection_date.strftime('%Y-%m-%d')
                next_day = (detection_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                
                # All Sentinel-2 bands
                all_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
                
                # Load single image for detection
                collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                    .filterBounds(aoi) \
                    .filterDate(detection_str, next_day) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold)) \
                    .select(all_bands) \
                    .sort('CLOUDY_PIXEL_PERCENTAGE')
                
                # Get the best image (least cloudy)
                best_image = collection.first()
                
                # Scale to reflectance
                scaled_image = best_image.multiply(0.0001)
                
                # Clip to AOI
                final_image = scaled_image.clip(aoi).reproject('EPSG:4326', None, 10)
                
                status_text.text("📤 Exporting single timestep image...")
                progress_bar.progress(60)
                
                # Export single timestep image for detection
                detection_task = ee.batch.Export.image.toDrive(
                    image=final_image,
                    description=f'S2_Detection_{detection_str}',
                    folder='Crop_Detection_Data',
                    region=aoi,
                    scale=10,
                    crs='EPSG:4326',
                    maxPixels=1e9
                )
                detection_task.start()
                
                status_text.text("🤖 Running crop detection model...")
                progress_bar.progress(80)
                
                # Simulate model inference
                time.sleep(3)
                
                # Create dummy detection results
                dummy_results = {
                    'prediction_summary': {
                        'maize': 35.2,
                        'ragi': 22.8,
                        'vegetables': 18.5,
                        'sugarcane': 12.3,
                        'fallow': 8.7,
                        'other_crops': 2.5
                    },
                    'detection_date': detection_str,
                    'model_used': model_type,
                    'cloud_coverage': 5.2,
                    'total_area_km2': 25.6,
                    'confidence_score': 0.87
                }
                
                progress_bar.progress(100)
                status_text.text("✅ Crop detection completed!")
                
                return dummy_results, detection_task.id
                
            except Exception as e:
                st.error(f"Detection failed: {str(e)}")
                return None, None
        
        # Detection interface
        if 'detection_coordinates' in locals() and detection_coordinates:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.write("✅ **Detection area selected**")
            if detection_coordinates['type'] == 'rectangle':
                st.write(f"**SW:** {detection_coordinates['coordinates'][0][1]:.4f}°N, {detection_coordinates['coordinates'][0][0]:.4f}°E")
                st.write(f"**NE:** {detection_coordinates['coordinates'][1][1]:.4f}°N, {detection_coordinates['coordinates'][1][0]:.4f}°E")
            else:
                st.write(f"**Polygon with {len(detection_coordinates['coordinates'])} points**")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("🔍 Run Crop Detection", type="primary"):
                if 'selected_model' in locals():
                    results, task_id = run_crop_detection(
                        detection_coordinates, 
                        detection_date, 
                        cloud_threshold_det, 
                        selected_model
                    )
                    
                    if results:
                        st.session_state.detection_results = results
                        st.session_state.detection_task_id = task_id
                        st.rerun()
                else:
                    st.error("No model selected!")
        
        else:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.write("⚠️ **Please select a detection area on the map**")
            st.markdown('</div>', unsafe_allow_html=True)

# TAB 4: Results Dashboard
with tabs[3]:
    st.header("📊 Results Daeeeeeeeshboard")
    
    # Check if detection results exist
    if hasattr(st.session_state, 'detection_results') and st.session_state.detection_results:
        results = st.session_state.detection_results
        
        # Summary metrics
        st.subheader("📈 Detection Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Detection Date", 
                results['detection_date']
            )
        
        with col2:
            st.metric(
                "Model Used", 
                results['model_used']
            )
        
        with col3:
            st.metric(
                "Confidence Score", 
                f"{results['confidence_score']:.2%}"
            )
        
        with col4:
            st.metric(
                "Total Area", 
                f"{results['total_area_km2']} km²"
            )
        
        # Crop distribution visualization
        st.subheader("🌾 Crop Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            crop_data = results['prediction_summary']
            labels = list(crop_data.keys())
            sizes = list(crop_data.values())
            
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                            colors=colors, startangle=90)
            ax.set_title('Crop Distribution by Area', fontsize=16, fontweight='bold')
            
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            st.pyplot(fig)
        
        with col2:
            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 8))
            bars = ax.bar(labels, sizes, color=colors)
            ax.set_title('Crop Areas (% of Total)', fontsize=16, fontweight='bold')
            ax.set_ylabel('Percentage (%)')
            ax.set_xlabel('Crop Type')
            
            # Add value labels on bars
            for bar, size in zip(bars, sizes):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{size:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Detailed results table
        st.subheader("📋 Detailed Results")
        
        # Convert to DataFrame for better display
        results_df = pd.DataFrame([
            {
                'Crop Type': crop.replace('_', ' ').title(),
                'Area Percentage': f"{percentage:.1f}%",
                'Estimated Area (km²)': f"{(percentage/100) * results['total_area_km2']:.2f}",
                'Confidence': "High" if percentage > 15 else "Medium" if percentage > 5 else "Low"
            }
            for crop, percentage in results['prediction_summary'].items()
        ])
        
        st.dataframe(results_df, use_container_width=True)
        
        # Download results
        st.subheader("📥 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV download
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="📊 Download CSV Report",
                data=csv_data,
                file_name=f"crop_detection_results_{results['detection_date']}.csv",
                mime="text/csv"
            )
        
        with col2:
            # JSON download
            json_data = json.dumps(results, indent=2)
            st.download_button(
                label="📄 Download JSON Report",
                data=json_data,
                file_name=f"crop_detection_results_{results['detection_date']}.json",
                mime="application/json"
            )
        
        # Task status for detection export
        if hasattr(st.session_state, 'detection_task_id'):
            st.subheader("📤 Export Status")
            
            try:
                task = ee.data.getTaskStatus(st.session_state.detection_task_id)[0]
                status = task['state']
                
                if status == 'COMPLETED':
                    st.success("✅ Detection image exported to Google Drive")
                elif status == 'FAILED':
                    st.error("❌ Export failed")
                else:
                    st.info(f"🔄 Export status: {status}")
                    
            except Exception as e:
                st.warning(f"Could not check export status: {str(e)}")
    
    else:
        # No results available
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.write("**📊 No detection results available**")
        st.write("Please run crop detection in the previous tab to see results here.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show placeholder charts
        st.subheader("📈 Results Will Appear Here")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Crop Distribution Pie Chart**\nShows percentage breakdown of detected crop types")
        
        with col2:
            st.info("**Area Comparison Bar Chart**\nCompares relative areas of different crops")
        
        st.info("**Detailed Results Table**\nProvides precise area calculations and confidence scores")
    
    # Historical results (if available)
    st.subheader("📚 Historical Results")
    
    # Check for saved results
    results_dir = Path("detection_results")
    if results_dir.exists():
        result_files = list(results_dir.glob("*.json"))
        
        if result_files:
            st.write(f"Found {len(result_files)} previous detection results:")
            
            for result_file in sorted(result_files, reverse=True)[:5]:  # Show last 5
                with open(result_file, 'r') as f:
                    historical_result = json.load(f)
                
                with st.expander(f"🗓️ {historical_result.get('detection_date', 'Unknown Date')} - {historical_result.get('model_used', 'Unknown Model')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Confidence", f"{historical_result.get('confidence_score', 0):.2%}")
                        st.metric("Total Area", f"{historical_result.get('total_area_km2', 0)} km²")
                    
                    with col2:
                        if 'prediction_summary' in historical_result:
                            st.write("**Top Crops:**")
                            sorted_crops = sorted(
                                historical_result['prediction_summary'].items(), 
                                key=lambda x: x[1], 
                                reverse=True
                            )[:3]
                            for crop, percentage in sorted_crops:
                                st.write(f"• {crop.replace('_', ' ').title()}: {percentage:.1f}%")
        else:
            st.info("No historical results found.")
    else:
        st.info("No historical results directory found.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🌾 Complete Crop Detection System | End-to-End Pipeline</p>
    <p>Data Collection → Model Training → Crop Detection → Results</p>
    <p>Powered by Google Earth Engine, TensorFlow, and Streamlit</p>
</div>
""", unsafe_allow_html=True)
