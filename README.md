# UAV Geo-Localization System

A deep learning-based UAV geo-localization system that estimates the geographic location of UAV images by matching them against a satellite image database. The system combines **global image retrieval** using **EigenPlaces**, **local feature matching** using **LoFTR**, and **FAISS** for fast large-scale nearest neighbor search to achieve accurate visual localization. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

---

## Overview

This project consists of two main stages:

1. **Satellite Database Construction**
   - Extract deep visual descriptors from satellite images.
   - Build a FAISS vector database for efficient image retrieval.
   - Store tile metadata for geographic coordinate recovery.

2. **Batch UAV Localization**
   - Retrieve the most similar satellite images.
   - Refine the match using local feature correspondence.
   - Estimate UAV geographic coordinates.
   - Evaluate localization accuracy against ground truth.
   - Generate visualization results.

---

## System Pipeline

```
Satellite Images
       │
       ▼
EigenPlaces Feature Extraction
       │
       ▼
L2 Normalization
       │
       ▼
FAISS Vector Database
       │
       │
       ├─────────────────────────────┐
       │                             │
       ▼                             ▼
UAV Query Image              Metadata Database
       │
       ▼
EigenPlaces Retrieval (Top-3)
       │
       ▼
LoFTR Local Feature Matching
       │
       ▼
Homography Estimation
       │
       ▼
Pixel Coordinate Projection
       │
       ▼
Latitude / Longitude Estimation
       │
       ▼
Localization Error Evaluation
```

---

# Project Structure

```
.
├── build_database.py          # Build satellite feature database
├── batch_evaluate.py          # Batch localization evaluation
├── satellite_gallery_latest/  # Satellite image gallery
├── competition_gallery_true/  # UAV query images
├── satellite_gallery.index    # FAISS index
├── satellite_meta.json        # Satellite metadata
├── evaluation_results/        # Matching visualization results
└── README.md
```

---

# Features

## Satellite Feature Database Construction

The database generation script performs:

- Loading the pretrained EigenPlaces model
- Feature extraction from satellite images
- Feature normalization
- FAISS index generation
- Metadata generation
- Automatic feature dimension detection

Each satellite tile is associated with

- Zoom level
- Tile X coordinate
- Tile Y coordinate
- Filename
- FAISS index ID

The generated database enables efficient large-scale image retrieval. :contentReference[oaicite:2]{index=2}

---

## Deep Global Image Retrieval

Global retrieval is performed using **EigenPlaces**, a state-of-the-art visual place recognition model.

Features include

- ResNet-50 backbone
- 2048-dimensional feature vectors
- ImageNet normalization
- Cosine similarity search
- Top-3 candidate retrieval

EigenPlaces provides robust place recognition under changes in

- viewpoint
- illumination
- seasonal conditions
- scale

---

## Fast Similarity Search

The project employs **Facebook AI Similarity Search (FAISS)**.

Advantages include

- Large-scale indexing
- High-speed nearest neighbor search
- Inner-product similarity search
- Efficient memory usage

The gallery descriptors are stored in

```
satellite_gallery.index
```

---

## Local Feature Verification

To improve localization precision, the retrieved candidates are refined using **LoFTR (Detector-Free Local Feature Matching)**.

Advantages include

- Detector-free matching
- Dense correspondences
- Robust matching under viewpoint changes
- High matching accuracy

Only geometrically consistent matches are retained.

---

## Homography-Based Localization

For each retrieved candidate

1. Estimate feature correspondences
2. Compute homography using USAC_MAGSAC
3. Transform the UAV image center
4. Convert pixel coordinates into GPS coordinates

The candidate with the highest number of inliers is selected as the final localization result.

---

## Fallback Strategy

If reliable local matching cannot be established,

the system automatically falls back to the geographic center of the highest-ranked retrieved satellite tile.

This improves robustness in challenging scenarios while preventing localization failure. :contentReference[oaicite:3]{index=3}

---

# Evaluation

The evaluation script automatically computes

- Mean localization error
- Median error
- Minimum error
- Maximum error

Accuracy statistics include

- Error ≤ 5 meters
- Error ≤ 15 meters
- Error ≤ 50 meters

The localization error is calculated using the Haversine distance between

- predicted GPS coordinates
- ground truth GPS coordinates

---

# Visualization

For every evaluated UAV image, the system generates a visualization containing

- UAV query image
- Retrieved satellite image
- Feature correspondences
- Estimated coordinates
- Ground truth coordinates
- Localization error
- Number of inlier matches

The results are automatically saved into

```
evaluation_results/
```

---

# Dataset Format

## Satellite Gallery

```
satellite_gallery_latest/

18_219845_107432.jpg
18_219845_107433.jpg
...
```

Filename format

```
zoom_x_y.jpg
```

---

## UAV Test Images

```
competition_gallery_true/

18_219850_107430.jpg
18_219852_107431.jpg
...
```

Ground truth coordinates are automatically derived from the filename.

---

# Installation

## Requirements

- Python 3.9+
- PyTorch
- torchvision
- timm
- faiss
- Pillow
- OpenCV
- NumPy
- tqdm
- mercantile
- Kornia

Install all dependencies

```bash
pip install torch torchvision
pip install timm
pip install faiss-cpu
pip install pillow
pip install numpy
pip install opencv-python
pip install tqdm
pip install mercantile
pip install kornia
```

If CUDA is available, install the GPU version of FAISS and PyTorch for significantly faster inference.

---

# Usage

## Step 1 — Build the Database

```bash
python build_database.py
```

Outputs

```
satellite_gallery.index
satellite_meta.json
```

---

## Step 2 — Evaluate UAV Localization

```bash
python batch_evaluate.py
```

The script automatically

- loads the database
- extracts UAV features
- retrieves Top-3 satellite candidates
- performs LoFTR verification
- estimates GPS coordinates
- evaluates localization accuracy
- saves visualization results

---

# Core Technologies

- Python
- PyTorch
- EigenPlaces
- LoFTR
- FAISS
- OpenCV
- NumPy
- Pillow
- Mercantile

---

# Algorithm Summary

| Stage | Method |
|--------|--------|
| Global Retrieval | EigenPlaces |
| Feature Search | FAISS |
| Local Matching | LoFTR |
| Geometric Verification | USAC_MAGSAC Homography |
| Coordinate Estimation | Pixel-to-GPS Projection |
| Error Metric | Haversine Distance |

---

# Output

After evaluation, the project generates

```
evaluation_results/
├── image001.jpg
├── image002.jpg
├── image003.jpg
└── ...
```

Each result image visualizes

- Query image
- Retrieved satellite image
- Matching correspondences
- Estimated GPS coordinates
- Ground truth coordinates
- Localization error

---

# Future Improvements

Potential future enhancements include

- Support for larger satellite databases
- Multi-scale retrieval
- Rotation-invariant matching
- Temporal localization using UAV trajectories
- Multi-modal localization with IMU/GNSS fusion
- Real-time onboard deployment
- ONNX/TensorRT acceleration
- Interactive localization visualization
- ROS2 integration
- Web-based localization interface

---

# Citation

If you use this project in your research, please cite the original libraries and models:

- EigenPlaces
- LoFTR
- FAISS
- PyTorch
- OpenCV

---

# License

This project is intended for academic research, computer vision, and UAV geo-localization studies. Users are responsible for complying with the licenses of all third-party libraries and pretrained models.