
## Apply Federated Learning to Pre-classified images
[https://github.com/carlynlee/FLSim/blob/main/docs/tutorials/federated_learning_for_image_classification.ipynb](https://github.com/carlynlee/FLSim/blob/main/docs/tutorials/federated_learning_for_image_classification.ipynb)


# README

## Overview

This repository contains a set of Python scripts designed to process, analyze, and visualize image data stored in an Elasticsearch index. The scripts support operations such as decoding images, clustering, and visualizing images and their distributions across different clusters and devices.

### Scripts in the Repository

1. **`decode_and_save_images.py`**:
   - Extracts images from JSON files, decodes them from Base64, and saves them as PNG files.

2. **`cluster_images_from_elasticsearch.py`**:
   - Extracts images from an Elasticsearch index, processes them using a pre-trained ResNet50 model, clusters them using KMeans, and updates Elasticsearch with the cluster labels. The script can also visualize the clusters using PCA.

3. **`visualize_cluster_mean_images.py`**:
   - Fetches images from Elasticsearch by device and cluster IDs, computes mean images for each cluster, and displays them in a grid.

4. **`fetch_and_display_cluster_images.py`**:
   - Fetches images by cluster ID from Elasticsearch, displays them with associated metadata (timestamp and user ID), and saves the visualizations as PNG files.

5. **`plot_device_cluster_statistics.py`**:
   - Fetches the distribution of cluster IDs for different devices over a specific time period and visualizes the results in a bar plot.

## Requirements

- Python 3.x
- The following Python modules:
  - `os`
  - `json`
  - `base64`
  - `numpy`
  - `collections`
  - `Elasticsearch` (from `elasticsearch-py`)
  - `tensorflow` (for Keras models)
  - `sklearn` (for KMeans clustering and PCA)
  - `PIL` (from `Pillow`)
  - `joblib`
  - `matplotlib`

## Setup

Before running any of the scripts:

- Ensure your Elasticsearch instance is running with an index named `credo-detections`.
- Ensure the `path_src.py` file is correctly set up if needed for certain scripts.

## How to Run the Scripts

### 1. **`decode_and_save_images.py`**

Processes JSON files to extract and save images as PNG files.

```bash
python decode_and_save_images.py
