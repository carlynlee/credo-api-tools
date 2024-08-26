
## Apply Federated Learning to Pre-classified images
[https://github.com/carlynlee/FLSim/blob/main/docs/tutorials/federated_learning_for_image_classification.ipynb](https://github.com/carlynlee/FLSim/blob/main/docs/tutorials/federated_learning_for_image_classification.ipynb)

README
======

Overview
--------

This repository contains a set of Python scripts designed to process, analyze, and visualize image data stored in an Elasticsearch index. The scripts support operations such as decoding images, clustering, and visualizing images and their distributions across different clusters and devices.

### Scripts in the Repository

1.  **`decode_and_save_images.py`**:

    -   Extracts images from JSON files, decodes them from Base64, and saves them as PNG files.
2.  **`cluster_images_from_elasticsearch.py`**:

    -   Extracts images from an Elasticsearch index, processes them using a pre-trained ResNet50 model, clusters them using KMeans, and updates Elasticsearch with the cluster labels. The script can also visualize the clusters using PCA.
3.  **`visualize_cluster_mean_images.py`**:

    -   Fetches images from Elasticsearch by device and cluster IDs, computes mean images for each cluster, and displays them in a grid.
4.  **`fetch_and_display_cluster_images.py`**:

    -   Fetches images by cluster ID from Elasticsearch, displays them with associated metadata (timestamp and user ID), and saves the visualizations as PNG files.
5.  **`plot_device_cluster_statistics.py`**:

    -   Fetches the distribution of cluster IDs for different devices over a specific time period and visualizes the results in a bar plot.

Requirements
------------

-   Python 3.x
-   The following Python modules:
    -   `os`
    -   `json`
    -   `base64`
    -   `numpy`
    -   `collections`
    -   `Elasticsearch` (from `elasticsearch-py`)
    -   `tensorflow` (for Keras models)
    -   `sklearn` (for KMeans clustering and PCA)
    -   `PIL` (from `Pillow`)
    -   `joblib`
    -   `matplotlib`

Setup
-----

Before running any of the scripts:

-   Ensure your Elasticsearch instance is running with an index named `credo-detections`.
-   Ensure the `path_src.py` file is correctly set up if needed for certain scripts.

How to Run the Scripts
----------------------

### 1\. **`decode_and_save_images.py`**

Processes JSON files to extract and save images as PNG files.

bash

`python decode_and_save_images.py`

-   Adjust the `values` and `max` parameters in the `main()` function to control which JSON file is processed and how many images are saved.

### 2\. **`cluster_images_from_elasticsearch.py`**

Extracts images from Elasticsearch, performs feature extraction using ResNet50, clusters them using KMeans, and updates Elasticsearch with the cluster labels. The script also visualizes the clusters using PCA.

bash

`python cluster_images_from_elasticsearch.py`

-   Modify parameters such as `save_images` and `model_path` within the script to control behavior, such as saving images locally or using an existing KMeans model.

### 3\. **`visualize_cluster_mean_images.py`**

Fetches images based on device and cluster IDs, computes mean images for each cluster, and displays them in a grid.

bash

`python visualize_cluster_mean_images.py`

-   By default, the script processes device IDs `[4866, 4961, 5555, 5209, 4681, 5158]` and cluster IDs from `0` to `9`. Adjust these lists in the script to process different devices or clusters.

### 4\. **`fetch_and_display_cluster_images.py`**

Fetches images by cluster ID from Elasticsearch, displays them with associated metadata (timestamp and user ID), and saves the visualization as a PNG file.

bash

`python fetch_and_display_cluster_images.py`

-   The script fetches and displays images from a specified cluster and saves the resulting image grid to your `Downloads` directory.

### 5\. **`plot_device_cluster_statistics.py`**

Fetches the distribution of cluster IDs for different devices over October 2018 and visualizes the results in a bar plot.

bash

`python plot_device_cluster_statistics.py`

-   The script generates a bar plot showing the distribution of clusters for each device. Each subplot represents a device, with cluster IDs on the x-axis and counts on the y-axis.

Example Workflow
----------------

1.  **Cluster Images**:

    -   Use `cluster_images_from_elasticsearch.py` to cluster your images and update Elasticsearch.
2.  **Visualize Cluster Distributions**:

    -   Use `plot_device_cluster_statistics.py` to understand how images are distributed across different clusters for various devices.
3.  **Analyze Individual Clusters**:

    -   Use `fetch_and_display_cluster_images.py` to visualize and save images from specific clusters.

Notes
-----

-   Ensure images in Elasticsearch are correctly encoded in Base64 and stored under the `frame_content` field.
-   Customize the clustering parameters and queries based on your specific dataset and analysis goals.
-   PCA is used for visualization and may not capture all variance in high-dimensional data.

License
-------

These scripts are provided as-is, without any warranty or guarantee of fitness for any particular purpose. Use them at your own risk.
