import matplotlib.pyplot as plt
import numpy as np
import base64
from PIL import Image
from io import BytesIO
from elasticsearch import Elasticsearch

# Initialize Elasticsearch client
es = Elasticsearch()

def decode_image_base64(image_base64, size=(256, 256)):
    """Decode an image from a base64 encoding and resize it to the specified size."""
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data))
    image = image.convert('RGB')  # Convert image to RGB
    image = image.resize(size, Image.ANTIALIAS)  # Resize image to ensure uniformity
    return image

def fetch_images(device_id, cluster_id):
    """Fetch images from Elasticsearch based on device_id and cluster_id."""
    query = {
        "size": 1000,
        "query": {
            "bool": {
                "must": [
                    {"term": {"device_id": device_id}},
                    {"term": {"cluster": cluster_id}},
                    {"range": {"timestamp": {"gte": "2018-10-01T00:00:00", "lte": "2018-10-31T23:59:59"}}}
                ]
            }
        }
    }
    response = es.search(index="credo-detections", body=query)
    images = []
    for hit in response['hits']['hits']:
        image_base64 = hit['_source']['frame_content']
        image = decode_image_base64(image_base64)
        images.append(np.array(image))
    return images

def plot_mean_images_grid(device_ids, cluster_ids):
    num_devices = len(device_ids)
    num_clusters = len(cluster_ids)
    fig, axes = plt.subplots(nrows=num_devices, ncols=num_clusters, figsize=(num_clusters * 3, num_devices * 3))

    for i, device_id in enumerate(device_ids):
        for j, cluster_id in enumerate(cluster_ids):
            ax = axes[i][j]
            images = fetch_images(device_id, cluster_id)
            if images:
                image_stack = np.stack(images, axis=0)
                mean_image = np.mean(image_stack, axis=0).astype(np.uint8)
                ax.imshow(mean_image)
            else:
                ax.imshow(np.zeros((256, 256, 3), dtype=np.uint8))  # Show an empty image if no data
            ax.axis('off')

            if j == 0:  # Only add device labels in the first column
                ax.set_ylabel(f'Device {device_id}', fontsize=10, rotation=90, labelpad=15, horizontalalignment='right')
            if i == 0:  # Only add cluster labels in the first row
                ax.set_title(f'Cluster {cluster_id}')

    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05)
    plt.tight_layout()
    plt.show()

# List of device IDs to process
device_ids = [4866, 4961, 5555, 5209, 4681, 5158]
cluster_ids = range(10)  # Cluster IDs from 0 to 9

# Call the function to plot images grid
plot_mean_images_grid(device_ids, cluster_ids)
