import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from elasticsearch import Elasticsearch
from datetime import datetime
import os

# Initialize Elasticsearch client
es = Elasticsearch()

# Function to decode base64 and return an image object
def decode_image_base64(image_base64):
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data))
    image = image.convert('RGB')  # Convert image to RGB
    return image

# Function to fetch images by cluster label
def fetch_images_by_cluster(cluster_id, index_name, max_results=100):
    query = {
        "size": max_results,
        "query": {
            "bool": {
                "must": [
                    {"match": {"cluster": cluster_id}}
                ]
            }
        }
    }
    response = es.search(index=index_name, body=query)
    images = []
    metadata = []
    for hit in response['hits']['hits']:
        image_base64 = hit['_source']['frame_content']
        image = decode_image_base64(image_base64)
        timestamp = datetime.fromtimestamp(int(hit['_source']['timestamp']) / 1000).strftime('%Y-%m-%d %H:%M:%S')
        user_id = hit['_source']['user_id']
        images.append(image)
        metadata.append((timestamp, user_id))
    return images, metadata

# Function to display and save a list of images with metadata, including cluster ID in the footer
def display_and_save_images(images, metadata, cluster_id, n_cols=5):
    if not images:
        print("No images to display.")
        return
    n_rows = (len(images) + n_cols - 1) // n_cols  # calculate the number of rows needed
    plt.figure(figsize=(15, 4 * n_rows + 0.5))  # adjust figure size based on rows, columns, and footer space
    for i, (image, (timestamp, user_id)) in enumerate(zip(images, metadata)):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Time: {timestamp}\nUser: {user_id}", fontsize=8)  # Display timestamp and user ID under each image
    plt.tight_layout()
    plt.figtext(0.5, 0.01, f'Cluster {cluster_id}', ha='center', fontsize=12)  # Footer text with cluster ID
    save_path = os.path.expanduser(f"~/Downloads/cluster{cluster_id}.png")
    plt.savefig(save_path)
    print(f"Saved cluster image to {save_path}")
    plt.show()

# Example usage: Display and save images from cluster 0
cluster_id = 3
index_name = 'credo-detections'  # Your Elasticsearch index name
images, metadata = fetch_images_by_cluster(cluster_id, index_name)
display_and_save_images(images, metadata, cluster_id)


'''


cluster_ids=range(0, 11)
for cluster_id in cluster_ids:
    index_name = 'credo-detections'  # Your Elasticsearch index name
    images, metadata = fetch_images_by_cluster(cluster_id, index_name)
    display_and_save_images(images, metadata, cluster_id)
'''
