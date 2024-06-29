from path_src import detections_path, pings_path
import os
import base64
import numpy as np
from elasticsearch import Elasticsearch
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import joblib  # Import joblib for model saving and loading

img_dir = pings_path + "img/"

# Settings
save_images = False  # Set this to False if you do not want to save images to disk
if save_images and not os.path.exists(img_dir):
    os.makedirs(img_dir)

# Initialize Elasticsearch client
es = Elasticsearch()

# Load pre-trained ResNet50 model for feature extraction
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
feature_extraction_model = Model(inputs=base_model.input, outputs=base_model.output)

# Function to decode base64 and return an image array
def decode_image_base64(image_base64, save_path=None):
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))
        image = image.convert('RGB')  # Ensure image is in RGB format
        if save_path:
            image.save(save_path)
        return image
    except (UnidentifiedImageError, IOError) as e:
        print(f"Failed to process image: {str(e)}")
        return None

# Function to preprocess and extract features directly from an image
def extract_features(image, model):
    try:
        image = image.resize((60, 60))  # Resize image to match model's expected input
        img_array = img_to_array(image)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_array_expanded)
        features = model.predict(img_preprocessed)
        return features.flatten()
    except Exception as e:
        print(f"Failed to extract features: {str(e)}")
        return None

# Path to the KMeans model
model_path = 'kmeans_model.pkl'
if os.path.exists(model_path):
    # Load the existing model
    kmeans = joblib.load(model_path)
    # Query for images from October 17, 2018
    query = {
    "query": {
        "bool": {
            "must": [
                {
                    "range": {
                        "timestamp": {
                            "gte": "2018-10-17T17:00:00",
                            "lte": "2018-10-17T23:59:59"
                        }
                    }
                }
            ],
            "must_not": [
                {
                    "exists": {
                        "field": "cluster"
                    }
                }
            ]
        }
    }
    }
else:
    # Query for all images if the model does not exist
    query = {"query": {"match_all": {}}}
    kmeans = KMeans(n_clusters=10, random_state=42)  # Initialize a new KMeans model

# Execute the search
resp = es.search(index="credo-detections", body=query, size=10000)

# Process images
features_list = []
doc_ids = []
for hit in resp['hits']['hits']:
    doc_id = hit['_id']
    image_base64 = hit['_source']['frame_content']
    save_path = f"{img_dir}{doc_id}.png" if save_images else None
    image = decode_image_base64(image_base64, save_path)
    if image:
        features = extract_features(image, feature_extraction_model)
        if features is not None:
            features_list.append(features)
            doc_ids.append(doc_id)
        else:
            print(f"Skipping image due to feature extraction failure: {doc_id}")
    else:
        print(f"Skipping image due to decoding failure: {doc_id}")


# Fit or predict clusters based on the model's existence
if not os.path.exists(model_path):
    if features_list:  # Ensure there is data to fit
        kmeans.fit(features_list)
        joblib.dump(kmeans, model_path)
else:
    if features_list:
        clusters = kmeans.predict(features_list)
        # Update Elasticsearch with cluster labels if not already labeled
        for doc_id, cluster_label in zip(doc_ids, clusters):
            es.update(index='credo-detections', id=doc_id, body={"doc": {"cluster": cluster_label}})
        print("Clustering complete and Elasticsearch updated with cluster labels.")