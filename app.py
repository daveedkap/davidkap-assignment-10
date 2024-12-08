import os
import io
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import open_clip

# Initialize the Flask application
search_app = Flask(__name__)

# Load the CLIP model and preprocessing tools
model_name = "ViT-B-32-quickgelu"
pretrained_source = "openai"
clip_model, _, preprocess_image = open_clip.create_model_and_transforms(
    model_name, pretrained=pretrained_source
)
clip_model.eval()

# Tokenizer for processing text inputs
text_tokenizer = open_clip.get_tokenizer(model_name)

# Set up device for computation
device_type = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = clip_model.to(device_type)

# Load precomputed image embeddings
embeddings_data = pd.read_pickle("image_embeddings.pickle")
image_embeddings = np.stack(embeddings_data['embedding'])  # [N, D]
image_filenames = embeddings_data['file_name'].tolist()

# Helper function to compute text embeddings
def compute_text_embedding(text_query):
    tokens = text_tokenizer([text_query])
    with torch.no_grad():
        text_embedding = clip_model.encode_text(tokens.to(device_type))
        text_embedding = F.normalize(text_embedding, p=2, dim=1)
    return text_embedding.cpu().numpy()  # [1, D]

# Helper function to compute image embeddings from a PIL image
def compute_image_embedding(pil_image):
    image_input = preprocess_image(pil_image).unsqueeze(0).to(device_type)
    with torch.no_grad():
        image_embedding = clip_model.encode_image(image_input)
        image_embedding = F.normalize(image_embedding, p=2, dim=1)
    return image_embedding.cpu().numpy()  # [1, D]

# Apply PCA for dimensionality reduction
def reduce_dimensions_with_pca(embedding_matrix, components):
    pca_model = PCA(n_components=components)
    reduced_embeddings = pca_model.fit_transform(embedding_matrix)
    return reduced_embeddings, pca_model

# Flask route for the homepage and handling search functionality
@search_app.route('/', methods=['GET', 'POST'])
def home():
    search_results = None

    if request.method == 'POST':
        # Extract form inputs
        text_query = request.form.get('text_query', '').strip()
        weight_text_image = request.form.get('lam', '0.5').strip()
        num_pca_components = request.form.get('pca_k', '').strip()

        # Set default weight if not provided
        if weight_text_image == '':
            weight_text_image = 0.5
        else:
            weight_text_image = float(weight_text_image)

        # Handle uploaded image
        uploaded_image = request.files.get('image_query', None)
        has_image_query = (uploaded_image is not None and uploaded_image.filename != '')
        has_text_query = (text_query != '')

        # Check if PCA is requested
        apply_pca_reduction = (num_pca_components != '')
        if apply_pca_reduction:
            num_pca_components = int(num_pca_components)
            reduced_embeddings, pca_instance = reduce_dimensions_with_pca(
                image_embeddings, num_pca_components
            )
        else:
            reduced_embeddings = image_embeddings

        # Combine queries based on inputs
        query_vector = None

        if has_text_query and not has_image_query:
            # Text query only
            text_embedding = compute_text_embedding(text_query)
            if apply_pca_reduction:
                text_embedding = pca_instance.transform(text_embedding)
            query_vector = text_embedding

        elif has_image_query and not has_text_query:
            # Image query only
            image_bytes = uploaded_image.read()
            image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_embedding = compute_image_embedding(image_pil)
            if apply_pca_reduction:
                image_embedding = pca_instance.transform(image_embedding)
            query_vector = image_embedding

        elif has_image_query and has_text_query:
            # Combined text and image query
            image_bytes = uploaded_image.read()
            image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_embedding = compute_image_embedding(image_pil)
            text_embedding = compute_text_embedding(text_query)

            if apply_pca_reduction:
                image_embedding = pca_instance.transform(image_embedding)
                text_embedding = pca_instance.transform(text_embedding)

            query_vector = (
                weight_text_image * text_embedding
                + (1.0 - weight_text_image) * image_embedding
            )

        # If a valid query vector is formed, compute similarities
        if query_vector is not None:
            similarity_scores = cosine_similarity(query_vector, reduced_embeddings)[0]
            top_indices = np.argsort(similarity_scores)[::-1][:5]
            search_results = [
                {
                    "file_name": image_filenames[idx],
                    "similarity": float(similarity_scores[idx]),
                }
                for idx in top_indices
            ]

    # Render the HTML page with results if available
    return render_template('index.html', results=search_results)
