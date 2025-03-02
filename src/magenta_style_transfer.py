import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import uuid  # Import UUID for unique filenames
from PIL import Image
import os

# Load the TensorFlow Hub model only once
MODEL_PATH = "models/google_magenta_model"
style_transfer_model = hub.load(MODEL_PATH)

def load_image(image_path):
    """Load an image, resize, and normalize it for TensorFlow."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((512, 512))  # Resize for faster processing
    img = np.array(img) / 255.0  # Normalize
    return tf.convert_to_tensor(img, dtype=tf.float32)[tf.newaxis, ...]

def stylize_image(content_path, style_path, output_folder="static/results/magenta"):
    """Apply style transfer and save the output image with a unique filename."""
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists

    content_image = load_image(content_path)
    style_image = load_image(style_path)

    # Perform style transfer
    stylized_image = style_transfer_model(tf.constant(content_image), tf.constant(style_image))[0]

    # ✅ Convert tensor to numpy array and format correctly
    stylized_image = np.array(stylized_image[0])  # Remove batch dimension
    stylized_image = (stylized_image * 255).astype(np.uint8)  # Convert to uint8 (0-255)

    # ✅ Generate a unique filename using UUID
    unique_filename = f"styled_{uuid.uuid4().hex[:8]}.jpg"  # Short UUID (8 characters)
    output_path = os.path.join(output_folder, unique_filename)

    # Save the final image
    Image.fromarray(stylized_image).save(output_path)

    return output_path, unique_filename  # Return new filename
