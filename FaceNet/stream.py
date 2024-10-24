import streamlit as st
import google.generativeai as genai
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import uuid
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import json
from pathlib import Path
from mtcnn import MTCNN  # Added MTCNN import
import requests

# --------------------------- Configuration ---------------------------

# Set Streamlit page configuration
st.set_page_config(
    page_title="HistoriClass",
    page_icon="🧑‍🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API Key for Google Generative AI (Consider using environment variables for security)
API_KEY = "AIzaSyCDb_mHYOsaR-99UeKvIyvkqJD9pfHHbWg"
genai.configure(api_key=API_KEY)

# Create 'uploads' directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# Initialize MTCNN detector
detector = MTCNN()

# --------------------------- Load Model ---------------------------

@tf.keras.utils.register_keras_serializable()
def l2_normalize(x, axis=None):
    return tf.linalg.l2_normalize(x, axis=axis)

@tf.keras.utils.register_keras_serializable()
def scaling(x, scale=1.0):
    return x * scale

def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/file/d/1QeyoWs19nE8Xfq6dx8zvNy_StyCbyN_J/view?usp=sharing"

    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

def download_model():
    # Google Drive file ID for your model
    file_id = "1QeyoWs19nE8Xfq6dx8zvNy_StyCbyN_J"  # Replace with your actual File ID
    model_dir = os.path.join(os.getcwd(), 'Model')
    model_path = os.path.join(model_dir, 'facenet_87img_profile_face1.h5')

    # Create directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Download the model if it's not already present
    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")
        download_file_from_google_drive(file_id, model_path)
        print("Model downloaded successfully!")
    else:
        print("Model already exists locally.")

    return model_path


# Load the pre-trained model with custom objects
@st.cache_resource
def load_trained_model():
    model_path = download_model()  # Ensure model is downloaded or present locally
    model = load_model(
        model_path,
        custom_objects={
            'l2_normalize': l2_normalize,  # Custom objects if needed
            'scaling': scaling
        }
    )
    return model

# Usage example
model = load_trained_model()

# --------------------------- Class Indices ---------------------------


@st.cache_data
def load_class_indices(json_path='config/class_indices.json'):
    """
    Loads class indices from a JSON file.

    Args:
        json_path (str): Path to the JSON file containing class indices.

    Returns:
        dict: A dictionary mapping class indices to personality names.
    """
    try:
        with open(json_path, 'r') as f:
            class_indices = json.load(f)
        # Convert string keys to integers
        class_indices = {int(k): v for k, v in class_indices.items()}
        return class_indices
    except FileNotFoundError:
        st.error(f"Class indices file not found at {json_path}.")
        return {}
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from the file {json_path}.")
        return {}

# Load class indices from external JSON file
class_indices = load_class_indices('config/class_indices.json')

# --------------------------- Functions ---------------------------

def prepare_image(image_bytes, target_size=(224, 224)):
    """
    Detects a face in the image using MTCNN, crops it, resizes, and preprocesses it.
    If no face is detected, uses the whole image.

    Args:
        image_bytes (bytes): Uploaded image in bytes.
        target_size (tuple): Desired image size.

    Returns:
        np.ndarray or None: Preprocessed image array or None if invalid.
    """
    # Convert image bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Invalid image uploaded. Please try another image.")
        return None

    # Convert the image from BGR (OpenCV format) to RGB (MTCNN works on RGB)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces in the image using MTCNN
    faces = detector.detect_faces(rgb_img)

    if len(faces) > 0:
        # Take the first detected face
        x, y, width, height = faces[0]['box']
        x, y = max(0, x), max(0, y)  # Ensure coordinates are positive
        face_img = img[y:y+height, x:x+width]
    else:
        # If no face is detected, use the whole image
        face_img = img

    # Convert to RGB (from BGR)
    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image and resize
    pil_img = Image.fromarray(face_img_rgb)
    pil_img = pil_img.resize(target_size)

    # Convert to array
    img_array = image.img_to_array(pil_img)

    # Scale the image by dividing by 255
    img_array = img_array / 255.0

    # Expand dimensions to match the expected input shape
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def fetch_description_from_gemini(personality: str) -> dict:
    """
    Fetches a structured biography from Google Gemini API for the given personality.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
    Provide a structured biography for the Indian historical personality {personality}. 
    The biography should be formatted as key-value pairs.
    """

    try:
        response = model.generate_content(prompt)
    except Exception as e:
        st.error(f"Error fetching description from Gemini API: {e}")
        return {
            "birth_date": "Unknown",
            "real_name": "Unknown",
            "alias": "Unknown",
            "achievements": "Unknown",
            "brief_description": "No description available."
        }

    if response.text:
        response_dict = parse_api_response(response.text)
        return response_dict
    else:
        return {
            "birth_date": "Unknown",
            "real_name": "Unknown",
            "alias": "Unknown",
            "achievements": "Unknown",
            "brief_description": "No description available."
        }

def parse_api_response(response: str) -> dict:
    """
    Converts the API response into a dictionary format.
    Excludes any unwanted lines like "Key:: Value:".
    """
    lines = response.split("\n")
    data = {}
    key = None
    for line in lines:
        if "**" in line and "|" in line:
            parts = line.split("|")
            if len(parts) == 2:
                key = parts[0].replace("**", "").strip()
                value = parts[1].strip()
                data[key] = value
        elif "*" in line and key:  # Handling bullet points under previous key
            if isinstance(data[key], str):
                data[key] = [data[key]]
            data[key].append(line.strip("* ").strip())
    return data

# Function to display the structured description
def display_personality_details(description: dict):
    # Main Section Title
    st.markdown("<h3 style='color: #FF4B4B;'>📚 Personality Details</h3>", unsafe_allow_html=True)

    # Iterate through the description dictionary and format it attractively
    for key, value in description.items():
        # Skip any undesired lines
        if key.lower() == "key:: value:":
            continue

        # Display each key in bold with a line separator
        st.markdown(f"<h4 style='color: #00BFFF;'>{key.replace('_', ' ').capitalize()}:</h4>", unsafe_allow_html=True)

        if isinstance(value, list):
            # Display lists as bullet points
            st.markdown("<ul>", unsafe_allow_html=True)
            for item in value:
                st.markdown(f"<li style='color: #FFFFFF;'>{item}</li>", unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)
        else:
            # Display single values
            st.markdown(f"<p style='color: #FFFFFF;'>{value}</p>", unsafe_allow_html=True)

@st.cache_data
def get_personality_images(personality_name: str, max_images: int = 2) -> list:
    """
    Retrieves up to `max_images` image paths for the given personality.

    Args:
        personality_name (str): Name of the predicted personality.
        max_images (int): Maximum number of images to retrieve.

    Returns:
        list: List of image file paths.
    """
    # Updated path to match data preprocessing
    images_dir = Path("./images/PERSONALITIES") / personality_name
    if not images_dir.exists() or not images_dir.is_dir():
        st.warning(f"No images found for {personality_name} in the dataset.")
        return []

    # Retrieve image paths with common image extensions
    image_paths = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
    if not image_paths:
        st.warning(f"No images found for {personality_name} in the dataset.")
        return []

    # Limit to `max_images`
    selected_images = image_paths[:max_images]
    return selected_images

def display_personality_images(image_paths: list):
    """
    Displays a list of images in a grid layout.

    Args:
        image_paths (list): List of image file paths to display.
    """
    if not image_paths:
        return

    st.markdown("<h3 style='color: #FF4B4B;'>📷 More Images of This Personality</h3>", unsafe_allow_html=True)

    # Determine the number of columns based on the number of images
    num_images = len(image_paths)
    cols = st.columns(num_images)

    for col, img_path in zip(cols, image_paths):
        try:
            image = Image.open(img_path)
            col.image(image, use_column_width=True, caption=Path(img_path).stem)
        except Exception as e:
            st.error(f"Error loading image {img_path}: {e}")

# --------------------------- Streamlit UI ---------------------------

def main():
    # Sidebar for uploading image
    with st.sidebar:
        st.title("🎨 Upload Image")
        st.write("Upload an image of an Indian historical personality to predict and learn more.")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Main UI Header
    st.title("🧑‍🏫 HistoriClass")

    # Columns for better layout
    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        # Display the uploaded image in column 1
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Read image bytes
        img_bytes = uploaded_file.read()

        # Preprocess the image
        img_array = prepare_image(img_bytes)

        if img_array is not None:
            # Show a spinner while processing
            with st.spinner("Analyzing Image..."):
                # Make prediction
                predictions = model.predict(img_array)
                class_idx = np.argmax(predictions[0])
                predicted_personality = class_indices.get(class_idx, "Unknown")

            # Display the predicted personality in column 2
            with col2:
                st.header(f"🔮 Predicted Personality: {predicted_personality}")

            # Fetch description from Gemini API
            description = fetch_description_from_gemini(predicted_personality)
            
            # Display the structured description
            display_personality_details(description)

            # Retrieve and display additional images
            additional_images = get_personality_images(predicted_personality)
            display_personality_images(additional_images)

            # Save the uploaded image to 'uploads' directory
            filename = f"{uuid.uuid4()}.png"
            image_path = os.path.join("uploads", filename)
            with open(image_path, "wb") as f:
                f.write(img_bytes)

    else:
        st.write("👈 Upload an image to get started.")

    # Additional Information Section
    st.markdown("---")
    st.subheader("ℹ️ How does this work?")
    st.write("""
    This application uses a pre-trained neural network model to predict the identity of Indian historical personalities based on uploaded images.
    Once an image is uploaded, the model analyzes it, predicts who the historical personality is, and then fetches a biography using Google's Gemini API.
    """)

if __name__ == "__main__":
    main()
