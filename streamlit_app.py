import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Define class labels
class_labels = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']

# Load model
@st.cache_resource
def load_cancer_model():
    try:
        model = load_model("lung5.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_cancer_model()

# Preprocessing function
def load_and_preprocess_image(img_file, target_size=(224, 224)):
    img = Image.open(img_file).convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Streamlit UI
st.title("🫁 Lung Cancer Detector")
st.write("Upload a lung CT image and let the model predict the class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Preprocess and predict
    img_array = load_and_preprocess_image(uploaded_file)
    predictions = model.predict(img_array)
    predicted_class = int(np.argmax(predictions[0]))
    predicted_label = class_labels[predicted_class]
    confidence = float(np.max(predictions[0]))

    st.subheader("Prediction Results")
    st.write(f"**Predicted Label:** `{predicted_label}`")
    st.write(f"**Confidence:** `{confidence:.2f}`")

    # Show all probabilities
    st.subheader("Class Probabilities")
    for label, prob in zip(class_labels, predictions[0]):
        st.write(f"{label}: `{prob:.4f}`")
