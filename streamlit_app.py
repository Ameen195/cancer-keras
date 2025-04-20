# # # import streamlit as st
# # # import numpy as np
# # # from tensorflow.keras.models import load_model
# # # from tensorflow.keras.preprocessing import image
# # # from PIL import Image
# # # import io

# # # # Define class labels
# # # class_labels = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']

# # # # Load model
# # # @st.cache_resource
# # # def load_cancer_model():
# # #     try:
# # #         model = load_model("lung5.keras")
# # #         return model
# # #     except Exception as e:
# # #         st.error(f"Error loading model: {e}")
# # #         return None

# # # model = load_cancer_model()

# # # # Preprocessing function
# # # def load_and_preprocess_image(img_file, target_size=(224, 224)):
# # #     img = Image.open(img_file).convert("RGB")
# # #     img = img.resize(target_size)
# # #     img_array = image.img_to_array(img)
# # #     img_array = np.expand_dims(img_array, axis=0)
# # #     img_array /= 255.0
# # #     return img_array

# # # # Streamlit UI
# # # st.title("🫁 Lung Cancer Detector")
# # # st.write("Upload a lung CT image and let the model predict the class.")

# # # uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# # # if uploaded_file and model:
# # #     st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

# # #     # Preprocess and predict
# # #     img_array = load_and_preprocess_image(uploaded_file)
# # #     predictions = model.predict(img_array)
# # #     predicted_class = int(np.argmax(predictions[0]))
# # #     predicted_label = class_labels[predicted_class]
# # #     confidence = float(np.max(predictions[0]))

# # #     st.subheader("Prediction Results")
# # #     st.write(f"**Predicted Label:** `{predicted_label}`")
# # #     st.write(f"**Confidence:** `{confidence:.2f}`")

# # #     # Show all probabilities
# # #     st.subheader("Class Probabilities")
# # #     for label, prob in zip(class_labels, predictions[0]):
# # #         st.write(f"{label}: `{prob:.4f}`")


# # import streamlit as st
# # import numpy as np
# # from tensorflow.keras.models import load_model
# # from tensorflow.keras.preprocessing import image
# # from PIL import Image
# # import os

# # # --- Page Configuration ---
# # st.set_page_config(
# #     page_title="Lung Cancer Detector",
# #     page_icon="🫁",
# #     layout="centered",
# #     initial_sidebar_state="collapsed"
# # )

# # # --- Styling ---
# # st.markdown("""
# #     <style>
# #     .title { font-size: 36px; font-weight: 700; color: #f63366; text-align: center; margin-bottom: 1rem; }
# #     .subtitle { font-size: 20px; color: #555; text-align: center; margin-bottom: 2rem; }
# #     .prediction-box {
# #         background-color: #f0f2f6;
# #         padding: 20px;
# #         border-radius: 12px;
# #         margin-top: 20px;
# #         text-align: center;
# #         box-shadow: 0px 0px 8px rgba(0,0,0,0.1);
# #     }
# #     .label {
# #         font-size: 28px;
# #         font-weight: bold;
# #         color: #2a9d8f;
# #     }
# #     .confidence {
# #         font-size: 20px;
# #         color: #264653;
# #     }
# #     .sample-grid img {
# #         border-radius: 10px;
# #         margin: 10px;
# #         width: 150px;
# #         height: 150px;
# #         object-fit: cover;
# #         box-shadow: 0 4px 8px rgba(0,0,0,0.1);
# #     }
# #     </style>
# # """, unsafe_allow_html=True)

# # # --- Title ---
# # st.markdown('<div class="title">🫁 Lung Cancer Detector</div>', unsafe_allow_html=True)
# # st.markdown('<div class="subtitle">Upload a lung CT image or try a sample to get a prediction.</div>', unsafe_allow_html=True)

# # # --- Class Labels ---
# # class_labels = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']

# # # --- Load model ---
# # @st.cache_resource
# # def load_cancer_model():
# #     try:
# #         model = load_model("lung5.keras")
# #         return model
# #     except Exception as e:
# #         st.error(f"Error loading model: {e}")
# #         return None

# # model = load_cancer_model()

# # # --- Image Preprocessing ---
# # def load_and_preprocess_image(img_file, target_size=(224, 224)):
# #     img = Image.open(img_file).convert("RGB")
# #     img = img.resize(target_size)
# #     img_array = image.img_to_array(img)
# #     img_array = np.expand_dims(img_array, axis=0)
# #     img_array = img_array / 255.0
# #     return img_array

# # # --- Sample Images Grid ---
# # with st.expander("🖼️ Try with a Sample Image"):
# #     sample_dir = "./samples"
# #     sample_files = sorted([f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])

# #     cols = st.columns(len(sample_files))
# #     selected_sample = None
# #     for i, file in enumerate(sample_files):
# #         with cols[i]:
# #             if st.button(f"Try a{i+1}", key=f"sample{i}"):
# #                 selected_sample = os.path.join(sample_dir, file)

# # # --- Upload Image or Use Sample ---
# # uploaded_file = st.file_uploader("Upload your lung CT image", type=["jpg", "jpeg", "png"])
# # final_file = uploaded_file if uploaded_file else selected_sample

# # # --- Prediction ---
# # if final_file and model:
# #     st.image(final_file, caption='Selected Image', use_column_width=True)

# #     img_array = load_and_preprocess_image(final_file)
# #     predictions = model.predict(img_array)
# #     predicted_class = int(np.argmax(predictions[0]))
# #     predicted_label = class_labels[predicted_class]
# #     confidence = float(np.max(predictions[0]))

# #     # Styled Output
# #     st.markdown(f"""
# #         <div class="prediction-box">
# #             <div class="label">{predicted_label}</div>
# #             <div class="confidence">Confidence: {confidence:.2f}</div>
# #         </div>
# #     """, unsafe_allow_html=True)

# #     st.progress(confidence)

# #     # Class probabilities
# #     st.subheader("🔍 Class Probabilities")
# #     for label, prob in zip(class_labels, predictions[0]):
# #         st.write(f"**{label}**: {prob:.4f}")


# import streamlit as st
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from PIL import Image
# import os

# # --- Page Config ---
# st.set_page_config(page_title="🫁 Lung Cancer Detector", layout="centered")

# # --- Custom CSS for styling ---
# st.markdown("""
#     <style>
#     html, body, [class*="css"]  {
#         font-family: 'Segoe UI', sans-serif;
#         background-color: #fafcff;
#     }
#     .title {
#         text-align: center;
#         font-size: 3rem;
#         font-weight: 800;
#         background: -webkit-linear-gradient(45deg, #0ea5e9, #9333ea);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         margin-bottom: 0.5rem;
#     }
#     .sub {
#         text-align: center;
#         color: #6b7280;
#         font-size: 1.1rem;
#         margin-bottom: 2rem;
#     }
#     .image-grid {
#         display: flex;
#         flex-wrap: wrap;
#         justify-content: center;
#         gap: 15px;
#         margin-bottom: 1rem;
#     }
#     .image-grid img {
#         border-radius: 12px;
#         width: 150px;
#         height: 150px;
#         object-fit: cover;
#         cursor: pointer;
#         transition: all 0.3s ease;
#         box-shadow: 0 4px 12px rgba(0,0,0,0.1);
#     }
#     .image-grid img:hover {
#         transform: scale(1.05);
#         box-shadow: 0 6px 20px rgba(0,0,0,0.15);
#     }
#     .result-box {
#         padding: 20px;
#         border-radius: 14px;
#         background-color: #f1f5f9;
#         text-align: center;
#         margin-top: 1.5rem;
#     }
#     .label {
#         font-size: 1.8rem;
#         font-weight: bold;
#         color: #16a34a;
#     }
#     .confidence {
#         font-size: 1.1rem;
#         color: #334155;
#         margin-top: 5px;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # --- Title ---
# st.markdown('<div class="title">🫁 Lung Cancer Detector</div>', unsafe_allow_html=True)
# st.markdown('<div class="sub">Upload a lung CT image or try one of our samples below to predict cancer type</div>', unsafe_allow_html=True)

# # --- Class Labels ---
# class_labels = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']

# # --- Load model ---
# @st.cache_resource
# def load_cancer_model():
#     try:
#         model = load_model("lung5.keras")
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None

# model = load_cancer_model()

# # --- Preprocessing ---
# def load_and_preprocess_image(img_file, target_size=(224, 224)):
#     img = Image.open(img_file).convert("RGB")
#     img = img.resize(target_size)
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0
#     return img_array

# # --- Sample Images ---
# selected_sample = None
# sample_dir = "./samples"
# if os.path.exists(sample_dir):
#     sample_files = sorted([f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

#     st.markdown("### 🧪 Try with a Sample Image")
#     st.markdown('<div class="image-grid">', unsafe_allow_html=True)
#     cols = st.columns(min(len(sample_files), 5))
#     for i, filename in enumerate(sample_files):
#         file_path = os.path.join(sample_dir, filename)
#         with cols[i % len(cols)]:
#             if st.button(f"Use Sample {i+1}", key=f"sample{i}"):
#                 selected_sample = file_path
#         st.image(file_path, use_column_width=True)
#     st.markdown('</div>', unsafe_allow_html=True)

# # --- Drag and Drop Image ---
# st.markdown("### 📤 Upload Image (Drag & Drop or Browse)")
# uploaded_file = st.file_uploader("Drop image here or click to upload", type=["jpg", "jpeg", "png"])

# # --- Final file to predict ---
# final_file = uploaded_file if uploaded_file else selected_sample

# # --- Prediction Section ---
# if final_file and model:
#     st.image(final_file, caption="Selected Image", use_column_width=True)

#     img_array = load_and_preprocess_image(final_file)
#     predictions = model.predict(img_array)
#     predicted_class = int(np.argmax(predictions[0]))
#     predicted_label = class_labels[predicted_class]
#     confidence = float(np.max(predictions[0]))

#     # Styled result box
#     st.markdown(f"""
#         <div class="result-box">
#             <div class="label">{predicted_label}</div>
#             <div class="confidence">Confidence: {confidence:.2%}</div>
#         </div>
#     """, unsafe_allow_html=True)

#     st.progress(confidence)

#     # Probabilities
#     st.markdown("### 🔬 Prediction Probabilities")
#     for label, prob in zip(class_labels, predictions[0]):
#         st.write(f"**{label.capitalize()}**: {prob:.4f}")


import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Streamlit config
st.set_page_config(page_title="🫁 Lung Cancer Detector", layout="centered")

# --- CSS Styling ---
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f9fafb;
    }
    .title {
        font-size: 2.4rem;
        text-align: center;
        font-weight: bold;
        background: linear-gradient(to right, #2563eb, #9333ea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        background-color: #f1f5f9;
        border-radius: 16px;
        text-align: center;
        margin-top: 2rem;
    }
    .label {
        font-size: 1.8rem;
        font-weight: 600;
        color: #0f766e;
    }
    .confidence {
        font-size: 1.1rem;
        color: #334155;
        margin-top: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# --- UI Title ---
st.markdown('<div class="title">🫁 Lung Cancer Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a lung CT image and choose model to detect cancer type</div>', unsafe_allow_html=True)

# --- Model Selection ---
model_option = st.selectbox("🔍 Select Model", [
    "Advanced: 4-Class (Adeno, Large, Squamous, Normal)",
    "Basic: 3-Class (Benign, Malignant, Normal)"
])

# --- Load Model ---
@st.cache_resource
def load_advanced_model():
    return load_model("lung5.keras")

@st.cache_resource
def load_basic_model():
    return load_model("model.weights.h5")

# --- Image Preprocessing ---
def preprocess_image(img_file):
    img = Image.open(img_file).convert("RGB").resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# --- File Upload ---
uploaded_file = st.file_uploader("📤 Upload CT Image", type=["jpg", "jpeg", "png"])

# --- Setup Based on Model ---
if model_option.startswith("Advanced"):
    model = load_advanced_model()
    class_labels = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']
else:
    model = load_basic_model()
    class_labels = ['Benign cases', 'Malignant cases', 'Normal cases']

# --- Run Prediction ---
if uploaded_file and model:
    img_array = preprocess_image(uploaded_file)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_index]
    confidence = float(np.max(predictions[0]))

    # --- Output ---
    st.markdown(f"""
        <div class="result-box">
            <div class="label">{predicted_label}</div>
            <div class="confidence">Confidence: {confidence:.2%}</div>
        </div>
    """, unsafe_allow_html=True)

    st.progress(confidence)

    # --- Class Probabilities ---
    st.markdown("### 📊 Class Probabilities")
    for label, prob in zip(class_labels, predictions[0]):
        st.write(f"**{label}**: `{prob:.4f}`")
