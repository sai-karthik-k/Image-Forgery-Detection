import streamlit as st
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from tensorflow.keras.models import load_model
import os
import time
import smtplib
from email.mime.text import MIMEText
import gdown  # Required for downloading model

# Download model from Google Drive if not present
model_path = './model.keras'
if not os.path.exists(model_path):
    st.info("Downloading model from Google Drive...")
    gdown.download('https://drive.google.com/uc?id=1frIR6crojKV86I_UtjIZmFinCCb2y6s_', model_path, quiet=False)

# Custom CSS
st.markdown("""
    <style>
    .main {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
        background-color: #1a1a1a;
        color: #ecf0f1;
    }
    .title {
        text-align: center;
        color: #ecf0f1;
    }
    .upload-container {
        display: flex;
        justify-content: center;
        margin: 2rem 0;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #2c3e50;
        text-align: center;
        color: #ecf0f1;
        margin-bottom: 1.5rem;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: #ecf0f1;
        padding: 1rem;
    }
    .sidebar .sidebar-content h3 {
        color: #ecf0f1;
    }
    .sidebar .stButton>button {
        background-color: #34495e;
        color: #ecf0f1;
        border: none;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        width: 100%;
    }
    .sidebar .stButton>button:hover {
        background-color: #46637f;
    }
    .confidence-explanation {
        font-style: italic;
        color: #bdc3c7;
        margin: 2rem 0;
    }
    .stFileUploader label {
        color: #ecf0f1;
        font-size: 1.1rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Session State Init
if 'page' not in st.session_state:
    st.session_state.page = 'About'

def get_opened_image(image):
    return Image.open(image).convert('RGB')

@st.cache_resource
def _loadmodel():
    try:
        return load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def difference(org):
    resaved_name = 'temp.jpg'
    try:
        org.save(resaved_name, 'JPEG', quality=92)
        resaved = Image.open(resaved_name)
        diff = ImageChops.difference(org, resaved)
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        diff = ImageEnhance.Brightness(diff).enhance(scale)
        return diff
    finally:
        if os.path.exists(resaved_name):
            os.remove(resaved_name)

def pred(img):
    model = _loadmodel()
    if model is None:
        return "Error", 0.0
    diff = np.array(difference(img).resize((128, 128))).flatten() / 255.0
    diff = diff.reshape(-1, 128, 128, 3)
    pred = model.predict(diff, verbose=0)[0]
    confidence = max(pred[0], pred[1]) * 100
    result = "Not Forged" if pred[0] > pred[1] else "Forged"
    return result, confidence

def send_feedback(name, email, message):
    sender = "everything9618@gmail.com"
    receiver = "krishnamsaikarthik@gmail.com"
    password = "tqak axsq gxag flfc"

    msg = MIMEText(f"Name: {name}\nEmail: {email}\nMessage: {message}")
    msg['Subject'] = "Feedback from Image Forgery Detection System"
    msg['From'] = sender
    msg['To'] = receiver

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        st.success("Feedback sent successfully!")
    except Exception as e:
        st.error(f"Failed to send feedback: {str(e)}")

# Sidebar Navigation
with st.sidebar:
    if st.button("About"):
        st.session_state.page = 'About'
    if st.button("Analysis"):
        st.session_state.page = 'Analysis'
    if st.button("Contact or Feedback"):
        st.session_state.page = 'Contact'

# About Page
if st.session_state.page == 'About':
    st.title("Image Forgery Detection System")
    st.markdown("""
        ### 🧠 About Image Forgery Detection System
        
        Welcome to the Image Forgery Detection System — a smart application built using deep learning and error-level analysis (ELA) to identify whether a JPG image has been manipulated or forged.

        This project uses the Xception model architecture with transfer learning and ELA preprocessing to highlight inconsistencies introduced by tampering. By analyzing pixel-level differences between the original and a resaved version of the image, our model predicts whether the image is genuine or forged with high confidence.

        #### 🔑 Key Features
        - 📤 Upload JPG images for analysis
        - 🔍 Automatically generates an ELA (Error Level Analysis) difference map
        - 🧠 Makes accurate predictions using a trained Xception-based CNN
        - 📊 Displays prediction confidence scores
        - 📥 Downloadable difference map for offline review

        #### 🛠 Tech Stack
        - 🖼 PIL for ELA-based image preprocessing
        - 🤖 TensorFlow/Keras for model training and prediction
        - 🌐 Streamlit for an interactive and intuitive user interface

        This tool is built with a focus on detecting image manipulations by combining image forensics with AI.

        For any questions or feedback, feel free to reach out or contribute!
    """)

# Analysis Page
elif st.session_state.page == 'Analysis':
    st.title("Image Forgery Detection System")
    st.markdown("Upload one or more JPG images to detect forgery using AI.")

    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image_files = st.file_uploader("Upload JPG Images", type='jpg', accept_multiple_files=True)

    if image_files:
        if st.button('Analyze Images'):
            for idx, image_file in enumerate(image_files):
                progress_bar = st.progress(0, text=f"Analyzing image {idx + 1}/{len(image_files)}")
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1, text=f"Analyzing image {idx + 1}/{len(image_files)}")
                try:
                    image = get_opened_image(image_file)
                    st.subheader(f"Image {idx + 1}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original")
                        st.image(image, use_container_width=True)
                    with col2:
                        st.subheader("ELA Difference Map")
                        diff_image = difference(image)
                        st.image(diff_image, use_container_width=True)
                    prediction, confidence = pred(image)
                    st.markdown(f"""
                        <div class="prediction-box">
                            <h3>Prediction: {prediction}</h3>
                            <p>Confidence Level: {confidence:.2f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    diff_image.save(f'diff_{idx}.jpg')
                    with open(f'diff_{idx}.jpg', 'rb') as file:
                        st.download_button(f"Download Map {idx+1}", file, file_name=f'diff_map_{idx+1}.jpg')
                except Exception as e:
                    st.error(f"Error analyzing image {idx+1}: {str(e)}")
            st.markdown("""
                <div class="confidence-explanation">
                    _The model gives confidence levels based on subtle pixel-level changes between the original and a re-saved version of each image. This helps identify potential forgery by analyzing differences that are often invisible to the human eye._
                </div>
            """, unsafe_allow_html=True)

# Contact Page
elif st.session_state.page == 'Contact':
    st.title("Contact or Feedback")
    st.markdown("We value your feedback! Please fill out the form below.")

    with st.form(key='contact_form'):
        name = st.text_input("Name")
        email = st.text_input("Email")
        message = st.text_area("Message")
        submit_button = st.form_submit_button(label="Submit")

    if submit_button and name and email and message:
        send_feedback(name, email, message)

# Footer
st.markdown("""
    ---
    <div style='text-align: center; color: #ecf0f1;'>
        Developed with ❤️ using Streamlit | Powered by TensorFlow
    </div>
""", unsafe_allow_html=True)