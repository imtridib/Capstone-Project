import streamlit as st
import torch
import time
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

from googletrans import Translator

# from deep_translator import GoogleTranslator


# Load model and processor
model_name = "Salesforce/blip-vqa-base"
model_path = "best_vqaEnglish_models.pth"  # Path to fine-tuned model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = BlipProcessor.from_pretrained(model_name)
model = BlipForQuestionAnswering.from_pretrained(model_name)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Streamlit UI Design
st.set_page_config(page_title="Multilingual VQA - Bangladesh", layout="wide")

# Custom Styling
st.markdown(
    """
    <style>
    body {
        background-color: #E3F2FD;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #1E3A8A;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        color: #374151;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        margin-top: 50px;
        color: #4B5563;
    }
    .upload-box {
        background-color: #F9FAFB;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #D1D5DB;
        text-align: center;
    }
    .container {
        background-color: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    .btn {
        width: 100%;
        padding: 10px;
        border-radius: 8px;
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Subtitle
st.markdown('<p class="title">A Multilingual Visual Question Answering for Cultural Heritage Sites of Bangladesh</p>',
            unsafe_allow_html=True)
st.markdown('<p class="subtitle">🔍 Explore the Rich Heritage of Bangladesh with AI-Powered Image Understanding</p>',
            unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([3, 5])

with col1:  # Left Side (File Upload)
    st.markdown('<p style="font-size:18px; font-weight:bold; text-align:center;">📤 Upload Your Image</p>',
                unsafe_allow_html=True)
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)

    upload_progress = st.empty()  # Placeholder for upload progress
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded_file:
        progress_bar = upload_progress.progress(0)
        for percent_complete in range(0, 101, 20):
            time.sleep(0.2)  # Simulating upload delay
            progress_bar.progress(percent_complete)

        upload_progress.empty()  # Remove progress bar after upload is complete
        st.success("✅ Image uploaded successfully!")

    st.markdown('</div>', unsafe_allow_html=True)

with col2:  # Right Side (Image Display & Question Input)
    with st.container():  # Adds a nice background and spacing
        st.markdown('<div class="container">', unsafe_allow_html=True)

        if uploaded_file:
            image_loading_progress = st.progress(0)
            status_text = st.empty()

            for percent_complete in range(0, 101, 20):
                time.sleep(0.1)  # Simulating image loading
                image_loading_progress.progress(percent_complete)
                status_text.text(f"Loading image... {percent_complete}%")

            # Display image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

            image_loading_progress.empty()
            status_text.text("✅ Image loaded successfully!")

        question = st.text_area("❓ Enter Your Question (Bangla & English Supported):", height=100)

        translator = Translator()
        # translator = GoogleTranslator()        
        if st.button("🔍 Ask", help="Click to generate an answer based on the uploaded image", key="ask_btn"):
            if uploaded_file is None:
                st.warning("⚠️ Please upload an image!")
            elif not question.strip():
                st.warning("⚠️ Please enter a question!")
            else:
                # Display progress bar for model inference
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Simulate step-by-step progress during model inference
                for percent_complete in range(0, 101, 10):
                    time.sleep(0.3)  # Simulating processing delay
                    progress_bar.progress(percent_complete)
                    status_text.text(f"Processing... {percent_complete}%")

                # Detect language
                detected_lang = translator.detect(question).lang

                if detected_lang == 'en':
                    inputs = processor(images=image, text=question, return_tensors="pt").to(device)

                    with torch.no_grad():
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_length=200,  # Increase length of generated text
                                temperature=0.7,  # Adjust randomness (higher = more creative)
                            )

                    # Complete progress bar
                    progress_bar.progress(100)
                    status_text.text("✅ Processing Complete!")

                    answer = processor.decode(outputs[0], skip_special_tokens=True)
                    st.success(f"💡 Answer: **{answer}**")
                else:
                    bangla_question = translator.translate(question, src='bn', dest='en')
                    inputs = processor(images=image, text=bangla_question.text, return_tensors="pt").to(device)

                    with torch.no_grad():
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_length=200,  # Increase length of generated text
                                temperature=0.7,  # Adjust randomness (higher = more creative)
                            )

                    # Complete progress bar
                    progress_bar.progress(100)
                    status_text.text("✅ Processing Complete!")

                    answer = processor.decode(outputs[0], skip_special_tokens=True)
                    translated_text = translator.translate(answer, src='en', dest='bn')
                    st.success(f"💡 Answer: **{translated_text.text}**")

        st.markdown('</div>', unsafe_allow_html=True)  # Closing the container

# Footer
st.markdown(
    # '<p class="footer">© 2025 Multilingual VQA System | Developed by Dr. Mohammad Rifat Ahmmad Rashid, Associate Professor, EWU Bangladesh</p>',
    '<p class="footer">© 2025 Multilingual VQA System | Developed by Dr. Mohammad Rifat Ahmmad Rashid, Associate Professor, EWU Bangladesh And His Capstone team.</p>',
    unsafe_allow_html=True)
