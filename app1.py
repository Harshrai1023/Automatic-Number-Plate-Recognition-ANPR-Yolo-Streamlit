import streamlit as st
from detection.main import recognize_number_plate_and_validate

# Page setup
st.set_page_config(page_title="Number Plate Detection", layout="wide", page_icon="🚘")

# Custom styles
st.markdown("""
    <style>
        /* Main background color */
        [data-testid="stAppViewContainer"] {
            background-color: #1e1e2f;
        }

        .center-title {
            text-align: center;
            font-size: 45px;
            font-weight: 800;
            color: #4fa3f7;
            margin-bottom: 5px;
        }
        .subtext {
            text-align: center;
            font-size: 18px;
            color: #aaa;
            margin-bottom: 30px;
        }
        .upload-box {
            border: 2px dashed #4fa3f7;
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            background-color: #111;
            margin: auto;
            width: 60%;
        }
        .upload-text {
            font-size: 18px;
            color: #ccc;
            margin-top: 15px;
        }
        .output-box {
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            padding: 15px;
            border-radius: 15px;
            text-align: center;
            font-size: 26px;
            font-weight: bold;
            color: #1a1a1a;
            box-shadow: 0px 0px 20px rgba(0,0,0,0.2);
        }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown("<div class='center-title'>🚘 Vehicle Number Plate Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Upload an image and let the magic happen ✨</div>", unsafe_allow_html=True)

# Custom Upload Module
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📂 Drag and drop or browse your vehicle image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
st.markdown("<div class='upload-text'>Supported formats: JPG, PNG, WEBP · Max 200MB</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Handling uploaded image
if uploaded_file is None:
    st.markdown("<p class='subtext'>📷 Waiting for image... Your vehicle’s number plate is one click away!</p>", unsafe_allow_html=True)
else:
    with st.spinner("🧠 Processing... detecting number plate..."):
        image, output_image, output_text = recognize_number_plate_and_validate(uploaded_file)

    st.balloons()

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, use_column_width=True)
        st.markdown("<div class='subtext'>📥 Uploaded Image</div>", unsafe_allow_html=True)

    with col2:
        st.image(output_image, use_column_width=True)
        st.markdown(f"<div class='output-box'>🔍 Detected Plate: {output_text}</div>", unsafe_allow_html=True)