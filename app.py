import streamlit as st
from detection.main import recognize_number_plate_and_validate

# Set the page layout to wide
st.set_page_config(layout="wide")

# Center the title using HTML and CSS
st.markdown("<h1 style='text-align: center;'>Vehicle Number Plate Detection</h1>", unsafe_allow_html=True)

# Sidebar for image upload
st.sidebar.title("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image of a vehicle...", type=["jpg", "jpeg", "png","webp"])

# Main page content
if uploaded_file is None:
    st.markdown("<p style='text-align: center;'>Please upload an image of a vehicle to detect the number plate.</p>", unsafe_allow_html=True)
else:
    # Show a spinner while processing the image
    with st.spinner('Processing image...'):
        image , output_image, output_text = recognize_number_plate_and_validate(uploaded_file)
    
    # Display the uploaded image and output_image image side by side
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.image(output_image, caption=output_text, use_column_width=True)
