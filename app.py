import streamlit as st

# Set up page configuration
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Add a header with logo (optional)
# st.image("resources/logo.png", width=150)  # Replace with your project's logo
st.title("Breast Cancer Prediction")
st.subheader("Using Vision Transformer (ViT)")

# Sidebar for additional navigation or information
st.sidebar.header("About")
st.sidebar.info(
    "This application allows users to upload medical images to predict breast cancer. "
    "Currently, it is a frontend demo and does not connect to the backend."
)

# Main form for file upload
with st.form("prediction_form"):
    st.markdown("### Upload Image")
    uploaded_file = st.file_uploader("Choose a mammogram or ultrasound image", type=["png", "jpg", "jpeg"])

    # Add any additional inputs if needed
    st.markdown("### Patient Information")
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    gender = st.selectbox("Gender", options=["Female", "Male"])
    symptoms = st.text_area("Describe any symptoms (optional)")

    # Submit button
    submit_button = st.form_submit_button("Submit")

# Handle form submission
if submit_button:
    if uploaded_file:
        st.success("File uploaded successfully!")
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write("**Age**:", age)
        st.write("**Gender**:", gender)
        st.write("**Symptoms**:", symptoms or "No symptoms provided")
        st.info("The backend is not connected. Results will be displayed here once connected.")
    else:
        st.error("Please upload an image before submitting.")

# Footer
st.markdown("---")
st.write("© 2024 Breast Cancer Prediction App")
