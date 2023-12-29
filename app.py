import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# loading the model
model_path = 'model.h5'
model = load_model(model_path)

# fixed image size for model
input_height, input_width = 224, 224
    
# threshold for prediction
threshold = 0.5

# Function for image preprocessing
def preprocess_image(image):
    # resizing image
    resized_image = tf.image.resize(image, (input_height, input_width))
    # setting values inbetween 0 to 255. {normalisation}
    normalized_image = resized_image / 255.0
    return normalized_image

# Streamlit App
st.set_page_config(
    page_title="Crack Detection App",
    page_icon="🕵️‍♂️",
    layout="centered"
)

st.title('Crack Detection with Deep Learning')

uploaded_file = st.file_uploader("Choose an image...", type="jpg", key="file")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # changing imgage sizes as per model
    image = np.array(image)
    image = preprocess_image(image)

    # predicting based on pretrained model
    try:
        predictions = model.predict(np.expand_dims(image, axis=0))
        # 0: prob of no crack present   1:prob of crack present
        # predictions results
        probability_no_crack = predictions[0][0]
        probability_crack = predictions[0][1]

        # Choose the higher prediction value
        result_text = "Crack is Not Present" if probability_no_crack > probability_crack else "Crack is Present"

        # Display result
        st.subheader('Prediction Result:')
        st.write(f"Probability of No Crack: {probability_no_crack:.2%}")
        st.write(f"Probability of Crack: {probability_crack:.2%}")
        st.write(result_text)

        # User contact info
        if "Crack is Present" in result_text:
            st.subheader('Contact Information:')
            st.write("We have detected a crack in the image and we need your contact details for further process to inform the local authorities.")
            user_name = st.text_input("Your Name:")
            user_locality = st.text_input("Your Locality:")
            user_email = st.text_input("Your Email Address:")
            user_mobile = st.text_input("Your Mobile Number:")
            crack_location = st.text_input("Location of the Crack:")

            # checks if user has filled all info bars
            if user_name and user_locality and user_email and user_mobile and crack_location:
                st.success("Thank you! Your contact information has been recorded. Our team will contact you soon.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Try clicking the image again with better lighting or try reuploading the image.")
