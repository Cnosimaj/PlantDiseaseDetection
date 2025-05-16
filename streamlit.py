import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import streamlit as st
from st_on_hover_tabs import on_hover_tabs

st.set_page_config(layout="wide")

# Importing stylesheet
st.markdown('<style>' + open('./style.css').read() +
            '</style>', unsafe_allow_html=True)

# Tensorflow Model Prediction


def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = Image.open(test_image)
    image = image.resize((224, 224))
    input_arr = np.array(image).astype('float32') / 255.0
    input_arr = np.expand_dims(input_arr, axis=0)
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index


# Sidebar
with st.sidebar:
    tabs = on_hover_tabs(tabName=['Home', 'About', 'Disease Recognition'],
                         iconName=['home', 'information-outline', 'coronavirus'], default_choice=0)

app_mode = tabs

if tabs == 'Home':
    pass

elif tabs == 'About':
    pass

elif tabs == 'Disease Recognition':
    pass

# Home Page
if (app_mode == "Home"):
    st.header("Plant Disease Recognition Model")
    image_path = "Tomato Plant Disease.png"
    st.image(image_path, use_container_width=True)
    st.markdown("""# 🌿 Welcome to the Plant Disease Recognition App

Your best resource for Plant Disease Recognition. Our model quickly and accurately detect diseases in plants using just an image.

---

## How It Works

1. **Capture or Upload an Image**: Take a clear picture of your plant's leaf or upload an existing image.
2. **Let the Model Analyze**: Our AI-powered model analyzes the image and identifies potential diseases.
3. **Get Instant Results**: Receive the name of the disease, a short description, and treatment suggestions.

---

##  Why Choose My Model?

-  **High Accuracy**: Trained on thousands of labeled images across various plant species.
-  **Fast Inference**: Get results in seconds with minimal resource usage.
-  **AI-Powered Insights**: Uses advanced deep learning techniques for high precision.
-  **Easy to Use**: Built with simplicity in mind.

---

##  Get Started

1. Click on **"Upload Image"** the Disease Recognition page.
2. Wait a few seconds while the model processes your image.
3. View what class the diseased leaf belongs to.

---

##  About Us



👉🏿 Learn more on my About page. 👈🏿

---""")

# About Page
elif (app_mode == "About"):
    st.header("PlantVillage Dataset")
    st.subheader("Dataset of diseased Plant Leafs")
    st.markdown(""" Human society needs to increase food production by an estimated 70% by 2050 to feed an expected population size that is predicted to be over 9 billion people. Currently, infectious diseases reduce the potential yield by an average of 40% with many farmers in the developing world experiencing yield losses as high as 100%. The widespread distribution of smartphones among crop growers around the world with an expected 5 billion smartphones by 2020 offers the potential of turning the smartphone into a valuable tool for diverse communities growing food. One potential application is the development of mobile disease diagnostics through machine learning and crowdsourcing. Here we announce the release of over 50,000 expertly curated images on healthy and infected leaves of crops plants through the existing online platform PlantVillage. We describe both the data and the platform. These data are the beginning of an on-going, crowdsourcing effort to enable computer vision approaches to help solve the problem of yield losses in crop plants due to infectious diseases.
    #### Content
    1. Train (43456 images)
    2. Valid (10849 images)
    

""")
    st.subheader("Plant Disease Recognition Model")

# Disease Recognition Page
elif (app_mode == "Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    # Define Class
    class_names = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
        'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
        'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

    if (st.button("Show Image")):
        st.image(test_image, use_container_width=True)
    # Prediction Button
    if (st.button("Predict")):
        st.write("Our Prediction")

        result_index = model_prediction(test_image)
        if result_index is not None and 0 <= result_index < len(class_names):
            st.success("Model is Predicting it's a {}".format(
                class_names[result_index]))
