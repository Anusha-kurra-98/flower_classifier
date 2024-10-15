import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import streamlit as st

st.header(":black[Flower Classification Model]")
flower_names = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']
# Define your model
img_size = 180

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal", input_shape=(img_size, img_size, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

model = tf.keras.Sequential([
    data_augmentation,
    layers.Rescaling(1. / 255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(5)  # Adjust the number of output units based on your classes
])

# Load the model checkpoint
checkpoint_dir = './Flower_Recog_Model'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(model=model)

# Restore the model from the latest checkpoint
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()


# Function to classify images
def classify_images(image_path):
    input_image = load_img(image_path, target_size=(img_size, img_size))
    input_image_array = img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)  # Create batch axis

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = f'The Image belongs to {flower_names[np.argmax(result)]} with a score of {np.max(result) * 100:.2f}%'
    return outcome


# Streamlit file uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Classify the uploaded image
    outcome = classify_images("temp_image.jpg")
    st.write(outcome)
