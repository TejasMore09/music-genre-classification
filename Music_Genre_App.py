import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import os
from tensorflow.image import resize

# Function to load the model
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model("Trained_model.h5")
    return model

# Function to load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)

    # Define the duration of each chunk and overlap
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram_db = np.expand_dims(mel_spectrogram_db, axis=-1)
        mel_spectrogram_resized = tf.image.resize(mel_spectrogram_db, target_shape).numpy()
        data.append(mel_spectrogram_resized)

    return np.array(data)

# TensorFlow Model Prediction
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_elements = unique_elements[counts == np.max(counts)]
    return max_elements[0]

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Home Page
if app_mode == "Home":
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #181646;
            color: white;
        }
        h2, h3 {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(""" ## Welcome to the,\n
    ## Music Genre Classification System! 🎶🎧""")
    st.image("music_genre_home.png", use_column_width=True)
    st.markdown("""
**Our goal is to help in identifying music genres from audio tracks efficiently. Upload an audio file, and our system will analyze it to detect its genre. Discover the power of AI in music analysis!**

### How It Works
1. **Upload Audio:** Go to the **Genre Classification** page and upload an audio file.
2. **Analysis:** Our system will process the audio using advanced algorithms to classify it into one of the predefined genres.
3. **Results:** View the predicted genre along with related information.

### Why Choose Us?
- **Accuracy:** Our system leverages state-of-the-art deep learning models for accurate genre prediction.
- **User-Friendly:** Simple and intuitive interface for a smooth user experience.
- **Fast and Efficient:** Get results quickly, enabling faster music categorization and exploration.

### Get Started
Click on the **Genre Classification** page in the sidebar to upload an audio file and explore the magic of our Music Genre Classification System!

### About Us
Learn more about the project, our team, and our mission on the **About** page.
""")

# About Project Page
elif app_mode == "About Project":
    st.markdown("""
### About Project
Music. Experts have been trying for a long time to understand sound and what differentiates one song from another. How to visualize sound. What makes a tone different from another.

This data hopefully can give the opportunity to do just that.

### About Dataset
#### Content
1. **genres original** - A collection of 10 genres with 100 audio files each, all having a length of 30 seconds (the famous GTZAN dataset, the MNIST of sounds)
2. **List of Genres** - blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
3. **images original** - A visual representation for each audio file. Audio files were converted to Mel Spectrograms to make classification with CNN possible.
4. **2 CSV files** - Containing features of the audio files. One for full-length songs and another for chunked versions.
""")

# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

    if uploaded_file is not None:
        temp_dir = "Test_Music"
        os.makedirs(temp_dir, exist_ok=True)
        filepath = os.path.join(temp_dir, uploaded_file.name)

        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Play Audio"):
            st.audio(uploaded_file)

        if st.button("Predict"):
            with st.spinner("Please Wait..."):
                X_test = load_and_preprocess_data(filepath)
                result_index = model_prediction(X_test)
                st.balloons()
                labels = ['blues', 'classical', 'country', 'disco', 'hiphop',
                          'jazz', 'metal', 'pop', 'reggae', 'rock']
                predicted_genre = labels[result_index]
                st.markdown(f"**:blue[Model Prediction:] It's a :red[{predicted_genre}] music**")
