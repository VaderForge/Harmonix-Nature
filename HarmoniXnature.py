import streamlit as st
import requests
import librosa
import numpy as np
import soundfile as sf
import os
from io import BytesIO

# Set up Streamlit UI
st.title("AI Nature Sound Generator")
st.write("Generate custom nature sounds based on your choice!")

# API Key and Base URL for Freesound
API_KEY = "Freesound-API"
BASE_URL = "https://freesound.org/apiv2/search/text/"

# Utility functions for processing audio
def normalize_audio(y):
    return y / np.max(np.abs(y))

def noise_reduction(y):
    return librosa.effects.preemphasis(y)

def add_echo(y, sr, delay=0.2, decay=0.5):
    delay_samples = int(sr * delay)
    echo = np.zeros(len(y) + delay_samples)
    echo[delay_samples:] = y * decay
    return y + echo[:len(y)]

def change_volume(y, factor):
    return y * factor

def low_pass_filter(y, cutoff_freq, sr):
    from scipy.signal import butter, lfilter

    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    return lowpass_filter(y, cutoff_freq, sr)

# Fetch and process audio based on a prompt
def fetch_and_process_audio(prompt):
    params = {
        "query": prompt,
        "fields": "id,name,previews",
        "filter": "type:wav",
        "sort": "score",
        "page_size": 3,
        "token": API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    response_data = response.json()
    
    audio_data = []
    sample_rates = []

    for sound in response_data['results']:
        preview_url = sound['previews']['preview-hq-mp3']
        audio_response = requests.get(preview_url)
        audio_path = sound['name'] + ".mp3"
        with open(audio_path, 'wb') as f:
            f.write(audio_response.content)

        y, sr = librosa.load(audio_path, sr=None)
        audio_data.append(normalize_audio(y))
        sample_rates.append(sr)
        os.remove(audio_path)

    return audio_data, sample_rates

# Combine and add effects
def process_and_combine(audio_data, sample_rates, volume, echo_decay, lp_cutoff):
    processed_audio = []

    for y, sr in zip(audio_data, sample_rates):
        y = noise_reduction(y)
        y = add_echo(y, sr, decay=echo_decay)
        y = change_volume(y, volume)
        y = low_pass_filter(y, lp_cutoff, sr)
        processed_audio.append(y)

    max_length = max(len(y) for y in processed_audio)
    final_audio = np.zeros(max_length)

    for y in processed_audio:
        y_padded = np.pad(y, (0, max_length - len(y)), 'constant')
        final_audio += y_padded

    return normalize_audio(final_audio), sample_rates[0]

# User inputs
prompt = st.text_input("Enter a nature sound prompt (e.g., 'rain forest', 'ocean waves', 'bird chirping')", "")
volume = st.slider("Volume Factor", 0.5, 2.0, 1.2)
echo_decay = st.slider("Echo Decay", 0.1, 1.0, 0.5)
lp_cutoff = st.slider("Low-Pass Filter Cutoff Frequency", 500, 5000, 3000)

# Generate button
if st.button("Generate Sound"):
    if prompt:
        with st.spinner("Fetching and processing audio..."):
            audio_data, sample_rates = fetch_and_process_audio(prompt)
            if audio_data:
                final_audio, sr = process_and_combine(audio_data, sample_rates, volume, echo_decay, lp_cutoff)
                
                output_path = "generated_nature_sound.wav"
                sf.write(output_path, final_audio, sr)

                # Show audio player
                st.audio(output_path)
                st.success(f"Generated audio saved as: {output_path}")
            else:
                st.error("No audio found for the prompt.")
    else:
        st.warning("Please enter a prompt to generate sound.")
