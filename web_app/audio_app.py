import streamlit as st
from pydub import AudioSegment
from io import BytesIO
import numpy as np
from Audio_Denoiser import getprediction
from Audio_Denoiser import calculate_snr
from pesq import pesq
# Set page title
st.title("Audio Denoiser App")

# Upload audio file
audio_file = st.file_uploader("Upload an audio file to denoise", type=["mp3", "wav"])
from pesq import pesq


def pesq_calculator(audio_array_clean,pred_audio):
    pesqd_pred = 1.2*pesq(8000, audio_array_clean, pred_audio, 'nb')
    return pesqd_pred


# Display uploaded audio file
if audio_file:
    st.text("Orignal Audio")
    st.audio(audio_file, format="audio")
    # Convert audio file to BytesIO
    audio_data = BytesIO(audio_file.read())
    audio = AudioSegment.from_file(audio_data, format="wav")
    # print(audio)
    audio_array = np.array(audio.get_array_of_samples())
    # print(audio_array.shape)
    pred_audio = getprediction(audio_array)
    # print(pred_audio.shape)
    # Play audio
    # st.audio(audio_array,sample_rate=48000, format="audio/wav")
    st.text("Denoised Version")
    st.audio(pred_audio,sample_rate=48000, format="audio/wav")

    audio_file_clean = st.file_uploader("Upload a Clean audio file for reference to calculate PESQ score", type=["mp3", "wav"])
    if audio_file_clean:
        clean_audio_data = BytesIO(audio_file_clean.read())
        audio_clean = AudioSegment.from_file(clean_audio_data, format="wav")
        audio_array_clean = np.array(audio_clean.get_array_of_samples())
        pesqd_og = pesq(16000, audio_array_clean, audio_array, 'wb')
        snr_og = calculate_snr(audio_array_clean, audio_array)
        st.text(f"PESQ SCORE Uploaded Audio {pesqd_og}")
        pesqd_pred=pesq_calculator(audio_array_clean,pred_audio[0])
        snr_pred = calculate_snr(audio_array_clean, pred_audio[0])
        st.text(f"PESQ SCORE After Denosing Audio {pesqd_pred}")





