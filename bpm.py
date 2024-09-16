import streamlit as st
import librosa
import numpy as np
import time
import soundfile as sf
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@st.cache_data
def load_and_analyze_audio(file):
    y, sr = librosa.load(file, sr=None)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return y, sr, tempo, beat_frames, chroma_stft, spectral_centroid


def downsample_audio(y, sr, target_length=100000):
    if len(y) > target_length:
        y_downsampled = librosa.resample(
            y, orig_sr=sr, target_sr=int(sr * target_length / len(y))
        )
        return y_downsampled
    return y


def create_visualizations(y, sr, tempo, beat_frames, chroma_stft, spectral_centroid):
    y_vis = downsample_audio(y, sr)

    tempo_scalar = tempo.item() if isinstance(tempo, np.ndarray) else tempo

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            f"Waveform and Beats (Tempo: {tempo_scalar:.2f} BPM)",
            "Chromagram",
            "Spectral Centroid",
        ),
    )

    times = np.linspace(0, len(y) / sr, len(y_vis))
    fig.add_trace(go.Scatter(x=times, y=y_vis, name="Waveform"), row=1, col=1)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    fig.add_trace(
        go.Scatter(
            x=beat_times,
            y=np.ones_like(beat_times),
            mode="markers",
            name="Beats",
            marker=dict(color="red", symbol="line-ns", size=10, line=dict(width=2)),
        ),
        row=1,
        col=1,
    )

    chroma_times = librosa.times_like(chroma_stft)
    fig.add_trace(
        go.Heatmap(
            z=chroma_stft,
            x=chroma_times,
            y=librosa.hz_to_note(librosa.midi_to_hz(range(12))),
            colorscale="Viridis",
        ),
        row=2,
        col=1,
    )

    cent_times = librosa.times_like(spectral_centroid)
    fig.add_trace(
        go.Scatter(x=cent_times, y=spectral_centroid[0], name="Spectral Centroid"),
        row=3,
        col=1,
    )

    fig.update_layout(height=900, width=800, title_text="Audio Analysis")
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Hz", row=2, col=1)

    return fig


def get_file_info(file, sr, y):
    return {
        "Filename": file.name,
        "File Size": f"{file.size/1024:.2f} KB",
        "Sample Rate": sr,
        "Duration": f"{len(y)/sr:.2f} seconds",
    }


st.title("Enchanced BPM with Audio Playback and Visualization")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    try:
        with st.spinner("Processing audio..."):
            y, sr, tempo, beat_frames, chroma_stft, spectral_centroid = (
                load_and_analyze_audio(uploaded_file)
            )
        st.audio(uploaded_file, format="audio/wav")

        st.subheader("File Information")
        file_info = get_file_info(uploaded_file, sr, y)
        for key, value in file_info.items():
            st.text(f"{key}: {value}")

        tempo_scalar = tempo.item() if isinstance(tempo, np.ndarray) else tempo
        st.write(f"Estimated BPM: {tempo_scalar:.2f}")

        show_plot = st.button("Show/Hide Visualizations")
        if show_plot:
            if "Show_visualizations" not in st.session_state:
                st.session_state.show_visualizations = True
            else:
                st.session_state.show_visualizations = (
                    not st.session_state.show_visualizations
                )

        if (
            "show_visualizations" in st.session_state
            and st.session_state.show_visualizations
        ):
            fig = create_visualizations(
                y, sr, tempo, beat_frames, chroma_stft, spectral_centroid
            )
            st.plotly_chart(fig)

        st.subheader("Manual BPM Tapping")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Tap"):
                if "tap_times" not in st.session_state:
                    st.session_state.tap_times = []
                st.session_state.tap_times.append(time.time())
                if len(st.session_state.tap_times) > 1:
                    intervals = np.diff(st.session_state.tap_times)
                    bpm = 60 / np.mean(intervals)
                    st.write(f"Tapped BPM: {bpm:.2f}")
        with col2:
            if st.button("Reset Tapping"):
                st.session_state.tap_times = []
                st.write("Tapping reset.")
    except Exception as e:
        st.error(f"An Error Occurred: {str(e)}")
        st.error("Please make sure you've uploaded a valid audio file.")
else:
    st.write("Please upload an audio file to begin.")
