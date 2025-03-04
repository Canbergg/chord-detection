import streamlit as st
from pytube import YouTube
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

st.title("🎸 YouTube Akor Algılama Sistemi")
st.write("Bir YouTube şarkı linki girin, otomatik olarak akorlarını analiz edelim!")

# Kullanıcıdan YouTube linkini al
video_url = st.text_input("🎶 YouTube URL'sini buraya yapıştırın:")

if st.button("Şarkıyı İşle"):
    if video_url:
        try:
            # YouTube'dan sesi indir
            st.write("🎵 Ses dosyası indiriliyor...")
            yt = YouTube(video_url)
            audio_stream = yt.streams.filter(only_audio=True).first()
            file_path = audio_stream.download(filename="downloaded_audio.mp4")
            
            # Librosa ile ses dosyasını yükle
            st.write("🔍 Ses analizi yapılıyor...")
            y, sr = librosa.load(file_path, sr=22050)

            # Chromagram oluştur
            chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chromagram, axis=1)

            # Nota isimleri
            notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

            # En baskın notayı bul
            dominant_notes = [notes[i] for i in np.argsort(chroma_mean)[-3:][::-1]]  # En baskın 3 nota
            st.write(f"🎶 Bulunan Notalar: {dominant_notes}")

            # Chromagram görselleştirme
            fig, ax = plt.subplots(figsize=(10, 4))
            librosa.display.specshow(chromagram, sr=sr, x_axis="time", y_axis="chroma")
            plt.colorbar()
            plt.title("Chromagram")
            st.pyplot(fig)

            # Geçici dosyayı sil
            os.remove(file_path)

        except Exception as e:
            st.error(f"Hata oluştu: {e}")
    else:
        st.warning("Lütfen bir YouTube URL'si girin!")

