import streamlit as st
import yt_dlp
import librosa
import numpy as np
import os
import tempfile
import time
import threading

st.title("🎸 Gerçek Zamanlı Akor Algılama")

# Kullanıcıdan YouTube URL’sini al
video_url = st.text_input("🎶 YouTube URL'sini girin:")

# Akorları belirleme fonksiyonu
def detect_chord(audio_buffer, sr=22050):
    chromagram = librosa.feature.chroma_stft(y=audio_buffer, sr=sr)
    chroma_mean = np.mean(chromagram, axis=1)
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    dominant_notes = [notes[i] for i in np.argsort(chroma_mean)[-3:][::-1]]
    return dominant_notes

# YouTube'dan sesi indirip analiz etmek için fonksiyon
def process_audio(video_url):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': True
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])

            # Dosya adını bul
            audio_file = None
            for file in os.listdir(temp_dir):
                if file.endswith('.mp3'):
                    audio_file = os.path.join(temp_dir, file)
                    break
            
            if not audio_file:
                st.error("🎵 Ses dosyası bulunamadı!")
                return
            
            # Ses dosyasını yükle ve analiz et
            y, sr = librosa.load(audio_file, sr=22050)

            # Akorları belirleme
            chunk_size = sr * 2  # 2 saniyelik parçalar
            for i in range(0, len(y), chunk_size):
                buffer = y[i:i+chunk_size]
                chords = detect_chord(buffer, sr)
                st.write(f"🎶 {round(i/sr, 1)} sn - {round((i+chunk_size)/sr, 1)} sn: {chords}")
                time.sleep(2)  # Akorları anlık olarak göster
            
    except Exception as e:
        st.error(f"Hata oluştu: {e}")

# Kullanıcı "Başlat" butonuna basarsa
if st.button("Başlat"):
    if video_url:
        threading.Thread(target=process_audio, args=(video_url,)).start()
    else:
        st.warning("Lütfen bir YouTube URL'si girin!")
