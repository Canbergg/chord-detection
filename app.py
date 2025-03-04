import streamlit as st
import yt_dlp
import librosa
import numpy as np
import sounddevice as sd
import queue
import threading
import time

st.title("🎸 Gerçek Zamanlı Akor Algılama")

# Kullanıcıdan YouTube URL'sini al
video_url = st.text_input("🎶 YouTube URL'sini girin:")

# Global değişkenler
audio_queue = queue.Queue()
running = False

# Akorları belirleme fonksiyonu
def detect_chord(audio_buffer, sr=22050):
    chromagram = librosa.feature.chroma_stft(y=audio_buffer, sr=sr)
    chroma_mean = np.mean(chromagram, axis=1)
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    dominant_notes = [notes[i] for i in np.argsort(chroma_mean)[-3:][::-1]]
    return dominant_notes

# YouTube'dan sesi stream etmek için fonksiyon
def stream_audio(video_url):
    global running
    running = True
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': 'audio.mp3',
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    
    y, sr = librosa.load("audio.mp3", sr=22050)
    
    # Anlık akorları belirleme
    chunk_size = sr * 2  # 2 saniyelik parçalar
    for i in range(0, len(y), chunk_size):
        if not running:
            break
        buffer = y[i:i+chunk_size]
        chords = detect_chord(buffer, sr)
        audio_queue.put(chords)
        time.sleep(2)

# Akorları gösterme fonksiyonu
def display_chords():
    while running:
        if not audio_queue.empty():
            chords = audio_queue.get()
            st.write(f"🎶 Çalan Akorlar: {chords}")
        time.sleep(1)

# Kullanıcı "Başlat" butonuna basarsa
if st.button("Başlat"):
    if video_url:
        # Arka planda YouTube sesini stream eden thread başlat
        threading.Thread(target=stream_audio, args=(video_url,)).start()
        threading.Thread(target=display_chords).start()
    else:
        st.warning("Lütfen bir YouTube URL'si girin!")

# Kullanıcı "Durdur" butonuna basarsa
if st.button("Durdur"):
    running = False
    st.write("🎵 Ses akışı durduruldu.")
