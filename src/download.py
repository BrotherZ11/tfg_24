#Opcion 1
import yt_dlp
import os


def download_audio(youtube_url):
    # Directorio de salida
    output_path = os.path.join('data', 'input', '%(title)s.%(ext)s')
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

# Ejemplo de uso
download_audio("https://www.youtube.com/watch?v=xxhET61yB1A")

#Opcion 2
from pytube import YouTube
import os

def download_audio_from_youtube(url):
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_file = audio_stream.download(output_path="audio_files")
    base, ext = os.path.splitext(audio_file)
    new_file = base + '.mp3'
    os.rename(audio_file, new_file)
    return new_file

audio_path = download_audio_from_youtube('https://www.youtube.com/watch?v=12345')
