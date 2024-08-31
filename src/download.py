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
