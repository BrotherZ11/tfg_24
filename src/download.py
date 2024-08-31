import yt_dlp
import os

# Direccion de descarga
#carpeta = os.path.join(os.path.expanduser('C:\Users\dzarz\Documents\GitHub\tfg_24\data\input'))
def download_audio(youtube_url):
    # Especificar el directorio de salida
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
download_audio("https://www.youtube.com/watch?v=7maJOI3QMu0")
