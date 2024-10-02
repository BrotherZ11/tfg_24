import yt_dlp
import os


def download_audio(youtube_url):
    # Directorio de salida
    output_path = os.path.join('uploads', '%(title)s.%(ext)s')
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=True)
        audio_file = ydl.prepare_filename(info_dict).replace(".webm", ".mp3")
        return audio_file

# Ejemplo de uso
#download_audio("https://www.youtube.com/watch?v=FZtBwlxL0Aw&ab_channel=ValentinaLisitsaQORRecordsOfficialchannel")
