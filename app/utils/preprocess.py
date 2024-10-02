# src/audio_processing.py
from pydub import AudioSegment
import os
def convert_to_wav(input_audio_path):
    if os.path.exists(input_audio_path):
        sound = AudioSegment.from_mp3(input_audio_path)
        wav_path = input_audio_path.replace(".mp3", ".wav")
        sound.export(wav_path, format="wav")
        return wav_path
    else:
        raise FileNotFoundError(f"El archivo {input_audio_path} no existe.")

#input_audio_path="./data/input/chopin.mp3"