# app.py
from flask import Flask, render_template, Blueprint, request,redirect, url_for
from .utils.download import download_audio
from .utils.preprocess import convert_to_wav
import os

main = Blueprint('main', __name__)

# Directorios de archivos
UPLOAD_FOLDER = 'uploads/'

@main.route("/")
def index():
    return render_template("index.html")


@main.route('/transcribir', methods=['POST'])
def transcribir():
    if 'audio_file' in request.files and request.files['audio_file'].filename != '':
        audio = request.files['audio_file']
        file_path = os.path.join(UPLOAD_FOLDER, audio.filename)
        audio.save(file_path)
         # Convertir el archivo subido a WAV
        if file_path.endswith('.mp3'):
            wav_path = convert_to_wav(file_path)
        else:
            return "Solo se permite archivos en formato MP3", 400

        return f"Archivo convertido a {wav_path}"
    elif 'youtube_url' in request.form and request.form['youtube_url'].strip() != '':
        url = request.form['youtube_url']
        mp3_path = download_audio(url)
        # Convertir el MP3 descargado a WAV
        print(mp3_path)
        if mp3_path.endswith('.mp3'):
            wav_path = convert_to_wav(mp3_path)
            return f"Archivo descargado y convertido a {wav_path}"
        else:
            return "Error al descargar el archivo de YouTube", 500

    return redirect(url_for("main.index"))

