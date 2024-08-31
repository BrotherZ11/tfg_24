import os
import librosa
import soundfile as sf

def preprocess_audio(input_path, output_path, sample_rate=22050):
    # Cargar el audio
    y, sr = librosa.load(input_path, sr=sample_rate)
    
    # Normalizar el audio
    y_normalized = librosa.util.normalize(y)
    
    # Guardar el archivo de audio preprocesado
    sf.write(output_path, y_normalized, sr)
    print(f"Audio preprocesado guardado en {output_path}")

if __name__ == "__main__":
    input_dir = "data/input/"
    output_dir = "data/output/preprocessed/"
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".mp3") or filename.endswith(".wav"):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, os.path.splitext(filename)[0] + "_preprocessed.wav")
            preprocess_audio(input_file, output_file)
