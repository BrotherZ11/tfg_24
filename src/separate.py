import os
from spleeter.separator import Separator

def separate_audio(input_path, output_dir):
    # Inicializar el separador de Spleeter con el modelo de 4 stems (pistas)
    separator = Separator('spleeter:4stems')
    
    # Separar el audio y guardar las pistas en la carpeta de salida
    separator.separate_to_file(input_path, output_dir)
    print(f"Audio separado guardado en {output_dir}")

if __name__ == "__main__":
    input_dir = "data/output/preprocessed/"
    output_dir = "data/output/separated/"
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            input_file = os.path.join(input_dir, filename)
            track_output_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
            os.makedirs(track_output_dir, exist_ok=True)
            separate_audio(input_file, track_output_dir)


