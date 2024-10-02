from music21 import converter, instrument, note, chord, stream

def midi_a_partitura(midi_file):
    # Cargar el archivo MIDI
    midi = converter.parse(midi_file)

    # Separar las partes por instrumento
    parts = instrument.partitionByInstrument(midi)

    # Crear un stream vacío para almacenar las notas y acordes
    partitura = stream.Score()

    # Si hay varias partes por instrumento
    if parts:  
        for part in parts.parts:
            # Iterar sobre los eventos del archivo MIDI
            for elemento in part.recurse():
                # Si el evento es una nota, añadirla al stream de la partitura
                if isinstance(elemento, note.Note):
                    print(f"Nota: {elemento}")
                    partitura.append(elemento)
                # Si el evento es un acorde, añadirlo al stream de la partitura
                elif isinstance(elemento, chord.Chord):
                    print(f"Acorde: {elemento}")
                    partitura.append(elemento)
    else:
        # Si no hay partes instrumentales, iterar sobre el midi directamente
        for elemento in midi.flat.notes:
            print(f"Elemento: {elemento}")
            partitura.append(elemento)

    # Intentar corregir la notación (esto puede resolver algunos problemas)
    try:
        partitura.makeNotation(inPlace=True)
    except Exception as e:
        print(f"Error corrigiendo la notación: {e}")

    # Mostrar la partitura en formato texto
    partitura.show('text')
    

    # Guardar la partitura en formato MusicXML
    try:
        partitura.write('pdf', fp='partitura.pdf')
        print("Partitura guardada correctamente.")
    except Exception as e:
        print(f"Error guardando la partitura: {e}")

# Ruta del archivo MIDI
archivo_midi = 'hola.mid'

# Convertir a partitura
midi_a_partitura(archivo_midi)
