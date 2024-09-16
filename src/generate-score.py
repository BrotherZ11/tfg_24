import os
from music21 import stream, note, chord, duration, midi

def parse_transcription(file_path):
    notes = []
    with open(file_path, 'r') as f:
        for line in f:
            pitch, start_time, end_time = line.strip().split(', ')
            notes.append({
                'pitch': int(pitch),
                'start_time': float(start_time),
                'end_time': float(end_time)
            })
    return notes

def create_midi_from_transcription(notes, output_path):
    midi_stream = stream.Stream()
    
    for n in notes:
        pitch = n['pitch']
        start_time = n['start_time']
        end_time = n['end_time']
        note_duration = end_time - start_time
        
        # Crear la nota con la duración correspondiente
        new_note = note.Note(pitch)
        new_note.duration = duration.Duration(note_duration)
        
        midi_stream.append(new_note)
    
    # Guardar la partitura en formato MIDI
    midi_fp = midi.translate.music21ObjectToMidiFile(midi_stream)
    midi_fp.open(output_path, 'wb')
    midi_fp.write()
    midi_fp.close()
    print(f"Partitura MIDI guardada en {output_path}")

if __name__ == "__main__":
    input_dir = "data/output/transcriptions/"
    output_dir = "data/output/scores/"
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, os.path.splitext(filename)[0] + ".mid")
            
            notes = parse_transcription(input_file)
            create_midi_from_transcription(notes, output_file)
