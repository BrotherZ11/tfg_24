from music21 import converter

def midi_to_musicxml(midi_file):
    score = converter.parse(midi_file)
    score.write('musicxml', 'output.xml')
    

midi_to_musicxml('output.mid')
