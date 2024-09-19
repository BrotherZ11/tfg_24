import tensorflow._api.v2.compat.v1 as tf
import os
from note_seq import audio_io
from note_seq import midi_io
from note_seq.protobuf import music_pb2
import note_seq
from magenta.models.onsets_frames_transcription import audio_label_data_utils
from magenta.models.onsets_frames_transcription import configs
from magenta.models.onsets_frames_transcription import data
from magenta.models.onsets_frames_transcription import infer_util
from magenta.models.onsets_frames_transcription import train_util

# Desactivar las características de TensorFlow 2 (ya que usamos TensorFlow 1)
tf.disable_v2_behavior()

# Paso 1: Inicialización del modelo MAESTRO y carga del checkpoint
checkpoint_dir = './train'  # Cambia esto a la ruta donde están tus checkpoints

# Configurar el modelo MAESTRO
config = configs.CONFIG_MAP['onsets_frames']
hparams = config.hparams
hparams.use_cudnn = False  # CUDNN puede causar problemas en algunos sistemas, así que lo desactivamos
hparams.batch_size = 1

# Crear el dataset de ejemplos
examples = tf.placeholder(tf.string, [None])

dataset = data.provide_batch(
    examples=examples,
    preprocess_examples=True,
    params=hparams,
    is_training=False,
    shuffle_examples=False,
    skip_n_initial_records=0
)

# Crear el estimador que manejará la inferencia (transcripción)
estimator = train_util.create_estimator(
    config.model_fn, checkpoint_dir, hparams
)

# Crear un iterador para leer los datos de audio
iterator = tf.data.make_initializable_iterator(dataset)
next_record = iterator.get_next()

# Paso 2: Cargar y procesar el archivo de audio local (WAV)
audio_path = 'hola.wav'  # Ruta del archivo WAV
wav_data = open(audio_path, 'rb').read()  # Leer el archivo de audio

# Procesar el archivo de audio para convertirlo en un ejemplo
example_list = list(
    audio_label_data_utils.process_record(
        wav_data=wav_data,
        sample_rate=hparams.sample_rate,
        ns=note_seq.NoteSequence(),
        example_id=os.path.basename(audio_path),
        min_length=0,
        max_length=-1,
        allow_empty_notesequence=True
    )
)
assert len(example_list) == 1  # Asegúrate de que solo haya un ejemplo
to_process = [example_list[0].SerializeToString()]

# Inicializar una sesión de TensorFlow
sess = tf.Session()

# Inicializar las variables y el iterador para la inferencia
sess.run([tf.initializers.global_variables(), tf.initializers.local_variables()])
sess.run(iterator.initializer, {examples: to_process})

# Función para obtener el siguiente batch de datos
def transcription_data(params):
    del params
    return tf.data.Dataset.from_tensors(sess.run(next_record))

# Preparar la función de entrada para la inferencia
input_fn = infer_util.labels_to_features_wrapper(transcription_data)

# Paso 3: Ejecutar la inferencia (transcripción de audio a MIDI)
prediction_list = list(
    estimator.predict(
        input_fn,
        yield_single_examples=False
    )
)
assert len(prediction_list) == 1  # Solo esperamos una predicción

# Convertir la predicción a un NoteSequence (secuencia de notas)
sequence_prediction = note_seq.NoteSequence.FromString(
    prediction_list[0]['sequence_predictions'][0]
)

# Paso 4: Visualizar y guardar el archivo MIDI generado

# Mostrar la secuencia de notas en un gráfico
note_seq.plot_sequence(sequence_prediction)

# Guardar la predicción como archivo MIDI
midi_filename = 'output.mid'
midi_io.sequence_proto_to_midi_file(sequence_prediction, midi_filename)

print(f'El archivo MIDI se ha guardado como {midi_filename}')
