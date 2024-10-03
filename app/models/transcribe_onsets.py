# models/transcribe_onsets_frames.py
import os
import tempfile
from magenta.models.onsets_frames_transcription import audio_label_data_utils
from magenta.models.onsets_frames_transcription import configs
from magenta.models.onsets_frames_transcription import data
from magenta.models.onsets_frames_transcription import infer_util
from magenta.models.onsets_frames_transcription import train_util
from note_seq import midi_io
from note_seq.protobuf import music_pb2
import tensorflow._api.v2.compat.v1 as tf
import six

# Desactivar las características de TensorFlow 2 (ya que usamos TensorFlow 1)
tf.disable_v2_behavior()

MODEL_DIR = './train'  # Define la ruta del modelo aquí
CHECKPOINT_PATH = './train'  # Define la ruta del checkpoint aquí

def create_example(filename, sample_rate, load_audio_with_librosa):
    """Procesa el archivo de audio en un Example proto."""
    wav_data = tf.gfile.Open(filename, 'rb').read()
    example_list = list(
        audio_label_data_utils.process_record(
            wav_data=wav_data,
            sample_rate=sample_rate,
            ns=music_pb2.NoteSequence(),
            example_id=six.ensure_text(filename, 'utf-8'),
            min_length=0,
            max_length=-1,
            allow_empty_notesequence=True,
            load_audio_with_librosa=load_audio_with_librosa))
    assert len(example_list) == 1
    return example_list[0].SerializeToString()

def transcribe_onsets_frames(audio_file_path):
    """Realiza la transcripción de un archivo de audio usando Onsets and Frames y genera un archivo MIDI."""
    config = configs.CONFIG_MAP['onsets_frames']
    hparams = config.hparams
    hparams.batch_size = 1
    hparams.use_cudnn = False  # CUDNN puede causar problemas en algunos sistemas, así que lo desactivamos

    with tf.Graph().as_default():
        examples = tf.placeholder(tf.string, [None])

        dataset = data.provide_batch(
            examples=examples,
            preprocess_examples=True,
            params=hparams,
            is_training=False,
            shuffle_examples=False,
            skip_n_initial_records=0)

        estimator = train_util.create_estimator(config.model_fn,
                                                os.path.expanduser(MODEL_DIR),
                                                hparams)

        iterator = tf.data.make_initializable_iterator(dataset)
        next_record = iterator.get_next()

        with tf.Session() as sess:
            sess.run([tf.initializers.global_variables(),
                      tf.initializers.local_variables()])

            # Procesar el archivo de audio
            sess.run(iterator.initializer, {
                examples: [create_example(audio_file_path, hparams.sample_rate, False)]
            })

            def transcription_data(params):
                del params
                return tf.data.Dataset.from_tensors(sess.run(next_record))
            
            input_fn = infer_util.labels_to_features_wrapper(transcription_data)

            prediction_list = list(
                estimator.predict(input_fn, checkpoint_path=CHECKPOINT_PATH, yield_single_examples=False)
            )
            assert len(prediction_list) == 1

            sequence_prediction = music_pb2.NoteSequence.FromString(
                prediction_list[0]['sequence_predictions'][0])

            # Guardar el archivo MIDI
            midi_filename = audio_file_path + '.midi'
            midi_io.sequence_proto_to_midi_file(sequence_prediction, midi_filename)

            return midi_filename
