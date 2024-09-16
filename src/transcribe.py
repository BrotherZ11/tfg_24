import functools
import numpy as np
import librosa
import note_seq
from mt3 import metrics_utils
from mt3 import models
from mt3 import network
from mt3 import note_sequences
from mt3 import preprocessors
from mt3 import spectrograms
from mt3 import vocabularies


SAMPLE_RATE = 16000
SF2_PATH = 'SGM-v2.01-Sal-Guit-Bass-V1.3.sf2'  # Ruta a tu archivo SoundFont

def load_model(model_type='mt3', checkpoint_path='./checkpoints/mt3/'):
    """Load the MT3 model."""
    if model_type == 'ismir2021':
        num_velocity_bins = 127
        encoding_spec = note_sequences.NoteEncodingSpec
        inputs_length = 512
    elif model_type == 'mt3':
        num_velocity_bins = 1
        encoding_spec = note_sequences.NoteEncodingWithTiesSpec
        inputs_length = 256
    else:
        raise ValueError(f'Unknown model_type: {model_type}')
    
    gin_files = ['mt3/gin/model.gin', f'mt3/gin/{model_type}.gin']
    
    # Set up model
    partitioner = t5x.partitioning.PjitPartitioner(num_partitions=1)
    spectrogram_config = spectrograms.SpectrogramConfig()
    codec = vocabularies.build_codec(vocab_config=vocabularies.VocabularyConfig(num_velocity_bins=num_velocity_bins))
    vocabulary = vocabularies.vocabulary_from_codec(codec)
    output_features = {
        'inputs': seqio.ContinuousFeature(dtype=tf.float32, rank=2),
        'targets': seqio.Feature(vocabulary=vocabulary),
    }
    
    # Load and restore model
    with gin.unlock_config():
        gin.parse_config_files_and_bindings(gin_files, finalize_config=False)
    
    model_config = gin.get_configurable(network.T5Config)()
    module = network.Transformer(config=model_config)
    model = models.ContinuousInputsEncoderDecoderModel(
        module=module,
        input_vocabulary=output_features['inputs'].vocabulary,
        output_vocabulary=output_features['targets'].vocabulary,
        optimizer_def=t5x.adafactor.Adafactor(decay_rate=0.8, step_offset=0),
        input_depth=spectrograms.input_depth(spectrogram_config)
    )
    
    # Restore from checkpoint
    train_state_initializer = t5x.utils.TrainStateInitializer(
        optimizer_def=model.optimizer_def,
        init_fn=model.get_initial_variables,
        input_shapes={'encoder_input_tokens': (8, inputs_length), 'decoder_input_tokens': (8, 1024)},
        partitioner=partitioner
    )
    
    restore_checkpoint_cfg = t5x.utils.RestoreCheckpointConfig(
        path=checkpoint_path,
        mode='specific',
        dtype='float32'
    )
    
    train_state_axes = train_state_initializer.train_state_axes
    predict_fn = partitioner.partition(
        lambda params, batch, decode_rng: model.predict_batch_with_aux(
            params, batch, decoder_params={'decode_rng': None}
        ),
        in_axis_resources=(train_state_axes.params, t5x.partitioning.PartitionSpec('data',), None),
        out_axis_resources=t5x.partitioning.PartitionSpec('data',)
    )
    
    train_state = train_state_initializer.from_checkpoint_or_scratch(
        [restore_checkpoint_cfg],
        init_rng=jax.random.PRNGKey(0)
    )
    
    return model, predict_fn, train_state, codec, encoding_spec, output_features

def preprocess_audio(audio_path):
    """Preprocess audio file for MT3 model."""
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    audio = np.expand_dims(audio, axis=0)  # Add batch dimension
    return audio

def transcribe_audio(model, predict_fn, train_state, audio, codec, encoding_spec, output_features):
    """Transcribe audio using MT3 model."""
    def audio_to_dataset(audio):
        frames, frame_times = _audio_to_frames(audio)
        return tf.data.Dataset.from_tensors({
            'inputs': frames,
            'input_times': frame_times,
        })

    def _audio_to_frames(audio):
        frame_size = spectrograms.SpectrogramConfig().hop_width
        padding = [0, frame_size - len(audio) % frame_size]
        audio = np.pad(audio, padding, mode='constant')
        frames = spectrograms.split_audio(audio, spectrograms.SpectrogramConfig())
        num_frames = len(audio) // frame_size
        times = np.arange(num_frames) / spectrograms.SpectrogramConfig().frames_per_second
        return frames, times

    def preprocess(ds):
        pp_chain = [
            functools.partial(
                t5.data.preprocessors.split_tokens_to_inputs_length,
                sequence_length={'inputs': 256, 'targets': 1024},
                output_features=output_features,
                feature_key='inputs',
                additional_feature_keys=['input_times']),
            preprocessors.add_dummy_targets,
            functools.partial(
                preprocessors.compute_spectrograms,
                spectrogram_config=spectrograms.SpectrogramConfig())
        ]
        for pp in pp_chain:
            ds = pp(ds)
        return ds

    ds = audio_to_dataset(audio)
    ds = preprocess(ds)

    model_ds = model.FEATURE_CONVERTER_CLS(pack=False)(
        ds, task_feature_lengths={'inputs': 256, 'targets': 1024}
    )
    model_ds = model_ds.batch(8)

    inferences = (tokens for batch in model_ds.as_numpy_iterator()
                  for tokens in predict_fn(train_state.params, batch, jax.random.PRNGKey(0)))

    predictions = []
    for example, tokens in zip(ds.as_numpy_iterator(), inferences):
        predictions.append(_postprocess(tokens, example))

    result = metrics_utils.event_predictions_to_ns(
        predictions, codec=codec, encoding_spec=encoding_spec
    )
    return result['est_ns']

def _postprocess(tokens, example):
    tokens = _trim_eos(tokens)
    start_time = example['input_times'][0]
    start_time -= start_time % (1 / vocabularies.DECODED_EOS_ID)
    return {
        'est_tokens': tokens,
        'start_time': start_time,
        'raw_inputs': []
    }

def _trim_eos(tokens):
    tokens = np.array(tokens, np.int32)
    if vocabularies.DECODED_EOS_ID in tokens:
        tokens = tokens[:np.argmax(tokens == vocabularies.DECODED_EOS_ID)]
    return tokens

def save_transcription(note_sequence, output_path):
    """Save the transcribed note sequence to MIDI file."""
    note_seq.sequence_proto_to_midi_file(note_sequence, output_path)
    print(f"Transcription saved to {output_path}")

if __name__ == "__main__":
    audio_file = "path/to/your/audio_file.wav"  # Cambia esto a la ruta de tu archivo de audio
    output_file = "path/to/output_file.mid"     # Cambia esto a la ruta de salida deseada
    checkpoint_path = './checkpoints/mt3/'

    model, predict_fn, train_state, codec, encoding_spec, output_features = load_model(
        model_type='mt3', checkpoint_path=checkpoint_path
    )
    audio = preprocess_audio(audio_file)
    note_sequence = transcribe_audio(model, predict_fn, train_state, audio, codec, encoding_spec, output_features)
    save_transcription(note_sequence, output_file)
