import argparse
import os
import sys

from pydub import AudioSegment
import numpy as np
import soundfile
from pydub.utils import make_chunks
from mir_eval.util import midi_to_hz

from onsets_and_frames import *


def load_and_process_audio(flac_path, sequence_length, device):

    random = np.random.RandomState(seed=42)


    song = AudioSegment.from_file(flac_path, format='flac')
    song_normal = set_loudness(song, -20)
    print(f"before: {song.dBFS}     after: {song_normal.dBFS}")

    name = flac_path.split('/')[-1]
    path = flac_path.split('/')

    path = path[:len(path)-1]

    path = os.path.join(*path)
    path = path + '_normalized'
    if not os.path.isdir(path):
        os.makedirs(path)

    flac_path = os.path.join(path, name)
    # save the output
    song_normal.export(os.path.join(path, name), "flac")


    audio, sr = soundfile.read(flac_path, dtype='int16')
    assert sr == SAMPLE_RATE

    audio = torch.ShortTensor(audio)

    if sequence_length is not None:
        audio_length = len(audio)
        step_begin = random.randint(audio_length - sequence_length) // HOP_LENGTH
        n_steps = sequence_length // HOP_LENGTH

        begin = step_begin * HOP_LENGTH
        end = begin + sequence_length

        audio = audio[begin:end].to(device)
    else:
        audio = audio.to(device)

    audio = audio.float().div_(SEQUENCE_LENGTH)

    return audio

def get_loudness(sound, slice_size=20*1000):
    return max(chunk.dBFS for chunk in make_chunks(sound, slice_size))

def set_loudness(sound, target_dBFS):
    loudness_difference = target_dBFS - get_loudness(sound)
    return sound.apply_gain(loudness_difference)

def transcribe(model, audio):

    mel = melspectrogram(audio.reshape(-1, audio.shape[-1])[:, :-1]).transpose(-1, -2)
    onset_pred, velocity_pred = model(mel)

    predictions = {
            'onset': onset_pred.reshape((onset_pred.shape[1], onset_pred.shape[2])),
            'velocity': velocity_pred.reshape((velocity_pred.shape[1], velocity_pred.shape[2]))
        }

    return predictions


def transcribe_file(model_file, flac_paths, save_path, sequence_length,
                  onset_threshold, frame_threshold, device):

    model = torch.load(model_file, map_location=device).eval()
    summary(model)
 
    for flac_path in flac_paths:
        print(f'Processing {flac_path}...', file=sys.stderr)
        audio = load_and_process_audio(flac_path, sequence_length, device)
        predictions = transcribe(model, audio)

        p_est, i_est, v_est = extract_notes(predictions['onset'], predictions['velocity'], onset_threshold)

        scaling = HOP_LENGTH / SAMPLE_RATE

        i_est = (i_est * scaling).reshape(-1, 2)

        p_est = np.array([midi_to_hz(HIT_MAPS_ENCODE[midi]) for midi in p_est])
        
        os.makedirs(save_path, exist_ok=True)
        pred_path = os.path.join(save_path, os.path.basename(flac_path) + '.pred.png')
        # save_pianoroll(pred_path, predictions['onset'])
        midi_path = os.path.join(save_path, os.path.basename(flac_path) + '.pred.mid')
        save_midi(midi_path, p_est, i_est, v_est)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('flac_paths', nargs='+', type=str)
    parser.add_argument('--save-path', type=str, default='.')
    parser.add_argument('--sequence-length', default=None, type=int)
    parser.add_argument('--onset-threshold', default=0.5, type=float)
    parser.add_argument('--frame-threshold', default=0.5, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        transcribe_file(**vars(parser.parse_args()))