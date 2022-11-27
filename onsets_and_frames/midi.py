import multiprocessing
import sys

import mido
import numpy as np
from joblib import Parallel, delayed
from mido import Message, MidiFile, MidiTrack
from .constants import *
from mir_eval.util import hz_to_midi
from tqdm import tqdm

def parse_midi(path):
    """open midi file and return np.array of (onset, offset, note, velocity) rows"""
    midi = mido.MidiFile(path)

    time = 0
    events = []
    for message in midi:
        time += message.time

        if 'note' in message.type:
            velocity = message.velocity if message.type == 'note_on' else 0
            note = ""
            for hits in HIT_MAPS['8-hit']:
                if(message.note in hits):
                    note = hits[0]
                    event = dict(index=len(events), time=time, type='note', note=HIT_MAPS_ENCODE_INVERT[note], velocity=velocity)
                    events.append(event)
                    break

    notes = []
    for i, onset in enumerate(events):
        if onset['velocity'] == 0:
            continue

        note = (onset['time'], onset['note'], onset['velocity'])
        notes.append(note)

    notes = np.array(notes)
    velocities = notes[:, 2]
    if(int(np.max(velocities) - np.min(velocities)) != 0):
        velocities = (velocities - np.min(velocities)) / (np.max(velocities) - np.min(velocities))
        notes[:, 2] = velocities
    return notes


def save_midi(path, pitches, intervals, velocities):
    """
    Save extracted notes as a MIDI file
    Parameters
    ----------
    path: the path to save the MIDI file
    pitches: np.ndarray of bin_indices
    intervals: list of (onset_index, offset_index)
    velocities: list of velocity values
    """
    file = MidiFile()
    track = MidiTrack()
    file.tracks.append(track)
    ticks_per_second = int((file.ticks_per_beat * 2) * (2/3))

    events = []
    for i in range(len(pitches)):
        events.append(dict(type='on', pitch=pitches[i], time=intervals[i][0], velocity=velocities[i]))
        events.append(dict(type='off', pitch=pitches[i], time=intervals[i][1], velocity=velocities[i]))
    events.sort(key=lambda row: row['time'])

    last_tick = 0
    for event in events:
        current_tick = int(event['time'] * ticks_per_second)
        velocity = int(event['velocity'] * 127)
        if velocity > 127:
            velocity = 127
        pitch = int(round(hz_to_midi(event['pitch'])))
        track.append(Message('note_' + event['type'], note=pitch, velocity=velocity, time=current_tick - last_tick))
        last_tick = current_tick

    file.save(path)


if __name__ == '__main__':

    def process(input_file, output_file):
        midi_data = parse_midi(input_file)
        np.savetxt(output_file, midi_data, '%.6f', '\t', header='onset\toffset\tnote\tvelocity')


    def files():
        for input_file in tqdm(sys.argv[1:]):
            if input_file.endswith('.mid'):
                output_file = input_file[:-4] + '.tsv'
            elif input_file.endswith('.midi'):
                output_file = input_file[:-5] + '.tsv'
            else:
                print('ignoring non-MIDI file %s' % input_file, file=sys.stderr)
                continue

            yield (input_file, output_file)

    Parallel(n_jobs=multiprocessing.cpu_count())(delayed(process)(in_file, out_file) for in_file, out_file in files())
