import json
import os
from abc import abstractmethod
from glob import glob

import numpy as np
import soundfile
from torch.utils.data import Dataset
from tqdm import tqdm
import math
from torch import nn

from .constants import *
from .midi import parse_midi


class PianoRollAudioDataset(Dataset):
    def __init__(self, path, groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)

        self.data = []
        print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {path}")
        for group in groups:
            for input_files in tqdm(self.files(group), desc='Loading group %s' % group):
                self.data.append(self.load(*input_files))

    def __getitem__(self, index):
        data = self.data[index]
        result = dict(path=data['path'])

        audio_length = len(data['audio'])

        shorter = False

        if self.sequence_length is not None and audio_length > self.sequence_length:
            step_begin = self.random.randint(audio_length - self.sequence_length) // HOP_LENGTH
            n_steps = self.sequence_length // HOP_LENGTH

            step_end = step_begin + n_steps

            begin = step_begin * HOP_LENGTH
            end = begin + self.sequence_length

            result['audio'] = data['audio'][begin:end].to(self.device)
            result['label'] = data['label'][step_begin:step_end, :].to(self.device)
            result['velocity'] = data['velocity'][step_begin:step_end, :].to(self.device)
        else:
            shorter = True
            result['audio'] = data['audio'].to(self.device)
            result['label'] = data['label'].to(self.device)
            result['velocity'] = data['velocity'].to(self.device).float()

        result['audio'] = result['audio'].float().div_(32768.0)
        result['onset'] = (result['label'] == 1).float()
        result['velocity'] = result['velocity'].float()
    
        # print(f'audio: {result["audio"].shape} onset: {result["onset"].shape}')
        
        if shorter:
            pad = 0
            if((self.sequence_length - result['audio'].shape[0]) % 2 != 0):
                pad = 1

            result['audio'] = nn.ConstantPad1d(((self.sequence_length - result['audio'].shape[0]) // 2, (self.sequence_length - result['audio'].shape[0]) // 2 + pad), 0)(result['audio'])

            pad = 0
            if((self.sequence_length // HOP_LENGTH - result['onset'].shape[0]) % 2 != 0):
                pad = 1
            
            result['onset'] = nn.ConstantPad2d((0, 0, (self.sequence_length // HOP_LENGTH - result['onset'].shape[0]) // 2, (self.sequence_length // HOP_LENGTH - result['onset'].shape[0]) // 2 + pad), 0)(result['onset'])
            result['velocity'] = nn.ConstantPad2d((0, 0, (self.sequence_length // HOP_LENGTH - result['velocity'].shape[0]) // 2, (self.sequence_length // HOP_LENGTH - result['velocity'].shape[0]) // 2 + pad), 0)(result['velocity'])
            result['label'] = nn.ConstantPad2d((0, 0, (self.sequence_length // HOP_LENGTH - result['label'].shape[0]) // 2, (self.sequence_length // HOP_LENGTH - result['label'].shape[0]) // 2 + pad), 0)(result['label'])

        # print(f'result[audio]: {result["audio"].shape} result[onset]: {result["onset"].shape} result[velocity]: {result["velocity"].shape}')

        return result

    def __len__(self):
        return len(self.data)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """return the list of input files (audio_filename, tsv_filename) for this group"""
        raise NotImplementedError

    def load(self, audio_path, tsv_path):
        """
        load an audio track and the corresponding labels

        Returns
        -------
            A dictionary containing the following data:

            path: str
                the path to the audio file

            audio: torch.ShortTensor, shape = [num_samples]
                the raw waveform

            label: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains the onset labels encoded as:
                1 = onset, 0 = all else

            velocity: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains MIDI velocity values at the onset locations
        """
        saved_data_path = audio_path.replace('.flac', '.pt').replace('.wav', '.pt')
        if os.path.exists(saved_data_path):
            return torch.load(saved_data_path)

        audio, sr = soundfile.read(audio_path, dtype='int16')
        assert sr == SAMPLE_RATE

        audio = torch.ShortTensor(audio)
        audio_length = len(audio)

        n_keys = 8
        n_steps = (audio_length - 1) // HOP_LENGTH

        label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        velocity = torch.zeros(n_steps, n_keys, dtype=torch.float)

        tsv_path = tsv_path
        midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)

        for onset, note, vel in midi:
            left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
            onset_right = min(n_steps, left + HOPS_IN_ONSET)

            f = int(note)
            label[left:onset_right, f] = 1
            velocity[left:onset_right, f] = vel

        data = dict(path=audio_path, audio=audio, label=label, velocity=velocity)
        torch.save(data, saved_data_path)
        return data

class GROOVE(PianoRollAudioDataset):

    def __init__(self, path='data/GROOVE', groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE):
        super().__init__(path, groups if groups is not None else ['validation_normalized'], sequence_length, seed, device)

    @classmethod
    def available_groups(cls):
        return ['train']

    def files(self, group):
        flacs = sorted(glob(os.path.join(self.path, group, '**', '*.flac'), recursive=True))
        if len(flacs) == 0:
            flacs = sorted(glob(os.path.join(self.path, group, '**', '*.wav'), recursive=True))

        midis = sorted(glob(os.path.join(self.path, group, '**', '*.mid'), recursive=True))

        files = list(zip(flacs, midis))
        if len(files) == 0:
            raise RuntimeError(f'Group {group} is empty')

        result = []
        for audio_path, midi_path in files:
            tsv_filename = midi_path.replace('.midi','.tsv').replace('.mid', '.tsv')
            if not os.path.exists(tsv_filename):
                midi = parse_midi(midi_path) 
                np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,note,velocity')
            result.append((audio_path, tsv_filename))
        return result
