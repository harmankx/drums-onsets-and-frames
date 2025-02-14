"""
A rough translation of Magenta's Onsets and Frames implementation [1].

    [1] https://github.com/tensorflow/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py
"""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from .lstm import BiLSTM
from .mel import melspectrogram


class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class OnsetsAndFrames(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48):
        super().__init__()

        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)

        self.onset_stack = nn.Sequential(
            ConvStack(input_features, model_size//4),
            sequence_model(model_size//4, model_size//2),
            nn.Linear(model_size//2, output_features),
            nn.Sigmoid()
        )

        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        
    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        velocity_pred = self.velocity_stack(mel)
        return onset_pred, velocity_pred

    def run_on_batch(self, batch):
        audio_label = batch['audio']
        onset_label = batch['onset']
        velocity_label = batch['velocity']

        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)
        onset_pred,velocity_pred = self(mel)

        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            'velocity': velocity_pred.reshape(*velocity_label.shape)
        }

        pred_on_label = predictions['velocity'] * onset_label

        losses = {
            'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
            'loss/velocity': F.binary_cross_entropy(pred_on_label, velocity_label)
        }

        return predictions, losses