import torch


SAMPLE_RATE = 16000
HOP_LENGTH = SAMPLE_RATE * 32 // 1000
ONSET_LENGTH = SAMPLE_RATE * 32 // 1000
OFFSET_LENGTH = SAMPLE_RATE * 32 // 1000
HOPS_IN_ONSET = ONSET_LENGTH // HOP_LENGTH
HOPS_IN_OFFSET = OFFSET_LENGTH // HOP_LENGTH
MIN_MIDI = 21
MAX_MIDI = 108

N_MELS = 229
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2
WINDOW_LENGTH = 2048

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

HIT_MAPS = {
    '8-hit': [
        # Kick
        [36],

        # Snare,X-stick, handclap
        [38, 40, 37, 39],

        # Toms + (Low_Conga extra)
        [48, 50, 45, 47, 43, 58, 64],

        # HH + Tambourine + (Maracas extra)
        [46, 26, 42, 22, 44, 54, 70],

        # Ride
        [51, 59],

        # Ride bell+ cow bell
        [53, 56],

        # Crashes
        [49, 55, 57, 52],

        # Clave / Sticks
        [75],
    ],
}

HIT_MAPS_ENCODE = {
    1 : 36, # Kick
    2 : 38, # Snare
    3 : 48, # Toms
    4 : 46, # HH
    5 : 51, # Ride
    6 : 53, # Ride Bell
    7 : 49, # Crash
    8 : 75  # Sticks
}

HIT_MAPS_ENCODE_INVERT = {
    36 : 1, # Kick
    38 : 2, # Snare
    48 : 3, # Toms
    46 : 4, # HH
    51 : 5, # Ride
    53 : 6, # Ride Bell
    49 : 7, # Crash
    75 : 8  # Sticks
}

