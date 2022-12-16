import pandas as pd
import numpy as np
import os
import shutil

train = pd.read_csv('./GROOVE_441/info.csv').query('split == "train" and audio_filename != ""')
valid = pd.read_csv('./GROOVE_441/info.csv').query('split == "validation" and audio_filename != ""')
test = pd.read_csv('./GROOVE_441/info.csv').query('split == "test" and audio_filename != ""')

lengths = []

print(train.shape[0])
print(valid.shape[0])
print(test.shape[0])

all = pd.read_csv('./GROOVE_441/info.csv').replace(r'^\s*$', np.nan, regex=True)

split = all.query('audio_filename != ""').dropna().loc[:, ['split']].to_numpy()
filename = all.query('audio_filename != ""').dropna().loc[:, ['midi_filename']].to_numpy()

count = 0
for s, f in zip(split, filename):
    s = s[0]
    f = os.path.join('GROOVE_441', f[0])
    path = os.path.join('GROOVE_441', s)
    if not os.path.isdir(path):
        os.makedirs(path)

    flac_path = f.replace('.mid', '.flac')
    shutil.copy(flac_path, path) 
    shutil.copy(f, path)
