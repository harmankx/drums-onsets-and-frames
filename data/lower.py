from pydub import AudioSegment
import os
import glob
import shutil
from tqdm import tqdm

folders = ['train', 'test', 'validation']

for folder in folders:
    p = os.path.join('./GROOVE_441', folder, '*.mid')
    files = glob.glob(p)
    print(f'Lowering volume of {folder} set')
    for f in tqdm(files):
        path = os.path.join('GROOVE_441', folder + '_lower_10db')
        if not os.path.isdir(path):
            os.makedirs(path)
        flac_path = f.replace('.mid', '.flac')

        song = AudioSegment.from_file(flac_path, format='flac')
        name = flac_path.split('/')
        name = name[-1]
        song = song - 10

        # save the output
        song.export(os.path.join(path, name), "flac")
        shutil.copy(f, path)