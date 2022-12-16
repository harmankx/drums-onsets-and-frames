from pydub import AudioSegment
import os
import glob
import shutil
from pydub.utils import make_chunks
from tqdm import tqdm

def get_loudness(sound, slice_size=20*1000):
    return max(chunk.dBFS for chunk in make_chunks(sound, slice_size))

def set_loudness(sound, target_dBFS):
    loudness_difference = target_dBFS - get_loudness(sound)
    return sound.apply_gain(loudness_difference)

folders = ['train', 'validation', 'test']

for folder in folders:
    p = os.path.join('GROOVE_441/' + folder, '*.mid')
    files = glob.glob(p)
    print(f'Normalizing {folder} set')
    for f in tqdm(files):
        path = os.path.join("GROOVE_441", folder + '_normalizedTo15db')
        if not os.path.isdir(path):
            os.makedirs(path)
        flac_path = f.replace('.mid', '.flac')

        # print(flac_path)
        song = AudioSegment.from_file(flac_path, format='flac')
        duration_seconds = song.duration_seconds
        song_normal = set_loudness(song, -15)
        # print(f"before: {song.dBFS}     after: {song_normal.dBFS}")

        name = flac_path.split('/')[-1]

        # save the output
        song_normal.export(os.path.join(path, name), "flac")
        shutil.copy(f, path)