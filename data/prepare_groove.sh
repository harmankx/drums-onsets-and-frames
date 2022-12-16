#! /bin/bash
set -e

echo Downloading the GROOVE dataset \(5 GB\) ...
curl -O https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0.zip

echo Extracting the files ...
unzip -o groove-v1.0.0.zip | awk 'BEGIN{ORS=""} {print "\rExtracting " NR "/2383 ..."; system("")} END {print "\ndone\n"}'

rm groove-v1.0.0.zip
mv groove GROOVE_441

echo Converting the audio files to FLAC ...
COUNTER=0
for f in GROOVE_441/*/*/*.wav; do
    COUNTER=$((COUNTER + 1))
    echo -ne "\rConverting ($COUNTER/1184) ..."
    ffmpeg -y -loglevel fatal -i $f -ac 1 -ar 44100 ${f/\.wav/.flac}
    rm -f $f
done

echo Normalizing audio files to 15db ...
python3.9 split.py
python3.9 normalize_volume.py

echo
echo Preparation complete!
