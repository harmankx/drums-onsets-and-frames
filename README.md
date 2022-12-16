# PyTorch Implementation of Oaf-Drums

This is a [PyTorch](https://pytorch.org/) implementation of Google's [Oaf-Drums](https://magenta.tensorflow.org/oaf-drums) model modified to predict relative velocities. It is based on [Jong Wook Kim's](https://github.com/jongwook) [implentation](https://github.com/jongwook/onsets-and-frames) of Onsets and Frames

### Downloading Dataset

To download the dataset, run the ./prepare_groove.sh file

### Training

All package requirements are contained in `requirements.txt`. To train the model, run:

```bash
pip install -r requirements.txt
python train.py
```

### Testing

To evaluate the trained model, run the following command:

```bash
python evaluate.py runs/LOGDRUMS/model-150000.pt
```
