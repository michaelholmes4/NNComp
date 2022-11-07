import numpy as np
import torch
from model import AudioDataSet, Trainer 
import librosa
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from scipy.io import wavfile as wav
import argparse


#Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('Model', help="Path to model file", type=str)
parser.add_argument('Audio', help="Path to audio input file", type=str)
parser.add_argument('--ingain', '-i', help="Input gain dB", type=float)
parser.add_argument('--outgain', '-g', help="Output gain dB", type=float)
parser.add_argument('--out', '-o', help="Output path", type=str)
args = parser.parse_args()
if args.out:
    out = args.out
else:
    out = "./"
if args.ingain:
    ingain = args.ingain
else:
    ingain = 0.0
if args.outgain:
    outgain = args.outgain
else:
    outgain = 0.0

audio_path = args.Audio
model_path = args.Model

#import audio
audio, sr = librosa.load(audio_path, mono=True, sr=48000, dtype=np.float32)

#apply model
data = Dataset()
dataset = DataLoader(data)
trainer = Trainer(dataset, dataset)
trainer.load_model(model_path)

#predict
output = trainer.predict(audio.astype(np.float32) * 10**(ingain/20)) * 10**(outgain/20)

#save file
wav.write(out + "predict_{}.wav".format(datetime.now().strftime("_%Y%d%m_%H%M")), sr, output)