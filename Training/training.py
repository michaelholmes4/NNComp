import numpy as np
import torch
from torch import nn
from model import AudioDataSet, LstmModel, Trainer #Custom classes and code
from scipy.io import wavfile as wav
import os
import re
import matplotlib.pyplot as plt
import librosa
from torch.utils.data import DataLoader
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import argparse

#Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('Type', help="Model type: eg. lstm-32-1 or gru-16-2 ...", type=str)
parser.add_argument('--path', '-p', help="Path to dataset", type=str)
parser.add_argument('--epochs', '-e', help="Number of epochs to train for. Default = 1000", type=int)
parser.add_argument('--batch_size', '-b', help="Training mini-batch size. Default = 32", type=int)
parser.add_argument('--out', '-o', help="Output path", type=str)
args = parser.parse_args()
if args.path:
    path = args.path 
else:
    path = '../DatasetCreation/GeneratedDatasets/out/'
if args.epochs:
    epochs = args.epochs
else:
    epochs = 1000
if args.batch_size:
    batch_size = args.batch_size
else:
    batch_size = 32
if args.out:
    out = args.out
else:
    out = "out/"

type = args.Type

#Import dataset
train_data = AudioDataSet(path + 'train/')
val_data = AudioDataSet(path + 'val/')

#Create torch dataloader to put data into batches and shuffle
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=True)

#Create model name
name = type + datetime.now().strftime("_%Y%d%m_%H%M")

#Create trainer
trainer = Trainer(train_dataloader, val_dataloader)
trainer.create_model(type, name)
trainer.create_optimiser(learning_rate=3e-4)

#Path to save model to
save_to = out + 'model_' + name + ".pth" #create file to save model to

#Train model
trainer.fit(epochs=epochs) #train model

#Save model
trainer.save_model(save_to)
