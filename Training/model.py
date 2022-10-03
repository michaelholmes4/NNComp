from copy import deepcopy
import os
import re
import torch
import librosa
from scipy.io import wavfile as wav
import numpy as np
from torch import nn, optim
import auraloss
from torch.utils.tensorboard import SummaryWriter
import gc
import torch


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

class AudioDataSet(torch.utils.data.Dataset):
    """
    Dataset class for audio samples. Import single audio file and target. Splits audio into samples of length sample_length. Leaves silence gap between samples val_split is the percentage of data to use for validation. If data=True, class represents the the training portion, false the class represents the validation portion.
    """
    def __init__(self, input_path, sr=48000):
        #self.input, self.sr = librosa.load(input_path, sr=None, mono = True, dtype=np.float32)
        self.input_path = input_path
        self.sr = sr
        self.setup() #split input samples into training and validation sets
        
    def __len__(self):
        """
        Number of audio samples in dataset
        """
        return self.len

    def __getitem__(self, idx):
        """
        Get sample at idx
        """

        #Read audio files
        _ , input = wav.read(self.input_path + 'input{}.wav'.format(idx))
        _ , target = wav.read(self.input_path + 'target{}.wav'.format(idx))

        input = torch.tensor(input)[:, None]
        target = torch.tensor(target)[:, None]

        return input, target

    def setup(self):
        """
        Sets up dataset from file structure
        """
        #Read files in input path
        files = sorted_alphanumeric(os.listdir(self.input_path))
        num_samples = int(files[-1][6:-4])

        #Generate indices
        self.len = num_samples

    def get_copy(self, train=True):
        """
        Returns a copy of this instance. Can change change the train=False to get a copy that represents the validation samples.
        """
        ret = deepcopy(self)
        ret.train = train
        return ret

        
class LstmModel(nn.Module):
    """
    Neural Network Class
    """
    def __init__(self, input_size=1, output_size=1, hidden_size=32, num_layers=1):
        super(LstmModel, self).__init__()

        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=False, bidirectional=False)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(1,0,2)
        out, _ = self.lstm(x)
        out = torch.tanh(self.linear(out))
        out = out.permute(1,0,2)
        return out

class GruModel(nn.Module):
    """
    Neural Network Class
    """
    def __init__(self, input_size=1, output_size=1, hidden_size=32, num_layers=1):
        super(GruModel, self).__init__()

        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=False, bidirectional=False)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(1,0,2)
        out, _ = self.gru(x)
        out = torch.tanh(self.linear(out))
        out = out.permute(1,0,2)
        return out

class RnnModel(nn.Module):
    """
    Neural Network Class
    """
    def __init__(self, input_size=1, output_size=1, hidden_size=32, num_layers=1):
        super(RnnModel, self).__init__()

        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=False, bidirectional=False)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(1,0,2)
        out, _ = self.rnn(x)
        out = torch.tanh(self.linear(out))
        out = out.permute(1,0,2)
        return out

class Trainer:
    """
    Class to handle training of model
    """
    def __init__(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader

        #loss functions
        self.l1 = torch.nn.L1Loss()
        self.stft = auraloss.freq.STFTLoss()
        
        self.current_epoch = 0

        #configure GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu" #else "mps" if torch.has_mps else "cpu"
        print(f"Using {self.device} device")      

    def create_model(self, type, name):
        """
        Setup NN for different model types.

        Name to call model, typically type + date.

        Accepts input:
            "lstm-32-1"
            "lstm-16-2"
            "lstm-8-4"

            "lstm-16-1"
            "lstm-8-2"
            "lstm-4-4"

            "lstm-8-1"
            "lstm-4-2"
            "lstm-2-4"

            "gru-32-1"
            "gru-16-2"
            "gru-8-4"

            "gru-16-1"
            "gru-8-2"
            "gru-4-4"

            "gru-8-1"
            "gru-4-2"
            "gru-2-4"

            "rnn-32-1"
            "rnn-16-2"
            "rnn-8-4"

            "rnn-16-1"
            "rnn-8-2"
            "rnn-4-4"

            "rnn-8-1"
            "rnn-4-2"
            "rnn-2-4"
        """
        inputs = type.split('-')
        if len(inputs) == 3 and (inputs[0] == 'lstm' or inputs[0] == 'gru' or inputs[0] == 'rnn'):
            if inputs[0] == 'lstm':
                self.model = LstmModel(hidden_size=int(inputs[1]), num_layers=(int(inputs[2]))).to(self.device)
            elif inputs[0] == 'gru':
                self.model = GruModel(hidden_size=int(inputs[1]), num_layers=(int(inputs[2]))).to(self.device)
            else:
                self.model = RnnModel(hidden_size=int(inputs[1]), num_layers=(int(inputs[2]))).to(self.device)
        else:
            raise ValueError("Not a valid model type")

        self.tb = SummaryWriter("./runs/" + name)
            
    def create_optimiser(self, learning_rate = 3e-4):
            self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, verbose=True)

    def save_model(self, path):
        torch.save(self.model, path)
    
    def load_model(self, path):
        if 'model' in locals():
            self.model.cpu()
            torch.cuda.empty_cache()
        
        self.model = torch.load(path, map_location='cpu')
        self.model.to(self.device)
        
    def get_model(self):
        return self.model

    def loss_fn(self, output, target, type="both"):
        """
        Loss function that uses combination of MAE and STFT
        """
        output = output.reshape(1,-1)
        target = target.reshape(1,-1)
        l1 = self.l1(output, target)
        stft = self.stft(output, target)
        
        if type == "stft":
            return stft
        elif type == "l1":
            return l1
        else:
            return l1 + stft
    
    def train_loop(self):
        """
        Training Loop
        """
        size = len(self.train_loader.dataset)
        avg = 0
        for batch, (X, y) in enumerate(self.train_loader):

            #move to GPU
            X = X.to(self.device)
            y = y.to(self.device)

            # Compute prediction and loss
            #print(torch.cuda.memory_summary(device=None, abbreviated=False))
            pred = self.model(X)
            loss = self.loss_fn(pred, y)    

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()

            #Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=70.0)
            self.optimizer.step()

            loss, current = loss.item(), batch * len(X)
            avg += loss
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", end='\r')

        self.tb.add_scalar("Train Loss", avg / batch, self.current_epoch)

    def test_loop(self):
        """
        Test data on validation set
        """
        size = len(self.val_loader.dataset)
        num_batches = len(self.val_loader)
        test_loss = 0

        with torch.no_grad():
            for X, y in self.val_loader:
                #move to GPU
                X = X.to(self.device)
                y = y.to(self.device)
                
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()

        test_loss /= num_batches
        self.tb.add_scalar("Test Loss", test_loss, self.current_epoch)
        print(f"Val loss: {test_loss:>8f} \n")

    def loss_metrics(self):
        """
        Perform a single test iteration on validation dataset and returns l1, stft and combined loss
        """
        size = len(self.val_loader.dataset)
        num_batches = len(self.val_loader)
        l1 = 0
        stft = 0
        comb = 0

        with torch.no_grad():
            for X, y in self.val_loader:
                #move to GPU
                X = X.to(self.device)
                y = y.to(self.device)
                
                pred = self.model(X)
                l1 += self.loss_fn(pred, y, "l1").item()
                stft += self.loss_fn(pred, y, "stft").item()
                comb += self.loss_fn(pred, y).item()

                X.cpu().detach()
                y.cpu().detach()
                pred.cpu().detach()

        torch.cuda.empty_cache()
        l1 /= num_batches
        stft /= num_batches
        comb /= num_batches
        return l1, stft, comb


    def predict(self, audio, start=0, end=-1):
        """
        Apply model to audio
        """
        data = torch.tensor(audio)[None, start:end, None].to(self.device)
        output = self.model(data)
        output = output.cpu().detach().numpy()[0,:,0]
        return output

    def fit(self, epochs):
        """
        Run train and test loops for number of epochs.
        Autosaves model every epoch.
        """
        for self.current_epoch in range(epochs):
            print(f"Epoch {self.current_epoch+1}\n-------------------------------")
            self.train_loop()
            self.test_loop()
            self.save_model('autosave.pth')
        print("Done!")