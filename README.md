# Real-Time Virtual Analog Modelling of an Audio Compressor Using Recurrent Neural Networks
![](https://drive.hcloudh.com/s/fMMaQw5z6gT4iea/download/Poster1.pdf)
This is the code base for the Honours thesis project submitted to the University of Queensland by Michael Holmes 2022.

#### Acknowledgements
The model creation and training was largely inspired by the fantastic [paper](https://arxiv.org/abs/2102.06200) by Christian Steinmetz and Joshua Reiss.

## Introduction
This repo can be used to generate audio training data for virtual analog modelling, train RNN, LSTM and GRU neural networks and convert these networks into efficient C++ code for use in audio plugins. Additionally pre-trained models are also supplied.

An audio plugin was also created using [iPlug2](https://github.com/iPlug2/iPlug2). The project folder for this is also supplied. Alternatively, pre-built binaries for the plugin can be downloaded [here](https://hcloudh.com/nextcloud/s/mxBCBoyYKQzmjcL).


## Contents
This repo is split into 3 modules/folders: Training, DatasetTools and Plugin. 

### Training
