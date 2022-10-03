# Real-Time Virtual Analog Modelling of an Audio Compressor Using Recurrent Neural Networks
This is the code base for the undergraduate thesis project submitted to the University of Queensland by Michael Holmes.

#### Acknowledgements
A large amount of inspiration was taken from the fantastic [paper](https://arxiv.org/abs/2102.06200) by Christian Steinmetz and Joshua Reiss. I tried to expand upon this work with my own ideas.

## Introduction
This codebase can be used to generate audio training data for virtual analog modelling, train RNN, LSTM and GRU neural networks and convert these networks into efficient C++ code for use in audio plugins. Additionally the pre-trained models are also supplied.

An audio plugin was also created using [iPlug2](https://github.com/iPlug2/iPlug2). The project folder for this is also supplied. Alternatively, pre-built binaries for the plugin can be downloaded [here](https://hcloudh.com/nextcloud/s/mxBCBoyYKQzmjcL) (for MacOs only at this point.)


## Contents
This repo is split into 3 modules/folders: Training, DatasetTools and Plugin. 

### Training
