# Real-Time Virtual Analog Modelling of an Audio Compressor Using Recurrent Neural Networks
This is the code base for the undergraduate thesis project submitted to the University of Queensland by Michael Holmes.
## Introduction
This codebase can be used to generate audio training data for virtual analog modelling, train RNN, LSTM and GRU neural networks and convert these models into efficient C++ code for use in audio plugins. 

An audio plugin was also created using [iPlug2](https://github.com/iPlug2/iPlug2) and the project folder is supplied. Alternatively, pre-built binaries for the plugin can be downloaded [here](https://hcloudh.com/nextcloud/s/mxBCBoyYKQzmjcL) for MacOs.

A large amount of inspiration was taken from the fantastic [paper](https://arxiv.org/abs/2102.06200) by Christian Steinmetz and Joshua Reiss. I tried to expand upon this work with my own ideas.

## Usage
### Training
