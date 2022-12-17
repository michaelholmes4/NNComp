# Real-Time Virtual Analog Modelling of an Audio Compressor Using Recurrent Neural Networks
<p align="center">
  <img src="https://user-images.githubusercontent.com/99728921/200828493-54f9ee7d-5127-4c0e-a326-13283a6a058b.png" width="600">
</p>

This is the code base for the Honours [thesis](https://github.com/michaelholmes4/NNComp/files/10250577/Thesis.pdf) project submitted to the University of Queensland by Michael Holmes 2022.

### Acknowledgements
The following papers were highly influential in guiding this project:
* [Efficient neural networks for real-time modeling of analog dynamic range compression](https://arxiv.org/abs/2102.06200) by Christian Steinmetz and Joshua Reiss.
* [Real-time black-box modelling with recurrent neural networks](http://dafx.de/paper-archive/2019/DAFx2019_paper_43.pdf) by Alec Wright, Eero-Pekka Damskägg, and Vesa Välimäki 

## Introduction
This repo can be used to train RNN, LSTM and GRU neural networks and convert these networks into efficient C++ code for use in audio plugins. 

The trained models from the thesis project can be downloaded [here](https://hcloudh.com/nextcloud/s/9rWEe6EHQq4x4DS/download/All_Models.zip).

A demo audio plugin was also created using [iPlug2](https://github.com/iPlug2/iPlug2) and can be downloaded [here](https://hcloudh.com/nextcloud/s/mxBCBoyYKQzmjcL).


## Contents
This repo is split into 2 modules: `Training` and `Plugin`. Detailed usage instructions are available inside each module.

### Training
Code for training and testing the PyTorch models. A script is supplied for converting these models into C++ headers to use with the `Plugin` module.

### Plugin
The iPlug2 project file is supplied along with quick C++ implementations that can be used in other projects.  
