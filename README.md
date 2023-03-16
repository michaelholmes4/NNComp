# Real-Time Virtual Analog Modelling of an Audio Compressor Using Recurrent Neural Networks
This is the code base for the Honours [thesis](https://files.hcloudh.com/s/zrpJqozJeN6Y25c) project submitted to the University of Queensland by Michael Holmes 2022.

<p align="center">
  <img src="https://user-images.githubusercontent.com/99728921/200828493-54f9ee7d-5127-4c0e-a326-13283a6a058b.png" width="400">
</p>

### Update 2023
New GUI!!  
Now the plugin has a modern look and the weights of each neural network can be visualised in real-time.
<p align="center">
<img width="600" alt="Screenshot" src="https://user-images.githubusercontent.com/99728921/224527763-4ca6901a-88c7-493e-839e-b351a2d7af7a.png">
</p>

### Acknowledgements
The following papers were highly influential in guiding this project:
* [Efficient neural networks for real-time modeling of analog dynamic range compression](https://arxiv.org/abs/2102.06200) by Christian Steinmetz and Joshua Reiss.
* [Real-time black-box modelling with recurrent neural networks](http://dafx.de/paper-archive/2019/DAFx2019_paper_43.pdf) by Alec Wright, Eero-Pekka Damskägg, and Vesa Välimäki 

## Introduction
This repo can be used to train RNN, LSTM and GRU neural networks and convert these networks into efficient C++ code for use in audio plugins. 

The trained models from the thesis project can be downloaded [here](https://files.hcloudh.com/s/qcNFYJYneXgY8X5).

An audio plugin was created using [iPlug2](https://github.com/iPlug2/iPlug2) and can be downloaded in the releases tab.


## Contents
This repo is split into 2 modules: `Training` and `Plugin`. Detailed usage instructions are available inside each module.

### Training
Code for training and testing the PyTorch models. A script is supplied for converting these models into C++ headers to use with the `Plugin` module.

### Plugin
The iPlug2 project file is supplied along with quick C++ implementations that can be used in other projects.  
