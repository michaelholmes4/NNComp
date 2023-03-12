# Plugin
In this module is both code to load and use the C++ headers generated in `Training` as well as the iPlug2 project file used in the thesis project.

<p align="center">
<img width="500" alt="Screenshot" src="https://user-images.githubusercontent.com/99728921/224518583-933bb6e9-eb92-44fe-b582-400ca899a421.png">
</p>

## Contents
* `dsh.h` - Provides an example of how to load model header files and process an input sample 
* `layers.h` - C++ implementation of LSTM, GRU and RNN layer equations
* `pretrained_models` - Header files for pretrained models from the thesis project
* `iPlug2_Project` - iPlug2 project file for plugin

## Usage
### Use C++ implementation in own project
An efficient C++ implementation is provided for LSTM, RNN and GRU networks. They can be run directly using the generated `.h` files from the `Training` module. Alternatively, a model from `pretrained_models` can be used.
1. Include the model `.h` file in the project.
2. Download the Eigen linear algebra library [here](https://eigen.tuxfamily.org/).
3. Ensure the `Eigen/Dense` and `layers.h` are included in the project paths.

Audio can then be processed on sequential audio samples by repeatedly calling the `apply_model(x, y);` function in the model class. 

### iPlug2 Project
The iPlug2 library can be a bit tricky to build. These are the rough steps that need to be followed to build the plugin:

1. Clone [iPlug2](https://github.com/iPlug2/iPlug2).
2. Copy `iPlug2_Project` to `iPlug2/Project/`.
3. Rename `iPlug2_Project` folder to `NNCompressor`.
4. Open the `NNCompressor.xcworkspace` in Xcode or `NNCompressor.code-workspace` in Visual Studio in the project folder. 
5. Build the plugin. There may be additional steps required as documented in the iPlug2 documentation, such as downloading pre-built [dependencies](https://github.com/iPlug2/iPlug2/tree/master/Dependencies).

Pre-built versions of the plugin are available [here](https://files.hcloudh.com/s/NiYCKSnC34xxbSW).
