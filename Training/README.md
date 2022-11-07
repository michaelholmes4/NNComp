# Training
This module can be used to train RNN, GRU and LSTM networks on audio datasets.

## Contents
* `training.py` - Command line utility to train new models.
* `create_headers.py` - Create C++ header file from trained model `.pth` file.
* `model.py` - All model, dataset and training classes
* `create_all_headers.sh` - Shell script to create headers for all model in local directory

## Setup
1. Clone this repo and cd to the `Training` folder
2. Setup a new conda environment. An `environment.yml` file is supplied with required dependencies.

   ```
   conda env create -f environment.yml
   conda activate nncomp
   
   ```

## Usage
### Train a new model
1. Create a folder with training dataset. Folder must contain mono `.wav` files all with the same length. Files must be named `input****.wav` and `target****.wav` where `****` corresponds to matching input target pairs indices.
2. Run the train script using the cmd line arguments presented below.
3. Tensorboard is supported and training is saved to `./runs`. You can launch tensorboard using: `tensorboard --logdir ./`
4. Once training has finished, the model will be saved as `.pth` in the specified directory. Additionally every epoch an `autosave.pth` file is also created.

Parameters for `training.py`

| Parameter                  | Short |          | Default                   | Description |
| ----------------           | ----- | -------- | ------------------------- | ----------- |
| _type_                     |       | required |                           | Model type eg,: lstm-32-1 format: type-hidden-layers types:lstm,gru,rnn hidden:8,16,32 layers:1,2,4|
| _--path_                   |    -p | optional | dataset/                  | Path to dataset folder |
| _--epochs_                 | -e    | optional | 1000                      | Number of epochs to train for |
| _--batch_size_             | -b    | optional | 32                        | Training batch size |
| _--out_                    | -o    | optional | out/                      | Output path to save model |

Eg:
```bash
$ python3 training.py lstm-32-1
```

### Predict Audio
To be added...

### Create C++ Model
`create_headers.py` can be used to generate a C++ header file from a saved `.pth` model to use with the supplied `Plugin` module. 

Parameters for `create_headers.py`

| Parameter                  | Short |          | Default                   | Description |
| ----------------           | ----- | -------- | ------------------------- | ----------- |
| _path_                     |       | required |                           | Path to `.pth` model |

Eg:
```bash
$ python3 generate_headers.py ./out/some_model.pth
```