# Project Title

A simple demo implementation of IUPG to train an MNIST classifier with or without noise.

## Prerequisites

Use Python 3. This project has been tested on a Linux CPU-only environment and with CUDA-capable GPUs.

This project has been tested with the following package settings:

* tensorflow==1.15.4
* scikit-learn==0.23.1
* Pillow==7.2.0
* matplotlib==3.2.2
* progress==1.5
* argparse==1.4.0
* mnist==0.2.2
* scipy==1.5.1
* pandas==1.0.5
* seaborn==0.10.1
* configparse==0.1.5

## Getting the datasets

Run

```
python3 get_data.py
```

To create and download all the datasets to this project directory. All datasets (stored as compressed Numpy arrays with extension .npz) will be stored inside `data/`. In total, 9 files will be created. The train, validation, and test splits will be saved. Versions of train, val, and test with Gaussian noise and random stroke noise seperately will also be saved. Global variables inside `get_data.py` control this process.

## Training a model

Use `train_IUPG.py` to train a model. This script takes in a configuration file. A template as well as two example config files are provided in `configs/`. An example call is

```
python3 train_IUPG.py --config_files train_without_noise
```

The above call will train an IUPG model and save everything to the directory specified in `save_dir` in the config file. With this example config file, all models and results are saved to `cnn_results/no_noise/`. To train on GPU 0 on your machine, add the following flag as such.

```
python3 train_IUPG.py --config_files train_without_noise --gpu_id 0
```

The resulting directory contains several self-explanatory CSV files which summarize performance. Additionally,

* `kmeans_plots` will contain the prototype initializations which were discovered by clustering if that option was chosen.
* `models` will contain the snapshot of the optimal model found during training.
* `perf_plots` will contain accompanying plots of the summary files.
* `proto_plots` will contain snapshot plots of the prototypes with a frequency that is specified in the config file.
* `tsne_plots` will contain snapshot plots of some training data in the output vector space with a frequency that is specified in the config file. t-SNE is used to visualize this high dimensional space. You may need to edit the t-SNE parameters inside `IUPG_Builder.py` to get this to work well.

## Running inference


## Analyzing results


## License

This project is licensed under the MIT License - see the LICENSE.md file for details
