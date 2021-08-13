# Attention-EEG

Set of models for attention estimation from EEG: transformer EEG, RNN and resnet based CNN.

## Instruction

The three proposed models are direcly available in [models.py](models.py):
* Transformer based approach as described in the paper [MultiTransformer.ipynb](MultiTransformer.ipynb). 
* Multi dimensional RNN [MultiTransformer.ipynb](MultiTransformer.ipynb).
* Resnet based CNN [CNN_EEG.ipynb](CNN_EEG.ipynb).

The considered inputs for the two considered datasets are proposed in the corresponding directory (car for the driving EEG dataset and phydaa for name related dataset). The file feature file for the first dataset being too voluminous and in a concern of reproducibility, we provide also the preprocessing scripts to extract the differential entropy feature matrices (preprocessing/). For the CNN approach, it is necessary to first generate the image by running [CNN_EEG.ipynb](CNN_EEG.ipynb) for the first time.

During the training, the metrics evolution are reported in runs directory with tensorboard (https://www.tensorflow.org/tensorboard/) and the final training results are saved in res/. 

## Installation and Dependencies

[Pytorch 1.7](https://pytorch.org/get-started/locally)

[MNE](https://mne.tools/stable/install/mne_python.html#install-python-and-mne-python)

[Cuda 10.2](https://developer.nvidia.com/cuda-toolkit)

Installation with pip: `pip install -r requirement.txt`

Import of the environment with conda: `conda env create -f environment.yml`

## Remarks

If you are interested in our work, don't hesitate to contact us. 

Wish you the best in your research projects!
