# Explainable AI in Drug Discovery: Self-interpretable Graph Neural Network for molecular property prediction using Concept Whitening

This is the official repository of the paper "Explainable AI in Drug Discovery: Self-interpretable Graph Neural Network for molecular property prediction using Concept Whitening. _Michela Proietti_,_Alessio Ragno_ _Biagio La Rosa_, _Rino Ragno_, and _Roberto Capobianco_. Machine Learning (2024)".

## Description
This repository implements the first concept-based explanability method for graph neural networks, obtained by adapting the code for (https://github.com/zhiCHEN96/ConceptWhitening "Concept Whitening").
It consists of a module which can be inserted straight after a convolutional layer in a GNN, in place of batch normalization layers. The module aligns the axes of the latent space with known concept of interest, in our case molecular properties that are known to influence the bioactivity of molecules.

## Environment setup
1) Download and install <a href="https://docs.docker.com/engine/install/">Docker</a> and <a href="https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html">Anaconda</a>.
2) Clone this repo:
```
git clone https://github.com/KRLGroup/GraphCW.git
```
4) Move inside the Dockerfile directory:
```
cd Dockerfile
```
6) Build the custom image:
```
docker build -t graphcw:1.0 .
```
8) Run the docker machine, substituting the absolute path to the cloned repo to <path_to_project_dir>:
```
docker run -v <path_to_project_dir>:<path_to_project_dir> -w <path_to_project_dir> -it --gpus all graphcw:1.0 bash
```
10) Inside the container, run the following commands to create a conda environment with the required libraries:
```
conda create -n graphcw python==3.9
conda activate graphcw
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install pyg -c pyg
pip install hydra-core==1.1.1 omegaconf==2.1.1 numpy==1.22 pandas==1.4.1 captum==0.2.0 networkx==2.7.1 scikit-learn==0.24.2 scipy==1.10.1 seaborn==0.11.2
pip install zipp==3.20 Pillow==9.0.1 matplotlib==3.5.1 matplotlib-inline==0.1.2 tqdm==4.63.0 scikit-image==0.18.3
pip install dive-into-graphs==0.1.2 deepchem==2.8.0 rdkit-pypi==2021.9.5.1
```

## Usage
Change the _config.yaml_ file to design your own experiments.
In order to use concept whitening, you will need to extract the molecular properties from the dataset's molecules.
This can be done by executing the command:
```
python extract_molecular_concepts.py
```

To train a new model or test a pre-trained one, execute the command
```
python train_gnns.py
```
after setting the desired parameters by changing the files in the config folder.

## Citation
```

@article{proietti2024explainable,
  title={Explainable AI in drug discovery: self-interpretable graph neural network for molecular property prediction using concept whitening},
  author={Proietti, Michela and Ragno, Alessio and Rosa, Biagio La and Ragno, Rino and Capobianco, Roberto},
  journal={Machine Learning},
  volume={113},
  number={4},
  pages={2013--2044},
  year={2024},
  publisher={Springer}
}
```

## Contact
If you have any question, do not hesitate to contact us at mproietti[at]diag[dot]uniroma1[dot]it.
