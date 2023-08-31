# Explainable AI in Drug Discovery: Self-interpretable Graph Neural Network for molecular property prediction using Concept Whitening

This is the official repository of the paper "Explainable AI in Drug Discovery: Self-interpretable Graph Neural Network for molecular property prediction using Concept Whitening. _Michela Proietti_,_Alessio Ragno_ _Biagio La Rosa_, _Rino Ragno_, and _Roberto Capobianco_. Springer Machine Learning Journal (2023)".

## Description
This repository implements the first concept-based explanability method for graph neural networks, obtained by adapting the code for (https://github.com/zhiCHEN96/ConceptWhitening "Concept Whitening").
It consists of a module which can be inserted straight after a convolutional layer in a GNN, in place of batch normalization layers. The module aligns the axes of the latent space with known concept of interest, in our case molecular properties that are known to influence the bioactivity of molecules.

## Usage
In the _config.yaml_ file in the config folder, you need to change the paths in the _base_dir_, _statistics_dir_, and _concept_dir_ items.
In order to use concept whitening, you will need to extract the molecular properties from the dataset's molecules.
This can be done by executing the command:
'''
python extract_molecular_concepts.py
'''

To train a new model or test a pre-trained one, execute the command
'''
python train_gnns.py
'''
after setting the desired parameters by changing the files in the config folder.

## Contact
If you have any question, do not hesitate to contact us at mproietti[at]diag[dot]uniroma1[dot]it.
