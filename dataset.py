import os
import torch
import numpy as np
from torch.utils.data import random_split, Subset
from torch_geometric.loader import DataLoader
from dig.xgraph.dataset import MoleculeDataset, SynGraphDataset, SentiGraphDataset, BA_LRP
from torch import default_generator
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split


def get_dataset(dataset_root, dataset_name, pre_filter=None):
    if dataset_name.lower() in list(MoleculeDataset.names.keys()):
        return MoleculeDataset(root=dataset_root, name=dataset_name, pre_filter=pre_filter)
    elif dataset_name.lower() in ['graph_sst2', 'graph_sst5', 'twitter']:
        return SentiGraphDataset(root=dataset_root, name=dataset_name)
    elif dataset_name.lower() in list(SynGraphDataset.names.keys()):
        return SynGraphDataset(root=dataset_root, name=dataset_name)
    elif dataset_name.lower() in ['ba_lrp']:
        return BA_LRP(root=dataset_root)
    else:
        raise ValueError(f"{dataset_name} is not defined.")


def get_dataloader(dataset, batch_size, stratified, random_split_flag=True, data_split_ratio=None, seed=2):
    """
    Args:
        dataset:
        batch_size: int
        random_split_flag: bool
        data_split_ratio: list, training, validation and testing ratio
        seed: random seed to split the dataset randomly
    Returns:
        a dictionary of training, validation, and testing dataLoader
    """
    dataloader = dict()
    
    if not stratified:
        if not random_split_flag and hasattr(dataset, 'supplement'):
            assert 'split_indices' in dataset.supplement.keys(), "split idx"
            split_indices = dataset.supplement['split_indices']
            train_indices = torch.where(split_indices == 0)[0].numpy().tolist()
            dev_indices = torch.where(split_indices == 1)[0].numpy().tolist()
            test_indices = torch.where(split_indices == 2)[0].numpy().tolist()

            train = Subset(dataset, train_indices)
            eval = Subset(dataset, dev_indices)
            test = Subset(dataset, test_indices)
        else:
            num_train = int(data_split_ratio[0] * len(dataset))
            num_eval = int(data_split_ratio[1] * len(dataset))
            num_test = len(dataset) - num_train - num_eval

            train, eval, test = random_split(dataset,
                                             lengths=[num_train, num_eval, num_test],
                                             generator=default_generator)
            
        dataloader['train'] = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
        dataloader['eval'] = DataLoader(eval, batch_size=batch_size, shuffle=False, drop_last=True)
        dataloader['test'] = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)
    
    else:
        
        targets = []
        for sample in dataset:
            targets.append(torch.argmax(sample.y).item())

        train_idx, test_idx = train_test_split(range(len(targets)),
                                                test_size=(1-data_split_ratio[0]),
                                                random_state=seed,
                                                shuffle=True,
                                                stratify=targets)

        test_targets = []
        for idx in test_idx:
            test_targets.append(targets[idx])

        if data_split_ratio[1] == data_split_ratio[2]:

            valid_idx, test_idx = train_test_split(range(len(test_targets)),
                                                    test_size=0.5,
                                                    random_state=seed,
                                                    shuffle=True,
                                                    stratify=test_targets)

        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

        dataloader = dict()
        dataloader['train'] = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
        dataloader['test'] = DataLoader(dataset, batch_size=1, sampler=test_sampler, drop_last=True)

        if data_split_ratio[1] == data_split_ratio[2]:
            valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)
            dataloader['eval'] = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, drop_last=True)
    
    return dataloader

def get_concept_dataloader(concept_dir:str, concepts:str, batch_size:int):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    concept_loaders = [
                torch.utils.data.DataLoader(
                datasets.ImageFolder(os.path.join(concept_dir, concept), transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=False)
                for concept in concepts.split(',')
            ]
    
    return concept_loaders
