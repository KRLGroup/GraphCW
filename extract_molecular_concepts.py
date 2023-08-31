import os
import re
import shutil
import json
import random
import hydra
import torch
import pickle
import zipfile
import gzip
from dig.xgraph.dataset import MoleculeDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset, InMemoryDataset, download_url
from dataset import get_dataset

from rdkit.Chem.Descriptors import MolWt, NumValenceElectrons, NumRadicalElectrons, TPSA, MolLogP # Or MolWt for the average molecular weight of the molecule
from rdkit import Chem

mean_mol_weight = 0
mean_val_electrons = 0

x_map = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
}

def maybe_log(path, log=True):
    if log:
        print('Extracting', path)


def extract_gz(path, folder, log=True):
    maybe_log(path, log)
    with gzip.open(path, 'r') as r:
        with open(os.path.join(folder, '.'.join(os.path.basename(path).split('.')[:-1])), 'wb') as w:
            w.write(r.read())


def extract_zip(path, folder, log=True):
    r"""Extracts a zip archive to a specific folder.

    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    maybe_log(path, log)
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)
        

class ConceptDataset(InMemoryDataset):
    
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{}'
    mutag_url = 'https://github.com/divelab/DIG_storage/raw/main/xgraph/datasets/{}'

    # Format: name: [display_name, url_name, filename, smiles_idx, y_idx]
    names = {
        'mutag': ['MUTAG', 'MUTAG.zip', None, None],
        'esol': ['ESOL', 'delaney-processed.csv', 'delaney-processed.csv', -1, -2],
        'freesolv': ['FreeSolv', 'SAMPL.csv', 'SAMPL.csv', 1, 2],
        'lipo': ['Lipophilicity', 'Lipophilicity.csv', 'Lipophilicity.csv', 2, 1],
        'pcba': ['PCBA', 'pcba.csv.gz', 'pcba.csv', -1,
                 slice(0, 128)],
        'muv': ['MUV', 'muv.csv.gz', 'muv.csv', -1,
                slice(0, 17)],
        'hiv': ['HIV', 'HIV.csv', 'HIV.csv', 0, -1],
        'bace': ['BACE', 'bace.csv', 'bace.csv', 0, 2],
        'bbbp': ['BBBP', 'BBBP.csv', 'BBBP.csv', -1, -2],
        'tox21': ['Tox21', 'tox21.csv.gz', 'tox21.csv', -1,
                  slice(0, 12)],
        'toxcast':
        ['ToxCast', 'toxcast_data.csv.gz', 'toxcast_data.csv', 0,
         slice(1, 618)],
        'sider': ['SIDER', 'sider.csv.gz', 'sider.csv', 0,
                  slice(1, 28)],
        'clintox': ['ClinTox', 'clintox.csv.gz', 'clintox.csv', 0,
                    slice(1, 3)],
    }
    
    
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        
        if Chem is None:
            raise ImportError('`MoleculeNet` requires `rdkit`.')

        self.name = name.lower()
        assert self.name in self.names.keys()
        
        super(ConceptDataset, self).__init__(root, transform, pre_transform, pre_filter)
        
        self.data = torch.load(self.processed_paths[0])
        
    
    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        if self.name.lower() == 'MUTAG'.lower():
            return ['MUTAG_A.txt', 'MUTAG_graph_labels.txt', 'MUTAG_graph_indicator.txt',
                    'MUTAG_node_labels.txt', 'README.txt']
        else:
            return self.names[self.name][2]
    
    @property
    def processed_file_names(self):
         with open(self.raw_paths[0], 'r') as f:
                dataset = f.read().split('\n')[1:-1]
                dataset = [x for x in dataset if len(x) > 0]
                
                return [f'data_{idx}.pt' for idx in range(len(dataset))]
            
    def download(self):
        if self.name.lower() == 'MUTAG'.lower():
            url = self.mutag_url.format(self.names[self.name][1])
        else:
            url = self.url.format(self.names[self.name][1])
        path = download_url(url, self.raw_dir)
        if self.names[self.name][1][-2:] == 'gz':
            extract_gz(path, self.raw_dir)
            os.unlink(path)
        elif self.names[self.name][1][-3:] == 'zip':
            extract_zip(path, self.raw_dir)
            os.unlink(path)
    
    def process(self):
        if self.name.lower() == 'MUTAG'.lower():
            with open(os.path.join(self.raw_dir, 'MUTAG_node_labels.txt'), 'r') as f:
                nodes_all_temp = f.read().splitlines()
                nodes_all = [int(i) for i in nodes_all_temp]

            adj_all = np.zeros((len(nodes_all), len(nodes_all)))
            with open(os.path.join(self.raw_dir, 'MUTAG_A.txt'), 'r') as f:
                adj_list = f.read().splitlines()
            for item in adj_list:
                lr = item.split(', ')
                l = int(lr[0])
                r = int(lr[1])
                adj_all[l - 1, r - 1] = 1

            with open(os.path.join(self.raw_dir, 'MUTAG_graph_indicator.txt'), 'r') as f:
                graph_indicator_temp = f.read().splitlines()
                graph_indicator = [int(i) for i in graph_indicator_temp]
                graph_indicator = np.array(graph_indicator)

            with open(os.path.join(self.raw_dir, 'MUTAG_graph_labels.txt'), 'r') as f:
                graph_labels_temp = f.read().splitlines()
                graph_labels = [int(i) for i in graph_labels_temp]

            self.data_list = []
            for i in range(1, 189):
                idx = np.where(graph_indicator == i)
                graph_len = len(idx[0])
                adj = adj_all[idx[0][0]:idx[0][0] + graph_len, idx[0][0]:idx[0][0] + graph_len]
                label = int(graph_labels[i - 1] == 1)
                feature = nodes_all[idx[0][0]:idx[0][0] + graph_len]
                nb_clss = 7
                targets = np.array(feature).reshape(-1)
                one_hot_feature = np.eye(nb_clss)[targets]
                data_example = Data(x=torch.from_numpy(one_hot_feature).float(),
                                    edge_index=dense_to_sparse(torch.from_numpy(adj))[0],
                                    y=label)
                self.data_list.append(data_example)
        else:
            with open(self.raw_paths[0], 'r') as f:
                dataset = f.read().split('\n')[1:-1]
                dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

            self.data_list = []
            for line in dataset:
                line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
                line = line.split(',')

                smiles = line[self.names[self.name][3]]
                ys = line[self.names[self.name][4]]
                ys = ys if isinstance(ys, list) else [ys]

                ys = [float(y) if len(y) > 0 else float('NaN') for y in ys]
                y = torch.tensor(ys, dtype=torch.float).view(1, -1)

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                xs = []
                for atom in mol.GetAtoms():
                    x = []
                    x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
                    x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
                    x.append(x_map['degree'].index(atom.GetTotalDegree()))
                    x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
                    x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
                    x.append(x_map['num_radical_electrons'].index(
                        atom.GetNumRadicalElectrons()))
                    x.append(x_map['hybridization'].index(
                        str(atom.GetHybridization())))
                    x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
                    x.append(x_map['is_in_ring'].index(atom.IsInRing()))
                    xs.append(x)

                x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

                edge_indices, edge_attrs = [], []
                for bond in mol.GetBonds():
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()

                    e = []
                    e.append(e_map['bond_type'].index(str(bond.GetBondType())))
                    e.append(e_map['stereo'].index(str(bond.GetStereo())))
                    e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

                    edge_indices += [[i, j], [j, i]]
                    edge_attrs += [e, e]

                edge_index = torch.tensor(edge_indices)
                edge_index = edge_index.t().to(torch.long).view(2, -1)
                edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

                # Sort indices.
                if edge_index.numel() > 0:
                    perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                    edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                            smiles=smiles)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                self.data_list.append(data)
                
        for idx, sample in enumerate(self.data_list):
            torch.save(sample, os.path.join(self.processed_dir, f'data_{idx}.pt'))
            
    def __repr__(self):
        return '{}({})'.format(self.names[self.name][0], len(self))
            
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

def molecular_weight_filtering(sample: Data):
    mol_weight = MolWt(Chem.MolFromSmiles(sample.smiles))
    
    if mol_weight < 500: # 1st Lipinski rule # mean_mol_weight:
        return True
 
    return False

def valence_electrons_filtering(sample: Data):
    val_electrons = NumValenceElectrons(Chem.MolFromSmiles(sample.smiles))
    
    if val_electrons > mean_val_electrons:
        return True
 
    return False

def radical_electrons_filtering(sample: Data):
    rad_electrons = NumRadicalElectrons(Chem.MolFromSmiles(sample.smiles))
    
    if rad_electrons > mean_rad_electrons:
        return True
 
    return False

def tpsa_filtering(sample: Data):
    tpsa = TPSA(Chem.MolFromSmiles(sample.smiles))
    
    if tpsa > mean_tpsa:
        return True
 
    return False

def logp_filtering(sample: Data):
    logp = MolLogP(Chem.MolFromSmiles(sample.smiles))
    
    if logp < 5:
        return True
 
    return False

def qed_filtering(sample: Data):
    qed = Chem.QED.qed(Chem.MolFromSmiles(sample.smiles))
    
    if qed > mean_qed:
        return True
 
    return False

def hbd_filtering(sample: Data):
    hbd = Chem.rdMolDescriptors.CalcNumHBD(Chem.MolFromSmiles(sample.smiles))
        
    if hbd < 5:
        return True
 
    return False

def hba_filtering(sample: Data):
    hba = Chem.rdMolDescriptors.CalcNumHBA(Chem.MolFromSmiles(sample.smiles))
        
    if hba < 10:
        return True
    
    return False

def heteroatoms_filtering(sample: Data):
    heteroatoms = Chem.rdMolDescriptors.CalcNumHeteroatoms(Chem.MolFromSmiles(sample.smiles))
        
    if heteroatoms < mean_n_heteroatoms:
        return True
    
    return False

def aliphatic_heterocycles_hydroxyGroups_filtering(sample: Data):
    hydroxy_groups = Chem.Fragments.fr_Al_OH_noTert(Chem.MolFromSmiles(sample.smiles))
    heterocycles = Chem.rdMolDescriptors.CalcNumAliphaticHeterocycles(Chem.MolFromSmiles(sample.smiles))
        
    if (heterocycles < 3 and hydroxy_groups < 2) or (heterocycles < 2 and (hydroxy_groups == 2 or hydroxy_groups == 5)) or (heterocycles > 2 and (hydroxy_groups == 0 or hydroxy_groups == 8)):
        return True
    
    return False
                                                                                                                           
def compute_nO(sample: Data):
    nO = 0
    mol = Chem.MolFromSmiles(sample.smiles)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "O":
            nO += 1

    return nO
                                                                                                                           
def nO_filtering(sample: Data):
    nO = compute_nO(sample)
    
    if nO < mean_nO:
        return True
                                                                                                                           
    return False

def NOCount_filtering(sample: Data):
    NOCount = Chem.Lipinski.NOCount(Chem.MolFromSmiles(sample.smiles))
    
    if NOCount < mean_NOCount:
        return True
    
    return False

def NHOHCount_filtering(sample: Data):
    NHOHCount = Chem.Lipinski.NHOHCount(Chem.MolFromSmiles(sample.smiles))
    
    if NHOHCount < mean_NHOHCount:
        return True
    
    return False


## Just for toxicity
def compute_nCl(sample: Data):
    nCl = 0
    mol = Chem.MolFromSmiles(sample.smiles)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "Cl":
            nCl += 1
            
    return nCl
    
def nCl_filtering(sample: Data):
    nCl = compute_nCl(sample)
    
    if nCl < mean_nCl:
        return True
                                                                                                                           
    return False
    
    
def compute_nBondsD2(sample: Data):
    bond_number = 0
    
    mol = Chem.MolFromSmiles(sample.smiles)
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            bond_number += 1
    
    return bond_number
    
def n2bonds_filtering(sample: Data):
    n2bonds = compute_nBondsD2(sample)
    
    if n2bonds < mean_n2bonds:
        return True
                                                                                                                           
    return False

def compute_aromatic_bonds(sample: Data):
    bond_number = 0
    
    mol = Chem.MolFromSmiles(sample.smiles)
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.AROMATIC:
            bond_number += 1
    
    return bond_number
    
def aromatic_bonds_filtering(sample: Data):
    aromatic_bonds = compute_aromatic_bonds(sample)
    
    if aromatic_bonds < mean_aromatic_bonds:
        return True
                                                                                                                           
    return False

def crippen_logp_filtering(sample: Data):
    logp,_ = Chem.rdMolDescriptors.CalcCrippenDescriptors(Chem.MolFromSmiles(sample.smiles))
    
    if logp < mean_crippen_logp:
        return True
    else:
        return False
    
def crippen_mr_filtering(sample: Data):
    _,mr = Chem.rdMolDescriptors.CalcCrippenDescriptors(Chem.MolFromSmiles(sample.smiles))
    
    if mr < mean_crippen_mr:
        return True
    else:
        return False
    
def complexity_filtering(sample: Data):
    complexity = Chem.GraphDescriptors.BertzCT(Chem.MolFromSmiles(sample.smiles))
    
    if complexity < mean_complexity:
        return True
    else:
        return False
    

def rotable_bonds_filtering(sample: Data):
    rot_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(Chem.MolFromSmiles(sample.smiles), True)
    
    if rot_bonds < mean_rotable_bonds:
        return True
    else:
        return False
    
def heavy_atoms_filtering(sample: Data):
    heavy_atoms = Chem.rdMolDescriptors.CalcNumHeavyAtoms(Chem.MolFromSmiles(sample.smiles))
    
    if heavy_atoms < mean_heavy_atoms:
        return True
    else:
        return False
    
def n_rings_filtering(sample: Data):
    n_rings = Chem.rdMolDescriptors.CalcNumRings(Chem.MolFromSmiles(sample.smiles))
    
    if n_rings < mean_n_rings:
        return True
    else:
        return False
    
def hall_kier_alpha_filtering(sample: Data):
    HallKierAlpha = Chem.rdMolDescriptors.CalcHallKierAlpha(Chem.MolFromSmiles(sample.smiles))
    
    if HallKierAlpha < mean_HallKierAlpha:
        return True
    else:
        return False
    
def aromatic_rings_filtering(sample: Data):
    aromatic_rings = Chem.rdMolDescriptors.CalcNumAromaticRings(Chem.MolFromSmiles(sample.smiles))
    
    if aromatic_rings < mean_aromatic_rings:
        return True
    else:
        return False
    
def amide_bonds_filtering(sample: Data):
    amide_bonds = Chem.rdMolDescriptors.CalcNumAmideBonds(Chem.MolFromSmiles(sample.smiles))
    
    if amide_bonds < mean_amide_bonds:
        return True
    else:
        return False
    
    
def aromatic_heterocycles_filtering(sample: Data):
    aromatic_heterocycles = Chem.rdMolDescriptors.CalcNumAromaticHeterocycles(Chem.MolFromSmiles(sample.smiles))
    
    if aromatic_heterocycles < mean_aromatic_heterocycles:
        return True
    else:
        return False
    
def aromatic_carbocycles_filtering(sample: Data):
    aromatic_carbocycles = Chem.rdMolDescriptors.CalcNumAromaticCarbocycles(Chem.MolFromSmiles(sample.smiles))
    
    if aromatic_carbocycles < mean_aromatic_carbocycles:
        return True
    else:
        return False
                                                                                                                          

@hydra.main(config_path="config", config_name="config")
def main(config):
    
    dataset = get_dataset(dataset_root=config.datasets.dataset_root,
                          dataset_name=config.datasets.dataset_name)
    
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    
    valence_electrons = []
    radical_electrons = []
    tpsas = []
    qeds = []
    n_heteroatoms = []
    nOs = []
    NOCounts = []
    NHOHCounts = []

    crippen_logps = []
    crippen_mrs = []
    nCls = []
    nDoubleBonds = []
    nAromaticBonds = []
    complexities = []
    rotable_bonds = []
    heavy_atoms = []
    n_rings = []
    HallKierAlphas = []
    aromatic_rings = []
    amide_bonds = []
    aromatic_heterocycles = []
    aromatic_carbocycles = []

    for sample in dataset:
        smiles = sample.smiles
        val_electrons = NumValenceElectrons(Chem.MolFromSmiles(smiles))
        rad_electrons = NumRadicalElectrons(Chem.MolFromSmiles(smiles))
        tpsa = TPSA(Chem.MolFromSmiles(smiles))
        qed = Chem.QED.qed(Chem.MolFromSmiles(smiles))
        n_heteroatom = Chem.rdMolDescriptors.CalcNumHeteroatoms(Chem.MolFromSmiles(sample.smiles))
        nO = compute_nO(sample)
        NOCount = Chem.Lipinski.NOCount(Chem.MolFromSmiles(sample.smiles))
        NHOHCount = Chem.Lipinski.NHOHCount(Chem.MolFromSmiles(sample.smiles))

        crippen_logp, crippen_mr = Chem.rdMolDescriptors.CalcCrippenDescriptors(Chem.MolFromSmiles(sample.smiles))
        nCl = compute_nCl(sample)
        nDoubleBond = compute_nBondsD2(sample)
        nAromaticBond = compute_aromatic_bonds(sample)
        complexity = Chem.GraphDescriptors.BertzCT(Chem.MolFromSmiles(smiles))
        rotable_bond = Chem.rdMolDescriptors.CalcNumRotatableBonds(Chem.MolFromSmiles(sample.smiles), True)
        heavy_atom = Chem.rdMolDescriptors.CalcNumHeavyAtoms(Chem.MolFromSmiles(sample.smiles))
        n_ring = Chem.rdMolDescriptors.CalcNumRings(Chem.MolFromSmiles(sample.smiles))
        HallKierAlpha = Chem.rdMolDescriptors.CalcHallKierAlpha(Chem.MolFromSmiles(sample.smiles))
        aromatic_ring = Chem.rdMolDescriptors.CalcNumAromaticRings(Chem.MolFromSmiles(sample.smiles))
        amide_bond = Chem.rdMolDescriptors.CalcNumAmideBonds(Chem.MolFromSmiles(sample.smiles))
        aromatic_heterocycle = Chem.rdMolDescriptors.CalcNumAromaticHeterocycles(Chem.MolFromSmiles(sample.smiles))
        aromatic_carbocycle = Chem.rdMolDescriptors.CalcNumAromaticCarbocycles(Chem.MolFromSmiles(sample.smiles))

        valence_electrons.append(val_electrons)
        radical_electrons.append(rad_electrons)
        tpsas.append(tpsa)
        qeds.append(qed)
        n_heteroatoms.append(n_heteroatom)
        nOs.append(nO)
        NOCounts.append(NOCount)
        NHOHCounts.append(NHOHCount)

        crippen_logps.append(crippen_logp)
        crippen_mrs.append(crippen_mr)
        nCls.append(nCl)
        nDoubleBonds.append(nDoubleBond)
        nAromaticBonds.append(nAromaticBond)
        complexities.append(complexity)
        rotable_bonds.append(rotable_bond)
        heavy_atoms.append(heavy_atom)
        n_rings.append(n_ring)
        HallKierAlphas.append(HallKierAlpha)
        aromatic_rings.append(aromatic_ring)
        amide_bonds.append(amide_bond)
        aromatic_heterocycles.append(aromatic_heterocycle)
        aromatic_carbocycles.append(aromatic_carbocycle)
        

    global mean_val_electrons, mean_rad_electrons, mean_tpsa, mean_qed, mean_n_heteroatoms, mean_nO, mean_NOCount, mean_NHOHCount
    global mean_crippen_logp, mean_crippen_mr, mean_n2bonds, mean_aromatic_bonds, mean_nCl, mean_complexity, mean_rotable_bonds, mean_heavy_atoms, mean_n_rings, mean_HallKierAlpha
    global mean_aromatic_heterocycles, mean_aromatic_carbocycles, mean_aromatic_rings, mean_amide_bonds

    mean_val_electrons = sum(valence_electrons)/len(valence_electrons)
    mean_rad_electrons = sum(radical_electrons)/len(radical_electrons)
    mean_tpsa = sum(tpsas)/len(tpsas)
    mean_qed = sum(qeds)/len(qeds)
    mean_n_heteroatoms = sum(n_heteroatoms)//len(n_heteroatoms)
    mean_nO = sum(nOs)//len(nOs)
    mean_NOCount = sum(NOCounts)//len(NOCounts)
    mean_NHOHCount = sum(NHOHCounts)//len(NHOHCounts)

    mean_crippen_logp = sum(crippen_logps)/len(crippen_logps)
    mean_n2bonds = sum(nDoubleBonds)/len(nDoubleBonds)
    mean_aromatic_bonds = sum(nAromaticBonds)/len(nAromaticBonds)
    mean_nCl = sum(nCls)/len(nCls)
    mean_complexity = sum(complexities)/len(complexities)
    mean_rotable_bonds = sum(rotable_bonds)//len(rotable_bonds)
    mean_heavy_atoms = sum(heavy_atoms)/len(heavy_atoms)
    mean_n_rings = sum(n_rings)/len(n_rings)
    mean_HallKierAlpha = sum(HallKierAlphas)/len(HallKierAlphas)
    mean_crippen_mr = sum(crippen_mrs)/len(crippen_mrs)
    mean_aromatic_rings = sum(aromatic_rings)/len(aromatic_rings)
    mean_amide_bonds = sum(amide_bonds)/len(amide_bonds)
    mean_aromatic_heterocycles = sum(aromatic_heterocycles)/len(aromatic_heterocycles)
    mean_aromatic_carbocycles = sum(aromatic_carbocycles)/len(aromatic_carbocycles)


    mol_weight_dataset = ConceptDataset(root= config.concept_dir + 'mol_weight/',
                                     name=config.datasets.dataset_name,
                                     pre_filter=molecular_weight_filtering)


    val_electrons_dataset = ConceptDataset(root= config.concept_dir + 'val_electrons/',
                                     name=config.datasets.dataset_name,
                                     pre_filter=valence_electrons_filtering)

    if config.datasets.dataset_name != 'bace':
        rad_electrons_dataset = ConceptDataset(root= config.concept_dir + 'rad_electrons/',
                                         name=config.datasets.dataset_name,
                                         pre_filter=radical_electrons_filtering)

    tpsa_dataset = ConceptDataset(root= config.concept_dir + 'tpsa/',
                                     name=config.datasets.dataset_name,
                                     pre_filter=tpsa_filtering)

    logp_dataset = ConceptDataset(root= config.concept_dir + 'logp/',
                                     name=config.datasets.dataset_name,
                                     pre_filter=logp_filtering)

    qed_dataset = ConceptDataset(root= config.concept_dir + 'qed/',
                                     name=config.datasets.dataset_name,
                                     pre_filter=qed_filtering)

    hba_dataset = ConceptDataset(root= config.concept_dir + 'hba/',
                             name=config.datasets.dataset_name,
                             pre_filter=hba_filtering)

    hbd_dataset = ConceptDataset(root= config.concept_dir + 'hbd/',
                                     name=config.datasets.dataset_name,
                                     pre_filter=hbd_filtering)

    n_heteroatom_dataset = ConceptDataset(root= config.concept_dir + 'n_heteroatoms/',
                                     name=config.datasets.dataset_name,
                                     pre_filter=heteroatoms_filtering)

    aliphatic_heterocycles_hydroxyGroups_dataset = ConceptDataset(root= config.concept_dir + 'al_heterocycles_OH/',
                                     name=config.datasets.dataset_name,
                                     pre_filter=aliphatic_heterocycles_hydroxyGroups_filtering)

    nO_dataset = ConceptDataset(root= config.concept_dir + 'nO/',
                                     name=config.datasets.dataset_name,
                                     pre_filter=nO_filtering)

    NOCount_dataset = ConceptDataset(root= config.concept_dir + 'NOCount/',
                                     name=config.datasets.dataset_name,
                                     pre_filter=NOCount_filtering)

    NHOHCount_dataset = ConceptDataset(root= config.concept_dir + 'NHOHCount/',
                                     name=config.datasets.dataset_name,
                                     pre_filter=NHOHCount_filtering)

    crippenlogp_dataset = ConceptDataset(root= config.concept_dir + 'crippenlogp/',
                                     name=config.datasets.dataset_name,
                                     pre_filter=crippen_logp_filtering)
    
    crippenmr_dataset = ConceptDataset(root= config.concept_dir + 'crippen_mr/',
                                     name=config.datasets.dataset_name,
                                     pre_filter=crippen_mr_filtering)

    nCl_dataset = ConceptDataset(root= config.concept_dir + 'nCl/',
                                     name=config.datasets.dataset_name,
                                     pre_filter=nCl_filtering)

    n2bonds_dataset = ConceptDataset(root= config.concept_dir + 'nBondsD2/',
                                     name=config.datasets.dataset_name,
                                     pre_filter=n2bonds_filtering)

    aromatic_bonds_dataset = ConceptDataset(root= config.concept_dir + 'aromatic_bonds/',
                                     name=config.datasets.dataset_name,
                                     pre_filter=aromatic_bonds_filtering)

    complexity_dataset = ConceptDataset(root= config.concept_dir + 'complexity/',
                                     name=config.datasets.dataset_name,
                                     pre_filter=complexity_filtering)

    rotable_bonds_dataset = ConceptDataset(root= config.concept_dir + 'rotable_bonds/',
                                     name=config.datasets.dataset_name,
                                     pre_filter=rotable_bonds_filtering)

    heavy_atoms_dataset = ConceptDataset(root= config.concept_dir + 'heavy_atoms/',
                                     name=config.datasets.dataset_name,
                                     pre_filter=heavy_atoms_filtering)

    n_rings_dataset = ConceptDataset(root= config.concept_dir + 'n_rings/',
                                     name=config.datasets.dataset_name,
                                     pre_filter=n_rings_filtering)
    
    hall_kier_alpha_dataset = ConceptDataset(root= config.concept_dir + 'hall_kier_alpha/',
                                     name=config.datasets.dataset_name,
                                     pre_filter=hall_kier_alpha_filtering)
    
    
    aromatic_rings_dataset = ConceptDataset(root= config.concept_dir + 'aromatic_rings/',
                                     name=config.datasets.dataset_name,
                                     pre_filter=aromatic_rings_filtering)
    
    amide_bonds_dataset = ConceptDataset(root= config.concept_dir + 'amide_bonds/',
                                     name=config.datasets.dataset_name,
                                     pre_filter=amide_bonds_filtering)
    
    aromatic_heterocycles_dataset = ConceptDataset(root= config.concept_dir + 'aromatic_heterocycles/',
                                     name=config.datasets.dataset_name,
                                     pre_filter=aromatic_heterocycles_filtering)
    
    aromatic_carbocycles_dataset = ConceptDataset(root= config.concept_dir + 'aromatic_carbocycles/',
                                     name=config.datasets.dataset_name,
                                     pre_filter=aromatic_carbocycles_filtering)


    print(f'Mean number of valence electrons = {mean_val_electrons}')
    print(f'Mean number of radical electrons = {mean_rad_electrons}')
    print(f'Mean TPSA = {mean_tpsa}')
    print(f'Mean QED = {mean_qed}')
    print(f'Mean number of heteroatoms = {mean_n_heteroatoms}')
    print(f'Mean number of oxygen atoms = {mean_nO}')
    print(f'Mean number of NO = {mean_NOCount}')
    print(f'Mean number of NHOH = {mean_NHOHCount}')

    print(f'Mean crippen logp = {mean_crippen_logp}')
    print(f'Mean crippen MR = {mean_crippen_mr}')
    print(f'Mean number of Cl atoms = {mean_nCl}')
    print(f'Mean number of double bonds = {mean_n2bonds}')
    print(f'Mean number of aromatic bonds = {mean_aromatic_bonds}')
    print(f'Mean complexity = {mean_complexity}')
    print(f'Mean number of rotable bonds = {mean_rotable_bonds}')
    print(f'Mean number of heavy atoms = {mean_heavy_atoms}')
    print(f'Mean number of rings = {mean_n_rings}')
    print(f'Mean hall_kier_alpha = {mean_HallKierAlpha}')
    print(f'Mean number of aromatic rings = {mean_aromatic_rings}')
    print(f'Mean number of amide bonds = {mean_amide_bonds}')
    print(f'Mean number of aromatic heterocycles = {mean_aromatic_heterocycles}')
    print(f'Mean number of aromatic carbocycles = {mean_aromatic_carbocycles}')
    
if __name__ == '__main__':
    main()