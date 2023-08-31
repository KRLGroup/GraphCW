"""
Plot functions to generate figures in the paper and additional ones.
Some functions have been adapted from those present in https://github.com/zhiCHEN96/ConceptWhitening/blob/final_version/plot_functions.py

"""


import os
import shutil
import seaborn as sns
from shutil import copyfile
import numpy as np
from numpy import linalg as LA
import seaborn as sns
from PIL import ImageFile, Image
from skimage.transform import resize
from sklearn.metrics import roc_auc_score
from utils import AverageMeter, accuracy
from sklearn.linear_model import LogisticRegression, SGDClassifier
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import matplotlib
import skimage.measure
import random
import cv2
matplotlib.use('Agg')

#from train_places import AverageMeter, accuracy

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from gnnNets import *
from dataset import get_dataset, get_dataloader, get_concept_dataloader


from rdkit.Chem.Descriptors import MolWt, NumValenceElectrons, NumRadicalElectrons, TPSA, MolLogP # Or MolWt for the average molecular weight of the molecule
from rdkit import Chem
from rdkit.Chem import Draw


'''
    This function finds the top 50 images that gets the greatest activations with respect to the concepts.
    (We have a folder for each concept containing the 50 images that have the greatest activations for that concept).
    Concept activation values are obtained based on iternorm_rotation module outputs.
    Since concept corresponds to channels in the output, we look for the top50 images whose kth channel activations
    are high.
'''
def plot_concept_topN(config, test_loader, N, model, whitened_layers, plot_cpt, mode = 'max', print_other = False, activation_mode = 'mean'):
    # switch to evaluate mode
    model.eval()
    
    dst = config.base_dir+'plot/' + '_'.join(plot_cpt) + '_' + mode + '_N' + str(N) + '_l'
    if not os.path.exists(dst):
        os.mkdir(dst)
    layer_list = whitened_layers
    
    folder = dst + '_'.join([str(l) for l in layer_list])
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    outputs= []
    def hook(module, input, output):
        from iterative_normalization import iterative_normalization_py
        X_hat = iterative_normalization_py.apply(input[0], module.running_mean, module.running_wm, module.num_channels, module.T,
                                                 module.eps, module.momentum, module.training)
        size_X = X_hat.size()
        size_R = module.running_rot.size()
        X_hat = X_hat.view(size_X[0], size_R[0], size_R[2], *size_X[2:])

        X_hat = torch.einsum('bgc,gdc->bgd', X_hat, module.running_rot)
        X_hat = X_hat.view(*size_X)

        outputs.append(X_hat.cpu().numpy())
    
    for layer in layer_list:
        assert layer < len(model.norm_layers)
        model.norm_layers[layer].register_forward_hook(hook)

    begin = 0
    end = len(plot_cpt)
    if print_other:
        begin = print_other
        end = begin + 1
    concept_vals = None
    with torch.no_grad():
        for k in range(begin, end):
            if k < len(plot_cpt):
                output_path = os.path.join(folder, plot_cpt[k])
            else:
                output_path = os.path.join(folder, 'other_dimension_'+str(k))
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            samples = []
            vals = None
            for i, sample in enumerate(test_loader):
                samples += list(sample.smiles)
                input_var = sample.cuda()
                outputs = []
                model(input_var)
                val = []
                for output in outputs:
                    if activation_mode == 'mean':
                        val = np.concatenate((val,output[:,k])) # 2,3 correspond to the last two dimensions (7,7), which is the
                                                                            # channel that corresponds to the k-th concept.
                                                                            # The mean gives a 512x1 vector where the k-th element is the
                                                                            # activation for the k-th concept
                    elif activation_mode == 'max':
                        val = np.concatenate((val,output[:,k]))
                    elif activation_mode == 'pos_mean':
                        pos_bool = (output > 0).astype('int32')
                        act = (output * pos_bool)/(pos_bool+0.0001)
                        val = np.concatenate((val,act[:,k]))

                val = val.reshape((len(outputs),-1))
                if i == 0:
                    vals = val
                else:
                    vals = np.concatenate((vals,val),1)
            
            layers_activations = []
            for i, layer in enumerate(layer_list):
                arr = list(zip(list(vals[i,:]),list(samples)))
                if mode == 'max':
                    arr.sort(key = lambda t: t[0], reverse = True)
                else:
                    arr.sort(key = lambda t: t[0], reverse = False)

                for j in range(N):
                    src = arr[j][1]
                    molecule = Chem.MolFromSmiles(src)
                    img = Draw.MolToImage(molecule)
                    plt.imshow(img)
                    plt.savefig(output_path+'/'+'layer'+str(layer)+'_'+str(j+1)+'.jpg')
                    
                layers_activations.append(arr[:N])
                    
            if k == 0:
                concept_vals = np.array([layers_activations])
            else:
                concept_vals = np.concatenate((concept_vals, np.array([layers_activations])), 0)

    return concept_vals

'''
    This method gets the activations of output from iternorm_rotation for images (from val_loader) at channel (cpt_idx)
'''
def get_layer_representation(config, model, device, test_loader, layer_idx, cpt_idx):
    
    with torch.no_grad():        
        model.eval()
        
        outputs= []
    
        def hook(module, input, output):
            from iterative_normalization import iterative_normalization_py
            #print(input)
            X_hat = iterative_normalization_py.apply(input[0], module.running_mean, module.running_wm, module.num_channels, module.T,
                                                    module.eps, module.momentum, module.training)
            size_X = X_hat.size()
            size_R = module.running_rot.size()
            X_hat = X_hat.view(size_X[0], size_R[0], size_R[2], *size_X[2:])

            X_hat = torch.einsum('bgc,gdc->bgd', X_hat, module.running_rot)
            X_hat = X_hat.view(*size_X)

            outputs.append(X_hat.cpu().numpy())
        
        
        model.norm_layers[layer_idx].register_forward_hook(hook)

        samples = []
        vals = None
        for i, sample in enumerate(test_loader):
            samples.append([sample])
            outputs = []
            model(sample.to(device))
            
            val = []
            for output in outputs:
                new_output = np.expand_dims(output, 0).sum(1)[:, cpt_idx]
                val.append(new_output)
            val = np.array(val)
            
            if i == 0:
                vals = val
            else:
                vals = np.concatenate((vals,val),1)
            
        del model
    return samples, vals

def get_layer_accuracy(config, test_loader, layer_idx, cpt_idx):
    if torch.cuda.is_available():
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')
        
    model = load_model(config)
    model.to(device)
    
    with torch.no_grad():        
        model.eval()
        
        outputs= []
    
        def hook(module, input, output):
            from iterative_normalization import iterative_normalization_py
            #print(input)
            X_hat = iterative_normalization_py.apply(input[0], module.running_mean, module.running_wm, module.num_channels, module.T,
                                                    module.eps, module.momentum, module.training)
            size_X = X_hat.size()
            size_R = module.running_rot.size()
            X_hat = X_hat.view(size_X[0], size_R[0], size_R[2], *size_X[2:])

            X_hat = torch.einsum('bgc,gdc->bgd', X_hat, module.running_rot)
            X_hat = X_hat.view(*size_X)

            outputs.append(X_hat.cpu().numpy())
        
        
        model.norm_layers[layer_idx].register_forward_hook(hook)

        samples, accs, preds = [], [], []
        vals = None
        for i, sample in enumerate(test_loader):
            samples.append([sample])
            outputs = []
            sample.x = sample.x.float()
            sample.y = sample.y.squeeze().float()
            model(sample.cuda())
            
            val, acc = [], []
            for output in outputs:
                output = np.reshape(output, (1, output.shape[1], output.shape[0]), order='F')
                new_output = output.sum((2))[:, cpt_idx]
                pred = torch.round(torch.sigmoid(torch.tensor(new_output)))
                preds.append(pred.item())
                accs.append(pred[0] == 1.0)
                val.append(new_output)
            val = np.array(val)
            if i == 0:
                vals = val
            else:
                vals = np.concatenate((vals,val),1)
            
        del model
        
    return samples, vals, sum(accs)/len(accs)

def concept_accuracy_by_layer(config, concept_loaders, whitened_layers=[0], plot_cpt = ['tpsa']):
    dst = os.path.join(config.base_dir+'plot/', '_'.join(config.concepts.split(',')) + '/concept_accuracies/')
    if not os.path.exists(dst):
        os.makedirs(dst)
    
    for concept_index, concept_loader in enumerate(concept_loaders):
        concept_accs = []
        for layer_index in whitened_layers:
            samples, vals, acc = get_layer_accuracy(config, concept_loader['train'], layer_index, concept_index)
            print(vals.shape)
            concept_accs.append(acc)
                    
        fig = plt.figure(figsize = (20, 10))
 
        # creating the bar plot
        layer_names = ['layer_'+str(l) for l in whitened_layers]
        plt.bar(layer_names, concept_accs, color ='blue', width = 0.4)
        plt.xticks(rotation='vertical')
        plt.ylabel("Classification accuracy")
        plt.ylim([0.0, 1.0])
        plt.title(plot_cpt[concept_index])
        plt.savefig('{}{}.jpg'.format(dst, plot_cpt[concept_index]))
        

''' This scatter plot shows how well samples belonging to two different classes are separated considering two concepts. '''

def scatter_plot(args, test_loader, whitened_layers, plot_cpt = ['mol_weight','val_electrons'], N = 3, normalize_by='layer'):
    
    if torch.cuda.is_available():
        device = torch.device('cuda', index=args.device_id)
    else:
        device = torch.device('cpu')
        
    dst = os.path.join(args.base_dir+'plot/', '_'.join(args.concepts.split(',')) + '/scatter_plot_1/')
    if not os.path.exists(dst):
        os.makedirs(dst)
    
    model = load_model(args)
    model = model.to(device)
    
    concepts = args.concepts.split(',')
    
    for i in range(len(concepts)):
        for j in range(i,len(concepts)):
            if i == j:
                continue
            samples1, vals1 = get_layer_representation(args, model, device, test_loader, 0, i)
            samples2, vals2 = get_layer_representation(args, model, device, test_loader, 0, j)

            labels = [sample.y.item() for sample in test_loader]

            fig = plt.figure(figsize=(10,5))

            plt.scatter(vals1, vals2, c=labels)

            plt.savefig('{}{}_{}_layer{}.jpg'.format(dst, concepts[i], concepts[j], 1))
            

''' This method returns a scatter plot with the most and least activated molecules on specified concepts'''

def show_scatter_plot(args, test_loader, whitened_layers, plot_cpt = ['mol_weight','val_electrons'], N = 3, normalize_by='layer'):
    dst = os.path.join(args.base_dir+'plot/', '_'.join(args.concepts.split(',')) + '/scatter_plot/')
    if not os.path.exists(dst):
        os.makedirs(dst)
    
    if torch.cuda.is_available():
        device = torch.device('cuda', index=args.device_id)
    else:
        device = torch.device('cpu')
        
    model = load_model(args)
    model = model.to(device)
    
    # We retrieve the 3 most and least activated samples
    act_max = plot_concept_topN(args, test_loader, N, model, whitened_layers, plot_cpt, mode = 'max', print_other = False, activation_mode = 'max')
    act_min = plot_concept_topN(args, test_loader, N, model, whitened_layers, plot_cpt, mode = 'min', print_other = False, activation_mode = 'max')
    
    activations = np.concatenate((act_min, act_max), 2) #(5,3,6,2)
    
    n_concepts, n_layers, n_samples, values = activations.shape
    external_loop = n_concepts
    internal_loop = n_layers
    
    if normalize_by == 'concept':
        activations = np.reshape(activations, (n_layers, n_concepts, n_samples, values), order='F')
        external_loop = n_layers
        internal_loop = n_concepts

    
    for i in range(external_loop): # for each concept
        layer_acts = np.squeeze(activations[:,:,:,0]).astype(np.float)
        smiless = np.squeeze(activations[:,:,:,1]).astype(np.str)
        max_act = np.max(layer_acts)
        min_act = np.min(layer_acts)
        layer_acts = (layer_acts-min_act)/(max_act-min_act)
        
        for j in range(internal_loop): # for each layer
            acts = np.array([float(elem) for elem in layer_acts[i][j]])
            smiles = smiless[i][j]
                                 
            fig,ax = plt.subplots()
            ax.set_xlim([0,7])
            ax.set_ylim([-0.5,1.5])
            ax.set_xlabel('Samples')
            ax.set_ylabel('Activation values')
            ax.scatter([i+1 for i in range(6)],acts)          
            
            for n, smile in enumerate(smiles):
                molecule = Chem.MolFromSmiles(smile)
                img = Draw.MolToImage(molecule)
                img = OffsetImage(img, zoom=0.15)
                if n == 0:
                    y = acts[n] + 0.4
                elif n < 3:
                    y = acts[n] + (n+1)*0.2
                elif n == 3:
                    y = acts[n] - 0.4
                else:
                    y = acts[n] - (n-2)*0.2
                
                ab = AnnotationBbox(img, (n+1,y), frameon=False)
                ax.add_artist(ab)
                
            plt.savefig('{}{}_layer{}.jpg'.format(dst, plot_cpt[i], j))


'''
    This function plots the relative activations of a image on two different concepts. 
'''
def plot_trajectory(args, test_loader, plt_cpt = ['tpsa','logp']):
    
    if torch.cuda.is_available():
        device = torch.device('cuda', index=args.device_id)
    else:
        device = torch.device('cpu')
    
    model = load_model(args)
    model = model.to(device)
    
    dst = os.path.join(args.base_dir+'plot/', '_'.join(args.concepts.split(',')) + '/trajectory_all/')
    if not os.path.exists(dst):
        os.makedirs(dst)
    concepts = args.concepts.split(',')
    cpt_idx = [concepts.index(cpt) for cpt in plt_cpt]
    vals = None 
    n_layers = len(args.models.param.gnn_latent_dim)
    for layer_idx in range(n_layers):
        if layer_idx == 0:
            samples, vals = get_layer_representation(args, model, device, test_loader, layer_idx, cpt_idx)
        else:
            _, val = get_layer_representation(args, model, device, test_loader, layer_idx, cpt_idx)
            vals = np.concatenate((vals,val),0)
    try:
        os.mkdir('{}{}'.format(dst,'_'.join(plt_cpt)))
    except:
        pass

    num_examples = vals.shape[1]
    num_layers = vals.shape[0]
    max_vals = np.amax(vals, axis=1)
    min_vals = np.amin(vals, axis=1)
    vals = vals.transpose((1,0,2))
    sort_idx = vals.argsort(0)
    for i in range(num_layers):
        for j in range(2):
            vals[sort_idx[:,i,j],i,j] = np.arange(num_examples)/num_examples
    idx = np.arange(num_examples)
    np.random.shuffle(idx)
    for k, i in enumerate(idx):
        if k==300:
            break
        fig = plt.figure(figsize=(10,5))
        ax2 = plt.subplot(1,2,2)
        ax2.set_xlim([0.0,1.0])
        ax2.set_ylim([0.0,1.0])
        ax2.set_xlabel(plt_cpt[0])
        ax2.set_ylabel(plt_cpt[1])
        if plt_cpt[0] == 'qed':
            cpt1 = Chem.QED.qed(Chem.MolFromSmiles(samples[i][0].smiles[0]))
        elif plt_cpt[0] == 'tpsa':
            cpt1 = TPSA(Chem.MolFromSmiles(samples[i][0].smiles[0]))
        elif plt_cpt[0] == 'logp':
            cpt1 = MolLogP(Chem.MolFromSmiles(samples[i][0].smiles[0]))
        elif plt_cpt[0] == 'NOCount':
            cpt1 = Chem.Lipinski.NOCount(Chem.MolFromSmiles(samples[i][0].smiles[0]))
        elif plt_cpt[0] == 'n_heteroatoms':
            cpt1 = Chem.rdMolDescriptors.CalcNumHeteroatoms(Chem.MolFromSmiles(samples[i][0].smiles[0]))
            
        if plt_cpt[1] == 'qed':
            cpt2 = Chem.QED.qed(Chem.MolFromSmiles(samples[i][0].smiles[0]))
        elif plt_cpt[1] == 'tpsa':
            cpt2 = TPSA(Chem.MolFromSmiles(samples[i][0].smiles[0]))
        elif plt_cpt[1] == 'logp':
            cpt2 = MolLogP(Chem.MolFromSmiles(samples[i][0].smiles[0]))
        elif plt_cpt[1] == 'NOCount':
            cpt2 = Chem.Lipinski.NOCount(Chem.MolFromSmiles(samples[i][0].smiles[0]))
        elif plt_cpt[1] == 'n_heteroatoms':
            cpt2 = Chem.rdMolDescriptors.CalcNumHeteroatoms(Chem.MolFromSmiles(samples[i][0].smiles[0]))
        
        ax2.set_title(f'Class: {str(samples[i][0].y[0].item())}, '+plt_cpt[0]+' = '+str(cpt1)+', '+plt_cpt[1]+' = '+str(cpt2))
        plt.scatter(vals[i,:,0],vals[i,:,1])
        start_x = vals[i,0,0]
        start_y = vals[i,0,1]
        for j in range(1, num_layers):
            dx, dy = vals[i,j,0]-vals[i,j-1,0],vals[i,j,1]-vals[i,j-1,1]
            plt.arrow(start_x, start_y, dx, dy, length_includes_head=True, head_width=0.01, head_length=0.02)
            start_x, start_y = vals[i,j,0], vals[i,j,1]
        ax1 = plt.subplot(1,2,1)
        ax1.axis('off')
        
        smiles = samples[i][0].smiles[0]
        molecule = Chem.MolFromSmiles(smiles)
        img = Draw.MolToImage(molecule)
        plt.imshow(img)
        plt.savefig('{}{}/{}.jpg'.format(dst,'_'.join(plt_cpt), k))
        print('saving in '+'{}{}/{}.jpg'.format(dst,'_'.join(plt_cpt), k))


def concept_gradient_importance(args, test_loader, layer, num_classes=2):
    dst = os.path.join(args.base_dir+'plot/', '_'.join(args.concepts.split(',')) + '/')
    if not os.path.exists(dst):
        os.mkdir(dst)
            
    if torch.cuda.is_available():
        device = torch.device('cuda', index=args.device_id)
    else:
        device = torch.device('cpu')
        
    model = load_model(args)
    
    model = model.to(device)
    model.eval()
    
    outputs = []

    # Instead of taking the input of ReLU, we directly take the output of the cw layer
    def hook(module, grad_input, grad_output):
        outputs.append(grad_output[0])
    
    layer = int(layer)
    model.norm_layers[layer].register_backward_hook(hook)

    class_count = [0] * num_classes
    concept_importance_per_class = [None] * num_classes

    for i, batch in enumerate(test_loader):
        batch = batch.to(device)
        batch.x = torch.autograd.Variable(batch.x)
        batch.x = batch.x.float()
        batch.y = torch.autograd.Variable(batch.y)
        batch.y = batch.y.squeeze().float()

        output = model(batch)
        model.zero_grad()
        prediction_result = torch.argmax(output, dim=1).flatten().tolist()[0]
        class_count[prediction_result] += 1
        output[:,prediction_result].backward()
        directional_derivatives = torch.unsqueeze(outputs[0],0).mean(dim=1).flatten().cpu().numpy()
        is_positive = (directional_derivatives > 0).astype(np.int64)
        if concept_importance_per_class[prediction_result] is None:
            concept_importance_per_class[prediction_result] = is_positive 
        else:
            concept_importance_per_class[prediction_result] += is_positive
        outputs = []

    for i in range(num_classes):
        concept_importance_per_class[i] = concept_importance_per_class[i].astype(np.float32)
        concept_importance_per_class[i] /= class_count[i]
            
    concepts = args.concepts.split(',')
    n_concepts = len(concepts)
    
    concepts_importance = concept_importance_per_class[0][:n_concepts]
                
    fig = plt.figure(figsize=(15,7))
    plt.rc('axes', labelsize=24)
    plt.rc('xtick', labelsize=22) 
    plt.rc('ytick', labelsize=22)
    barlist = plt.bar(args.concepts.split(','), concepts_importance)
    colors = ['b','r','g','c','m']
    for i in range(len(args.concepts.split(','))):
        barlist[i].set_color(colors[i])
    plt.title('Layer '+str(layer), fontsize=24)
    plt.xlabel('Concept', fontsize=24)
    plt.ylabel('Concept importance', fontsize=24)
    plt.savefig(dst+'/concepts_importance_layer'+str(layer)+'.jpg')
    
    return concepts_importance

# This method compares the intra concept group dot product with inter concept group dot product
def intra_concept_dot_product_vs_inter_concept_dot_product(args, loader, layer, plot_cpt = ['qed','logp','n_heteroatoms'], cw = True):
    dst = args.base_dir + 'plot/' + '_'.join(args.concepts.split(','))
    if not os.path.exists(dst):
        os.mkdir(dst)
    
    dst = dst + '/inner_product/'
    if not os.path.exists(dst):
        os.mkdir(dst)
            
    if torch.cuda.is_available():
        device = torch.device('cuda', index=args.device_id)
    else:
        device = torch.device('cpu')
        
    model = load_model(args)
    
    model = model.to(device)
    model.eval()
    
    representations = {}
    for cpt in plot_cpt:
        representations[cpt] = []
    
    for c, cpt in enumerate(plot_cpt):
        with torch.no_grad():
            
            outputs= []
        
            def hook(module, input, output):
                if not cw:
                    outputs.append(output.cpu().numpy())
                else:
                    from iterative_normalization import iterative_normalization_py
                    X_hat = iterative_normalization_py.apply(input[0], module.running_mean, module.running_wm, module.num_channels, module.T,
                                                            module.eps, module.momentum, module.training)
                    size_X = X_hat.size()
                    size_R = module.running_rot.size()
                    X_hat = X_hat.view(size_X[0], size_R[0], size_R[2], *size_X[2:])

                    X_hat = torch.einsum('bgc,gdc->bgd', X_hat, module.running_rot)
                    X_hat = X_hat.view(*size_X)

                    outputs.append(X_hat.cpu().numpy())
                    
                    
            layer = int(layer)
            assert layer < len(model.norm_layers)
            model.norm_layers[layer].register_forward_hook(hook)

            for j, batch in enumerate(loader):
                batch = batch.to(device)
                batch.x = torch.autograd.Variable(batch.x)
                batch.x = batch.x.float()
                
                labels = batch.y.cpu().numpy().flatten().astype(np.int32).tolist()
                batch.y = torch.autograd.Variable(batch.y)
                
                outputs = []
                model(batch)
                for instance_index in range(len(labels)): # batch size
                    smiles = batch.smiles[0]
                    # For BBBP
                    if (cpt == 'qed' and Chem.QED.qed(Chem.MolFromSmiles(smiles)) > 0.6244) or \
                        (cpt == 'tpsa' and TPSA(Chem.MolFromSmiles(smiles)) > 70.73) or \
                        (cpt == 'logp' and MolLogP(Chem.MolFromSmiles(smiles)) < 5) or \
                        (cpt == 'NOCount' and Chem.Lipinski.NOCount(Chem.MolFromSmiles(smiles)) < 5) or \
                        (cpt == 'n_heteroatoms' and Chem.rdMolDescriptors.CalcNumHeteroatoms(Chem.MolFromSmiles(smiles)) < 6):
                        
                        output_shape = outputs[0].shape
                        representation_mean = outputs[0].mean(axis=0) # mean of all features of that instance
                        representations[cpt].append(representation_mean) # get the cpt_index channel of the output

    # representation of concepts in matrix form
    dot_product_matrix = np.zeros((len(plot_cpt),len(plot_cpt))).astype('float')
    m_representations = {}
    m_representations_normed = {}
    intra_dot_product_means = {}
    intra_dot_product_means_normed = {}
    for i, concept in enumerate(plot_cpt):
        m_representations[concept] = np.stack(representations[concept], axis=0) # n * (h*w)
        m_representations_normed[concept] = m_representations[concept]/LA.norm(m_representations[concept], axis=1, keepdims=True)
        intra_dot_product_means[concept] = np.matmul(m_representations[concept], m_representations[concept].transpose()).mean()
        intra_dot_product_means_normed[concept] = np.matmul(m_representations_normed[concept], m_representations_normed[concept].transpose()).mean()
        dot_product_matrix[i,i] = 1.0

    inter_dot_product_means = {}
    inter_dot_product_means_normed = {}
    for i in range(len(plot_cpt)):
        for j in range(i+1, len(plot_cpt)):
            cpt_1 = plot_cpt[i]
            cpt_2 = plot_cpt[j]
            inter_dot_product_means[cpt_1 + '_' + cpt_2] = np.matmul(m_representations[cpt_1], m_representations[cpt_2].transpose()).mean()
            inter_dot_product_means_normed[cpt_1 + '_' + cpt_2] = np.matmul(m_representations_normed[cpt_1], m_representations_normed[cpt_2].transpose()).mean()
            dot_product_matrix[i,j] = abs(inter_dot_product_means_normed[cpt_1 + '_' + cpt_2]) / np.sqrt(abs(intra_dot_product_means_normed[cpt_1]*intra_dot_product_means_normed[cpt_2]))
            dot_product_matrix[j,i] = dot_product_matrix[i,j]
    
    plt.figure(figsize=(12,6))
    ticklabels  = [s.replace('_',' ') for s in plot_cpt]
    sns.set(font_scale=1.4)
    ax = sns.heatmap(dot_product_matrix, vmin = 0, vmax = 1, xticklabels = ticklabels, yticklabels = ticklabels, annot=True)
    ax.figure.tight_layout()
    plt.savefig(dst + '_' + str(cw) + '_' + str(layer) +'.jpg', dpi=1200)

    return intra_dot_product_means, inter_dot_product_means, intra_dot_product_means_normed, inter_dot_product_means_normed
   

def load_model(config):
    
    if torch.cuda.is_available():
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')
        
    dataset = get_dataset(dataset_root=config.datasets.dataset_root,
                          dataset_name=config.datasets.dataset_name)
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    
    # Defining the model
    model = get_gnnNets(dataset.num_node_features, dataset.num_classes, config.models, config.concept_whitening)
    model = model.to(device)
    
    # Replace all BatchNorm layers with CW layers
    if config.concept_whitening:
        model.replace_norm_layers()

    # CHANGE MODEL TO LOAD HERE
    model_path = os.path.join('/home/michelaproietti/thesis_last/trained_models/'+dataset.name+'/', config.models.gnn_name+"_baseline.pth")
    print(f'Loading model: {model_path}')

    if os.path.isfile(model_path):
        state_dict = torch.load(model_path)['net']
        model.load_state_dict(state_dict)
    else:
        raise Exception("checkpoint {} not found!".format(model_path))
    
    return model
