import os
import json
import hydra
import torch
import shutil
import warnings
from torch.optim import Adam, SGD
from omegaconf import OmegaConf
from utils import check_dir, load_trained_model
from gnnNets import get_gnnNets
from dataset import get_dataset, get_dataloader, get_concept_dataloader
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch_geometric.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from plot_functions import *


class TrainModel(object):
    def __init__(self,
                 model,
                 dataset,
                 concept_loaders,
                 device,
                 statistics_dir,
                 graph_classification=True,
                 save_dir=None,
                 save_name='model',
                 concept_whitening=False,
                 ** kwargs):
        self.model = model
        self.dataset = dataset  # train_mask, eval_mask, test_mask
        self.concept_loaders = concept_loaders
        self.loader = None
        self.device = device
        self.graph_classification = graph_classification
        self.node_classification = not graph_classification
        self.concept_whitening = concept_whitening
        self.statistics_dir = statistics_dir

        self.optimizer = None
        self.save = save_dir is not None
        self.save_dir = save_dir
        self.save_name = save_name
        check_dir(self.save_dir)

        if self.graph_classification:
            dataloader_params = kwargs.get('dataloader_params')
            self.loader = get_dataloader(dataset, **dataloader_params)
    
    def adjust_learning_rate(self, epoch, lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        new_lr = lr * (0.1 ** (epoch // 10))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def __loss__(self, logits, labels):
        if self.dataset.name == 'tox21':
            loss = torch.nn.BCEWithLogitsLoss()
            return loss(logits, labels)
        else:
            return F.cross_entropy(logits, labels)

    def _train_batch(self, data, labels):
        logits = self.model(data=data)
        #print(f'logits = {logits.shape}') # [32, 2] = [batch_size, logits for both classes]
        if self.graph_classification:
            loss = self.__loss__(logits, labels)
        else:
            loss = self.__loss__(logits[data.train_mask], labels[data.train_mask])

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=2.0)
        self.optimizer.step()
        
        if self.dataset.name == 'tox21':
            loss = loss.mean()
            
        return loss.item(), logits.argmax(-1), logits

    def _eval_batch(self, data, labels, **kwargs):
        self.model.eval()
        logits = self.model(data)
        if self.graph_classification:
            loss = self.__loss__(logits, labels)
        else:
            mask = kwargs.get('mask')
            if mask is None:
                warnings.warn("The node mask is None")
                mask = torch.ones(labels.shape[0])
            loss = self.__loss__(logits[mask], labels[mask])
            
        if self.dataset.name == 'tox21':
            loss = loss.mean()
            
        loss = loss.item()
        preds = logits.argmax(-1)
        return loss, preds, logits

    def eval(self):
        self.model.to(self.device)
        self.model.eval()
        if self.graph_classification:
            losses, accs = [], []
            labels, pred_probs = [], []
            for i, batch in enumerate(self.loader['eval']):
                batch = batch.to(self.device)
                
                if self.dataset.name == 'clintox':
                    batch.y = batch.y.float()
                    batch.y = torch.argmax(batch.y, 1)
                elif self.dataset.name == 'tox21':
                    batch.y = batch.y.float()
                    for i_s,s in enumerate(batch.y):
                        new_s = s
                        new_s[new_s < 0] = 0
                        new_s[new_s > 0] = 1
                        batch.y[i_s] = new_s
                        
                loss, batch_preds, logits = self._eval_batch(batch, batch.y)
                
                if self.dataset.name == 'tox21':
                    batch_preds = torch.round(torch.sigmoid(logits))

                    acc_ = torch.zeros((batch.y.shape[0], 1))
                    for idx, target in enumerate(batch.y):
                        acc_[idx] = batch_preds[idx].eq(target).sum()/target.shape[0]

                    accs.append(acc_.mean())
                else:
                    accs.append(batch_preds == batch.y)
                
                losses.append(loss)

                if self.dataset.name == 'tox21':
                    probs = torch.sigmoid(logits)
                else:
                    probs = torch.softmax(logits, 1)
                
                if self.dataset.name != 'tox21':
                    probs = probs[:, 1]
                    
                if i == 0:
                    labels = batch.y.cpu()
                    pred_probs = probs.cpu().detach()
                else:
                    labels = torch.cat([labels, batch.y.cpu()])
                    pred_probs = torch.cat([pred_probs, probs.cpu().detach()])
                
            eval_loss = torch.tensor(losses).mean().item()
            
            if self.dataset.name == 'tox21':
                eval_acc = torch.tensor(accs).float().mean().item()
            else:
                eval_acc = torch.cat(accs, dim=-1).float().mean().item()
            
            if self.dataset.name == 'tox21':
                eval_auc = roc_auc_score(labels, pred_probs, multi_class='ovr')
            else:
                eval_auc = roc_auc_score(labels, pred_probs)
            
        else:
            data = self.dataset.data.to(self.device)
            eval_loss, preds, logits = self._eval_batch(data, data.y, mask=data.val_mask)
            eval_acc = (preds == data.y).float().mean().item()
            
            probs = torch.softmax(logits, 1)[:, 1]
            
            if self.dataset.name == 'tox21':
                eval_auc = (torch.tensor(roc_auc_score(data.y.cpu(), probs.cpu().detach().numpy(), multi_class='ovr'))).float().mean().item()
            else:
                eval_auc = (torch.tensor(roc_auc_score(data.y.cpu(), probs.cpu().detach().numpy()))).float().mean().item()
            
        return eval_loss, eval_acc, eval_auc

    def test(self):
        state_dict = torch.load(os.path.join(self.save_dir, f'{self.save_name}_best.pth'))['net']
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        if self.graph_classification:
            losses, preds, accs = [], [], []
            for i, batch in enumerate(self.loader['test']):
                if batch.batch.shape[0] == 1:
                    continue
                    
                batch = batch.to(self.device)
                
                if self.dataset.name == 'clintox':
                    batch.y = batch.y.float()
                    batch.y = torch.argmax(batch.y, 1)
                elif self.dataset.name == 'tox21':
                    batch.y = batch.y.float()
                    for i_s,s in enumerate(batch.y):
                        new_s = s
                        new_s[new_s < 0] = 0
                        new_s[new_s > 0] = 1
                        batch.y[i_s] = new_s
                
                loss, batch_preds, logits = self._eval_batch(batch, batch.y)
                
                if self.dataset.name == 'tox21':
                    batch_preds = torch.round(torch.sigmoid(logits))

                    acc_ = torch.zeros((batch.y.shape[0], 1))
                    for idx, target in enumerate(batch.y):
                        acc_[idx] = batch_preds[idx].eq(target).sum()/target.shape[0]

                    accs.append(acc_.mean())
                else:
                    accs.append(batch_preds == batch.y)
                
                losses.append(loss)

                if self.dataset.name == 'tox21':
                    probs = torch.sigmoid(logits)
                else:
                    probs = torch.softmax(logits, 1)
                
                if self.dataset.name != 'tox21':
                    probs = probs[:, 1]
                    
                if i == 0:
                    labels = batch.y.cpu()
                    pred_probs = probs.cpu().detach()
                else:
                    labels = torch.cat([labels, batch.y.cpu()])
                    pred_probs = torch.cat([pred_probs, probs.cpu().detach()])
                
            test_loss = torch.tensor(losses).mean().item()
            
            if self.dataset.name == 'tox21':
                test_acc = torch.tensor(accs).float().mean().item()
            else:
                test_acc = torch.cat(accs, dim=-1).float().mean().item()
            
            if self.dataset.name == 'tox21':
                test_auc = roc_auc_score(labels, pred_probs, multi_class='ovr')
            else:
                test_auc = roc_auc_score(labels, pred_probs)
        else:
            data = self.dataset.data.to(self.device)
            test_loss, preds, logits = self._eval_batch(data, data.y, mask=data.test_mask)
            test_acc = (preds == data.y).float().mean().item()

            probs = torch.softmax(logits, 1)[:, 1]
            
            if self.dataset.name == 'tox21':
                test_auc = (torch.tensor(roc_auc_score(data.y.cpu(), probs.cpu().detach().numpy(), multi_class='ovr'))).float().mean().item()
            else:
                test_auc = (torch.tensor(roc_auc_score(data.y.cpu(), probs.cpu().detach().numpy()))).float().mean().item()
            
        print(f"Test loss: {test_loss:.4f}, test acc {test_acc:.4f}, test_auc {test_auc:.4f}")
        return test_loss, test_acc, test_auc, preds

    def train(self, train_params=None, optimizer_params=None):
        num_epochs = train_params['num_epochs']
        num_early_stop = train_params['num_early_stop']
        milestones = train_params['milestones']
        gamma = train_params['gamma']

        if optimizer_params is None:
            self.optimizer = Adam(self.model.parameters()) #SGD(self.model.parameters())
        else:
            self.optimizer = Adam(self.model.parameters(), **optimizer_params) #SGD(self.model.parameters(), **optimizer_params)
        
        if milestones is not None and gamma is not None:
            lr_schedule = MultiStepLR(self.optimizer,
                                      milestones=milestones,
                                      gamma=gamma)
        else:
            lr_schedule = None
        
        
        self.model.to(self.device)
        best_eval_acc = 0.0
        best_eval_auc = 0.0
        best_eval_loss = 10000.0
        early_stop_counter = 0
        for epoch in range(num_epochs):
                        
            is_best = False
            self.model.train()
            if self.graph_classification:
                losses, accs = [], []
                
                for i, batch in enumerate(self.loader['train']):
                    batch = batch.to(self.device)
                    
                    if self.concept_whitening:
                        if (i + 1) % 20 == 0:
                            self.model.eval()
                            with torch.no_grad():
                                # update the gradient matrix G
                                for concept_index, concept_loader in enumerate(self.concept_loaders):
                                    self.model.change_mode(concept_index)
                                    for j, concept_batch in enumerate(concept_loader['train']):
                                        concept_batch.x = torch.autograd.Variable(concept_batch.x)
                                        concept_batch.x = concept_batch.x.float()
                                        concept_batch.y = concept_batch.y.squeeze().float()
                                        concept_batch = concept_batch.to(self.device)

                                        preds = self.model(concept_batch)
                                        break
                                self.model.update_rotation_matrix()
                                # change to ordinary mode
                                self.model.change_mode(-1)
                            self.model.train()
                    
                    if self.dataset.name == 'clintox':
                        batch.y = batch.y.float()
                        batch.y = torch.argmax(batch.y, 1)
                    elif self.dataset.name == 'tox21':
                        batch.y = batch.y.float()
                        for i_s,s in enumerate(batch.y):
                            new_s = s
                            new_s[new_s < 0] = 0
                            new_s[new_s > 0] = 1
                            batch.y[i_s] = new_s
                        
                    loss, preds, logits = self._train_batch(batch, batch.y)
                    
                    if self.dataset.name == 'tox21':
                        preds = torch.round(torch.sigmoid(logits))
                        
                        acc_ = torch.zeros((batch.y.shape[0], 1))
                        for idx, target in enumerate(batch.y):
                            acc_[idx] = preds[idx].eq(target).sum()/target.shape[0]
                            
                        accs.append(acc_.mean().item())
                    else:
                        accs.append(preds == batch.y)
                        
                    losses.append(loss)
                    
                    if self.dataset.name == 'tox21':
                        probs = torch.sigmoid(logits)
                    else:
                        probs = torch.softmax(logits, 1)
                    
                    if self.dataset.name != 'tox21':
                        probs = probs[:, 1]
                    
                    if i == 0:
                        labels = batch.y.cpu()
                        pred_probs = probs.cpu().detach()
                    else:
                        labels = torch.cat([labels, batch.y.cpu()])
                        pred_probs = torch.cat([pred_probs, probs.cpu().detach()])
                    
                    # This was used to check class balance
                    #class1.append(torch.sum((batch.y.cpu() == torch.zeros(batch.y.shape)).int()).numpy())
                    #preds1.append(torch.sum((preds.cpu() == torch.zeros(preds.shape)).int()).numpy())
                    #n_batches += 1
                    #print(batch.y)
                    #print(torch.sum((batch.y.cpu() == torch.zeros(batch.y.shape)).int()))
                    
                    
                train_loss = torch.FloatTensor(losses).mean().item()
                
                if self.dataset.name == 'tox21':
                    train_acc = torch.tensor(accs).float().mean().item()
                else:
                    train_acc = torch.cat(accs, dim=-1).float().mean().item()
                
                if self.dataset.name == 'tox21':
                    train_auc = roc_auc_score(labels, pred_probs, multi_class='ovr')
                else:
                    train_auc = roc_auc_score(labels, pred_probs)

            else:
                data = self.dataset.data.to(self.device)
                train_loss, preds, logits = self._train_batch(data, data.y)
                train_acc = (preds == data.y).float().mean().item()
                
                probs = torch.softmax(logits, 1)[:, 1]
                
                if self.dataset.name == 'tox21':
                    train_auc = (torch.tensor(roc_auc_score(data.y.cpu(), probs.cpu().detach().numpy(), multi_class='ovr'))).float().mean().item()
                else:
                    train_auc = (torch.tensor(roc_auc_score(data.y.cpu(), probs.cpu().detach().numpy()))).float().mean().item()
                
            eval_loss, eval_acc, eval_auc = self.eval()
            print(f'Epoch:{epoch}, Training_loss:{train_loss:.4f}, Training_acc:{train_acc:.4f}, Training_auc:{train_auc:.4f}, Eval_loss:{eval_loss:.4f}, Eval_acc:{eval_acc:.4f}, Eval_auc:{eval_auc:.4f}')
            if num_early_stop > 0:
                if eval_loss <= best_eval_loss:
                    best_eval_loss = eval_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                if epoch > (num_epochs / 2) and early_stop_counter > num_early_stop:
                    break
            if lr_schedule:
                lr_schedule.step()

            # Uncomment a different if clause to use a different metric for early stopping
            '''
            if eval_loss <= best_eval_loss:
                best_eval_loss = eval_loss
                is_best = True
            recording = {'epoch': epoch, 'is_best': str(is_best)}
            
            if best_eval_acc < eval_acc:
                is_best = True
                best_eval_acc = eval_acc
            recording = {'epoch': epoch, 'is_best': str(is_best)}
            '''
            if best_eval_auc < eval_auc:
                is_best = True
                best_eval_auc = eval_auc
            recording = {'epoch': epoch, 'is_best': str(is_best)}
            
            if self.save:
                self.save_model(is_best, recording=recording)
        
        # Saving losses and metrics values on json file
        listObj = []
        
        with open(self.statistics_dir, 'r') as fp:
            if len(fp.readlines()) != 0:
                fp.seek(0)
                listObj = json.load(fp)
        
        listObj.append({
                          "Epoch": epoch,
                          "train_loss": train_loss,
                          "train_acc": train_acc,
                          "train_auc": train_auc,
                          "val_loss": eval_loss,
                          "val_acc": eval_acc,
                          "val_auc": eval_auc,
                        })
        with open(self.statistics_dir, 'w') as json_file:
            json.dump(listObj, json_file, indent=4, separators=(',',': '))
        

    def save_model(self, is_best=False, recording=None):
        self.model.to('cpu')
        state = {'net': self.model.state_dict()}
        for key, value in recording.items():
            state[key] = value
        latest_pth_name = f"{self.save_name}_latest.pth"
        best_pth_name = f'{self.save_name}_best.pth'
        ckpt_path = os.path.join(self.save_dir, latest_pth_name)
        torch.save(state, ckpt_path)
        if is_best:
            print('saving best...')
            shutil.copy(ckpt_path, os.path.join(self.save_dir, best_pth_name))
        self.model.to(self.device)

    def load_model(self):
        state_dict = torch.load(os.path.join(self.save_dir, f"{self.save_name}_best.pth"))['net']
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)


@hydra.main(config_path="config", config_name="config")
def main(config):
    config.models.gnn_saving_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    config.models.param = config.models.param[config.datasets.dataset_name]
    
    if os.path.isdir(config.statistics_dir) is False:
        print('Creating new folder for statistics')
        os.mkdir(config.statistics_dir)
    
    if os.path.isdir(config.statistics_dir+config.datasets.dataset_name+'/') is False:
        print('Creating new folder for a certain dataset')
        os.mkdir(config.statistics_dir+config.datasets.dataset_name+'/')
        
    if os.path.isfile(config.statistics_dir+config.datasets.dataset_name+'/'+config.statistics_file) is False:
        print('Creating new saving file')
        open(os.path.join(config.statistics_dir+config.datasets.dataset_name+'/', config.statistics_file), 'a').close()
        
    config.statistics_dir = os.path.join(config.statistics_dir+config.datasets.dataset_name+'/', config.statistics_file)

    if torch.cuda.is_available():
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')
        
    print(device)

    dataset = get_dataset(dataset_root=config.datasets.dataset_root,
                          dataset_name=config.datasets.dataset_name)
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    if config.models.param.graph_classification:
        dataloader_params = {'batch_size': config.models.param.batch_size,
                             'stratified': config.stratified,
                             'random_split_flag': config.datasets.random_split_flag,
                             'data_split_ratio': config.datasets.data_split_ratio,
                             'seed': config.datasets.seed}
        
    if config.train_flag:
    
        if config.concept_whitening:
            data_lists = []
            print(f'Concepts: {config.concepts.split(",")}')
            for concept in config.concepts.split(','):
                files = os.listdir(os.path.join(config.concept_dir, concept+'/'+config.datasets.dataset_name+'/processed/'))
                files = [f for f in files if 'data' in f]

                data_list = [torch.load(os.path.join(config.concept_dir+concept+'/'+config.datasets.dataset_name+'/processed/', file)) for file in files]
                data_lists.append(data_list)


            concept_loaders = [get_dataloader(data_list, dataloader_params['batch_size'], data_split_ratio=[0.8,0.0,0.2], stratified=False) for data_list in data_lists]
            
            model = load_trained_model(config, dataset)
            train_params = {'num_epochs': 50,
                            'num_early_stop': 20,
                            'milestones': config.models.param.milestones,
                            'gamma': config.models.param.gamma}
            optimizer_params = {'lr': 0.0005,
                                'weight_decay': 0}
            
        else:
            concept_loaders = [None]

            model = get_gnnNets(dataset.num_node_features, dataset.num_classes, config.models, config.concept_whitening)
            train_params = {'num_epochs': config.models.param.num_epochs,
                            'num_early_stop': config.models.param.num_early_stop,
                            'milestones': config.models.param.milestones,
                            'gamma': config.models.param.gamma}
            optimizer_params = {'lr': config.models.param.learning_rate,
                                'weight_decay': config.models.param.weight_decay}
                                #'momentum': 0.9}

        if config.models.param.graph_classification:
            trainer = TrainModel(model=model,
                                 dataset=dataset,
                                 concept_loaders=concept_loaders,
                                 device=device,
                                 concept_whitening=config.concept_whitening,
                                 statistics_dir=config.statistics_dir,
                                 graph_classification=config.models.param.graph_classification,
                                 save_dir=os.path.join(config.models.gnn_saving_dir,
                                                       config.datasets.dataset_name),
                                 save_name=f'{config.models.gnn_name}_{len(config.models.param.gnn_latent_dim)}l',
                                 dataloader_params=dataloader_params)
        else:
            trainer = TrainModel(model=model,
                                 dataset=dataset,
                                 concept_loaders=concept_loaders,
                                 device=device,
                                 concept_whitening=config.concept_whitening,
                                 statistics_dir=config.statistics_dir,
                                 graph_classification=config.models.param.graph_classification,
                                 save_dir=os.path.join(config.models.gnn_saving_dir,
                                                       config.datasets.dataset_name),
                                 save_name=f'{config.models.gnn_name}_{len(config.models.param.gnn_latent_dim)}l')
        trainer.train(train_params=train_params, optimizer_params=optimizer_params)
        test_loss, test_acc, test_auc, _ = trainer.test()
        
        # Saving losses and metrics values on json file
        listObj = []
        
        with open(config.statistics_dir, 'r') as fp:
          listObj = json.load(fp)
        
        n = len(listObj)-1
        listObj[n]['test_loss'] = test_loss
        listObj[n]['test_acc'] = test_acc
        listObj[n]['test_auc'] = test_auc

        with open(config.statistics_dir, 'w') as json_file:
            json.dump(listObj, json_file, indent=4, separators=(',',': '))
        
    else:
        if config.models.param.graph_classification:
            loader = get_dataloader(dataset, **dataloader_params)
            
        plot_dst = os.path.join(config.base_dir, 'plot/')
        if not os.path.exists(plot_dst):
            os.mkdir(plot_dst)
        
        print("Computing scatter plot...")
        scatter_plot(config, loader['test'], whitened_layers=[2], plot_cpt = config.concepts.split(','), N = 3, normalize_by='layer')
        print("Computing scatter plot...")
        show_scatter_plot(config, loader['test'], whitened_layers=[0,1,2], plot_cpt = config.concepts.split(','), N = 3, normalize_by='concept')
        print("Computing trajectory...")
        plot_trajectory(config, loader['test'], plt_cpt = config.concepts.split(','))
        print("Computing intra- and inter-concept similarity")
        intra_concept_dot_product_vs_inter_concept_dot_product(config, loader['test'], 2, plot_cpt = config.concepts.split(','), cw = config.concept_whitening)
        print("... Done")


if __name__ == '__main__':
    main()
