import os
import torch
from gnnNets import get_gnnNets


def check_dir(save_dirs):
    if save_dirs:
        if os.path.isdir(save_dirs):
            pass
        else:
            os.makedirs(save_dirs)

def load_trained_model(config, dataset):
    model_dir = config.base_dir + 'trained_models/'
    
    if torch.cuda.is_available():
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')
    
    # Defining the model
    model = get_gnnNets(dataset.num_node_features, dataset.num_classes, config.models, config.concept_whitening)
    model = model.to(device)
    
    # CHANGE MODEL TO LOAD HERE
    model_path = os.path.join(model_dir+dataset.name+'/', config.models.gnn_name+"_graphnorm_baseline.pth")
    print(f'Loading model: {model_path}')

    if os.path.isfile(model_path):
        state_dict = torch.load(model_path)['net']
        model.load_state_dict(state_dict)
    else:
        raise Exception("checkpoint {} not found!".format(model_path))
       
    # Replace BatchNorm with CW layer
    model.replace_norm_layers()

    return model

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

