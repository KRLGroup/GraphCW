import statistics
import argparse
import json
import os

parser = argparse.ArgumentParser(description='Computing statistics over 15 runs')
parser.add_argument('path', default='', type=str, metavar='PATH',
                    help='path to statistics file')
parser.add_argument('--mode', default='', type=str)
parser.add_argument('--path_out', default='', type=str)
parser.add_argument('--index', default=-1, type=int)
parser.add_argument('--concepts', default='', type=str)

def compute_statistics(path, path_out, mode, index, concepts):
    listObj = []
        
    with open(path, 'r') as fp:
        if len(fp.readlines()) != 0:
            fp.seek(0)
            listObj = json.load(fp)
            
    train_losses, train_accs, train_aucs = [], [], []
    val_losses, val_accs, val_aucs = [], [], []
    test_losses, test_accs, test_aucs = [], [], []
    
    for entry in listObj:
        if 'train_loss' in entry.keys():
            train_losses.append(entry['train_loss'])
            train_accs.append(entry['train_acc'])
            train_aucs.append(entry['train_auc'])

            val_losses.append(entry['val_loss'])
            val_accs.append(entry['val_acc'])
            val_aucs.append(entry['val_auc'])

            test_losses.append(entry['test_loss'])
            test_accs.append(entry['test_acc'])
            test_aucs.append(entry['test_auc'])
        
    if mode == 'top10':
        d = {}
        for i in range(len(test_accs)):
            d[i] = test_accs[i]
        sorted_d = {k : v for k, v in sorted(d.items(), key = lambda v: v[1], reverse=True)}
        
        top10 = list(sorted_d.items())[:10]
        
        train_losses_10, train_accs_10, train_aucs_10 = [], [], []
        val_losses_10, val_accs_10, val_aucs_10 = [], [], []
        test_losses_10, test_accs_10, test_aucs_10 = [], [], []
        
        for elem in top10:
            train_losses_10.append(train_losses[elem[0]])
            train_accs_10.append(train_accs[elem[0]])
            train_aucs_10.append(train_aucs[elem[0]])

            val_losses_10.append(val_losses[elem[0]])
            val_accs_10.append(val_accs[elem[0]])
            val_aucs_10.append(val_aucs[elem[0]])

            test_losses_10.append(test_losses[elem[0]])
            test_accs_10.append(test_accs[elem[0]])
            test_aucs_10.append(test_aucs[elem[0]])
            
        train_losses, train_accs, train_aucs = train_losses_10, train_accs_10, train_aucs_10
        val_losses, val_accs, val_aucs = val_losses_10, val_accs_10, val_aucs_10
        test_losses, test_accs, test_aucs = test_losses_10, test_accs_10, test_aucs_10
        
    if mode == 'median':
        mean_train_loss = statistics.median(train_losses)
        mean_train_acc = statistics.median(train_accs)
        mean_train_auc = statistics.median(train_aucs)

        mean_val_loss = statistics.median(val_losses)
        mean_val_acc = statistics.median(val_accs)
        mean_val_auc = statistics.median(val_aucs)

        mean_test_loss = statistics.median(test_losses)
        mean_test_acc = statistics.median(test_accs)
        mean_test_auc = statistics.median(test_aucs)
        
        test_acc_stddev = statistics.pstdev(test_accs)
        test_auc_stddev = statistics.pstdev(test_aucs)
    
    else:
        mean_train_loss = sum(train_losses)/len(train_losses)
        mean_train_acc = sum(train_accs)/len(train_accs)
        mean_train_auc = sum(train_aucs)/len(train_aucs)

        mean_val_loss = sum(val_losses)/len(val_losses)
        mean_val_acc = sum(val_accs)/len(val_accs)
        mean_val_auc = sum(val_aucs)/len(val_aucs)

        mean_test_loss = sum(test_losses)/len(test_losses)
        mean_test_acc = sum(test_accs)/len(test_accs)
        mean_test_auc = sum(test_aucs)/len(test_aucs)
        
        test_acc_stddev = statistics.pstdev(test_accs)
        test_auc_stddev = statistics.pstdev(test_aucs)
    
    listObj = []
    print(index)
    
    if index != -1:
        if os.path.isfile(path_out) is False:
            print('Creating new saving file')
            open(path_out, 'a').close()
            
        with open(path_out, 'r') as fp:
            if len(fp.readlines()) != 0:
                fp.seek(0)
                listObj = json.load(fp)
        
        listObj.append({
                      "Index": index,
                      "Concepts:": concepts,
                      "mean_train_loss": mean_train_loss,
                      "mean_train_acc": mean_train_acc,
                      "mean_train_auc":mean_train_auc,
                      "mean_val_loss": mean_val_loss,
                      "mean_val_acc": mean_val_acc,
                      "mean_val_auc": mean_val_auc,
                      "mean_test_loss": mean_test_loss,
                      "mean_test_acc": mean_test_acc,
                      "test_acc_stddev": test_acc_stddev,
                      "test_auc_stddev": test_auc_stddev,
                      "mean_test_auc": mean_test_auc,
                    })

            
    elif index == -1:
        path_out = path
        
        with open(path_out, 'r') as fp:
            if len(fp.readlines()) != 0:
                fp.seek(0)
                listObj = json.load(fp)
            
        listObj.append({
                      "Runs": len(train_losses),
                      "mean_train_loss": mean_train_loss,
                      "mean_train_acc": mean_train_acc,
                      "mean_train_auc":mean_train_auc,
                      "mean_val_loss": mean_val_loss,
                      "mean_val_acc": mean_val_acc,
                      "mean_val_auc": mean_val_auc,
                      "mean_test_loss": mean_test_loss,
                      "mean_test_acc": mean_test_acc,
                      "test_acc_stddev": test_acc_stddev,
                      "test_auc_stddev": test_auc_stddev,
                      "mean_test_auc": mean_test_auc,
                    })
    
    with open(path_out, 'w') as json_file:
        json.dump(listObj, json_file, indent=4, separators=(',',': '))
        
    print(f'Mean train loss = {mean_train_loss}')
    print(f'Mean train accuracy = {mean_train_acc}')
    print(f'Mean train auc roc = {mean_train_auc}')
    
    print(f'Mean val loss = {mean_val_loss}')
    print(f'Mean val accuracy = {mean_val_acc}')
    print(f'Mean val auc roc = {mean_val_auc}')
    
    print(f'Mean test loss = {mean_test_loss}')
    print(f'Mean test accuracy = {mean_test_acc}')
    print(f'Mean test auc roc = {mean_test_auc}')
    
    print(f'Test acc std dev = {test_acc_stddev}')
    print(f'Test auc std dev = {test_auc_stddev}')
    
def main():
    args = parser.parse_args()
    
    compute_statistics(args.path, args.path_out, args.mode, args.index, args.concepts)
    
if __name__ == '__main__':
    main()