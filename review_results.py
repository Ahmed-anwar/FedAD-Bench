import matplotlib
matplotlib.use('Agg') # This is to avoid segmentation fault at process exit
import matplotlib.pyplot as plt
import torch 
import pickle
import os
import numpy as np
def get_results_from_pickle(path):
    
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

def plot_metric(metric_name, values, path=None):
    fz = 13
    plt.figure(figsize=(8, 6))
    steps = [value[0] for value in values]
    metric_values = [value[1] for value in values]
    plt.xlabel('Round Number', fontsize=fz)
    plt.ylabel(metric_name, fontsize=fz)
    if torch.is_tensor(metric_values):
        metric_values = metric_values.detach().cpu().numpy() 
    plt.plot(steps, metric_values)
    if path is None:
        plt.show()
    else:
        plt.savefig(path)

    plt.close('all')

def plot_metric_all_exps(metric_name, exps_results, exps_names, path=None):
    fz = 13
    plt.figure(figsize=(8, 6))
    # ls = ['-','--',':','dotted']
    for i in range(len(exps_results)):
        exp = exps_results[i]
        exp_name = exps_names[i]
        
        values = exp['metrics_centralized'][metric_name]
        steps = [value[0] for value in values]
        metric_values = [value[1] for value in values]
        plt.plot(steps, np.array(metric_values)*(i+1), label=exp_name)

    ylabel = metric_name
    if metric_name == 'auc_roc':
        ylabel = 'Area Under ROC curve'
    elif metric_name == 'precision':
        ylabel = 'Precision'
    elif metric_name == 'recall':
        ylabel = 'Recall'
    elif metric_name == 'eval_loss':
        ylabel = 'Evaluation Loss'

    plt.xlabel('Round Number', fontsize=fz)
    plt.ylabel(ylabel, fontsize=fz)
    plt.legend()
    plt.grid()

    if path is None:
        plt.show()
    else:
        plt.savefig(path)

if __name__ == '__main__':
    all_paths = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk('pickled_results/')
                             for f in filenames 
                             if os.path.join(dirpath,f)[-3:] == 'pkl']
    all_paths.sort()
    all_results = [get_results_from_pickle(path) for path in all_paths]
    results_names = [path.split('/')[-2] for path in all_paths]

    metrics = all_results[0]['metrics_centralized'].keys()
    print('Number of experiments: ',len(results_names))
    print('Metrics:',  list(metrics))
    count = 1
    base_path = 'figures/1'
    os.makedirs(base_path, exist_ok=True)
    for i, metric in enumerate(metrics):
        if metric in ['AUROC','Recall', 'Precision', 'F1-Score', 'eval_loss']:
            plot_metric_all_exps(metric, all_results, results_names, path=f'{base_path}/{metric}.png') # plt.subplot(int(f'12{count}'))
            count += 1
            plt.legend()
