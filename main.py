import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import flwr as fl

from server import get_evaluate_fn, get_on_fit_config, get_on_eval_config, start_server
from client import generate_client_fn
from model.bootstrap import resolve_model_trainer
from review_results import plot_metric
from dataset.datamanager.dataset import *

from pathlib import Path
import pickle
import os
import torch
import random
import numpy
import wandb

import warnings
warnings.filterwarnings("ignore")

def aggregation(res):
    ret = {}
    metrics = list(res[0][1].keys())
    for metric in metrics:
        ret[metric] = 0

    for _, metrics_dict in res:
        for metric, value in metrics_dict.items():    
            ret[metric] += value

    for k in ret.keys():
        ret[k] /= len(res)
        
    return ret

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    #save_path = HydraConfig.get().runtime.output_dir
    save_path = 'pickled_results'
    save_path = os.path.join(save_path, cfg.experiment)

    cfg.results_path = os.path.join(cfg.results_path, str(cfg.num_clients), cfg.model_name)
    

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    

    # Uncomment if you want to use WandB
    # Instantiate Wandb
    
    # wandb.login(key="2182ae8b1eac63f54fd46c909a906d8ca8c4b97a")
    # wandb.init(
    #     project="FedAD-Bench",
    #     config={
    #         "expertiment": cfg.experiment,
    #         "learning_rate": cfg.config_fit.lr,
    #         "model": cfg.model,
    #         "dataset": cfg.dataset.dataset,
    #         "rounds": cfg.num_rounds,
    #         "clients": cfg.num_clients,
    #         "local_epochs": cfg.config_fit.local_epochs,
    #         "resample": cfg.resampler,
    #         "name": cfg.experiment
    #      }
    # )


    # prepare dataset
    partition_algorithm = cfg.partition

    # Dynamically load the Dataset instance
    dataset_name = cfg.dataset.dataset
    dataset_path = cfg.dataset.dataset_path
    normal_size = cfg.normal_size
    models_path = cfg.model_path
    results_path = cfg.results_path
    

    clsname = globals()[f'{dataset_name}Dataset']
    dataset = clsname(path=dataset_path, normal_size=normal_size)
    anomaly_thresh = 1 - dataset.anomaly_ratio

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loaders, test_loader, _ = dataset.loaders_FL(
                    batch_size=cfg.batch_size,
                    seed=cfg.seed,
                    contamination_rate=cfg.rho,
                    validation_ratio=cfg.val_ratio,
                    holdout=cfg.hold_out,
                    drop_last_batch=cfg.drop_last_batch,
                    num_clients = cfg.num_clients
                )
    
    # prepare client generating function
    client_fn = generate_client_fn(
                        cfg,
                        dataset, 
                        train_loaders, 
                        metrics_cfg = {},
                    )
    
    # prepare strategy 
    model, model_trainer = resolve_model_trainer(
        model_name=cfg.model_name,
        model_params=cfg.model,
        dataset=dataset,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        weight_decay=cfg.weight_decay,
        learning_rate=cfg.lr,
        device=device
    )

    params = [val.cpu().numpy() for _, val in model.state_dict().items()]
    initial_parameters = fl.common.ndarrays_to_parameters(params)
    
    strategy = instantiate(cfg.strategy, evaluate_fn=get_evaluate_fn(cfg, test_loader, dataset),
                on_fit_config_fn=get_on_fit_config(cfg.config_fit),_recursive_=False,
                on_evaluate_config_fn=get_on_eval_config(cfg.config_eval),
                initial_parameters=initial_parameters,
                fit_metrics_aggregation_fn=aggregation)

    # start simulation
    if cfg.id == -2:
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=cfg.num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": cfg.num_cpus, "num_gpus": cfg.num_gpus/cfg.num_clients},
        )
        
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(cfg.results_path, exist_ok=True)
        
        # save result       
        results_path = Path(save_path) / f'{cfg.experiment}.pkl'
        results = {
                    'name': cfg.experiment,
                    'metrics_centralized': history.metrics_centralized, 
                    'loss_centralized': history.losses_centralized, 
                    'metrics_distributed': history.metrics_distributed               
                   }
        with open(str(results_path), "wb") as h:
            pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

        plot_metric('loss', history.losses_centralized, path=os.path.join(save_path, 'loss.png'))
        print('####### Centralized Results ########')
        for metric, values in history.metrics_centralized.items():
            plot_metric(metric, values, path=os.path.join(save_path, f'{metric}.png'))
            print(f'{metric}: {values[-1]}')
               
        print('####### Distributed Results ########')
        for metric, values in history.metrics_distributed_fit.items():
            plot_metric(metric, values, path=os.path.join(save_path, f'{metric}_dist.png'))
            print(f'{metric}_distributed: {values[-1]}', flush=True)
        
        with open(os.path.join(cfg.results_path, f'{cfg.model_name}_{dataset_name}_{cfg.num_clients}.txt'), "a") as fi:
            fi.write(f'{save_path} \n')
            for metric, values in history.metrics_centralized.items():
                fi.write(f'{metric}: {values[-5:]}\n')
            for metric, values in history.metrics_distributed_fit.items():
                fi.write(f'{metric}: {values[-5:]}\n')

    elif cfg.id == -1:
        start_server(strategy)
    
    else:
        client = client_fn(cfg.id)
        fl.client.start_numpy_client(server_address='localhost:5040', client=client)
    
    # Uncomment if you use WandB
    # wandb.finish()
    
if __name__ == "__main__":
    main()
