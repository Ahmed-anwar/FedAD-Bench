

from omegaconf import DictConfig
import torch
from collections import OrderedDict
import flwr as fl

from model.bootstrap import resolve_model_trainer
from model.utils import metrics


def start_server(strategy):
    fl.server.start_server(
    server_address="localhost:5040",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
    )

def get_on_fit_config(config: DictConfig):
    """Return a function to configure the client's fit."""

    def fit_config_fn(server_round: int):
        lr = config.lr
        return {
            'lr': lr,
            'local_epochs': config.local_epochs,
            'local_optimizer': str(config.local_optimizer),
            'metrics': str(config.metrics),
            'undersampling_p': config.undersampling_p,
            'oversampling_p': config.oversampling_p,
            'printing': config.printing,
            'fedprox_mu': config.fedprox_mu
        }

    return fit_config_fn

def get_evaluate_fn(cfg, test_loader, dataset):
    """Return a function to evaluate the global model."""

    def evaluate_fn(server_round: int, parameters, config):
        if server_round < 1:
            return 0.0, {}
        
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model, trainer = resolve_model_trainer(
            model_name=cfg.model_name,
            model_params=cfg.model,
            dataset=dataset,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            weight_decay=cfg.weight_decay,
            learning_rate=cfg.lr,
            device=device
        )
        
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        trainer.model = model
        trainer.before_training(test_loader)
        y_test_true, test_scores, _ = trainer.test(test_loader)
        results = metrics.estimate_optimal_threshold(test_scores, y_test_true)

        loss = trainer.eval(test_loader)
        results['eval_loss'] = loss
        return loss, results
    
    return evaluate_fn

def get_on_eval_config(config: DictConfig):
    """Return a function to configure the client's eval."""

    def eval_config_fn(server_round: int):
        return {'metrics': str(config.metrics)}
    
    return eval_config_fn
