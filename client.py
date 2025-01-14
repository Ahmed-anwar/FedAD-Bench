from collections import OrderedDict
import flwr
from flwr.common.logger import log
import torch 
from model.bootstrap import resolve_model_trainer

from logging import INFO

class FlowerClient(flwr.client.NumPyClient):
    def __init__(
                self, 
                cid, 
                cfg,
                dataset, 
                train_loader, 
                metrics_cfg = {}, 
    ):
        super().__init__()
        self.cid = cid
        self.train_loader = train_loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model, self.model_trainer = resolve_model_trainer(
            model_name=cfg.model_name,
            model_params=cfg.model,
            dataset=dataset,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            weight_decay=cfg.weight_decay,
            learning_rate=cfg.lr,
            device=self.device
        )
        self.model_trainer.before_training(self.train_loader)
        self.metrics_cfg = metrics_cfg

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def fit(self, parameters, config):
        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)
        lr = config['lr']
        epochs = config['local_epochs']
        mu = config['fedprox_mu']
        loss = self.model_trainer.train(self.train_loader, mu)

        return self.get_parameters({}), len(self.train_loader), {'train_loss': loss}


def generate_client_fn(
                        cfg,
                        dataset, 
                        train_loaders, 
                        metrics_cfg = {}
                    ):
    
    def client_fn(cid):
        log(INFO, f'{"-"*20} Client {cid} {"-"*20}')

        return FlowerClient(cid, 
                            cfg,
                            dataset, 
                            train_loaders[int(cid)],
                            metrics_cfg, 
                            )
        
    return client_fn