'''
This script has been either partially or fully inspired by the repository:

https://github.com/ireydiak/anomaly_detection_NRCAN

Alvarez, M., Verdier, J.C., Nkashama, D.K., Frappier, M., Tardif, P.M.,
Kabanza, F.: A revealing large-scale evaluation of unsupervised anomaly
detection algorithms
'''

import numpy as np
import torch

from abc import ABC, abstractmethod
from typing import Union
from sklearn import metrics as sk_metrics
from torch.utils.data.dataloader import DataLoader
from torch import optim
from tqdm import trange
import os, sys
# path = os.getcwd()
# print(path)
# sys.path.append("..")
# print(os.getcwd())
# os.chdir("..")
# print(os.getcwd())
# from ... import BaseModel
from model.utils import metrics
import matplotlib.pyplot as plt

def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

class BaseTrainer(ABC):
    name = "BaseTrainer"

    def __init__(self,
                 model,
                 batch_size,
                 lr: float = 1e-4,
                 n_epochs: int = 200,
                 n_jobs_dataloader: int = 0,
                 device: str = "cuda",
                 anomaly_label=1,
                 ckpt_root: str = None,
                 validation_ldr=None,
                 **kwargs):
        self.device = device
        self.model = model.to(device)
        self.batch_size = batch_size
        self.n_jobs_dataloader = n_jobs_dataloader
        self.n_epochs = n_epochs
        self.lr = lr
        self.anomaly_label = anomaly_label
        self.weight_decay = kwargs.get('weight_decay', 0)
        self.optimizer = self.set_optimizer(weight_decay=kwargs.get('weight_decay', 0))
        self.ckpt_root = ckpt_root
        self.validation_ldr = validation_ldr
        self.metric_values = {"precision": [], "recall": [], "f1-score": [], "aupr": []}
        self.epoch = -1

    # @staticmethod
    # def load_from_file(fname: str, device: str = None):
    #     device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     ckpt = torch.load(fname, map_location=device)
    #     metric_values = ckpt["metric_values"]
    #     model = BaseModel.load_from_ckpt(ckpt)
    #     trainer = BaseTrainer(model=model, batch_size=ckpt["batch_size"], device=device)
    #     trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    #     trainer.metric_values = metric_values

    #     return trainer, model

    def save_ckpt(self, fname: str):
        general_params = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metric_values": self.metric_values
        }
        trainer_params = self.get_params()
        model_params = self.model.get_params()
        torch.save(dict(**general_params, **model_params, **trainer_params), fname)
        self.optimizer = self.set_optimizer(self.weight_decay)

    @abstractmethod
    def train_iter(self, sample: torch.Tensor):
        pass

    @abstractmethod
    def score(self, sample: torch.Tensor):
        pass

    def after_training(self):
        """
        Perform any action after training is done
        """
        pass

    def before_training(self, dataset: DataLoader):
        """
        Optionally perform pre-training or other operations.
        """
        pass

    def set_optimizer(self, weight_decay):
        return optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)

    def train(self, dataset: DataLoader, mu=-1):
        self.model.train(mode=True)
        self.before_training(dataset)
        assert self.model.training, "Model not in training mode. Aborting"
        global_params = list(self.model.parameters()).copy()
        print("Started training")
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            self.epoch = epoch
            assert self.model.training, "model not in training mode, aborting"
            with trange(len(dataset)) as t:
                for sample in dataset:
                    X, _, _ = sample
                    #print(len(X), len(X[0]))
                    # X, _ = sample
                    X = X.to(self.device).float()

                    # Reset gradient
                    self.optimizer.zero_grad()

                    loss = self.train_iter(X)
                    
                    # Apply FedProx
                    if mu != -1 and global_params != None:
                        print('Applying FedProx', flush=True)
                        proximal_term = 0
                        for local_weights, global_weights in zip(self.model.parameters(), global_params):
                            proximal_term += (local_weights - global_weights).norm(2)
                        loss += (mu / 2) * proximal_term
                        
                    # Backpropagation
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    t.set_postfix(
                        loss='{:.3f}'.format(epoch_loss / (epoch + 1)),
                        epoch=epoch + 1
                    )
                    t.update()

            if self.ckpt_root and epoch % 5 == 0:
                self.save_ckpt(
                    os.path.join(self.ckpt_root, "{}_epoch={}.pt".format(self.model.name.lower(), epoch + 1))
                )

            if self.validation_ldr is not None and (epoch % 5 == 0 or epoch == 0):
                self.validate()

        self.after_training()
        print('TRAIN LOSS', epoch_loss / (self.n_epochs + 1), flush=True)
        return epoch_loss / (self.n_epochs + 1)

    def eval(self, dataset: DataLoader):
        self.model.eval()
        with torch.no_grad():
            loss = 0
            for row in dataset:
                X, _,_ = row
                # X, _ = row
                X = X.to(self.device).float()
                loss += self.train_iter(X)
            loss /= len(dataset)
        self.model.train()
        print('EVAL LOSS', loss, flush=True)
        return loss.detach().cpu().numpy()

    def test(self, dataset: DataLoader) -> Union[np.array, np.array]:
        self.model.eval()
        y_true, scores, labels = [], [], []
        # print('START TEST IN  BASE', flush=True)
        with torch.no_grad():
            for row in dataset:
                X, label, _ = row
                # X, label = row
                X = X.to(self.device).float()
                score = self.score(X)
                y_true.extend(label.cpu().tolist())
                labels.extend(list(label))
                scores.extend(score.cpu().tolist())
        self.model.train()

        return np.array(y_true), np.array(scores), np.array(labels)

    def get_params(self) -> dict:
        return {
            "lr": self.lr,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "epoch": self.epoch,
        }

    def predict(self, scores: np.array, thresh: float):
        return (scores >= thresh).astype(int)

    def evaluate(
            self,
            y_true: np.array,
            scores: np.array,
            thresh: float = None,
            pos_label: int = 1
    ) -> dict:
        res = {"Precision": -1, "Recall": -1, "F1-Score": -1, "AUROC": -1, "AUPR": -1}

        thresh = thresh or (y_true == self.anomaly_label) / len(y_true)
        thresh = np.percentile(scores, thresh)
        y_pred = self.predict(scores, thresh)
        res["Precision"], res["Recall"], res["F1-Score"], _ = sk_metrics.precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=pos_label
        )
        res["AUROC"] = sk_metrics.roc_auc_score(y_true, scores)
        res["AUPR"] = sk_metrics.average_precision_score(y_true, scores)
        return res

    def plot_metrics(self, figname="fig1.png"):
        """
        Function that plots train and validation losses and accuracies after
        training phase
        """

        precision, recall = self.metric_values["precision"], self.metric_values["recall"]
        f1, aupr = self.metric_values["f1-score"], self.metric_values["aupr"]
        epochs = range(1, len(precision) + 1)

        f, ax1 = plt.subplots(figsize=(10, 5))

        ax1.plot(
            epochs, precision, '-o', label="Precision", c="b"
        )
        ax1.plot(
            epochs, recall, '-o', label="Recall", c="g"
        )
        ax1.plot(
            epochs, aupr, '-o', label="AUPR", c="c"
        )
        ax1.plot(
            epochs, f1, '-o', label="F1-Score", c="r"
        )
        ax1.set_xlabel("Epochs", fontsize=16)
        ax1.set_ylabel("Metrics", fontsize=16)
        ax1.legend(fontsize=14)

        f.savefig(figname)
        plt.show()

    def validate(self):
        y_true, scores, _ = self.test(self.validation_ldr)
        self.model.train(mode=True)
        res, _ = metrics.score_recall_precision_w_threshold(scores, y_true)
        self.metric_values["precision"].append(res["Precision"])
        self.metric_values["recall"].append(res["Recall"])
        self.metric_values["f1-score"].append(res["F1-Score"])
        self.metric_values["aupr"].append(res["AUPR"])


class BaseShallowTrainer(ABC):

    def __init__(self,
                 model,
                 batch_size,
                 lr: float = 1e-4,
                 n_epochs: int = 200,
                 n_jobs_dataloader: int = 0,
                 device: str = "cuda",
                 anomaly_label=1,
                 ckpt_root: str = None,
                 validation_ldr=None,
                 **kwargs
                 ):
        """
        Parameters are mostly ignored but kept for better code consistency

        Parameters
        ----------
        model
        batch_size
        lr
        n_epochs
        n_jobs_dataloader
        device
        """
        self.batch_size = None
        self.lr = None
        self.n_epochs = None
        self.n_jobs_dataloader = None
        self.device = None
        self.anomaly_label = 1
        self.ckpt_root = None
        self.validation_ldr = None
        self.model = model

    def train(self, dataset: DataLoader):
        self.model.clf.fit(dataset.dataset.dataset.X)

    def score(self, sample: torch.Tensor):
        return self.model.predict(sample.numpy())

    def test(self, dataset: DataLoader) -> Union[np.array, np.array]:
        y_true, scores, labels = [], [], []
        for row in dataset:
            X, y, label = row
            s = self.score(X)
            y_true.extend(y.cpu().tolist())
            scores.extend(s)
            labels.extend(list(label))

        return np.array(y_true), np.array(scores), np.array(labels)

    def get_params(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "lr": self.lr,
            "n_epochs": self.n_epochs
        }

    def predict(self, scores: np.array, thresh: float):
        return (scores >= thresh).astype(int)

    def evaluate(self, y_true: np.array, scores: np.array, threshold: float, pos_label: int = 1) -> dict:
        res = {"Precision": -1, "Recall": -1, "F1-Score": -1, "AUROC": -1, "AUPR": -1}

        thresh = np.percentile(scores, threshold)
        y_pred = self.predict(scores, thresh)
        res["Precision"], res["Recall"], res["F1-Score"], _ = sk_metrics.precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=pos_label
        )

        res["AUROC"] = sk_metrics.roc_auc_score(y_true, scores)
        res["AUPR"] = sk_metrics.average_precision_score(y_true, scores)
        return res
