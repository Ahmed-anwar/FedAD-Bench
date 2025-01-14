'''
This script has been either partially or fully inspired by the repository:

https://github.com/ireydiak/anomaly_detection_NRCAN

Alvarez, M., Verdier, J.C., Nkashama, D.K., Frappier, M., Tardif, P.M.,
Kabanza, F.: A revealing large-scale evaluation of unsupervised anomaly
detection algorithms
'''

import os

import numpy as np
import torch
import torch.nn as nn

from typing import Union
from torch import optim
from torch.nn import Parameter
from torch.utils.data import DataLoader
from tqdm import trange
from model.model.density import DSEBM
from model.trainer.base import BaseTrainer


class DSEBMTrainer(BaseTrainer):
    def __init__(self, score_metric="reconstruction", b_prime: torch.Tensor = None, **kwargs):
        assert score_metric == "reconstruction" or score_metric == "energy"
        super(DSEBMTrainer, self).__init__(**kwargs)
        self.score_metric = score_metric
        self.criterion = nn.BCEWithLogitsLoss()
        if b_prime is not None:
            self.b_prime = b_prime
        else:
            self.b_prime = Parameter(torch.Tensor(self.batch_size, self.model.in_features).to(self.device))
        torch.nn.init.xavier_normal_(self.b_prime)
        self.optim = optim.Adam(
            list(self.model.parameters()) + [self.b_prime],
            lr=self.lr, betas=(0.5, 0.999)
        )

    def save_ckpt(self, fname: str):
        general_params = {
            "epoch": self.epoch,
            "batch_size": self.batch_size,
            "model_state_dict": self.model.state_dict(),
            "b_prime": self.b_prime,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metric_values": self.metric_values
        }
        model_params = self.model.get_params()
        torch.save(dict(**general_params, **model_params), fname)

    @staticmethod
    def load_from_file(fname: str, device: str = None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(fname, map_location=device)
        metric_values = ckpt["metric_values"]
        model = DSEBM.load_from_ckpt(ckpt)
        trainer = DSEBMTrainer(model=model, batch_size=ckpt["batch_size"], device=device, b_prime=ckpt["b_prime"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.metric_values = metric_values

        return trainer, model

    def train_iter(self, X):
        noise = self.model.random_noise_like(X).to(self.device)
        X_noise = X + noise
        X.requires_grad_()
        X_noise.requires_grad_()
        out_noise = self.model(X_noise)
        energy_noise = self.energy(X_noise, out_noise)
        dEn_dX = torch.autograd.grad(energy_noise, X_noise, retain_graph=True, create_graph=True)
        fx_noise = (X_noise - dEn_dX[0])
        return self.loss(X, fx_noise)

    def train(self, dataset: DataLoader, mu=-1):
        self.model.train()
        global_params = list(self.model.parameters()).copy()

        print("Started training")
        for epoch in range(self.n_epochs):
            assert self.model.training, "model not in training mode, aborting"
            epoch_loss = 0.0
            self.epoch = epoch
            with trange(len(dataset)) as t:
                for sample in dataset:
                    X, _, _ = sample
                    X = X.to(self.device).float()

                    if len(X) < self.batch_size:
                        t.update()
                        break

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
                        loss='{:05.3f}'.format(epoch_loss/(epoch + 1)),
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

    def score(self, sample: torch.Tensor):
        # Evaluation of the score based on the energy
        with torch.no_grad():
            flat = sample - self.b_prime
            out = self.model(sample)
            energies = 0.5 * torch.sum(torch.square(flat), dim=1) - torch.sum(out, dim=1)

        # Evaluation of the score based on the reconstruction error
        sample.requires_grad_()
        out = self.model(sample)
        energy = self.energy(sample, out)
        dEn_dX = torch.autograd.grad(energy, sample)[0]
        rec_errs = torch.linalg.norm(dEn_dX, 2, keepdim=False, dim=1)
        return energies.cpu().numpy(), rec_errs.cpu().numpy()

    def test(self, dataset: DataLoader) -> Union[np.array, np.array]:
        self.model.eval()
        y_true, scores, labels = [], [], []
        scores_e, scores_r = [], []
        #print("IN TRAINER TEST 1", flush=True)
        for row in dataset:
            X, y, label = row
            X = X.to(self.device).float()

            score_e, score_r = self.score(X)

         #   print("IN TRAINER TEST 2", flush=True)
            y_true.extend(y.cpu().tolist())
            scores_e.extend(score_e)
            scores_r.extend(score_r)
            labels.extend(list(label))

        scores = scores_r if self.score_metric == "reconstruction" else scores_e
        return np.array(y_true), np.array(scores), np.array(labels)

    def loss(self, X, fx_noise):
        out = torch.square(X - fx_noise)
        out = torch.sum(out, dim=-1)
        out = torch.mean(out)
        return out

    def energy(self, X, X_hat):
        if len(X) != len(self.b_prime):
            return 0
        return 0.5 * torch.sum(torch.square(X - self.b_prime)) - torch.sum(X_hat)

    def eval(self, dataset: DataLoader):
        self.model.eval()
        loss = 0
        # print('IN EVAL 1', flush=True)
        for row in dataset:
            X, _,_ = row
            # X, _ = row
            if len(X) < self.batch_size:
                continue

            # print('IN EVAL 2', flush=True)
            X = X.to(self.device).float()
            # print('IN EVAL 3', flush=True)
            loss += self.train_iter(X)
            # print('IN EVAL 4', flush=True)
        loss /= len(dataset)
        self.model.train()
        print('EVAL LOSS', loss, flush=True)
        return loss.detach().cpu().numpy()