'''
This script has been either partially or fully inspired by the repository:

https://github.com/ireydiak/anomaly_detection_NRCAN

Alvarez, M., Verdier, J.C., Nkashama, D.K., Frappier, M., Tardif, P.M.,
Kabanza, F.: A revealing large-scale evaluation of unsupervised anomaly
detection algorithms
'''

import torch

from torch.utils.data import DataLoader
from .base import BaseShallowTrainer


class RecForestTrainer(BaseShallowTrainer):

    def score(self, sample: torch.Tensor):
        return self.model.clf.predict(sample.numpy())

    def train(self, dataset: DataLoader):
        self.model.clf.fit(dataset.dataset.dataset.X)

    def get_params(self) -> dict:
        return {
            **self.model.get_params()
        }


class OCSVMTrainer(BaseShallowTrainer):

    def score(self, sample: torch.Tensor):
        return -self.model.clf.predict(sample.numpy())

    def train(self, dataset: DataLoader):
        self.model.clf.fit(dataset.dataset.dataset.X)

    def get_params(self) -> dict:
        return {
            **self.model.get_params()
        }


class LOFTrainer(BaseShallowTrainer):

    def score(self, sample: torch.Tensor):
        return -self.model.clf.score_samples(sample.numpy())

    def train(self, dataset: DataLoader):
        self.model.clf.fit(dataset.dataset.dataset.X)


class PCATrainer(BaseShallowTrainer):

    def score(self, sample: torch.Tensor):
        sample = sample.numpy()
        z = self.model.clf.transform(sample)
        x_hat = self.model.clf.inverse_transform(z)
        return ((sample - x_hat) ** 2).sum(axis=1)

    def train(self, dataset: DataLoader):
        self.model.clf.fit(dataset.dataset.dataset.X)
