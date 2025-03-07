'''
This script has been either partially or fully inspired by the repository:

https://github.com/ireydiak/anomaly_detection_NRCAN

Alvarez, M., Verdier, J.C., Nkashama, D.K., Frappier, M., Tardif, P.M.,
Kabanza, F.: A revealing large-scale evaluation of unsupervised anomaly
detection algorithms
'''

import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from model.model.reconstruction import AutoEncoder, DAGMM, MemAutoEncoder
from model.trainer.base import BaseTrainer
from model.loss.EntropyLoss import EntropyLoss
from torch import nn


class AutoEncoderTrainer(BaseTrainer):

    @staticmethod
    def load_from_file(fname: str, device: str = None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(fname, map_location=device)
        metric_values = ckpt["metric_values"]
        model = AutoEncoder.load_from_ckpt(ckpt)
        trainer = AutoEncoderTrainer(model=model, batch_size=ckpt["batch_size"], device=device)
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.metric_values = metric_values

        return trainer, model

    def score(self, sample: torch.Tensor):
        _, X_prime = self.model(sample)
        return ((sample - X_prime) ** 2).sum(axis=1)

    def train_iter(self, X):
        code, X_prime = self.model(X)
        l2_z = code.norm(2, dim=1).mean()
        reg = 0.5
        loss = ((X - X_prime) ** 2).sum(axis=-1).mean() + reg * l2_z

        return loss


class DAGMMTrainer(BaseTrainer):
    def __init__(self, **kwargs) -> None:
        super(DAGMMTrainer, self).__init__(**kwargs)
        self.lamb_1 = self.model.lambda_1
        self.lamb_2 = self.model.lambda_2
        self.phi = None
        self.mu = None
        self.cov_mat = None
        self.covs = None
        self.reg_covar = self.model.reg_covar

    def get_params(self) -> dict:
        params = {
            "phi": self.phi,
            "mu": self.mu,
            "cov_mat": self.cov_mat,
            "covs": self.covs,
        }
        return dict(
            **super(DAGMMTrainer, self).get_params(),
            **params
        )

    @staticmethod
    def load_from_file(fname: str, device: str = None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(fname, map_location=device)
        metric_values = ckpt["metric_values"]
        model = DAGMM.load_from_ckpt(ckpt)
        trainer = DAGMMTrainer(model=model, batch_size=1, device=device)
        trainer.model = model
        trainer.phi = ckpt["phi"]
        trainer.cov_mat = ckpt["cov_mat"]
        trainer.covs = ckpt["covs"]
        trainer.mu = ckpt["mu"]
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.metric_values = metric_values

        return trainer, model

    def train_iter(self, sample: torch.Tensor):
        z_c, x_prime, _, z_r, gamma_hat = self.model(sample)
        phi, mu, cov_mat = self.compute_params(z_r, gamma_hat)
        energy_result, pen_cov_mat = self.estimate_sample_energy(
            z_r, phi, mu, cov_mat
        )
        self.phi = phi.data
        self.mu = mu.data
        self.cov_mat = cov_mat
        return self.loss(sample, x_prime, energy_result, pen_cov_mat)

    def loss(self, x, x_prime, energy, pen_cov_mat):
        rec_err = ((x - x_prime) ** 2).mean()
        return rec_err + self.lamb_1 * energy + self.lamb_2 * pen_cov_mat

    def test(self, dataset: DataLoader):
        """
        function that evaluate the model on the test set every iteration of the
        active learning process
        """
        self.model.eval()
        with torch.no_grad():
            scores, y_true, labels = [], [], []
            for row in dataset:
                X, y, label = row
                X = X.to(self.device).float()
                # forward pass
                z_c, x_prime, _, z_r, gamma_hat = self.model(X)
                phi, mu, cov_mat = self.compute_params(z_r, gamma_hat)
                energy_result, _ = self.estimate_sample_energy(
                    z_r, phi, mu, cov_mat, average_energy=False
                )
                if energy_result.ndim == 0:
                    energy_result = energy_result.unsqueeze(0)

                y_true.extend(y)
                labels.extend(list(label))
                scores.extend(energy_result.cpu().numpy())
    

        self.model.train(mode=True)
        return np.array(y_true), np.array(scores), np.array(labels)

    def weighted_log_sum_exp(self, x, weights, dim):
        """
        Inspired by https://discuss.pytorch.org/t/moving-to-numerically-stable-log-sum-exp-leads-to-extremely-large-loss-values/61938

        Parameters
        ----------
        x
        weights
        dim

        Returns
        -------

        """
        m, idx = torch.max(x, dim=dim, keepdim=True)
        return m.squeeze(dim) + torch.log(torch.sum(torch.exp(x - m) * (weights.unsqueeze(2)), dim=dim))

    def relative_euclidean_dist(self, x, x_prime):
        return (x - x_prime).norm(2, dim=1) / x.norm(2, dim=1)

    def compute_params(self, z: torch.Tensor, gamma: torch.Tensor):
        r"""
        Estimates the parameters of the GMM.
        Implements the following formulas (p.5):
            :math:`\hat{\phi_k} = \sum_{i=1}^N \frac{\hat{\gamma_{ik}}}{N}`
            :math:`\hat{\mu}_k = \frac{\sum{i=1}^N \hat{\gamma_{ik} z_i}}{\sum{i=1}^N \hat{\gamma_{ik}}}`
            :math:`\hat{\Sigma_k} = \frac{
                \sum{i=1}^N \hat{\gamma_{ik}} (z_i - \hat{\mu_k}) (z_i - \hat{\mu_k})^T}
                {\sum{i=1}^N \hat{\gamma_{ik}}
            }`

        The second formula was modified to use matrices instead:
            :math:`\hat{\mu}_k = (I * \Gamma)^{-1} (\gamma^T z)`

        Parameters
        ----------
        z: N x D matrix (n_samples, n_features)
        gamma: N x K matrix (n_samples, n_mixtures)


        Returns
        -------

        """
        N = z.shape[0]

        # K
        gamma_sum = torch.sum(gamma, dim=0)
        phi = gamma_sum / N

        # phi = torch.mean(gamma, dim=0)

        # K x D
        # :math: `\mu = (I * gamma_sum)^{-1} * (\gamma^T * z)`
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / gamma_sum.unsqueeze(-1)
        # mu = torch.linalg.inv(torch.diag(gamma_sum)) @ (gamma.T @ z)

        mu_z = z.unsqueeze(1) - mu.unsqueeze(0)
        cov_mat = mu_z.unsqueeze(-1) @ mu_z.unsqueeze(-2)
        cov_mat = gamma.unsqueeze(-1).unsqueeze(-1) * cov_mat
        cov_mat = torch.sum(cov_mat, dim=0) / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        return phi, mu, cov_mat

    def get_eigenvalues(self, matrix):
        eigen_vals = torch.linalg.eigvals(matrix).real
        return torch.abs(torch.min(eigen_vals, dim=1))
        negatives = eigen_vals.clone()
        positives = eigen_vals.clone()
        negatives[negatives >= 0] = float('-inf')
        positives[positives <= 0] = float('inf')
    
        # Get the highest negative value and lowest positive value per row
        highest_negatives = negatives.max(dim=1).values
        lowest_positives = positives.min(dim=1).values
        highest_negatives = eigen_vals.min(dim=1).values
        
        # Determine if there are negative values in each row
        has_negatives = (eigen_vals < 0).any(dim=1)
        
        # Use the highest negative value if exists, otherwise the lowest positive value
        result = torch.where(has_negatives, torch.abs(highest_negatives), torch.zeros_like(lowest_positives))

        return result
    
    def next_decimal_with_same_precision(self, y):
        x = y.detach().cpu().numpy()
        return round(x + 10**(-len(str(x).split('.')[1])), len(str(x).split('.')[1])) if '.' in str(x) else x + 1


    def make_psd(self, matrix):
        sym_matrix = (matrix + matrix.mT) / 2
        eigen_vals = torch.linalg.eigvals(sym_matrix).real
        min_evs = torch.min(eigen_vals, dim=1).values
        if min(min_evs) >= 0:
            return sym_matrix
        min_evs = torch.abs(min_evs)
        min_evs += torch.Tensor(list(map(self.next_decimal_with_same_precision, min_evs)))
        eyes = torch.eye(sym_matrix.shape[0]) * min_evs.view(-1,1,1)
        return sym_matrix + eyes        
        



    def estimate_sample_energy(self, z, phi=None, mu=None, cov_mat=None, average_energy=True):
        if phi is None:
            phi = self.phi
        if mu is None:
            mu = self.mu
        if cov_mat is None:
            cov_mat = self.cov_mat

        d = z.shape[1]
        # Avoid non-invertible covariance matrix by adding small values (self.reg_covar)
        cov_mat = cov_mat + (torch.eye(d)).to(self.device) * self.reg_covar


        cov_mat = self.make_psd(cov_mat)

        # N x K x D
        mu_z = z.unsqueeze(1) - mu.unsqueeze(0)

        # scaler
        print('ENERGY 1', torch.linalg.eigvals(cov_mat).real)
        # inv_cov_mat = torch.cholesky_inverse(torch.linalg.cholesky(cov_mat))
        inv_cov_mat = torch.inverse(cov_mat).unsqueeze(0)
        print('ENERGY 2', torch.linalg.eigvals(cov_mat).real)

        pi_cov_mat = 2 * np.pi * cov_mat

        #print('ENERGY 2.5', torch.linalg.eigvals(pi_cov_mat).real)
        #pi_cov_mat = self.make_psd(pi_cov_mat)
        print('ENERGY 3', torch.linalg.eigvals(pi_cov_mat).real)
        # inv_cov_mat = torch.linalg.inv(cov_mat)


        det_cov_mat = torch.linalg.cholesky(pi_cov_mat)
        print('ENERGY 4', torch.linalg.eigvals(cov_mat).real)
        det_cov_mat = torch.diagonal(det_cov_mat, dim1=1, dim2=2)
        det_cov_mat = torch.prod(det_cov_mat, dim=1)

        exp_term = torch.matmul(mu_z.unsqueeze(-2), inv_cov_mat)
        exp_term = torch.matmul(exp_term, mu_z.unsqueeze(-1))
        exp_term = - 0.5 * exp_term.squeeze()

        # Applying log-sum-exp stability trick
        # https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
        if exp_term.ndim == 1:
            exp_term = exp_term.unsqueeze(0)
        max_val = torch.max(exp_term.clamp(min=0), dim=1, keepdim=True)[0]
        exp_result = torch.exp(exp_term - max_val)

        log_term = phi * exp_result
        log_term /= det_cov_mat
        log_term = log_term.sum(axis=-1)

        # energy computation
        energy_result = - max_val.squeeze() - torch.log(log_term + self.reg_covar)

        if average_energy:
            energy_result = energy_result.mean()

        # penalty term
        cov_diag = torch.diagonal(cov_mat, dim1=1, dim2=2)
        pen_cov_mat = (1 / cov_diag).sum()

        return energy_result, pen_cov_mat

    def score(self, sample: torch.Tensor):
        _, _, _, z, _ = self.model(sample)
        return self.estimate_sample_energy(z)


class MemAETrainer(BaseTrainer):
    def __init__(self, **kwargs) -> None:
        super(MemAETrainer, self).__init__(**kwargs)
        self.alpha = self.model.alpha
        self.recon_loss_fn = nn.MSELoss().to(self.device)
        self.entropy_loss_fn = EntropyLoss().to(self.device)

    @staticmethod
    def load_from_file(fname: str, device: str = None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(fname, map_location=device)
        metric_values = ckpt["metric_values"]
        model = MemAutoEncoder.load_from_ckpt(ckpt)
        trainer = MemAETrainer(model=model, batch_size=1, device=device)
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.metric_values = metric_values

        return trainer, model

    def train_iter(self, sample: torch.Tensor):
        x_hat, w_hat = self.model(sample)
        R = self.recon_loss_fn(sample, x_hat)
        E = self.entropy_loss_fn(w_hat)
        return R + (self.alpha * E)

    def score(self, sample: torch.Tensor):
        x_hat, _ = self.model(sample)
        return torch.sum((sample - x_hat) ** 2, axis=1)


class SOMDAGMMTrainer(BaseTrainer):

    def before_training(self, dataset: DataLoader):
        self.train_som(dataset.dataset.dataset.X)

    def train_som(self, X):
        self.model.train_som(X)

    def train_iter(self, X):
        # SOM-generated low-dimensional representation
        code, X_prime, cosim, Z, gamma = self.model(X)

        phi, mu, Sigma = self.model.compute_params(Z, gamma)
        energy, penalty_term = self.model.estimate_sample_energy(Z, phi, mu, Sigma)

        return self.model.compute_loss(X, X_prime, energy, penalty_term)

    def test(self, dataset: DataLoader):
        """
        function that evaluate the model on the test set every iteration of the
        active learning process
        """
        self.model.eval()

        with torch.no_grad():
            scores, y_true, labels = [], [], []
            for row in dataset:
                X, y, label = row
                X = X.to(self.device).float()

                sample_energy, _ = self.score(X)

                y_true.extend(y)
                labels.extend(list(label))
                scores.extend(sample_energy.cpu().numpy())

            return np.array(y_true), np.array(scores), np.array(labels)

    def score(self, sample: torch.Tensor):
        code, x_prime, cosim, z, gamma = self.model(sample)
        phi, mu, cov_mat = self.model.compute_params(z, gamma)
        sample_energy, pen_cov_mat = self.model.estimate_sample_energy(
            z, phi, mu, cov_mat, average_energy=False
        )
        return sample_energy, pen_cov_mat
