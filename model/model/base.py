'''
This script has been either partially or fully inspired by the repository:

https://github.com/ireydiak/anomaly_detection_NRCAN

Alvarez, M., Verdier, J.C., Nkashama, D.K., Frappier, M., Tardif, P.M.,
Kabanza, F.: A revealing large-scale evaluation of unsupervised anomaly
detection algorithms
'''

import torch
from abc import ABC
from torch import nn


class BaseModel(nn.Module):

    def __init__(self, in_features: int, n_instances: int, device: str = None, **kwargs):
        super(BaseModel, self).__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_instances = n_instances
        self.in_features = in_features

    def get_params(self) -> dict:
        return {
            "n_instances": self.n_instances,
            "in_features": self.in_features,
        }

    def reset(self):
        self.apply(self.weight_reset)

    def weight_reset(self, m):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    @staticmethod
    def load_from_ckpt(ckpt):
        model = BaseModel(
            in_features=ckpt["in_features"],
            n_instances=ckpt["n_instances"],
        )
        return model

    def save(self, filename):
        # Save model to file (.pt)
        torch.save(
            self.state_dict(), filename
        )


class BaseShallowModel(ABC):

    def __init__(self, in_features: int, n_instances: int, device: str = None, **kwargs):
        self.device = device
        self.in_features = in_features
        self.n_instances = n_instances

    def resolve_params(self, dataset_name: str):
        pass

    def reset(self):
        """
        This function does nothing.
        It exists only for consistency with deep models
        """
        pass

    def save(self, filename):
        """
        This function does nothing.
        It exists only for consistency with deep models
        """
        pass
