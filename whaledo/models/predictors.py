from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from whaledo.models.base import ModelFactoryOut, PredictorFactory

__all__ = ["Fcn", "BiaslessLayerNorm", "Identity"]


class BiaslessLayerNorm(nn.Module):
    beta: Parameter

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.gamma = Parameter(torch.ones(input_dim))
        self.register_buffer("beta", torch.zeros(input_dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(
            x,
            normalized_shape=x.shape[-1:],
            weight=self.gamma,
            bias=self.beta,
        )


class Fcn(PredictorFactory):
    out_dim: int = 256
    num_hidden: int = 0
    hidden_dim: Optional[int] = 4096
    final_norm: bool = False

    def __call__(
        self,
        in_dim: int,
    ) -> ModelFactoryOut:
        predictor = nn.Sequential(nn.Flatten())
        if self.out_dim <= 0:
            out_dim = self.out_dim
            predictor.append(BiaslessLayerNorm(in_dim))
            if self.num_hidden > 0:
                hidden_dim = in_dim if self.hidden_dim is None else self.hidden_dim
                for _ in range(self.num_hidden):
                    predictor.append(nn.Linear(in_dim, hidden_dim))
                    predictor.append(BiaslessLayerNorm(in_dim))
                    predictor.append(nn.GELU())

            predictor.append(nn.Linear(in_dim, out_dim))
        else:
            predictor.append(nn.Identity())
            out_dim = in_dim
        if self.final_norm:
            predictor.append(BiaslessLayerNorm(out_dim))

        return predictor, self.out_dim


class Identity(PredictorFactory):
    def __call__(
        self,
        in_dim: int,
    ) -> ModelFactoryOut:
        return nn.Identity(), in_dim
