from dataclasses import dataclass
from enum import Enum
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


class NormType(Enum):
    BN = "batchnorm"
    LN = "layernorm"


@dataclass
class Fcn(PredictorFactory):
    out_dim: Optional[int] = 256
    num_hidden: int = 0
    hidden_dim: Optional[int] = None
    final_norm: bool = False
    norm: NormType = NormType.LN
    dropout_prob: float = 0.0

    def __call__(
        self,
        in_dim: int,
    ) -> ModelFactoryOut:
        predictor = nn.Sequential(nn.Flatten())
        curr_dim = in_dim
        if self.num_hidden > 0:
            hidden_dim = in_dim if self.hidden_dim is None else self.hidden_dim
            for _ in range(self.num_hidden):
                predictor.append(nn.Linear(curr_dim, hidden_dim))
                if self.norm is NormType.BN:
                    predictor.append(nn.BatchNorm1d(hidden_dim))
                else:
                    predictor.append(BiaslessLayerNorm(hidden_dim))
                predictor.append(nn.GELU())
                if self.dropout_prob > 0:
                    predictor.append(nn.Dropout(p=self.dropout_prob))
                curr_dim = hidden_dim

        if self.out_dim is None:
            predictor.append(nn.Identity())
            curr_dim = out_dim = in_dim
        else:
            out_dim = self.out_dim
            predictor.append(nn.Linear(curr_dim, out_dim))
        if self.final_norm:
            if self.norm is NormType.BN:
                predictor.append(nn.BatchNorm1d(out_dim, affine=False))
            else:
                predictor.append(BiaslessLayerNorm(out_dim))

        return predictor, out_dim


@dataclass
class Identity(PredictorFactory):
    def __call__(
        self,
        in_dim: int,
    ) -> ModelFactoryOut:
        return nn.Identity(), in_dim
