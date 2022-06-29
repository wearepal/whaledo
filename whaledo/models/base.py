from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, TypeVar

from torch import Tensor
import torch.nn as nn
from typing_extensions import TypeAlias

__all__ = [
    "BackboneFactory",
    "Model",
]

M = TypeVar("M", bound=nn.Module)
ModelFactoryOut: TypeAlias = Tuple[M, int]


@dataclass(unsafe_hash=True)
class Prediction:
    query_inds: Tensor
    database_inds: Tensor
    n_retrieved_per_query: Tensor
    scores: Tensor

    def __post_init__(self) -> None:
        if len(self.query_inds) != len(self.database_inds) != len(self.scores):
            raise AttributeError(
                "'query_inds', 'retrieved_inds', and 'scores' must be equal in length."
            )


@dataclass
class BackboneFactory:
    def __call__(self) -> ModelFactoryOut:
        ...


@dataclass
class PredictorFactory:
    def __call__(self, in_dim: int) -> ModelFactoryOut:
        ...


@dataclass(unsafe_hash=True)
class Model(nn.Module):
    backbone: nn.Module
    feature_dim: int
    out_dim: int
    predictor: nn.Module = field(default_factory=nn.Identity)

    def __new__(cls, *args: Any, **kwargs: Any) -> "Model":
        obj = object.__new__(cls)
        nn.Module.__init__(obj)
        return obj

    def forward(self, x: Tensor) -> Tensor:
        return self.predictor(self.backbone(x))

    def threshold_scores(self, scores: Tensor, *, threshold: float) -> Tensor:
        return scores > threshold

    def predict(
        self,
        queries: Tensor,
        *,
        db: Optional[Tensor] = None,
        k: int = 20,
        sorted: bool = True,
        temperature: float = 1.0,
        threshold: float = 0.0,
    ) -> Prediction:
        mask_diag = False
        if db is None:
            db = queries
            mask_diag = True

        sim_mat = queries @ db.T / temperature
        db_size = sim_mat.size(1)

        all_scores = sim_mat.float().softmax(dim=1)
        if mask_diag:
            # Mask the diagonal to prevent self matches.
            all_scores.fill_diagonal_(-float("inf"))
            db_size -= 1

        k = min(k, db_size)
        topk_scores, topk_inds = all_scores.topk(dim=1, k=k, sorted=sorted)
        mask = self.threshold_scores(scores=topk_scores, threshold=threshold)
        n_retrieved_per_query = mask.count_nonzero(dim=1)
        mask_inds = mask.nonzero(as_tuple=True)
        retrieved_scores, retrieved_inds = topk_scores[mask_inds], topk_inds[mask_inds]

        return Prediction(
            query_inds=mask_inds[0],
            database_inds=retrieved_inds,
            n_retrieved_per_query=n_retrieved_per_query,
            scores=retrieved_scores,
        )
