from dataclasses import dataclass, field
from typing import Optional

from conduit.data.structures import BinarySample
from conduit.logging import init_logger
from conduit.models.utils import prefix_keys
from conduit.types import Stage
import pytorch_lightning as pl
from ranzen import implements
from ranzen.torch.transforms import RandomMixUp
import torch
from torch import Tensor
import torch.nn.functional as F
from typing_extensions import TypeAlias

from whaledo.algorithms.base import Algorithm
from whaledo.models.predictors import Fcn, NormType
from whaledo.transforms import MultiViewPair
from whaledo.utils import to_item

from .loss import SupConReduction, soft_supcon_loss, supcon_loss
from .multicrop import MultiCropWrapper

__all__ = ["SimClr"]

LOGGER = init_logger(name=__file__)

TrainBatch: TypeAlias = BinarySample[MultiViewPair]


@dataclass(unsafe_hash=True)
class SimClr(Algorithm):
    dcl: bool = False
    student: MultiCropWrapper = field(init=False)
    manifold_mu: Optional[RandomMixUp] = None
    input_mu: Optional[RandomMixUp] = None
    soft_supcon: bool = False
    margin: float = 0.0
    reduction: SupConReduction = SupConReduction.MEAN
    q: float = 0.0

    proj_depth: int = 0
    mlp_dim: Optional[int] = None
    mlp_norm: NormType = NormType.LN
    final_norm: bool = True
    out_dim: int = 256

    def __post_init__(self) -> None:
        # initialise the encoders
        embed_dim = self.model.out_dim
        projector = Fcn(
            hidden_dim=self.mlp_dim,
            out_dim=self.out_dim,
            final_norm=self.final_norm,
            norm=self.mlp_norm,
            num_hidden=self.proj_depth,
        )(embed_dim)[0]
        self.student = MultiCropWrapper(backbone=self.model, head=projector)
        if self.soft_supcon and (self.manifold_mu is None) and (self.input_mu is None):
            self.manifold_mu = RandomMixUp.with_beta_dist(2.0, inplace=False)
        super().__post_init__()

    @implements(Algorithm)
    def on_after_batch_transfer(
        self,
        batch: TrainBatch,
        dataloader_idx: Optional[int] = None,
    ) -> TrainBatch:
        if self.training:
            if isinstance(batch.x, MultiViewPair):
                batch.x.v1 = self._apply_batch_transforms(batch.x.v1)
                batch.x.v2 = self._apply_batch_transforms(batch.x.v2)
            else:
                raise ValueError(
                    "Inputs from the training data must be 'MultiCropOutput' or 'MultiViewPair'"
                    " objects."
                )
        return batch

    @implements(pl.LightningModule)
    def training_step(
        self,
        batch: TrainBatch,
        batch_idx: int,
    ) -> Tensor:
        x1, x2 = batch.x.v1, batch.x.v2
        y_ohe = None
        if self.soft_supcon and (self.input_mu is not None):
            # Make y values contiguous in the range [0, card({y)}).
            y_unique, y_contiguous = batch.y.unique(return_inverse=True)
            y_ohe = F.one_hot(y_contiguous, num_classes=len(y_unique))
            dtype = x1.dtype
            x1, y1 = self.input_mu(x1.float(), targets=y_ohe.clone())
            x2, y2 = self.input_mu(x2.float(), targets=y_ohe)
            x1 = x1.to(dtype)
            x2 = x2.to(dtype)
            y = torch.cat((y1, y2), dim=0)
        else:
            y = batch.y.repeat(2).long()

        logits_v1 = self.student.forward(x1)
        logits_v2 = self.student.forward(x2)
        logits = torch.cat((logits_v1, logits_v2), dim=0)

        temp = self.temp
        if ((self.manifold_mu is None) and (self.input_mu is None)) or (not self.soft_supcon):
            loss = supcon_loss(
                anchors=logits,
                anchor_labels=y,
                temperature=temp,
                exclude_diagonal=True,
                margin=self.margin,
                dcl=self.dcl,
                reduction=self.reduction,
                q=self.q,
                normalize=True,
            )
        else:
            if self.manifold_mu is not None:
                if self.input_mu is None:
                    y_unique, y_contiguous = y.unique(return_inverse=True)
                    y = F.one_hot(y_contiguous, num_classes=len(y_unique))
            loss = soft_supcon_loss(z1=logits, p1=y, normalize=True)

        if not self.learn_temp:
            loss *= temp
        # Anneal the temperature parameter by one step.
        self.step_temp()

        logging_dict = {"supcon": to_item(loss)}
        logging_dict = prefix_keys(
            dict_=logging_dict,
            prefix=f"{str(Stage.fit)}/batch_loss",
            sep="/",
        )
        if isinstance(temp, Tensor):
            logging_dict["temperature"] = to_item(temp)

        self.log_dict(logging_dict)

        return loss
