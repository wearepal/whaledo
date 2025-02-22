from dataclasses import dataclass, field
from typing import Optional

from conduit.data.structures import BinarySample
from conduit.logging import init_logger
from conduit.models.utils import prefix_keys
from conduit.types import Stage
import pytorch_lightning as pl
from ranzen import implements
import torch
from torch import Tensor
import torch.nn.functional as F
from typing_extensions import TypeAlias

from whaledo.algorithms.base import Algorithm
from whaledo.transforms import MultiViewPair
from whaledo.utils import to_item

from .loss import supcon_loss
from .multicrop import MultiCropWrapper

__all__ = ["SimClr"]

LOGGER = init_logger(name=__file__)

TrainBatch: TypeAlias = BinarySample[MultiViewPair]


@dataclass(unsafe_hash=True)
class SimClr(Algorithm):
    dcl: bool = False
    student: MultiCropWrapper = field(init=False)
    proj_depth: int = 2
    replace_model: bool = False

    def __post_init__(self) -> None:
        # initialise the encoders
        embed_dim = self.model.feature_dim
        projector = self.build_mlp(
            input_dim=embed_dim,
            num_layers=self.proj_depth,
            hidden_dim=self.mlp_dim,
            out_dim=self.out_dim,
            final_norm=True,
        )
        self.student = MultiCropWrapper(backbone=self.model.backbone, head=projector)
        if self.replace_model:
            self.model.backbone = self.student
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
        logits_v1 = self.student.forward(batch.x.v1)
        logits_v2 = self.student.forward(batch.x.v2)
        logits = F.normalize(torch.cat((logits_v1, logits_v2), dim=0), dim=1, p=2)

        temp = self.temp.val
        loss = supcon_loss(
            anchors=logits,
            anchor_labels=batch.y.repeat(2),
            temperature=temp,
            exclude_diagonal=True,
            dcl=self.dcl,
        )
        loss *= 2 * temp

        # Anneal the temperature parameter by one step.
        self.temp.step()

        logging_dict = {"supcon": to_item(loss)}
        logging_dict = prefix_keys(
            dict_=logging_dict,
            prefix=f"{str(Stage.fit)}/batch_loss",
            sep="/",
        )

        self.log_dict(logging_dict)

        return loss
