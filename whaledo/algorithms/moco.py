from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Optional, TypeVar

from conduit.data.structures import BinarySample
from conduit.models.utils import prefix_keys
from conduit.types import MetricDict, Stage
import pytorch_lightning as pl
from ranzen import implements
from ranzen.torch.transforms import RandomMixUp
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing_extensions import Self, TypeAlias

from whaledo.algorithms.base import Algorithm
from whaledo.algorithms.mean_teacher import MeanTeacher
from whaledo.algorithms.memory_bank import MemoryBank
from whaledo.data.datamodule import WhaledoDataModule
from whaledo.transforms import MultiCropOutput, MultiViewPair
from whaledo.utils import to_item

from .base import Algorithm
from .loss import moco_v2_loss, simclr_loss, soft_supcon_loss, supcon_loss
from .multicrop import MultiCropWrapper

__all__ = ["Moco"]

M = TypeVar("M", MultiCropOutput, MultiViewPair)
TrainBatch: TypeAlias = BinarySample[M]


class LossFn(Enum):
    ID = "instance_discrimination"
    SUPCON = "supcon"


@dataclass(unsafe_hash=True)
class Moco(Algorithm):
    IGNORE_INDEX: ClassVar[int] = -100

    mb_capacity: int = 16_384
    ema_decay_start: float = 0.99
    ema_decay_end: float = 0.99
    ema_warmup_steps: int = 0

    cross_sample_only: bool = False
    loss_fn: LossFn = LossFn.SUPCON
    dcl: bool = False
    soft_supcon: bool = False

    proj_depth: int = 0
    pred_depth: int = 0

    logit_mb: Optional[MemoryBank] = field(init=False)
    label_mb: Optional[MemoryBank] = field(init=False)
    student: nn.Sequential = field(init=False)
    teacher: MeanTeacher = field(init=False)
    manifold_mu: Optional[RandomMixUp] = None

    replace_model: bool = False

    def __post_init__(self) -> None:
        if not (0 <= self.ema_decay_start <= 1):
            raise AttributeError("'ema_decay_start' must be in the range [0, 1].")
        if not (0 <= self.ema_decay_end <= 1):
            raise AttributeError("'ema_decay_end' must be in the range [0, 1].")
        if self.ema_warmup_steps < 0:
            raise AttributeError("'ema_warmup_steps' must be non-negative.")

        if self.mb_capacity < 0:
            raise AttributeError("'mb_capacity' must be non-negative.")

        # initialise the encoders
        embed_dim = self.model.out_dim
        projector = self.build_mlp(
            input_dim=embed_dim,
            num_layers=self.proj_depth,
            hidden_dim=self.mlp_dim,
            out_dim=self.out_dim,
            final_norm=True,
        )
        predictor = self.build_mlp(
            input_dim=self.out_dim,
            num_layers=self.pred_depth,
            hidden_dim=self.mlp_dim,
            out_dim=self.out_dim,
            final_norm=self.final_norm,
        )
        self.student = nn.Sequential(
            MultiCropWrapper(backbone=self.model, head=projector), predictor
        )
        if self.replace_model:
            self.model.backbone = self.student
        self.teacher = MeanTeacher(
            self.student,
            decay_start=self.ema_decay_start,
            decay_end=self.ema_decay_end,
            warmup_steps=self.ema_warmup_steps,
            auto_update=False,
        )

        # initialise the memory banks (if needed)
        self.logit_mb = None
        self.label_mb = None
        if self.mb_capacity > 0:
            self.logit_mb = MemoryBank.with_l2_hypersphere_init(
                dim=self.out_dim, capacity=self.mb_capacity
            )
            if self.loss_fn is LossFn.SUPCON:
                self.label_mb = MemoryBank.with_constant_init(
                    dim=1, capacity=self.mb_capacity, value=self.IGNORE_INDEX, dtype=torch.long
                )
        if self.soft_supcon and (self.manifold_mu is None):
            self.manifold_mu = RandomMixUp.with_beta_dist(0.5, inplace=False)
        super().__post_init__()

    @implements(Algorithm)
    def on_after_batch_transfer(
        self,
        batch: TrainBatch,
        dataloader_idx: Optional[int] = None,
    ) -> TrainBatch:
        if self.training:
            if isinstance(batch.x, MultiCropOutput):
                batch.x.global_views.v1 = self._apply_batch_transforms(batch.x.global_views.v1)
                batch.x.global_views.v2 = self._apply_batch_transforms(batch.x.global_views.v2)
                batch.x.local_views = self._apply_batch_transforms(batch.x.local_views)
            elif isinstance(batch.x, MultiViewPair):
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
        logging_dict: MetricDict = {}
        # Compute the student's logits using both the global and local crops.
        inputs = batch.x
        student_logits = self.student.forward(inputs.anchor)
        # Compute the teacher's logits using only the global crops.
        with torch.no_grad():
            self.teacher.update()
            teacher_logits = self.teacher.forward(inputs.target)

        temp = self.temp
        if self.loss_fn is LossFn.SUPCON:
            candidates = teacher_logits
            y = candidate_labels = batch.y
            if self.soft_supcon and (self.manifold_mu is not None):
                if self.label_mb is None:
                    # Make y values contiguous in the range [0, card({y)}).
                    y_unique, y_contiguous = y.unique(return_inverse=True)
                    y_ohe = F.one_hot(y_contiguous, num_classes=len(y_unique))
                else:
                    num_classes = self.label_mb.memory.size(1)
                    y_ohe = F.one_hot(y, num_classes=num_classes)
                student_logits, y = self.manifold_mu(
                    student_logits.float(), targets=y_ohe, group_labels=None
                )
                candidates, candidate_labels = self.manifold_mu(
                    candidates.float(), targets=y_ohe, group_labels=None
                )

            student_logits = F.normalize(student_logits, dim=1, p=2)
            candidates = F.normalize(candidates, dim=1, p=2)

            if (self.logit_mb is not None) and (self.label_mb is not None):
                self.logit_mb.push(candidates)
                lp_mask = (self.label_mb.memory != self.IGNORE_INDEX).any(dim=1)
                labels_past = self.label_mb.clone(lp_mask).squeeze(-1)
                self.label_mb.push(candidate_labels)
                candidate_labels = torch.cat((candidate_labels, labels_past), dim=0)
                logits_past = self.logit_mb.clone(lp_mask)
                candidates = torch.cat((candidates, logits_past), dim=0)

            if self.soft_supcon:
                loss = soft_supcon_loss(z1=student_logits, p1=y, z2=candidates, p2=candidate_labels)
            else:
                loss = supcon_loss(
                    anchors=student_logits,
                    anchor_labels=y,
                    candidates=candidates,
                    candidate_labels=candidate_labels,
                    temperature=temp,
                    dcl=self.dcl,
                    exclude_diagonal=self.cross_sample_only,
                )
        else:
            student_logits = F.normalize(student_logits, dim=1, p=2)
            teacher_logits = F.normalize(teacher_logits, dim=1, p=2)
            if self.logit_mb is None:
                loss = simclr_loss(
                    anchors=student_logits,
                    targets=teacher_logits,
                    temperature=temp,
                    dcl=self.dcl,
                )
            else:
                logits_past = self.logit_mb.clone()
                loss = moco_v2_loss(
                    anchors=student_logits,
                    positives=teacher_logits,
                    negatives=logits_past,
                    temperature=temp,
                    dcl=self.dcl,
                )
                self.logit_mb.push(teacher_logits)

        if not self.learn_temp:
            loss *= temp
        # Anneal the temperature parameter by one step.
        self.step_temp()

        logging_dict[self.loss_fn.value] = to_item(loss)
        logging_dict = prefix_keys(
            dict_=logging_dict,
            prefix=f"{str(Stage.fit)}/batch_loss",
            sep="/",
        )

        self.log_dict(logging_dict)

        return loss

    def _run_internal(
        self, datamodule: WhaledoDataModule, *, trainer: pl.Trainer, test: bool = True
    ) -> Self:
        if (self.label_mb is not None) and self.soft_supcon and (self.manifold_mu is not None):
            self.label_mb.memory = self.label_mb.memory.expand(-1, datamodule.max_y + 1)
        return super()._run_internal(datamodule, trainer=trainer, test=test)
