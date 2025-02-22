from abc import abstractmethod
from whaledo.schedulers import CosineWarmup
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
import operator
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, TypeVar, Union

from conduit.data.datamodules.vision.base import CdtVisionDataModule
from conduit.data.structures import BinarySample, NamedSample
from conduit.models.utils import prefix_keys
from conduit.types import LRScheduler, MetricDict, Stage
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
import pandas as pd  # type: ignore
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from ranzen import implements
from ranzen.torch.data import TrainingMode
import torch
from torch import Tensor, optim
import torch.nn as nn
from torch.nn.parameter import Parameter
from typing_extensions import Self

from whaledo.metrics import MeanAveragePrecision
from whaledo.models import MetaModel, Model
from whaledo.optimizers import Adafactor
from whaledo.transforms import BatchTransform
from whaledo.types import EvalEpochOutput, EvalOutputs, EvalStepOutput

__all__ = ["Algorithm"]

T = TypeVar("T", bound=Union[Tensor, NamedSample[Tensor]])


def exclude_from_weight_decay(
    named_params: Iterable[Tuple[str, Parameter]],
    weight_decay: float = 0.0,
    exclusion_patterns: Tuple[str, ...] = ("bias",),
) -> List[Dict[str, Union[List[Parameter], float]]]:
    params: List[Parameter] = []
    excluded_params: List[Parameter] = []

    for name, param in named_params:
        if not param.requires_grad:
            continue
        elif any(layer_name in name for layer_name in exclusion_patterns):
            excluded_params.append(param)
        else:
            params.append(param)

    return [
        {"params": params, "weight_decay": weight_decay},
        {
            "params": excluded_params,
            "weight_decay": 0.0,
        },
    ]


class Optimizer(Enum):
    ADAM = torch.optim.AdamW
    ADAFACTOR = Adafactor


@dataclass(unsafe_hash=True)
class Algorithm(pl.LightningModule):
    model: Union[Model, MetaModel]
    base_lr: float = 1.0e-4
    lr: float = field(default=base_lr, init=False)
    optimizer_cls: Optimizer = Optimizer.ADAM
    weight_decay: float = 0.0
    optimizer_kwargs: Optional[DictConfig] = None
    use_sam: bool = False
    sam_rho: float = 0.05
    scheduler_cls: Optional[str] = None
    scheduler_kwargs: Optional[DictConfig] = None
    lr_sched_interval: TrainingMode = TrainingMode.step
    lr_sched_freq: int = 1
    batch_transforms: Optional[List[BatchTransform]] = None
    test_on_best: bool = False

    out_dim: int = 128
    mlp_dim: int = 4096

    temp_start: float = 1.0
    temp_end: float = 1.0
    temp_warmup_steps: int = 0
    temp: CosineWarmup = field(init=False)

    def __new__(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
        obj = object.__new__(cls)
        pl.LightningModule.__init__(obj)
        return obj

    def __post_init__(self) -> None:
        if self.temp_start <= 0:
            raise AttributeError("'temp_start' must be positive.")
        if self.temp_end <= 0:
            raise AttributeError("'temp_end' must be positive.")
        if self.temp_warmup_steps < 0:
            raise AttributeError("'temp_warmup_steps' must be non-negative.")
        self.temp = CosineWarmup(
            start_val=self.temp_start, end_val=self.temp_end, warmup_steps=self.temp_warmup_steps
        )

    def _apply_batch_transforms(self, batch: T) -> T:
        if self.batch_transforms is not None:
            for tform in self.batch_transforms:
                if isinstance(batch, Tensor):
                    batch = tform(inputs=batch, targets=None)  # type: ignore
                else:
                    if isinstance(batch, BinarySample):
                        transformed_x, transformed_y = tform(inputs=batch.x, targets=batch.y)
                        batch.y = transformed_y
                    else:
                        transformed_x = tform(inputs=batch.x, targets=None)
                    batch.x = transformed_x
        return batch

    @implements(pl.LightningModule)
    def on_after_batch_transfer(
        self,
        batch: T,
        dataloader_idx: Optional[int] = None,
    ) -> T:
        if self.training:
            batch = self._apply_batch_transforms(batch)
        return batch

    @abstractmethod
    def training_step(
        self,
        batch: BinarySample,
        batch_idx: int,
    ) -> STEP_OUTPUT:
        ...

    @torch.no_grad()
    def inference_step(self, batch: BinarySample) -> EvalOutputs:
        logits = self.forward(batch.x)
        return EvalOutputs(
            logits=logits.cpu(),
            ids=batch.y.cpu(),
        )

    @implements(pl.LightningModule)
    @torch.no_grad()
    def validation_step(
        self,
        batch: BinarySample,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> EvalStepOutput:
        return self.inference_step(batch=batch)

    @torch.no_grad()
    def _evaluate(self, outputs: EvalOutputs) -> MetricDict:
        same_id = (outputs.ids.unsqueeze(1) == outputs.ids).long()
        preds = self.model.predict(queries=outputs.logits, k=MeanAveragePrecision.PREDICTION_LIMIT)
        pred_df = pd.DataFrame(
            {
                "query_id": preds.query_inds.numpy(),
                "database_image_id": preds.database_inds.numpy(),
                "score": preds.scores.numpy(),
            },
        )
        pred_df.set_index("query_id", inplace=True)

        gt_query_inds, gt_db_inds = same_id.nonzero(as_tuple=True)
        gt_df = pd.DataFrame(
            {
                "query_id": gt_query_inds.numpy(),
                "database_image_id": gt_db_inds.numpy(),
            },
        )
        gt_df.set_index("query_id", inplace=True)
        rmap = MeanAveragePrecision.score(predicted=pred_df, actual=gt_df)

        return {"mean_average_precision": rmap.item()}

    def _epoch_end(self, outputs: Union[List[EvalOutputs], EvalEpochOutput]) -> MetricDict:
        outputs_agg = reduce(operator.add, outputs)
        return self._evaluate(outputs_agg)

    @implements(pl.LightningModule)
    @torch.no_grad()
    def validation_epoch_end(self, outputs: EvalEpochOutput) -> None:
        results_dict = self._epoch_end(outputs=outputs)
        results_dict = prefix_keys(results_dict, prefix=str(Stage.validate))
        self.log_dict(results_dict)

    @implements(pl.LightningModule)
    @torch.no_grad()
    def test_step(
        self,
        batch: BinarySample,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> EvalStepOutput:
        return self.inference_step(batch=batch)

    @implements(pl.LightningModule)
    @torch.no_grad()
    def test_epoch_end(self, outputs: EvalEpochOutput) -> None:
        results_dict = self._epoch_end(outputs=outputs)
        results_dict = prefix_keys(results_dict, prefix=str(Stage.test))
        self.log_dict(results_dict)

    def predict_step(
        self, batch: BinarySample[Tensor], batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> BinarySample:

        return BinarySample(x=self.forward(batch.x), y=batch.y).to("cpu")

    @implements(pl.LightningModule)
    def configure_optimizers(
        self,
    ) -> Union[
        Tuple[
            Union[List[optim.Optimizer], optim.Optimizer],
            List[Mapping[str, Union[LRScheduler, int, TrainingMode]]],
        ],
        Union[List[optim.Optimizer], optim.Optimizer],
    ]:
        optimizer_config = DictConfig({"weight_decay": self.weight_decay, "lr": self.lr})
        if self.optimizer_kwargs is not None:
            optimizer_config.update(self.optimizer_kwargs)

        params = exclude_from_weight_decay(
            self.named_parameters(), weight_decay=optimizer_config["weight_decay"]
        )
        optimizer = self.optimizer_cls.value(**optimizer_config, params=params)

        if self.scheduler_cls is not None:
            scheduler_config = DictConfig({"_target_": self.scheduler_cls})
            if self.scheduler_kwargs is not None:
                scheduler_config.update(self.scheduler_kwargs)
            scheduler = instantiate(scheduler_config, optimizer=optimizer)
            scheduler_config = {
                "scheduler": scheduler,
                "interval": self.lr_sched_interval.name,
                "frequency": self.lr_sched_freq,
            }
            return [optimizer], [scheduler_config]
        return optimizer

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _run_internal(
        self, datamodule: CdtVisionDataModule, *, trainer: pl.Trainer, test: bool = True
    ) -> Self:
        eff_bs = trainer.num_devices * datamodule.train_batch_size
        self.lr = self.base_lr * eff_bs / 256  # linear scaling rule
        # Run routines to tune hyperparameters before training.
        trainer.tune(model=self, datamodule=datamodule)
        # Train the model
        trainer.fit(model=self, datamodule=datamodule)
        if test:
            # Test the model if desired
            trainer.test(
                model=self,
                ckpt_path="best" if self.test_on_best else None,
                datamodule=datamodule,
            )
        return self

    def run(
        self, datamodule: CdtVisionDataModule, *, trainer: pl.Trainer, test: bool = True
    ) -> Self:
        return self._run_internal(datamodule=datamodule, trainer=trainer, test=test)

    def build_mlp(
        self,
        input_dim: int,
        *,
        num_layers: int,
        hidden_dim: int,
        out_dim: int,
        final_norm: bool = True,
    ) -> nn.Module:
        if num_layers <= 0:
            return nn.Identity()
        else:
            mlp: List[nn.Module] = []
            for l in range(num_layers):
                dim1 = input_dim if l == 0 else hidden_dim
                dim2 = out_dim if l == num_layers - 1 else hidden_dim

                mlp.append(nn.Linear(dim1, dim2, bias=False))

                if l < (num_layers - 1):
                    mlp.append(nn.BatchNorm1d(dim2))
                    mlp.append(nn.GELU())
                elif final_norm:
                    mlp.append(nn.BatchNorm1d(dim2, affine=False))

            return nn.Sequential(*mlp)
