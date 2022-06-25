"""Whaledo data-module."""
from enum import Enum
from typing import Any, List, Optional, cast, final

import attr
from conduit.data.constants import IMAGENET_STATS
from conduit.data.datamodules.base import CdtDataModule
from conduit.data.datamodules.vision.base import CdtVisionDataModule
from conduit.data.datasets.utils import CdtDataLoader, ImageTform
from conduit.data.structures import TrainValTestSplit
from pytorch_lightning import LightningDataModule
from ranzen import implements
from ranzen.torch.data import SequentialBatchSampler
import torchvision.transforms as T  # type: ignore

from whaledo.data.dataset import SampleType, WhaledoDataset
from whaledo.data.samplers import QueryKeySampler
from whaledo.transforms import ResizeAndPadToSize

__all__ = ["WhaledoDataModule"]


class BaseSampler(Enum):
    WEIGHTED = "weighted"
    RANDOM = "random"


@attr.define(kw_only=True)
class WhaledoDataModule(CdtVisionDataModule[WhaledoDataset, SampleType]):
    """Data-module for the 'Where's Whale-do' dataset."""

    base_sampler: BaseSampler = BaseSampler.WEIGHTED
    image_size: int = 224
    use_qk_sampler: bool = True

    @property
    def _default_train_transforms(self) -> ImageTform:
        transform_ls: List[ImageTform] = [
            ResizeAndPadToSize(self.image_size),
            T.ToTensor(),
            T.Normalize(*IMAGENET_STATS),
        ]
        return T.Compose(transform_ls)

    @property
    def _default_test_transforms(self) -> ImageTform:
        return self._default_train_transforms

    @implements(LightningDataModule)
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        WhaledoDataset(root=self.root)

    @implements(CdtDataModule)
    def _get_splits(self) -> TrainValTestSplit[WhaledoDataset]:
        all_data = WhaledoDataset(root=self.root, transform=None)
        _, inverse, counts = all_data.y.unique(return_inverse=True, return_counts=True)
        singleton_mask = counts > 1
        all_data = all_data.subset(singleton_mask[inverse].nonzero().squeeze().tolist())
        train, test = all_data.train_test_split(prop=self.test_prop, seed=self.seed)

        return TrainValTestSplit(train=train, val=test, test=test)

    def train_dataloader(
        self, *, shuffle: bool = False, drop_last: bool = False, batch_size: Optional[int] = None
    ) -> CdtDataLoader[SampleType]:
        batch_size = self.train_batch_size if batch_size is None else batch_size
        batch_sampler = None
        if self.use_qk_sampler:
            base_ds = cast(WhaledoDataset, self._get_base_dataset())
            if batch_size & 1:
                self.logger.info(
                    "train_batch_size is not an even number: rounding down to the nearest multiple of "
                    "two to ensure the effective batch size is upperjbounded by the requested "
                    "batch size."
                )
            if self.base_sampler is BaseSampler.WEIGHTED:
                _, inverse, counts = base_ds.group_ids.unique(
                    return_counts=True, return_inverse=True
                )
                sample_weights = counts.reciprocal()[inverse]
            else:
                sample_weights = None
            batch_sampler = QueryKeySampler(
                data_source=base_ds,
                num_queries_per_batch=batch_size // 2,
                labels=base_ds.y,
                sample_weights=sample_weights,
            )
        else:
            batch_sampler = SequentialBatchSampler(
                data_source=self.train_data,
                batch_size=self.train_batch_size,
                drop_last=False,
                training_mode="step",
            )
        return self.make_dataloader(
            ds=self.train_data, batch_size=self.train_batch_size, batch_sampler=batch_sampler
        )

    @property
    @final
    def max_y(self) -> int:
        self._check_setup_called()
        return self._get_base_dataset().y.max()
