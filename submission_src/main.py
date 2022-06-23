import importlib
import math
from pathlib import Path
from typing import Dict, Literal, NamedTuple, Optional, Tuple, Union, cast

from PIL import Image
from loguru import logger
import pandas as pd  # type: ignore
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T  # type: ignore
from tqdm import tqdm  # type: ignore
from typing_extensions import Final, TypeAlias

from whaledo.models.base import BackboneFactory, Model

ROOT_DIRECTORY: Final[Path] = Path("/code_execution")
PREDICTION_FILE: Final[Path] = ROOT_DIRECTORY / "submission" / "submission.csv"
DATA_DIRECTORY: Final[Path] = ROOT_DIRECTORY / "data"
MODEL_PATH: Final[Path] = ROOT_DIRECTORY / "model.pt"
DEFAULT_IMAGE_SIZE: Final[int] = 256


class MeanStd(NamedTuple):
    mean: Tuple[float, ...]
    std: Tuple[float, ...]


IMAGENET_STATS: Final[MeanStd] = MeanStd(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)


class TestTimeWhaledoDataset(Dataset):
    """Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id and image tensors.
    """

    def __init__(
        self, metadata: pd.DataFrame, image_size: Optional[int] = DEFAULT_IMAGE_SIZE
    ) -> None:
        if image_size is None:
            image_size = DEFAULT_IMAGE_SIZE
        self.image_size = image_size
        self.metadata = metadata
        self.transform = T.Compose(
            [
                ResizeAndPadToSize(self.image_size),
                T.ToTensor(),
                T.Normalize(*IMAGENET_STATS),
            ]
        )

    def __getitem__(self, idx: int) -> Dict[str, Union[int, Image.Image]]:
        image = Image.open(DATA_DIRECTORY / self.metadata.path.iloc[idx]).convert("RGB")
        image = self.transform(image)
        return {"image_id": self.metadata.index[idx], "image": image}

    def __len__(self) -> int:
        return len(self.metadata)


_Resample: TypeAlias = Literal[0, 1, 2, 3, 4, 5]


class ResizeAndPadToSize:
    resample: Optional[_Resample]

    def __init__(self, size: int, *, resample: Optional[_Resample] = Image.BILINEAR) -> None:
        self.size = size
        self.resample = resample

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if h == w:
            img = img.resize(size=(self.size, self.size))
        if h > w:
            new_w = round(w / h * self.size)
            img = img.resize(size=(new_w, self.size), resample=self.resample)
            half_residual = (self.size - new_w) / 2
            left_padding = math.ceil(half_residual)
            right_padding = math.floor(half_residual)
            img = TF.pad(img, padding=[left_padding, 0, right_padding, 0])  # type: ignore
        else:
            new_h = round(h / w * self.size)
            img = img.resize(size=(self.size, new_h), resample=self.resample)
            half_residual = (self.size - new_h) / 2
            top_padding = math.ceil(half_residual)
            bottom_padding = math.floor(half_residual)
            img = TF.pad(img, padding=[0, top_padding, 0, bottom_padding])  # type: ignore
        return img


def load_model_from_artifact(
    filepath: Union[Path, str],
    *,
    project: Optional[str] = None,
    filename: str = "final_model.pt",
    target_dim: Optional[int] = None,
) -> Tuple[nn.Module, int, Optional[int]]:
    filepath = Path(filepath)
    if not filepath.exists():
        raise RuntimeError(
            f"No pre-existing model-artifact found at location '{filepath.resolve()}'"
            " and because no wandb run has been specified, it can't be downloaded."
        )
    state_dict = torch.load(filepath)
    backbone_conf = state_dict["config"]
    module, class_ = backbone_conf.pop("_target_").rsplit(sep=".", maxsplit=1)
    loaded_module = importlib.import_module(module)
    bb_fn: BackboneFactory = getattr(loaded_module, class_)(**backbone_conf)
    backbone, feature_dim = bb_fn()
    logger.info("Loading saved parameters and buffers...")
    backbone.load_state_dict(state_dict["state"]["backbone"])
    logger.info(f"Model artifact successfully loaded from '{filepath.resolve()}'.")
    return backbone, feature_dim, state_dict["image_size"]


def main() -> None:
    logger.info("Starting main script")
    # load test set data and pretrained model
    query_scenarios = cast(
        pd.DataFrame, pd.read_csv(DATA_DIRECTORY / "query_scenarios.csv", index_col="scenario_id")
    )
    metadata = cast(
        pd.DataFrame, pd.read_csv(DATA_DIRECTORY / "metadata.csv", index_col="image_id")
    )
    logger.info("Loading pre-trained model...")
    backbone, feature_dim, image_size = load_model_from_artifact(MODEL_PATH)
    model = Model(backbone=backbone, feature_dim=feature_dim)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # we'll only precompute embeddings for the images in the scenario files (rather than all images), so that the
    # benchmark example can run quickly when doing local testing. this subsetting step is not necessary for an actual
    # code submission since all the images in the test environment metadata also belong to a query or database.
    scenario_imgs = []
    for row in query_scenarios.itertuples():
        scenario_imgs.extend(pd.read_csv(DATA_DIRECTORY / row.queries_path).query_image_id.values)
        scenario_imgs.extend(
            pd.read_csv(DATA_DIRECTORY / row.database_path).database_image_id.values
        )
    scenario_imgs = sorted(set(scenario_imgs))
    metadata = metadata.loc[scenario_imgs]

    # instantiate dataset/loader and generate embeddings for all images
    dataset = TestTimeWhaledoDataset(metadata, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=16)
    embeddings = []
    model.eval()

    logger.info("Precomputing embeddings")
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            x = batch["image"].to(device)
            batch_embeddings = model(x)
            batch_embeddings_df = pd.DataFrame(
                batch_embeddings.cpu().detach().numpy(), index=batch["image_id"]
            )
            embeddings.append(batch_embeddings_df)

    embeddings = pd.concat(embeddings)
    logger.info(f"Precomputed embeddings for {len(embeddings)} images")
    logger.info("Generating image rankings")
    # process all scenarios
    results = []
    for row in query_scenarios.itertuples():
        # load query df and database images; subset embeddings to this scenario's database
        qry_df = cast(pd.DataFrame, pd.read_csv(DATA_DIRECTORY / row.queries_path))
        db_img_ids = cast(
            pd.DataFrame, pd.read_csv(DATA_DIRECTORY / row.database_path).database_image_id.values
        )
        db_embeddings = embeddings.loc[db_img_ids]

        # predict matches for each query in this scenario
        for qry in qry_df.itertuples():
            # get embeddings; drop query from database, if it exists
            qry_embedding = embeddings.loc[[qry.query_image_id]]
            _db_embeddings = db_embeddings.drop(qry.query_image_id, errors="ignore")
            with torch.no_grad():
                qry_embedding_t = torch.as_tensor(qry_embedding, device=device)
                db_embeddings_t = torch.as_tensor(_db_embeddings, device=device)
                prediction = model.predict(queries=qry_embedding_t, db=db_embeddings_t, k=20)
            # append result
            db_ids = _db_embeddings.index[prediction.database_inds.cpu().numpy()].to_numpy()
            qry_result = pd.DataFrame(
                {
                    "query_id": qry.query_id,
                    "database_image_id": db_ids,
                    "score": prediction.scores.cpu().numpy(),
                }
            )
            results.append(qry_result)

    logger.info(f"Writing predictions file to {PREDICTION_FILE}")
    submission = pd.concat(results)
    submission.to_csv(PREDICTION_FILE, index=False)


if __name__ == "__main__":
    main()
