from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import cast

import typer
import wandb

from submission_src.main import MODEL_PATH
from submission_src.whaledo.models.artifact import download_artifact


def main(
    artifact_name: str = typer.Argument(
        ...,
        help="Name of the artifact to download from wandb.",
    )
) -> None:
    run = wandb.init(entity="predictive-analytics-lab", project="whaledo")
    root = ""
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        artifact_path = cast(Path, download_artifact(name=artifact_name, run=run, root=root))
        dst = Path("submission_src") / MODEL_PATH.name
        if dst.exists():
            typer.echo(f"Removing existing model from {dst.resolve()}")
            dst.unlink()
        src = artifact_path / "final_model.pt"
        path = shutil.move(src=str(src), dst=dst)
        typer.echo(f"Artifact '{artifact_name}' downloaded and moved to '{dst.resolve()}'")


if __name__ == "__main__":
    typer.run(main)
