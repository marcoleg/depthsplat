import json
from pathlib import Path
import argparse

import torch
from tqdm import tqdm


def main(dataset_path: Path):
    for stage in ["test"]:
        stage = dataset_path / stage

        index = {}
        for chunk_path in tqdm(
            sorted(list(stage.iterdir())), desc=f"Indexing {stage.name}"
        ):
            if chunk_path.suffix == ".torch":
                chunk = torch.load(chunk_path)
                for example in chunk:
                    index[example["key"]] = str(chunk_path.relative_to(stage))
        with (stage / "index.json").open("w") as f:
            json.dump(index, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate index.json for dataset.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset folder (e.g., /path/to/DEPTHSPLAT_xxx)"
    )

    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    main(dataset_path)