"""All functionality related to logging data and model artifacts in Weights & Biases."""
from pathlib import Path

import numpy as np
import wandb
from PIL import Image


def create_data_table(path: Path) -> wandb.Table:
    """Create a table that can be logged to the Weights & Biases dashboard.

    Args:
        path (Path): Data directory

    Returns:
        wandb.Table: Data table
    """
    data = []
    pos_fns = list((path/"positive/data").iterdir())
    neg_fns = list((path/"negative/data").iterdir())
    test_img = Image.open(neg_fns[0])
    img_size = test_img.size
    test_img.close()
    for idx, file_name in enumerate(pos_fns):
        img = wandb.Image(str(file_name))
        mask = wandb.Image(str(path/f"positive/annotation/{file_name.name}"))
        row = [idx, file_name.name, img, mask, "positive"]
        data.append(row)

    for idx, file_name in enumerate(neg_fns, start=len(pos_fns)):
        img = wandb.Image(str(file_name))
        mask = wandb.Image(np.zeros(img_size))
        row = [idx, file_name.name, img, mask, "negative"]
        data.append(row)

    columns = ["id", "filename", "image", "mask", "label"]
    return wandb.Table(data=data, columns=columns)
