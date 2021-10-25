"""All functionality related to logging data and model artifacts in Weights & Biases."""
from pathlib import Path

from PIL import Image
import wandb


def create_data_table(path: Path) -> wandb.Table:
    """Create a table that can be logged to the Weights & Biases dashboard.

    Args:
        path (Path): Data directory

    Returns:
        wandb.Table: Data table
    """
    pos_imgs = [Image.open(f) for f in (path/"positive/data").iterdir()]
    pos_masks = [Image.open(f) for f in (path/"positive/annotation").iterdir()]
    neg_imgs = [Image.open(f) for f in (path/"negative/data").iterdir()]
    pos_fns = [f.name for f in (path/"positive/data").iterdir()]
    neg_fns = [f.name for f in (path/"negative/data").iterdir()]
    img_size = neg_imgs[0].size
    neg_masks = [Image.new("RGB", img_size) for _ in neg_imgs]

    ids = list(range(len(pos_imgs) + len(neg_imgs)))
    fns = pos_fns + neg_fns
    imgs = pos_imgs + neg_imgs
    masks = pos_masks + neg_masks
    labels = ["positive"] * len(pos_imgs) + ["negative"] * len(neg_imgs)
    data = [[id, fn, wandb.Image(img), wandb.Image(mask), label] for (
        id, fn, img, mask, label) in zip(ids, fns, imgs, masks, labels)]

    columns = ["id", "filename", "image", "mask", "label"]
    table = wandb.Table(data=data, columns=columns)
    return table
