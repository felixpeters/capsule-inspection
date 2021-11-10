import shutil
from pathlib import Path

from py7zr import pack_7zarchive, unpack_7zarchive
from fastai.data.external import untar_data
from fastai.data.transforms import get_image_files
from fastai.vision.data import SegmentationDataLoaders
from fastai.vision.learner import unet_learner
from fastai.vision.models import resnet18
from PIL import Image

SENSUM_SODF = "https://www.sensum.eu/resources/SensumSODF.7z"
MODEL_PATH = Path("../models/softgel_segmentation.model").absolute()
BATCH_SIZE = 8
NUM_EPOCHS = 10

shutil.register_archive_format(
    '7zip', pack_7zarchive, description='7zip archive')
shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive)


def label_func(fn):
    image_class = str(fn).split("/")[-3]
    if image_class == "positive":
        return path/f"softgel/positive/annotation/{fn.stem}{fn.suffix}"

    neg_masks_path = path/"softgel/negative/annotation"
    neg_masks_path.mkdir(exist_ok=True)
    mask_path = neg_masks_path/f"{fn.stem}{fn.suffix}"

    if mask_path.exists():
        return mask_path

    img = Image.open(fn)
    mask = Image.new("RGB", img.size)
    mask.save(mask_path)
    return mask_path


path = untar_data(SENSUM_SODF)

pos_fnames = get_image_files(path/"softgel/positive/data")
neg_fnames = get_image_files(path/"softgel/negative/data")
fnames = pos_fnames + neg_fnames

dls = SegmentationDataLoaders.from_label_func(
    path, bs=BATCH_SIZE, fnames=fnames, label_func=label_func, codes=["no_defect", "defect"])

learner = unet_learner(dls, resnet18)
learner.fit(NUM_EPOCHS)
learner.save(MODEL_PATH)
