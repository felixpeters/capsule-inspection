import shutil
from pathlib import Path

from py7zr import pack_7zarchive, unpack_7zarchive
from fastai.data.external import untar_data
from fastai.data.transforms import get_image_files
from fastai.vision.data import ImageDataLoaders
from fastai.vision.learner import cnn_learner
from fastai.vision.models import resnet18
from fastai.metrics import accuracy, RocAucBinary
from fastai.callback.tracker import SaveModelCallback
from PIL import Image

SENSUM_SODF = "https://www.sensum.eu/resources/SensumSODF.7z"
MODEL_PATH = Path("/workspace/experiments/models/softgel/classification/").absolute()
MODEL_PATH.mkdir(exist_ok=True, parents=True)
BATCH_SIZE = 64
NUM_EPOCHS = 20

shutil.register_archive_format(
    '7zip', pack_7zarchive, description='7zip archive')
shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive)


def label_func(fn):
    image_class = str(fn).split("/")[-3]
    return image_class == "positive"

print("Start downloading data.")
path = untar_data(SENSUM_SODF)
print("Finished downloading and unpacking data.")

pos_fnames = get_image_files(path/"softgel/positive/data")
print(f"Found {len(pos_fnames)} images from the positive class.")
neg_fnames = get_image_files(path/"softgel/negative/data")
print(f"Found {len(neg_fnames)} images from the negative class.")
fnames = pos_fnames + neg_fnames

dls = ImageDataLoaders.from_path_func(
    path, fnames, label_func, bs=BATCH_SIZE)
print("Created data loader")

roc_auc = RocAucBinary()
save_model = SaveModelCallback()
learner = cnn_learner(dls, resnet18, path=MODEL_PATH, model_dir="resnet18", metrics=[accuracy, roc_auc], cbs=[save_model])
print("Created learner")
print("Start model training")
learner.fit(NUM_EPOCHS)
print("Finished model training")
