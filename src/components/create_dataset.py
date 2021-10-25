from typing import Tuple

import wandb

from ..data import download_data
from ..config import URLs, WANDB


def main():
    """Downloads dataset from given URL and creates data artifacts.

    Args:
        url (str): Dataset URL
    """
    data_path = download_data(URLs.SENSUM_SODF)

    wandb.init(project=WANDB.PROJECT_NAME,
               job_type="data-collection", save_code=True)

    capsule_data = wandb.Artifact(WANDB.CAPSULE_DATA_ARTIFACT, type="dataset")
    capsule_data.add_dir(str(data_path/"capsule/"))
    wandb.log_artifact(capsule_data)

    softgel_data = wandb.Artifact(WANDB.SOFTGEL_DATA_ARTFIFACT, type="dataset")
    softgel_data.add_dir(str(data_path/"softgel/"))
    wandb.log_artifact(softgel_data)


if __name__ == "__main__":
    main()
