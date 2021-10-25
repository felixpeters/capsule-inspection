#pylint: skip-file
"""All things config, e.g., constants and hyperparameters."""


class URLs():
    """Global constants for dataset URLs."""
    SENSUM_SODF = "https://www.sensum.eu/resources/SensumSODF.7z"


class WANDB():
    """Global constants for all things related to Weights & Biases."""
    PROJECT_NAME = "capsule-inspection"
    CAPSULE_DATA_ARTIFACT = "capsule-data-raw"
    SOFTGEL_DATA_ARTFIFACT = "softgel-data-raw"
    CAPSULE_TABLE_ARTIFACT = "capsule-table-raw"
    SOFTGEL_TABLE_ARTIFACT = "softgel-table-raw"
