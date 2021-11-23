import numpy as np
import pytest
import torch
from src.models.architecture import SimpleConvNet
from src.models.classification import CapsuleClassificationModel

from ..data.classification import classification_data_module


@pytest.fixture(scope="session")
def classification_model():
    net = SimpleConvNet(img_size=(16, 16), num_filters=8, num_classes=2)
    model = CapsuleClassificationModel(net)
    return model


def test_forward(classification_model):
    x = torch.rand((4, 3, 16, 16))
    output = classification_model.forward(x).detach().numpy()
    assert output.shape == (4, 2)
    assert (output >= 0).all()
    assert (output <= 1).all()
    sums = output.sum(axis=1)
    np.testing.assert_allclose(sums, np.ones(sums.shape), rtol=1e-06)


def test_configure_optimizers(classification_model):
    optimizer = classification_model.configure_optimizers()
    assert len(optimizer.param_groups) == 1
    for param_group in optimizer.param_groups:
        assert param_group["lr"] == 1e-04


def test_training_step(classification_model):
    batch = (torch.rand((4, 3, 16, 16)), torch.ones((4,), dtype=torch.long))
    loss = classification_model.training_step(
        batch, 1)["loss"].detach().numpy()
    assert (loss >= 0).all()


def test_validation_step(classification_model):
    batch = (torch.rand((4, 3, 16, 16)), torch.ones((4,), dtype=torch.long))
    loss = classification_model.validation_step(
        batch, 1)["val_loss"].detach().numpy()
    assert (loss >= 0).all()
