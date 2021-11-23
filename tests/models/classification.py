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
