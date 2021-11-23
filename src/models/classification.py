from typing import Dict, Sequence, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


class CapsuleClassificationModel(pl.LightningModule):
    """Defines architecture and optimization strategy for a capsule
    classification model. """

    def __init__(self, model: nn.Module, lr: float = 1e-4):
        """Initialize classification module with given backbone.

        Args:
            model (nn.Module): Backbone to use for this model.
            lr (float): Initial learning rate to use for Adam optimizer.
        """
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss = CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """To be used for inference only.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
           torch.Tensor: Output tensor
        """
        output = self.model(x)
        return F.softmax(output, dim=1)

    def training_step(self, batch: Sequence[torch.Tensor], batch_idx: int) -> Dict:
        """The complete training loop.

        Args:
            batch (Sequence[torch.Tensor]): Batch to be used for training.
            batch_idx (int): Number of the current batch.

        Returns:
            Dict: Dictionary containing the training loss.
        """
        x, y = batch
        output = self.model(x)
        loss = self.loss(output, y)
        logits = F.softmax(output, dim=1)
        self.train_acc(logits, y)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc)
        return {"loss": loss}

    def validation_step(self, batch: Sequence[torch.Tensor], batch_idx: int) -> Dict:
        """The complete validation loop.

        Args:
            batch (Sequence[torch.Tensor]): Batch to be used for validation.
            batch_idx (int): Number of the current batch.

        Returns:
            Dict: Dictionary containing the validation loss.
        """
        x, y = batch
        output = self.model(x)
        loss = self.loss(output, y)
        logits = F.softmax(output, dim=1)
        self.val_acc(logits, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc)
        return {"val_loss": loss}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Creates the Adam optimizer to be used for training the model.

        Returns:
            torch.optim.Optimizer: Adam optimizer
        """
        optimizer = Adam(self.model.parameters(), self.lr)
        return optimizer
