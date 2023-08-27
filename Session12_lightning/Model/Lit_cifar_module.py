from Dataset.trainalbumentation import TrainAlbumentation
from Dataset.testalbumentation import TestAlbumentation
from pytorch_lightning import LightningModule
from Model import custom_resnet
from torchmetrics import Accuracy
from torchvision.datasets import CIFAR10
from torch.nn import functional as F
import torch
from torch.utils.data import DataLoader, random_split
from utils import *
import os
class LitCifar(LightningModule):
    def __init__(self, data_dir='./data', hidden_size=16, learning_rate=2e-4):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.train_transform = TrainAlbumentation()
        self.test_transform = TestAlbumentation()

        # Define PyTorch model
        device = getdevice()
        self.model = custom_resnet.Net('bn').to(device)

        self.accuracy = Accuracy('MULTICLASS', num_classes=10)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 10)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.train_transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=32, num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=32, num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=32, num_workers=os.cpu_count())