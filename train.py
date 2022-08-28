import os
from pathlib import Path

import fire
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms

from dataset import AmazonDataset


class Model(pl.LightningModule):
    def __init__(self, classes=17):
        super().__init__()
        self.save_hyperparameters()

        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in model.parameters():
            param.require_grad = False
        model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        model.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, classes),
        )
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"val_loss": avg_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class Datamodule(pl.LightningDataModule):
    def __init__(self, data_dir, bs=64):
        super().__init__()
        self.trsfms = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.data_dir = data_dir
        self.bs = bs

    def setup(self, stage=None):
        df = pd.read_csv(Path(self.data_dir) / "train_v2.csv")
        df["list_tags"] = df.tags.str.split(" ")
        encoder = MultiLabelBinarizer()
        train_data, val_data = train_test_split(df, test_size=0.2)
        ohe_train_tags = encoder.fit_transform(train_data.list_tags.values)
        ohe_val_tags = encoder.fit_transform(val_data.list_tags.values)
        self.train_dataset = AmazonDataset(
            train_data,
            ohe_tags=ohe_train_tags,
            transform=self.trsfms,
            path=Path(self.data_dir) / "train",
        )
        self.val_dataset = AmazonDataset(
            val_data,
            ohe_tags=ohe_val_tags,
            transform=self.trsfms,
            path=Path(self.data_dir) / "train",
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=64,
            num_workers=os.cpu_count(),
        )


def runner(path):
    model = Model()
    data = Datamodule(data_dir=path)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="ckpt",
        filename="model-epoch{epoch:02d}-val_loss{val/loss:.2f}",
        auto_insert_metric_name=True,
    )
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=20)],
        accelerator="gpu",
        enable_progress_bar=True,
        enable_model_summary=True,
        max_epochs=10,
        fast_dev_run=True,
        enable_checkpointing=True,
        log_every_n_steps=1,
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    fire.Fire(runner)
