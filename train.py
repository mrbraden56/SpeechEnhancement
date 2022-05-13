import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import torch.optim.lr_scheduler
from pytorch_lightning import Trainer
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.optim import adam
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset

class ConvAutoEncoder8_lightning(pl.LightningModule):
    def __init__(self):
        # N, 1, 64, 87
        super(ConvAutoEncoder8_lightning, self).__init__()
        self.learning_rate = 0.01
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 8)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(64, 128, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, (1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return decoder

    def add_channels(self, combined, clean):
        return combined.unsqueeze(dim=1).float(), clean.unsqueeze(dim=1).float()

    def training_step(self, batch, batch_idx):
        combined, clean = batch
        combined = combined[:, 1:, :]
        combined, clean = self.add_channels(combined, clean)
        output = self(combined)
        loss = F.l1_loss(output, clean)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        combined, clean = batch
        combined = combined[:, 1:, :]
        combined, clean = self.add_channels(combined, clean)
        output = self(combined)
        loss = F.mse_loss(output, clean)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        combined, clean = batch
        combined = combined[:, 1:, :]
        combined, clean = self.add_channels(combined, clean)
        output = self(combined)
        loss = F.mse_loss(output, clean)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-6,
                                     amsgrad=True)  # weight_decay=1e-6
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=.1),
            },
        }

class AudioDenoiserDataset(Dataset):

    def __init__(self, audio_dir):
        self.audio_dir=audio_dir
        self.audio_files=natsorted(os.listdir(self.audio_dir))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        audio_sample_path=self.get_audio_sample_path(index)
        combined, clean=self.get_labels(audio_sample_path)
        return combined, clean


    def get_audio_sample_path(self, index):
        path=os.path.join(self.audio_dir, self.audio_files[index])
        return path

    def get_labels(self, audio_sample_path):
        audio=np.load(audio_sample_path, allow_pickle=True)
        combined=audio[0]
        clean=audio[1]
        return combined, clean

    def add_column(self, audio):
        z = np.zeros((256,3), dtype=np.int64)
        audio=np.append(audio, z, axis=1)
        return audio

def lightning():
  dataset = AudioDenoiserDataset("/content/train_sr")
  train_loader = DataLoader(dataset=dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=2)
  dir = "/content/drive/MyDrive/JayHear"

  #1. Initialize Model
  model = ConvAutoEncoder8_lightning()

  #2. Initialize Trainer
  trainer = Trainer(gpus=1, default_root_dir=dir, max_epochs=150)
  trainer.fit(model, train_loader)


def main(arguments):
    """Main func."""
    lightning()

if __name__ == "__main__":
    main(sys.argv[1:])