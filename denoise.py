import sys
import librosa
import librosa.display
import scipy
import soundfile as sf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

class ConvAutoEncoder8BIG_lightning(pl.LightningModule):
    def __init__(self):
        # N, 1, 64, 87
        super(ConvAutoEncoder8BIG_lightning, self).__init__()
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
            nn.Conv2d(512, 1024, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(512),
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
                "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.1),
            },
        }


class ConvAutoEncoder8VAE_lightning(pl.LightningModule):
    def __init__(self):
        # N, 1, 64, 87
        super(ConvAutoEncoder8VAE_lightning, self).__init__()
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
            nn.Conv2d(512, 1024, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(512),
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

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = h, h
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x):
        encoder = self.encoder(x)
        z, mu, logvar = self.bottleneck(encoder)
        decoder = self.decoder(z)
        return decoder, mu, logvar

    def add_channels(self, combined, clean):
        return combined.unsqueeze(dim=1).float(), clean.unsqueeze(dim=1).float()

    def loss_vae(self, predicted, target, mu, logvar):
        kl_loss=-0.5*torch.sum(1+logvar-torch.square(mu)-torch.exp(logvar))
        rc_loss=F.l1_loss(predicted, target)
        combined_loss=1000*rc_loss+kl_loss
        return combined_loss

    def training_step(self, batch, batch_idx):
        combined, clean = batch
        combined = combined[:, 1:, :]
        combined, clean = self.add_channels(combined, clean)
        output, mu, logvar = self(combined)
        loss = self.loss_vae(output, clean, mu, logvar)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        combined, clean = batch
        combined = combined[:, 1:, :]
        combined, clean = self.add_channels(combined, clean)
        output = self(combined)
        loss = F.l1_loss(output, clean)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        combined, clean = batch
        combined = combined[:, 1:, :]
        combined, clean = self.add_channels(combined, clean)
        output = self(combined)
        loss = F.l1_loss(output, clean)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-6,
                                     amsgrad=True)  # weight_decay=1e-6
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.1),
            },
        }


class ConvAutoEncoder8_lightning(pl.LightningModule):
    def __init__(self):
        # N, 1, 64, 87
        super(ConvAutoEncoder8_lightning, self).__init__()
        self.learning_rate = 0.001

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

    # def validation_step(self, batch, batch_idx):
    #     combined, clean = batch
    #     combined = combined[:, 1:, :]
    #     combined, clean = self.add_channels(combined, clean)
    #     output = self(combined)
    #     loss = F.mse_loss(output, clean)
    #     self.log('val_loss', loss)
    #
    # def test_step(self, batch, batch_idx):
    #     combined, clean = batch
    #     combined = combined[:, 1:, :]
    #     combined, clean = self.add_channels(combined, clean)
    #     output = self(combined)
    #     loss = F.mse_loss(output, clean)
    #     self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-6,
                                     amsgrad=True)  # weight_decay=1e-6
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.1),
            },
        }
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', cooldown=2),
        #         "monitor": "train_loss"
        #     },
        # }



class AudioDenoiserDataset(Dataset):

    def __init__(self, audio_files):
        self.audio_files=audio_files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        return self.audio_files[index]



class ConvertToAudio:
    def __init__(self, model, noisy_audio_path):
        self.model = model.float()
        self.noisy_audio_path = noisy_audio_path
        self.window = scipy.signal.hamming(256, sym=False)
        self.noisy_log_spec = None
        self.noisy_phase = None
        self.noisy_log_spec_orig=None

    def adjust_shape(self, audio):  # Makes numpy input a tensor with dim [Batch, Tensor, height, width]
        audio = torch.tensor(audio)
        audio = audio.unsqueeze(dim=0).float()
        audio = audio.unsqueeze(dim=0).float()
        return audio

    def normalize(self, array):
        self.MinMax = MinMaxScaler(feature_range=(-1, 1))
        return self.adjust_shape(self.MinMax.fit_transform(array))

    def denormalize(self, norm):
        return self.MinMax.inverse_transform(norm)

    def to_log_spectrogram(self):
        noisy_signal, sr = librosa.load(path=self.noisy_audio_path, sr=8000, dtype='double')
        self.noisy_log_spec = librosa.stft(noisy_signal, n_fft=256, hop_length=round(256 * 0.25), win_length=256,
                                           window=self.window, center=True)
        self.noisy_phase = np.angle(self.noisy_log_spec)
        self.noisy_log_spec_orig=librosa.amplitude_to_db(np.abs(self.noisy_log_spec))
        self.noisy_log_spec = self.adjust_shape(librosa.amplitude_to_db(np.abs(self.noisy_log_spec)))

    def add_zeros_to_front(self, array):
        zeros = np.zeros((1, 1, 129, 7))
        array = torch.tensor(np.append(zeros, array, axis=3))
        return array

    def extract_frames(self, spectrogram) -> list[ndarray]:
        data=[]
        length=spectrogram.shape[3]-7
        spectrogram = spectrogram[0, 0, :, :]
        spectrogram = spectrogram.detach().numpy()
        for i in range(0, length):
            spectrogram_frames = spectrogram[:, i:i + 8]  # [0,1]
            data.append(np.array(spectrogram_frames))
        return data


    def create_batch(self, VAE=False):
        dataset=self.extract_frames(self.noisy_log_spec)
        dataset = AudioDenoiserDataset(dataset)
        train_loader = DataLoader(dataset=dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=16)
        batch=[]
        temp=[]
        for i, batch_spectrogram_frames in enumerate(train_loader):
            batch_spectrogram_frames=batch_spectrogram_frames.unsqueeze(dim=1).float()
            denoised_spectrogram_frames=self.model(batch_spectrogram_frames)
            denoised_spectrogram_frames = denoised_spectrogram_frames[:, 0, :, :]
            denoised_spectrogram_frames = denoised_spectrogram_frames.detach().numpy()
            batch.append(denoised_spectrogram_frames)
            print(f'Creating batch {i} of data')
        for i, arrays in enumerate(batch):
            for frames in arrays:
                self.modeled_audio = np.append(self.modeled_audio, frames, axis=1)

    def feed_multi_into_model(self):
        self.noisy_log_spec = self.normalize(self.noisy_log_spec[0, 0, :, :])
        self.noisy_log_spec = self.add_zeros_to_front(self.noisy_log_spec)
        self.modeled_audio = np.zeros((129, 1))
        self.create_batch(VAE=False)
        self.modeled_audio = self.modeled_audio[:, 1:]  # removes the layer of zeros we initially added
        return self.modeled_audio

    def apply_griffin(self):
        self.modeled_audio = self.denormalize(self.modeled_audio)
        self.modeled_audio=librosa.db_to_amplitude(self.modeled_audio)
        self.modeled_audio=np.squeeze(self.modeled_audio)
        #reconstructing phase
        self.modeled_audio=self.modeled_audio*np.exp(1j*self.noisy_phase)
        denoised_audio = librosa.istft(self.modeled_audio, hop_length=round(256 * 0.25),
                                     win_length=256,
                                     window=self.window, center=True)
        return denoised_audio


def show_spectrogram(clean_audio) -> None:
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)

    img = librosa.display.specshow(librosa.amplitude_to_db(clean_audio, ref=np.max), y_axis='linear',
                                   x_axis='time',
                                   ax=ax, sr=16000, hop_length=round(256 * 0.25))
    ax.set_title('Clean Spectrogram')
    ax.label_outer()

    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.savefig("/home/braden/Environments/JayHear_Production/project/denoised_audio_VAE")


def denoise_audio_files(model_path, noisy_audio_path, save_denoised_path):
    print('=> Denoising Audio')
    model = ConvAutoEncoder8_lightning()
    model = model.load_from_checkpoint(model_path)
    convert = ConvertToAudio(model, noisy_audio_path)
    convert.to_log_spectrogram()
    convert.feed_multi_into_model()
    clean_audio = convert.apply_griffin()
    # show_spectrogram(picture_spectrogram)
    sf.write(save_denoised_path, clean_audio, 8000, 'PCM_24', format="wav")
    print('=> Denoising Audio Complete')

def main(arguments):
    """Main func."""
    model_path="/home/braden/Environments/JayHear_Production/project/20_hours/epoch=9-step=320449.ckpt"
    noisy_audio_path="/home/braden/Work/JayHear/MS-SNSD/NoisySpeech_training/noisy13_SNRdb_0.0_clnsp13.wav"
    save_denoised_path='/home/braden/Environments/JayHear_Production/project/denoised_noisy13_SNRdb_0.wav'
    denoise_audio_files(model_path, noisy_audio_path, save_denoised_path)
if __name__ == "__main__":
    main(sys.argv[1:])