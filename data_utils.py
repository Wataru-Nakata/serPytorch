import torch
import random
from utils import load_filepaths_and_text, load_wav_to_torch
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning
import model
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import layers


class MelEmoLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text,split=",")
        self.sampling_rate = 22050
        self.max_wav_value = hparams.max_wav_value
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self.category_to_label = {"neu":0,"hap":1, "sad":2, "ang":3}

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, label = audiopath_and_text[1], audiopath_and_text[2]
        encoded = torch.tensor(self.category_to_label[label])
        mel = self.get_mel("../iemocap_wavs/" + audiopath + ".wav")
        return  mel, encoded

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if len(audio)  > sampling_rate*3:
                start = random.randint(0,len(audio)- sampling_rate*3)
                audio= audio[start:start+sampling_rate*3]
            else:
                dummy = torch.zeros(sampling_rate*3)
                dummy[:len(audio)] = audio
                audio = dummy
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)
