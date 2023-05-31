# Utils
from glob import glob
from pathlib import Path

# Numbers
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import librosa.display

# Machine learning
import torch
import torchaudio
from torchaudio.functional import resample

# Audio
import torchaudio
import librosa

def get_waveform(path, target_sr, duration=None):
    # Load waveform using torchaudio, if sample rate is different from target_sr, resample
    waveform, original_sr = torchaudio.load(path)
    if original_sr != target_sr:
        print(f'Resampling from {original_sr} to {target_sr}')
        waveform = resample(waveform, original_sr, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform[0,:] # Use left channel
    if duration is not None:
        waveform = waveform[:int(target_sr * duration)]  # Trim to custom duration
    return waveform


def get_specgram(waveform, win_length, hop_length, spec_in_db=True):
    F = torch.stft(waveform, n_fft=win_length, hop_length=hop_length,
                   win_length=win_length, return_complex=True).T
    if spec_in_db:
        S = 10*torch.log10(torch.abs(F)**2)
    else:
        S = torch.abs(F)
    return torch.angle(F), S


def get_spectrograms_from_audios(audio_path_list, target_sr, win_length, hop_length, db_min_norm=None, spec_in_db=True, normalize_each_audio=False):
    X = []
    y = []
    phases = []
    for i, filename in enumerate(audio_path_list):
        waveform = get_waveform(filename, target_sr)
        phase, S = get_specgram(waveform, win_length, hop_length, spec_in_db = spec_in_db)
        if normalize_each_audio:
            S = S / S.max()
        phases.append(phase)
        X.append(S)
        y.append(torch.ones(S.shape[0]) * i)
    phases = torch.vstack(phases)
    X = torch.vstack(X)
    if spec_in_db and db_min_norm is not None:
        X = X.clip(db_min_norm, None) - db_min_norm
    X_max = float(X.max().numpy())
    X = X / X_max
    y = torch.hstack(y)
    return X, phases, X_max, y


def save_specgram(specgram, hop_length, path):
    plt.figure(figsize=(14, 4))
    librosa.display.specshow(specgram.detach().numpy(
    ).T, y_axis='linear', x_axis='time', hop_length=hop_length)
    plt.colorbar()
    plt.savefig(Path(path, "predicted_spectrogram.png"))


def spectrogram2audio(Y, Sclip, phase, hop_length, win_length, in_db):
    if in_db:
        Y_ = torch.sqrt(10**((Y+Sclip)/10))*torch.exp(1j*phase)
    else:
        Y_ = Y*torch.exp(1j*phase)
    return torch.istft(Y_.T, hop_length=hop_length, n_fft=win_length)


def save_audio(Y, Sclip, phase, hop_length, win_length, samplerate, path, in_db):
    audio = spectrogram2audio(Y, Sclip, phase, hop_length, win_length, in_db)
    torchaudio.save(Path(path, 'reconstructed.mp3'), audio.reshape(
        1, -1), samplerate, format='mp3', compression=320)
