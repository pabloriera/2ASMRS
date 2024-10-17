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
        waveform = waveform[0, :]  # Use left channel
    if duration is not None:
        # Trim to custom duration
        waveform = waveform[:int(target_sr * duration)]
    return waveform


def get_specgram(waveform, win_length, hop_length, spec_in_db=True):
    F = torch.stft(waveform,
                   n_fft=win_length,
                   hop_length=hop_length,
                   win_length=win_length,
                   return_complex=True,
                   window=torch.hann_window(win_length)).T
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
        phase, S = get_specgram(waveform, win_length,
                                hop_length, spec_in_db=spec_in_db)
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

def save_latentscore(Z,  hop_length, sr, path):
    plt.figure(figsize=(14, 4))
    t = np.arange(0, Z.shape[0])*hop_length/sr
    plt.plot(t, Z + np.arange(Z.shape[1]) ,color='k')
    plt.savefig(Path(path, "Z_latent_score.png"))


def spectrogram2audio(Y, db_min_norm, phase, hop_length, win_length, in_db, griffinlim=False):
    if in_db:
        Y_ = torch.sqrt(10**((Y+db_min_norm)/10))*torch.exp(1j*phase)
    else:
        Y_ = Y*torch.exp(1j*phase)

    if griffinlim:
        audio = torch.tensor(librosa.griffinlim(Y_.numpy().T,
                                                hop_length=hop_length,
                                                win_length=win_length,
                                                window='hann'))
    else:
        audio = torch.istft(Y_.T,
                            hop_length=hop_length,
                            n_fft=win_length,
                            window=torch.hann_window(win_length).to(Y_.device))

    return audio


def save_audio(Y, db_min_norm, phase, hop_length, win_length, samplerate, path, in_db):
    audio = spectrogram2audio(Y, db_min_norm, phase,
                              hop_length, win_length, in_db)
    torchaudio.save(Path(path, 'reconstructed.mp3'), audio.reshape(
        1, -1), samplerate, format='mp3', compression=torchaudio.io.CodecConfig(bit_rate=320))
