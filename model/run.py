from fire import Fire
from pathlib import Path
import pandas as pd
import torch

from audio_utils import get_spectrograms_from_audios, save_audio, save_latentscore, save_specgram
from autoencoder import AutoEncoder


def train(audio_path_list, target_sampling_rate=22050,
		  hop_length_samples=512,
          win_length_samples=2048,
          encoder_layers=(512, 256, 128, 64, 32, 16, 8, 4),
          seed=42,
          db_min_norm=-60,
          spec_in_db=True,
          normalize_each_audio=False,
          validation_size=0.05,
          learning_rate=0.001,
          epochs=1000,
          batch_size=128,
          loss='MAE+MSE',
          log_path='logs',
          run_name='example',
          accelerator='cpu',
          pca_latent_space=False, 
          checkpoint_path=None):

    hop_length = hop_length_samples
    win_length = win_length_samples

    print(f'hop_length: {hop_length}, win_length: {win_length}')

    X, phases, Xmax, y = get_spectrograms_from_audios(audio_path_list,
                                                      target_sampling_rate,
                                                      win_length,
                                                      hop_length,
                                                      db_min_norm=db_min_norm,
                                                      spec_in_db=spec_in_db,
                                                      normalize_each_audio=normalize_each_audio)
    print(
        f'X shape: {X.shape}, phases shape: {phases.shape}, Xmax: {Xmax}, y shape: {y.shape}')

    if isinstance(encoder_layers, str):
        encoder_layers = tuple(map(int, encoder_layers[1:-1].split(',')))

    encoder_layers = (win_length//2+1,)+encoder_layers
    decoder_layers = encoder_layers[::-1][1:]


    ae = AutoEncoder(encoder_layers, decoder_layers, seed=seed, checkpoint_path=checkpoint_path)
    ae.load_data(X, y, Xmax, db_min_norm=db_min_norm, spec_in_db=spec_in_db)
    ae.split_data(validation_size=validation_size)

    # log all hyperparameters and paramters
    hps = {
        'encoder_layers': encoder_layers,
        'decoder_layers': decoder_layers,
        'win_length': win_length,
        'hop_length': hop_length,
        'loss': loss,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'spec_in_db': spec_in_db,
        'db_min_norm': db_min_norm,
        'normalize_each_audio': normalize_each_audio,
        'target_sampling_rate': target_sampling_rate,
        'Xmax': Xmax,

    }

    ae.log_hyperparameters(**hps)
    trainer, metrics_tracker = ae.train_model(
        loss,
        learning_rate,
        epochs,
        batch_size,
        log_path,
        run_name=run_name,
        accelerator=accelerator)

    pd.DataFrame(metrics_tracker.collection).astype(float).to_csv(
        Path(trainer.log_dir, 'metrics_history.csv'))

    ae.export_decoder(pca_latent_space)

    predicted_specgram = ae.predict(X)*Xmax

    save_audio(predicted_specgram, db_min_norm, phases, hop_length,
               win_length, target_sampling_rate,  trainer.log_dir, spec_in_db)    
    
    X = X[:2*60*target_sampling_rate//hop_length]
    predicted_specgram = ae.predict(X)*Xmax
    
    save_specgram(predicted_specgram, hop_length,  trainer.log_dir)

    with torch.no_grad():
        Z = ae.encoder(X).cpu().numpy()
    save_latentscore(Z, hop_length, target_sampling_rate,  trainer.log_dir)



def main(path=None, **kwargs):

    if path is None:
        print('No path provided, using example')
        # Download example from url if already not exists
        path = Path('data', 'Mozart25_2min.wav')
        if not path.exists():
            import requests
            path.parent.mkdir(parents=True, exist_ok=True)
            url = 'https://www.dropbox.com/s/iar9y2beo884zah/Mozart25_2min.wav?dl=1'
            r = requests.get(url, allow_redirects=True)
            path.write_bytes(r.content)

    path = Path(path)
    if path.is_file():
        audio_list = [path]
    else:
        # Load all wavfiles in directory
        audio_list = list(path.glob('*.*'))
    train(audio_list, **kwargs)


if __name__ == '__main__':
    Fire(main)
