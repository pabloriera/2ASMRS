# from kiwisolver import Variable
from pytorch_lightning.callbacks import Callback
from copy import copy
import numpy as np
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import json
from pathlib import Path
from sklearn.decomposition import PCA
from autoencoder import PCADecoder

eps = 1e-10

class MetricTracker(Callback):

    def __init__(self):
        self.collection = []
        self.epoch = 0

    def on_validation_epoch_end(self, trainer, module):
        metrics = {}
        for k,v in trainer.logged_metrics.items():
            metrics[k] = copy(v.cpu().detach().numpy()) 
        self.collection.append(metrics)


class MyEarlyStopping(EarlyStopping):
    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        if trainer.current_epoch < 10:
            pass
        else:
            self._run_early_stopping_check(trainer)

    def on_train_end(self, trainer, pl_module):
        # instead, do it at the end of training loop
        self._run_early_stopping_check(trainer)


def export_decoder(ae, X, run_name, hps, pca_latent_space=False):

    if pca_latent_space:
        Z = ae.encoder(X).detach().cpu().numpy()
        pca = PCA()
        Zpca = pca.fit_transform(Z)
        rangeZ = np.ceil(np.abs(Zpca).max(0))
        decoder = PCADecoder(torch.tensor(pca.components_), ae.decoder)
    else:
        Z = ae.encoder(X).detach().cpu().numpy()
        rangeZ = np.ceil(np.abs(Z).max(0))
        decoder = ae.decoder

    output_path = Path("exported_models",f"{run_name}.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    decoder.eval()
    example = torch.zeros(1, hps['encoder_layers'][-1])
    traced_script_module = torch.jit.trace(decoder, example)
    traced_script_module.save(output_path)
    
    data = {
        "parameters": {
                    "latent_dim": hps['encoder_layers'][-1],
                    "sClip": hps['db_min_norm'],
                    "sr": hps['target_sampling_rate'],
                    "win_length": hps['win_length'],
                    "xMax": hps['Xmax'],
                    "zRange": [{"max": int(v), "min": -int(v)} for v in rangeZ]
                },
                "model_path": str(Path("exported_models", f"{run_name}.pt").absolute())
            }
    output_path = Path('exported_models',f"{run_name}.json")

    with open(output_path, 'w') as fp:
        json.dump(data, fp)
