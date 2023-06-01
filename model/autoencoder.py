
import json
from pathlib import Path
import numpy as np
from utils import eps
import torch
from sklearn.model_selection import train_test_split
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, TensorDataset, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from utils import MetricTracker
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import seed_everything
from sklearn.decomposition import PCA


class Encoder(nn.Module):
    def __init__(self, layers_size):
        super().__init__()

        layers = []
        for i in range(len(layers_size)-1):
            layers.append(nn.Linear(layers_size[i], layers_size[i+1]))
            layers.append(nn.BatchNorm1d(layers_size[i+1]))
            layers.append(nn.ELU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, layers_size):
        super().__init__()
        layers = []
        for i in range(len(layers_size)-2):
            layers.append(nn.Linear(layers_size[i], layers_size[i+1]))
            layers.append(nn.BatchNorm1d(layers_size[i+1]))
            layers.append(nn.ELU())
        layers.append(nn.Linear(layers_size[-2], layers_size[-1]))
        layers.append(nn.BatchNorm1d(layers_size[-1]))
        # layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PCADecoder(nn.Module):
    def __init__(self, pca_matrix, decoder):
        super(PCADecoder, self).__init__()
        self.decoder = decoder
        self.pca_matrix = pca_matrix

    def forward(self, x):
        x = torch.matmul(x, self.pca_matrix)
        x = self.decoder(x)
        return x


class AutoEncoder(pl.LightningModule):
    def __init__(self, encoder_layers, decoder_layers, checkpoint_path=None, seed=None):
        super().__init__()
        self.seed = seed
        if seed:
            seed_everything(seed, workers=True)
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder = Encoder(self.encoder_layers)
        self.decoder = Decoder(
            (self.encoder_layers[-1],)+tuple(self.decoder_layers))
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def loss_fn(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.spec_in_db:
            mse = torch.mean((y_hat - x)**2)
        else:
            _y_hat = (10*torch.log10(y_hat**2).clip(self.db_min_norm,
                      None) - self.db_min_norm)/self.Xmax
            _x = (10*torch.log10(x**2).clip(self.db_min_norm, None) -
                  self.db_min_norm)/self.Xmax
            mse = torch.mean((_y_hat - _x)**2)

        if (self.loss_type == "MSE"):
            loss = torch.mean((y_hat - x)**2)
        elif (self.loss_type == "MAE"):
            loss = torch.mean(torch.abs(y_hat - x))
        elif (self.loss_type == "MSE_log"):
            loss = torch.mean(
                (torch.log10(y_hat.clip(0, None)+eps) - torch.log10(x+eps))**2)
        elif (self.loss_type == "MAE_log"):
            loss = torch.mean(torch.abs(torch.log10(
                torch.clip(y_hat, 0, None) + eps) - torch.log10(x+eps)))
        elif (self.loss_type == "MSRE"):
            loss = torch.mean((y_hat - x)**2) / (torch.mean(y_hat**2)+eps)
        elif (self.loss_type == "MAE+MSE"):
            loss = torch.mean(torch.abs(y_hat - x)) + \
                torch.sqrt(torch.mean((y_hat - x)**2))
        elif (self.loss_type == "MAE+MSE_log"):
            loss = torch.mean(torch.abs(y_hat - x)) + \
                torch.mean((torch.log10(y_hat.clip(0, None)+eps) -
                           torch.log10(x.clip(0, None)+eps))**2)
        else:
            raise Exception("Fill me")

        if torch.isnan(loss).any():
            raise Exception("NaNs")
        return loss, mse

    def training_step(self, batch, batch_idx):
        loss, mse = self.loss_fn(batch, batch_idx)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        self.log('train_mse', mse, on_step=False, on_epoch=True)

        self.loss = loss
        self.train_mse = mse
        return loss

    def validation_step(self, batch, batch_idx):
        loss, mse = self.loss_fn(batch, batch_idx)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_mse', mse, on_step=False, on_epoch=True)
        self.val_mse = mse

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def load_data(self, X, y, Xmax,  db_min_norm, spec_in_db):
        self.X = X
        self.y = y
        self.Xmax = Xmax
        self.db_min_norm = db_min_norm
        self.spec_in_db = spec_in_db

    def split_data(self, validation_size=0.05, shuffle=True):
        self.X_train, self.X_val, _, _ = train_test_split(
            self.X, self.y, test_size=validation_size, shuffle=shuffle)
        print('Train and val shapes', self.X_train.shape, self.X_val.shape)

    def log_hyperparameters(self, **kwargs):
        print(kwargs)

        for k, v in kwargs.items():
            self.hparams[k] = v
        self.hparams['parameters'] = sum(p.numel() for p in self.parameters())
        self.save_hyperparameters(ignore=['checkpoint_path'])

    def train_model(self, loss_type, learning_rate, epochs=120, batch_size=512, log_path="tb_logs",
                    run_name="mymodel", accelerator='cpu', use_early_stopping=True):

        self.run_name = run_name
        self.loss_type = loss_type

        self.learning_rate = learning_rate
        # train_dataloader
        silence = np.zeros(
            [int(self.X_train.shape[0]*0.01), self.X_train.shape[1]])
        X_data = torch.vstack([torch.tensor(silence), self.X_train]).float()

        dataset = TensorDataset(X_data, X_data)
        train_dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=2)
        # val_dataloader
        X_data = self.X_val.float()
        dataset = TensorDataset(X_data, X_data)
        val_dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=2)

        logger = TensorBoardLogger(log_path, name=run_name)
        metric_tracker = MetricTracker()

        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=1e-6, patience=100, verbose=True, mode="min")

        callbacks = [metric_tracker]
        if use_early_stopping:
            callbacks += [early_stop_callback]

        

        trainer = pl.Trainer(max_epochs=epochs, enable_model_summary=False, logger=logger,
                             callbacks=callbacks,
                             accelerator=accelerator)

        trainer.fit(self,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader)

        logger.log_metrics({"hp_metric": self.train_mse})
        # self.log("hp/loss", self.loss)
        # self.log("hp/mse", self.loss)

        return trainer, metric_tracker

    def load_checkpoint(self, path, device='cpu'):
        self.load_state_dict(torch.load(
            path, map_location=device)['state_dict'])
        self.checkpoint_path = path

    def predict(self, specgram):
        S_hat = self(specgram)
        return S_hat

    def export2onnx(self, path):
        self.eval()
        torch.onnx.export(self, self.X_val, path, verbose=False,
                          input_names="input", output_names="output")

    def add_PCA_layer(self):
        Z = self.encoder(self.X)
        pca = PCA()
        Zpca = pca.fit_transform(Z)
        # hay que agregar una capa linear al modelo que multiplique el input del decoder por pca.components_
        # self.decoder.layers.insert(0, nn.Linear(self.decoder_layers[0], self.decoder_layers[0]))

    def export_decoder(self, pca_latent_space=False):
        hps = self.hparams
        Z = self.encoder(self.X).detach().cpu().numpy()
        if pca_latent_space:
            pca = PCA()
            Zpca = pca.fit_transform(Z)
            rangeZ = np.ceil(np.abs(Zpca).max(0))
            decoder = PCADecoder(torch.tensor(pca.components_), self.decoder)
        else:
            rangeZ = np.ceil(np.abs(Z).max(0))
            decoder = self.decoder

        output_path = Path("exported_models",f"{self.run_name}.pt")
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
                    "model_path": str(Path("exported_models", f"{self.run_name}.pt").absolute()),
                    "ztrack": Z.tolist()
                }
        output_path = Path('exported_models',f"{self.run_name}.json")

        with open(output_path, 'w') as fp:
            json.dump(data, fp)
