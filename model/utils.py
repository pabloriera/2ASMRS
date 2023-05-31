# from kiwisolver import Variable
from pytorch_lightning.callbacks import Callback
from copy import copy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

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


