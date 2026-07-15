import torch
import torch.nn as nn
import lightning as L

from modules import *

class JEPA(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        pass

    def forward(self, X):
        pass

    def training_step(self, batch, batch_idx):
        pass
    
    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self, cfg):
        pass