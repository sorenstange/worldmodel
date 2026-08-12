import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lightning as L

from transformers import get_cosine_schedule_with_warmup

from modules import *
from util import *

class Actor(L.LightningModule):
    def __init__(self, cfg, jepa):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.jepa = jepa.eval()
        for param in self.jepa.parameters():
            param.requires_grad = False

        self.backbone = Predictor(
            input_dim = cfg['actor']['d_model'],
            d_model = cfg['actor']['d_model'],
            num_layers = cfg['actor']['backbone']['num_layers'],
            num_heads = cfg['actor']['backbone']['num_heads'],
            max_len = cfg['actor']['backbone']['max_len'],
            condition_dim = 2,
            dropout = cfg['actor']['backbone']['dropout']
        )

        self.actor_head = nn.Sequential(
            nn.Linear(cfg['actor']['d_model'], 2*cfg['actor']['d_model']),
            nn.LayerNorm(2*cfg['actor']['d_model']),       
            nn.SiLU(),
            nn.Dropout(cfg['actor']['action_head']['dropout']),     
            nn.Linear(2*cfg['actor']['d_model'], 2*cfg['actor']['d_model']),
            nn.LayerNorm(2*cfg['actor']['d_model']),       
            nn.SiLU(),
            nn.Dropout(2*cfg['actor']['action_head']['dropout']),    
            nn.Linear(2*cfg['actor']['d_model'], cfg['actor']['action_head']['num_bins'])
        )
        self.register_buffer('action_bins', 
                            torch.linspace(
                                cfg['actor']['action_head']['min_value'], 
                                cfg['actor']['action_head']['max_value'], 
                                cfg['actor']['action_head']['num_bins']
                                )
                            )

        self.MSELoss = nn.MSELoss()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

        self.lr     = cfg['actor']['training']['lr']
        self.epochs = cfg['actor']['training']['epochs']
        self.warmup = cfg['actor']['training']['warmup']

    def forward(self, Z, Cond):
        Z_hat = self.backbone(Z, Cond)
        return self.actor_head(Z_hat)

    def training_step(self, batch, batch_idx):
        X, Ret = batch['sample'], batch['return']
        Act, Act_target = batch['action'], batch['action_target']

        with torch.no_grad():
            Z = self.jepa.encode(X)

        Z_in = Z[:, :-1]
        Ret_in = Ret[:, :-1]
        Act_in = Act[:, :-1]

        cond = torch.cat((Ret_in, Act_in), dim=-1)
        act_logits = self(Z_in, cond)

        act_target = Act_target[:, 1:] 

        act_logits_flat = act_logits.reshape(-1, act_logits.size(-1))
        act_target_flat = act_target.reshape(-1).long()
        L = self.CrossEntropyLoss(act_logits_flat, act_target_flat) 

        self.log('train/loss', L)

        return L

    def validation_step(self, batch, batch_idx):
        X, Ret = batch['sample'], batch['return']
        Act, Act_target = batch['action'], batch['action_target']

        with torch.no_grad():
            Z = self.jepa.encode(X)

        Z_in = Z[:, :-1]
        Ret_in = Ret[:, :-1]
        Act_in = Act[:, :-1]

        cond = torch.cat((Ret_in, Act_in), dim=-1)
        act_logits = self(Z_in, cond)

        act_target = Act_target[:, 1:] 

        act_logits_flat = act_logits.reshape(-1, act_logits.size(-1))
        act_target_flat = act_target.reshape(-1).long()
        L_act = self.CrossEntropyLoss(act_logits_flat, act_target_flat) 

        self.log('val/loss', L_act, on_step=False, on_epoch=True, prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        total_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = self.warmup / self.epochs * total_steps
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps
        )
    
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
