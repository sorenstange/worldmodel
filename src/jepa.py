import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lightning as L

from transformers import get_cosine_schedule_with_warmup

from modules import *
from util import *

class JEPA(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.encoder = Encoder(
            input_dim = cfg['jepa']['input_dim'],
            d_model = cfg['jepa']['d_model'],
            num_layers = cfg['jepa']['encoder']['num_layers'],
            num_heads = cfg['jepa']['encoder']['num_heads'],
            max_len = cfg['jepa']['encoder']['max_len'],
            dropout = cfg['jepa']['encoder']['dropout']
        )

        self.predictor = Predictor(
            input_dim = cfg['jepa']['d_model'],
            d_model = cfg['jepa']['d_model'],
            num_layers = cfg['jepa']['predictor']['num_layers'],
            num_heads = cfg['jepa']['predictor']['num_heads'],
            max_len = cfg['jepa']['predictor']['max_len'],
            dropout = cfg['jepa']['predictor']['dropout']
        )

        self.return_head = nn.Sequential(
            nn.Linear(cfg['jepa']['d_model'], 2*cfg['jepa']['d_model']),
            nn.LayerNorm(2*cfg['jepa']['d_model']),       
            nn.SiLU(),
            nn.Dropout(cfg['jepa']['return_head']['dropout']),     
            nn.Linear(2*cfg['jepa']['d_model'], 2*cfg['jepa']['d_model']),
            nn.LayerNorm(2*cfg['jepa']['d_model']),       
            nn.SiLU(),
            nn.Dropout(2*cfg['jepa']['return_head']['dropout']),    
            nn.Linear(2*cfg['jepa']['d_model'], cfg['jepa']['return_head']['num_bins'])
        )
        self.register_buffer('return_bins', 
                            torch.linspace(
                                cfg['jepa']['return_head']['min_value'], 
                                cfg['jepa']['return_head']['max_value'], 
                                cfg['jepa']['return_head']['num_bins']
                                )
                            )

        self.MSELoss = nn.MSELoss()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

        self.lr     = cfg['jepa']['training']['lr']
        self.epochs = cfg['jepa']['training']['epochs']
        self.warmup = cfg['jepa']['training']['warmup']

    def encode(self, X):
        if X.dim() == 4:
            B, Seq, Win, D = X.shape
        elif X.dim() == 3:
            B = 1
            Seq, Win, D = X.shape
        
        X = X.view(B * Seq, Win, -1)
        Z = self.encoder(X)
        Z = Z.view(B, Seq, -1)
        return Z

    def predict(self, Z, Ret):
        Zp1 = self.predictor(Z, Ret)
        ret_logits = self.return_head(Zp1)
        return Zp1, ret_logits

    def forward(self, X, Ret):
        Z = self.encode(X)
        Zp1, ret_logits = self.predict(Z, Ret)
        return Z, Zp1, ret_logits

    def training_step(self, batch, batch_idx):
        X = batch['sample']
        Ret, Ret_target = batch['return'], batch['return_target']

        Z = self.encode(X)

        Z_in = Z[:, :-1]
        Ret_in = Ret[:, :-1]

        Z_hat, ret_logits = self.predict(Z_in, Ret_in)

        Z_target = Z[:, 1:]
        ret_target = Ret_target[:, 1:] 

        L_state = self.MSELoss(Z_hat, Z_target)

        ret_logits_flat = ret_logits.reshape(-1, ret_logits.size(-1))
        ret_target_flat = ret_target.reshape(-1).long()
        L_ret = self.CrossEntropyLoss(ret_logits_flat, ret_target_flat) 

        L = L_state + L_ret

        self.log('train/state_loss', L_state)
        self.log('train/return_loss', L_ret)
        self.log('train/loss', L)

        return L

    def validation_step(self, batch, batch_idx):
        X = batch['sample']
        Ret, Ret_target = batch['return'], batch['return_target']

        Z = self.encode(X)

        Z_in = Z[:, :-1]
        Ret_in = Ret[:, :-1]

        Z_hat, ret_logits = self.predict(Z_in, Ret_in)

        Z_target = Z[:, 1:]
        ret_target = Ret_target[:, 1:] 

        L_state = self.MSELoss(Z_hat, Z_target)

        ret_logits_flat = ret_logits.reshape(-1, ret_logits.size(-1))
        ret_target_flat = ret_target.reshape(-1).long()
        L_ret = self.CrossEntropyLoss(ret_logits_flat, ret_target_flat) 

        L = L_state + L_ret

        self.log('val/state_loss', L_state, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/return_loss', L_ret, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/loss', L, on_step=False, on_epoch=True, prog_bar=True)

    def dream(self, Z_prompt, Ret_prompt, horizon = 15, ret_temperature = 1.0):
        dream_states = []
        dream_ret, dream_ret_probs, dream_ret_sampled_idx = [], [], []

        Z_prompt = truncate(Z_prompt, self.predictor.max_len)
        Ret_prompt = truncate(Ret_prompt, self.predictor.max_len)

        for t in range(horizon):
            Z_next, ret_logits = self.predict(Z_prompt, Ret_prompt)

            new_Z = Z_next[:, -1:, :]
            ret_logits = ret_logits[:, -1:, :].squeeze(1)

            ret_logits = ret_logits / ret_temperature
            ret_probs = torch.softmax(ret_logits, dim=-1)
            ret_sampled_idx = torch.multinomial(ret_probs, num_samples=1)
            new_ret = self.return_bins[ret_sampled_idx].unsqueeze(-1)

            Z_prompt = torch.cat([Z_prompt, new_Z.detach()], dim=1)
            Ret_prompt = torch.cat([Ret_prompt, new_ret.detach()], dim=1)

            Z_prompt = truncate(Z_prompt, self.predictor.max_len)
            Ret_prompt = truncate(Ret_prompt, self.predictor.max_len)

            dream_states.append(new_Z)
            dream_ret.append(new_ret.detach())
            dream_ret_probs.append(ret_probs.unsqueeze(1))
            dream_ret_sampled_idx.append(ret_sampled_idx.detach().unsqueeze(1))

        return (
            torch.concatenate(dream_states, dim=1),
            torch.concatenate(dream_ret, dim=1),
            torch.concatenate(dream_ret_probs, dim=1),
            torch.concatenate(dream_ret_sampled_idx, dim=1)
        )
        
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
