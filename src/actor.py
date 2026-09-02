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
            condition_dim = cfg['jepa']['return_head']['num_bins'],
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

        with torch.no_grad():
            Zp1, Ret_logits = self.jepa.predict(Z_in, Ret_in)
            Ret_probs = torch.softmax(Ret_logits, dim=-1)

        act_logits = self(Zp1, Ret_probs)

        act_target = Act_target[:, 1:] 

        act_logits_flat = act_logits.reshape(-1, act_logits.size(-1))
        act_target_flat = act_target.reshape(-1).long()
        L = self.CrossEntropyLoss(act_logits_flat, act_target_flat) 

        self.log('train/actor_loss', L)

        return L

    def validation_step(self, batch, batch_idx):
        X, Ret = batch['sample'], batch['return']
        Act, Act_target = batch['action'], batch['action_target']

        with torch.no_grad():
            Z = self.jepa.encode(X)

        Z_in = Z[:, :-1]
        Ret_in = Ret[:, :-1]

        with torch.no_grad():
            Zp1, Ret_logits = self.jepa.predict(Z_in, Ret_in)
            Ret_probs = torch.softmax(Ret_logits, dim=-1)

        act_logits = self(Zp1, Ret_probs)

        act_target = Act_target[:, 1:] 

        act_logits_flat = act_logits.reshape(-1, act_logits.size(-1))
        act_target_flat = act_target.reshape(-1).long()
        L = self.CrossEntropyLoss(act_logits_flat, act_target_flat) 

        self.log('val/actor_loss', L, on_step=False, on_epoch=True, prog_bar=True)

        b_output = self.backtest(batch)
        self.log('val/mean_eq', torch.mean(b_output['end_equity']), on_step=False, on_epoch=True, prog_bar=True)
    
    def backtest(self, batch, act_temp = 1.0):
        X, Ret, Act, Act_target = batch['sample'], batch['return'], batch['action'], batch['action_target']

        with torch.no_grad():
            Z = self.jepa.encode(X)

        Z_in = Z[:, :-1]
        Ret_in = Ret[:, :-1]

        with torch.no_grad():
            Zp1, Ret_logits = self.jepa.predict(Z_in, Ret_in)
            Ret_probs = torch.softmax(Ret_logits, dim=-1)

        Act_out = Act[:, 1:]
        Act_out_idx = Act_target[:, 1:]
        Ret_out = Ret[:, 1:]

        act_logits = self(Zp1, Ret_probs)
        act_logits = act_logits / act_temp
        act_probs = torch.softmax(act_logits, dim=-1)
        
        B, Seq, d = act_probs.shape
        act_idx = torch.multinomial(act_probs.reshape(B*Seq, d), num_samples=1).reshape(B, Seq, 1)
        actions = self.action_bins[act_idx]

        act0 = torch.zeros((B, 1, 1), device=actions.device)
   
        E, e = equity(torch.cat([act0, actions], dim=1).squeeze(-1), Ret_out.squeeze(-1), c = 0.0005)
        t_E, t_e = equity(torch.cat([act0, Act_out], dim=1).squeeze(-1), Ret_out.squeeze(-1), c = 0.0005)
        
        return {
            'action' : actions,
            'action_prob' : act_probs,
            'action_idx' : act_idx,
            'return' : Ret_out,
            'return_prob' : Ret_probs,
            'equity' : E,
            'end_equity' : e,
            'opt_action' : Act_out,
            'opt_action_idx' : Act_out_idx,  
            'opt_equity' : t_E,
            'opt_end_equity' : t_e
        }


        
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

CTX_LEN = 1
PRED_STEPS = 63

class ActorAR(Actor):
    def training_step(self, batch, batch_idx):
        act_logits = self.step(batch, CTX_LEN, PRED_STEPS)
        Act_target = batch['action_target']

        act_target = Act_target[:, CTX_LEN:CTX_LEN+PRED_STEPS] 

        act_logits_flat = act_logits.reshape(-1, act_logits.size(-1))
        act_target_flat = act_target.reshape(-1).long()
        L = self.CrossEntropyLoss(act_logits_flat, act_target_flat) 

        self.log('train/actor_loss', L)

        return L

    def validation_step(self, batch, batch_idx):
        act_logits = self.step(batch, CTX_LEN, PRED_STEPS)
        Act_target = batch['action_target']

        act_target = Act_target[:, CTX_LEN:CTX_LEN+PRED_STEPS] 

        act_logits_flat = act_logits.reshape(-1, act_logits.size(-1))
        act_target_flat = act_target.reshape(-1).long()
        L = self.CrossEntropyLoss(act_logits_flat, act_target_flat) 

        self.log('val/actor_loss', L, on_step=False, on_epoch=True, prog_bar=True)

        b_output = self.backtest(batch, ctx_len = CTX_LEN)
        self.log('val/mean_eq', torch.mean(b_output['end_equity']), on_step=False, on_epoch=True, prog_bar=True)

    def step(self, batch, ctx_len, pred_steps):
        action_logits = []
        X, Ret = batch['sample'], batch['return']

        with torch.no_grad():
            Z = self.jepa.encode(X)

        B, Seq, D = Z.shape

        Z_p = Z[:, :ctx_len, :]
        Ret_p = Ret[:, :ctx_len, :]
        Act_p = torch.zeros_like(Ret_p)

        Z_p = truncate(Z_p, self.backbone.max_len)
        Ret_p = truncate(Ret_p, self.backbone.max_len)
        Act_p = truncate(Act_p, self.backbone.max_len)

        for t in range(pred_steps):
            with torch.no_grad():
                Zp1, Ret_logits = self.jepa.predict(Z_p, Ret_p)
                Ret_probs = torch.softmax(Ret_logits, dim=-1)

            cond = torch.cat((Ret_probs, Act_p), dim=-1)    
            act_logits = self(Z_p, cond)[:, -1:, :]

            act_idx = torch.argmax(act_logits, dim=-1)
            new_act = self.action_bins[act_idx].unsqueeze(-1)

            new_Z = Z[:, ctx_len+t, :].unsqueeze(1)
            new_Ret = Ret[:, ctx_len+t, :].unsqueeze(1)
            
            Z_p = torch.cat([Z_p, new_Z.detach()], dim=1)
            Ret_p = torch.cat([Ret_p, new_Ret.detach()], dim=1)
            Act_p = torch.cat([Act_p, new_act.detach()], dim=1)

            Z_p = truncate(Z_p, self.backbone.max_len)
            Ret_p = truncate(Ret_p, self.backbone.max_len)
            Act_p = truncate(Act_p, self.backbone.max_len)

            action_logits.append(act_logits)

        action_logits = torch.concatenate(action_logits, dim=1)
        
        return action_logits