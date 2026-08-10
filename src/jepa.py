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
            nn.Linear(2*cfg['jepa']['d_model'], cfg['jepa']['return_head']['num_bins'])
        )
        self.register_buffer('return_bins', 
                            torch.linspace(
                                cfg['jepa']['return_head']['min_value'], 
                                cfg['jepa']['return_head']['max_value'], 
                                cfg['jepa']['return_head']['num_bins']
                                )
                            )

        self.action_head = nn.Sequential(
            nn.Linear(cfg['jepa']['d_model'], 2*cfg['jepa']['d_model']),
            nn.LayerNorm(2*cfg['jepa']['d_model']),       
            nn.SiLU(),
            nn.Dropout(cfg['jepa']['action_head']['dropout']),         
            nn.Linear(2*cfg['jepa']['d_model'], cfg['jepa']['action_head']['num_bins'])
        )
        self.register_buffer('action_bins', 
                            torch.linspace(
                                cfg['jepa']['action_head']['min_value'], 
                                cfg['jepa']['action_head']['max_value'], 
                                cfg['jepa']['action_head']['num_bins']
                                )
                            )

        self.MSELoss = nn.MSELoss()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

        self.lr = cfg['jepa']['training']['lr']
        self.epochs = cfg['jepa']['training']['epochs']
        self.warmup = cfg['jepa']['training']['warmup']

        self.AR_lr = cfg['jepa']['ar_training']['lr']
        self.AR_epochs = cfg['jepa']['ar_training']['epochs']
        self.AR_warmup = cfg['jepa']['ar_training']['warmup']
        self.AR_horizon = cfg['jepa']['ar_training']['horizon']
        self.AR_action_temp = cfg['jepa']['ar_training']['action_temperature']
        self.AR_com_val = cfg['jepa']['ar_training']['commission_value']
        

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

    def predict(self, Z, Ret, Act):
        Zp1 = self.predictor(Z, Ret, Act)
        ret_logits = self.return_head(Zp1)
        act_logits = self.action_head(Zp1)
        return Zp1, ret_logits, act_logits

    def forward(self, X, Ret, Act):
        Z = self.encode(X)
        Zp1, ret_logits, act_logits = self.predict(Z, Ret, Act)
        return Z, Zp1, ret_logits, act_logits

    def training_step(self, batch, batch_idx):
        X = batch['sample']
        Ret, Ret_target = batch['return'], batch['return_target']
        Act, Act_target = batch['action'], batch['action_target']

        Z = self.encode(X)

        Z_in = Z[:, :-1]
        Ret_in = Ret[:, :-1]
        Act_in = Act[:, :-1]

        Z_hat, ret_logits, act_logits = self.predict(Z_in, Ret_in, Act_in)

        Z_target = Z[:, 1:]
        act_target = Act_target[:, 1:]
        ret_target = Ret_target[:, 1:] 

        L_state = self.MSELoss(Z_hat, Z_target)

        ret_logits_flat = ret_logits.reshape(-1, ret_logits.size(-1))
        ret_target_flat = ret_target.reshape(-1).long()
        L_ret = self.CrossEntropyLoss(ret_logits_flat, ret_target_flat) 

        act_logits_flat = act_logits.reshape(-1, act_logits.size(-1))
        act_target_flat = act_target.reshape(-1).long()
        L_act = self.CrossEntropyLoss(act_logits_flat, act_target_flat) 

        L = L_state + L_ret + L_act

        self.log('train/state_loss', L_state)
        self.log('train/return_loss', L_ret)
        self.log('train/action_loss', L_act)
        self.log('train/loss', L)

        return L

    def validation_step(self, batch, batch_idx):
        X = batch['sample']
        Ret, Ret_target = batch['return'], batch['return_target']
        Act, Act_target = batch['action'], batch['action_target']

        Z = self.encode(X)
        Z_in = Z[:, :-1]
        
        Ret_in = Ret[:, :-1]
        Act_in = Act[:, :-1]

        Z_hat, ret_logits, act_logits = self.predict(Z_in, Ret_in, Act_in)

        Z_target = Z[:, 1:]
        act_target = Act_target[:, 1:]
        ret_target = Ret_target[:, 1:] 

        L_state = self.MSELoss(Z_hat, Z_target)

        ret_logits_flat = ret_logits.reshape(-1, ret_logits.size(-1))
        ret_target_flat = ret_target.reshape(-1).long()
        L_ret = self.CrossEntropyLoss(ret_logits_flat, ret_target_flat) 

        act_logits_flat = act_logits.reshape(-1, act_logits.size(-1))
        act_target_flat = act_target.reshape(-1).long()
        L_act = self.CrossEntropyLoss(act_logits_flat, act_target_flat) 

        L = L_state + L_ret + L_act

        self.log('val/state_loss', L_state, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/return_loss', L_ret, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/action_loss', L_act, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/loss', L, on_step=False, on_epoch=True, prog_bar=True)

        h = 32
        _, _, acts, _, _ = self.backtest(Z, Ret, horizon = h)

        act0 = torch.full((Z.size(0), 1, 1), 0.0, dtype=acts.dtype, device=acts.device)
        b_a = torch.cat((act0, acts), dim=1).squeeze(-1)
        b_r = Ret[:, -h:, :].squeeze(-1)
        _, E = equity(b_a, b_r, c=0.0005)
        E = torch.mean(E)
        self.log('val/mean_equity', E, on_step=False, on_epoch=True, prog_bar=True)

    def dream(self, Z_prompt, Ret_prompt, Act_prompt, 
                horizon = 15, 
                ret_temperature = 1.0,
                act_temperature = 1.0):

        dream_states = []
        dream_ret, dream_ret_probs, dream_ret_sampled_idx = [], [], []
        dream_act, dream_act_probs, dream_act_sampled_idx = [], [], []

        Z_prompt = truncate(Z_prompt, self.predictor.max_len)
        Ret_prompt = truncate(Ret_prompt, self.predictor.max_len)
        Act_prompt = truncate(Act_prompt, self.predictor.max_len)

        for t in range(horizon):
            Z_next, ret_logits, act_logits = self.predict(Z_prompt, Ret_prompt, Act_prompt)

            new_Z = Z_next[:, -1:, :]
            ret_logits = ret_logits[:, -1:, :].squeeze(1)
            act_logits = act_logits[:, -1:, :].squeeze(1)

            ret_probs = torch.softmax(ret_logits, dim=-1) / ret_temperature
            ret_sampled_idx = torch.multinomial(ret_probs, num_samples=1)
            new_ret = self.return_bins[ret_sampled_idx].unsqueeze(-1)

            act_logits = act_logits / act_temperature
            act_probs = torch.softmax(act_logits, dim=-1)
            act_sampled_idx = torch.multinomial(act_probs, num_samples=1)
            new_act = self.action_bins[act_sampled_idx].unsqueeze(-1)

            Z_prompt = torch.cat([Z_prompt, new_Z.detach()], dim=1)
            Ret_prompt = torch.cat([Ret_prompt, new_ret.detach()], dim=1)
            Act_prompt = torch.cat([Act_prompt, new_act.detach()], dim=1)

            Z_prompt = truncate(Z_prompt, self.predictor.max_len)
            Ret_prompt = truncate(Ret_prompt, self.predictor.max_len)
            Act_prompt = truncate(Act_prompt, self.predictor.max_len)

            dream_states.append(new_Z)
            dream_ret.append(new_ret.detach())
            dream_ret_probs.append(ret_probs.unsqueeze(1))
            dream_ret_sampled_idx.append(ret_sampled_idx.detach().unsqueeze(1))
            dream_act.append(new_act.detach())
            dream_act_probs.append(act_probs.unsqueeze(1))
            dream_act_sampled_idx.append(act_sampled_idx.detach().unsqueeze(1))

        return (
            torch.concatenate(dream_states, dim=1),
            torch.concatenate(dream_ret, dim=1),
            torch.concatenate(dream_ret_probs, dim=1),
            torch.concatenate(dream_ret_sampled_idx, dim=1),
            torch.concatenate(dream_act, dim=1),
            torch.concatenate(dream_act_probs, dim=1),
            torch.concatenate(dream_act_sampled_idx, dim=1)
        )

    def backtest(self, Z, Ret, Act_prompt = None, horizon = 15, act_temperature = 1.0):
        out_Z, out_ret_probs, out_act, out_act_probs, out_act_sampled_idx = [], [], [], [], []

        B, Seq, D = Z.shape
        start_idx = Seq - horizon - 1

        Z_prompt = Z[:, :start_idx+1, :]
        Ret_prompt = Ret[:, :start_idx+1, :]
        if (Act_prompt is None or Act_prompt.size(1) != Ret_prompt.size(1)):
            print('Zero action')
            Act_prompt = torch.full((B, start_idx+1, 1), 0.0, dtype=Z_prompt.dtype, device=Z_prompt.device)

        Z_prompt = truncate(Z_prompt, self.predictor.max_len)
        Ret_prompt = truncate(Ret_prompt, self.predictor.max_len)
        Act_prompt = truncate(Act_prompt, self.predictor.max_len)

        for t in range(horizon):
            Z_next, ret_logits, act_logits = self.predict(Z_prompt, Ret_prompt, Act_prompt)

            new_Z = Z[:, start_idx+t+1, :].unsqueeze(1)
            ret_logits = ret_logits[:, -1:, :].squeeze(1)
            act_logits = act_logits[:, -1:, :].squeeze(1)

            ret_probs = torch.softmax(ret_logits, dim=-1)
            new_ret = Ret[:, start_idx+t+1, :].unsqueeze(-1)

            act_logits = act_logits / act_temperature
            act_probs = torch.softmax(act_logits, dim=-1)
            act_sampled_idx = torch.multinomial(act_probs, num_samples=1)
            new_act = self.action_bins[act_sampled_idx].unsqueeze(-1)

            Z_prompt = torch.cat([Z_prompt, new_Z.detach()], dim=1)
            Ret_prompt = torch.cat([Ret_prompt, new_ret.detach()], dim=1)
            Act_prompt = torch.cat([Act_prompt, new_act.detach()], dim=1)

            Z_prompt = truncate(Z_prompt, self.predictor.max_len)
            Ret_prompt = truncate(Ret_prompt, self.predictor.max_len)
            Act_prompt = truncate(Act_prompt, self.predictor.max_len)

            new_Z = Z_next[:, -1:, :]
            out_Z.append(new_Z)
            out_ret_probs.append(ret_probs.unsqueeze(1))
            out_act.append(new_act.detach())
            out_act_probs.append(act_probs.unsqueeze(1))
            out_act_sampled_idx.append(act_sampled_idx.detach().unsqueeze(1))

        return (
            torch.concatenate(out_Z, dim=1),
            torch.concatenate(out_ret_probs, dim=1),
            torch.concatenate(out_act, dim=1),
            torch.concatenate(out_act_probs, dim=1),
            torch.concatenate(out_act_sampled_idx, dim=1)
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

class JEPA_AR(JEPA):
    def training_step(self, batch, batch_idx):
        X = batch['sample']
        Ret, Ret_target = batch['return'], batch['return_target']
        Act, Act_target = batch['action'], batch['action_target']
        
        Z = self.encode(X)

        B, Seq, D = Z.shape
        horizon = Seq - 1

        Z_hat, ret_probs, _, act_probs, _ = self.backtest(
            Z, Ret, horizon = horizon, 
            act_temperature = self.AR_action_temp, 
            commission_value = self.AR_com_val)

        Z_target = Z[:, -horizon:, :]
        ret_target = Ret_target[:, -horizon:, :]
        act_target = Act_target[:, -horizon:, :]

        L_state = self.MSELoss(Z_hat, Z_target)

        ret_probs_flat = ret_probs.reshape(-1, ret_probs.size(-1))
        ret_target_flat = ret_target.reshape(-1).long()
        L_ret = self.CrossEntropyLoss(ret_probs_flat, ret_target_flat) 

        act_probs_flat = act_probs.reshape(-1, act_probs.size(-1))
        act_target_flat = act_target.reshape(-1).long()
        L_act = self.CrossEntropyLoss(act_probs_flat, act_target_flat) 

        L = L_state + L_ret + L_act

        self.log('train/state_loss', L_state)
        self.log('train/return_loss', L_ret)
        self.log('train/action_loss', L_act)
        self.log('train/loss', L)

        return L
    
    def validation_step(self, batch, batch_idx):
        X = batch['sample']
        Ret, Ret_target = batch['return'], batch['return_target']
        Act, Act_target = batch['action'], batch['action_target']

        Z = self.encode(X)

        B, Seq, D = Z.shape
        horizon = Seq - 1

        Z_hat, ret_probs, _, act_probs, _ = self.backtest(
            Z, Ret, horizon = horizon, 
            act_temperature = self.AR_action_temp, 
            commission_value = self.AR_com_val)

        Z_target = Z[:, -horizon:, :]
        ret_target = Ret_target[:, -horizon:, :]
        act_target = Act_target[:, -horizon:, :]

        L_state = self.MSELoss(Z_hat, Z_target)

        ret_probs_flat = ret_probs.reshape(-1, ret_probs.size(-1))
        ret_target_flat = ret_target.reshape(-1).long()
        L_ret = self.CrossEntropyLoss(ret_probs_flat, ret_target_flat) 

        act_probs_flat = act_probs.reshape(-1, act_probs.size(-1))
        act_target_flat = act_target.reshape(-1).long()
        L_act = self.CrossEntropyLoss(act_probs_flat, act_target_flat) 

        L = L_state + L_ret + L_act

        self.log('val/state_loss', L_state, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/return_loss', L_ret, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/action_loss', L_act, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/loss', L, on_step=False, on_epoch=True, prog_bar=True)

        B, seq, D = Z.shape
        b_a = acts.detach()
        act0 = torch.full((B, 1, 1), 0.0, dtype=b_a.dtype, device=b_a.device)
        b_a = torch.cat((act0, b_a), dim=1).squeeze(-1)
        b_r = Ret[:, -horizon:, :].squeeze(-1)
        _, E = equity(b_a, b_r, c=self.AR_com_val)
        E = torch.mean(E)
        self.log('val/mean_equity', E, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.AR_lr)
        total_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = self.AR_warmup / self.AR_epochs * total_steps
        
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

class JEPA_RL(JEPA):
    def training_step(self, batch, batch_idx):
        X = batch['sample']
        Ret, Ret_target = batch['return'], batch['return_target']
        Act, Act_target = batch['action'], batch['action_target']

        Z = self.encode(X)
        B, Seq, D = Z.shape
        horizon = Seq - 1

        Z_hat, ret_probs, acts, act_probs, act_idx = self.backtest(
            Z, Ret, horizon = horizon, 
            act_temperature = 1.0)

        Z_target = Z[:, -horizon:, :]
        ret_target = Ret_target[:, -horizon:, :]

        L_state = self.MSELoss(Z_hat, Z_target)

        ret_probs_flat = ret_probs.reshape(-1, ret_probs.size(-1))
        ret_target_flat = ret_target.reshape(-1).long()
        L_ret = self.CrossEntropyLoss(ret_probs_flat, ret_target_flat) 

        act0 = torch.full((B, 1, 1), 0.0, dtype=acts.dtype, device=acts.device)
        b_a = torch.cat((act0, acts), dim=1).squeeze(-1)
        b_r = Ret[:, -horizon:, :].squeeze(-1)
        print('b_a.shape', b_a.shape)
        print('b_r.shape', b_r.shape)
        dE = delta_equity(b_a.squeeze(-1), b_r, c = 0.0005)
        rewards = torch.log1p(dE)

        sharpe = torch.sum(rewards, dim=-1) / (torch.std(rewards, dim=-1) + 1e-6)
        rewards_with_bonus = rewards.clone()
        rewards_with_bonus[:, -1] = rewards_with_bonus[:, -1] + sharpe

        gamma = 0.99 # Diskonteringsfaktor
        G = torch.zeros_like(rewards_with_bonus)
        running_g = torch.zeros(B, device=rewards.device)
        
        for t in reversed(range(horizon)):
            running_g = rewards_with_bonus[:, t] + gamma * running_g
            G[:, t] = running_g

        if B > 1:
            G = (G - G.mean(dim=0, keepdim=True)) / (G.std(dim=0, keepdim=True) + 1e-8)

        chosen_act_probs = act_probs.gather(dim=-1, index=act_idx).squeeze(-1) # Form: [B, horizon]
        log_probs = torch.log(chosen_act_probs + 1e-8)

        L_policy = -100 * torch.mean(log_probs * G)

        L = L_state + L_ret + L_policy

        self.log('train/rl_state_loss', L_state)
        self.log('train/rl_return_loss', L_ret)
        self.log('train/rl_policy_loss', L_policy)
        self.log('train/rl_loss', L)

        return L
    
    def validation_step(self, batch, batch_idx):
        X = batch['sample']
        Ret, Ret_target = batch['return'], batch['return_target']
        Act, Act_target = batch['action'], batch['action_target']

        Z = self.encode(X)
        B, Seq, D = Z.shape
        horizon = Seq - 1

        Z_hat, ret_probs, acts, act_probs, act_idx = self.backtest(
            Z, Ret, horizon = horizon, 
            act_temperature = 1.0)

        Z_target = Z[:, -horizon:, :]
        ret_target = Ret_target[:, -horizon:, :]

        L_state = self.MSELoss(Z_hat, Z_target)

        ret_probs_flat = ret_probs.reshape(-1, ret_probs.size(-1))
        ret_target_flat = ret_target.reshape(-1).long()
        L_ret = self.CrossEntropyLoss(ret_probs_flat, ret_target_flat) 

        act0 = torch.full((B, 1, 1), 0.0, dtype=acts.dtype, device=acts.device)
        b_a = torch.cat((act0, acts), dim=1).squeeze(-1)
        b_r = Ret[:, -horizon:, :].squeeze(-1)
        print('b_a.shape', b_a.shape)
        print('b_r.shape', b_r.shape)
        dE = delta_equity(b_a.squeeze(-1), b_r, c = 0.0005)
        rewards = torch.log1p(dE)

        sharpe = torch.sum(rewards, dim=-1) / (torch.std(rewards, dim=-1) + 1e-6)
        rewards_with_bonus = rewards.clone()
        rewards_with_bonus[:, -1] = rewards_with_bonus[:, -1] + sharpe

        gamma = 0.99 # Diskonteringsfaktor
        G = torch.zeros_like(rewards_with_bonus)
        running_g = torch.zeros(B, device=rewards.device)
        
        for t in reversed(range(horizon)):
            running_g = rewards_with_bonus[:, t] + gamma * running_g
            G[:, t] = running_g

        if B > 1:
            G = (G - G.mean(dim=0, keepdim=True)) / (G.std(dim=0, keepdim=True) + 1e-8)

        chosen_act_probs = act_probs.gather(dim=-1, index=act_idx).squeeze(-1) # Form: [B, horizon]
        log_probs = torch.log(chosen_act_probs + 1e-8)

        L_policy = -torch.mean(log_probs * G)

        L = L_state + L_ret + L_policy

        self.log('val/rl_state_loss', L_state, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/rl_return_loss', L_ret, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/rl_policy_loss', L_policy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/rl_loss', L, on_step=False, on_epoch=True, prog_bar=True)

        _, E = equity(b_a, b_r, c=0.0005)
        E = torch.mean(E)
        self.log('val/mean_equity', E, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.0003)
        total_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = 100 / 1000 * total_steps
        
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

if __name__ == '__main__':
    model = JEPA_AR.load_from_checkpoint('./models/jepa/jepa-actor/last.ckpt')
    print(model)


