import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L

from transformers import get_cosine_schedule_with_warmup

from modules import *
from util import *


class Actor(L.LightningModule):
    """Allocation policy on top of the frozen world model.

    The AdaLN condition is [predicted return distribution, previous allocation].
    The previous allocation matters: commission is charged on |dx|, so a policy
    that cannot see its current position is being asked to make a cost-aware
    decision blind. This is also what makes the autoregressive variant below a
    drop-in rather than a different model.
    """

    def __init__(self, cfg, jepa):
        super().__init__()
        # `jepa` is a live module, not a hyperparameter -- it is reconstructed
        # and passed in explicitly at load time. See the note in JEPA.__init__.
        self.save_hyperparameters(ignore=['jepa'])
        self.jepa = jepa.eval()
        for param in self.jepa.parameters():
            param.requires_grad = False

        acfg = cfg['actor']
        d_model = acfg['d_model']
        head_drop = acfg['action_head']['dropout']

        # 61 return-distribution bins + 1 previous allocation.
        self.condition_dim = cfg['jepa']['return_head']['num_bins'] + 1

        self.backbone = Predictor(
            input_dim=cfg['jepa']['d_model'],
            d_model=d_model,
            num_layers=acfg['backbone']['num_layers'],
            num_heads=acfg['backbone']['num_heads'],
            max_len=acfg['backbone']['max_len'],
            condition_dim=self.condition_dim,
            dropout=acfg['backbone']['dropout'],
            ff_mult=acfg['backbone'].get('ff_mult', 4),
        )

        self.actor_head = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.LayerNorm(2 * d_model),
            nn.SiLU(),
            nn.Dropout(head_drop),
            nn.Linear(2 * d_model, 2 * d_model),
            nn.LayerNorm(2 * d_model),
            nn.SiLU(),
            nn.Dropout(head_drop),
            nn.Linear(2 * d_model, acfg['action_head']['num_bins']),
        )

        self.register_buffer(
            'action_bins',
            bin_centers(
                acfg['action_head']['min_value'],
                acfg['action_head']['max_value'],
                acfg['action_head']['num_bins'],
            ),
        )

        self.CrossEntropyLoss = nn.CrossEntropyLoss()

        self.commission = cfg['data']['actions']['commission_value']
        self.lr = acfg['training']['lr']
        self.weight_decay = acfg['training']['weight_decay']
        self.warmup_steps = acfg['training']['warmup_steps']
        self.sched_steps = acfg['training'].get('sched_steps', None)

    def forward(self, Z, Cond):
        return self.actor_head(self.backbone(Z, Cond))

    def world_state(self, batch):
        """Frozen-world-model features. Everything downstream reads these."""
        X, Ret = batch['sample'], batch['return']
        with torch.no_grad():
            Z = self.jepa.encode(X)
            Zp1, Ret_logits = self.jepa.predict(Z[:, :-1], Ret[:, :-1])
            Ret_probs = torch.softmax(Ret_logits, dim=-1)
        return Z, Zp1, Ret_probs

    def _teacher_forced_logits(self, batch):
        _, Zp1, Ret_probs = self.world_state(batch)
        # Position t of Zp1 predicts window t+1, so the allocation held going
        # into it is the oracle's allocation at t.
        prev_action = batch['action'][:, :-1]
        cond = torch.cat((Ret_probs, prev_action), dim=-1)
        return self(Zp1, cond)

    def _shared_step(self, batch, stage):
        act_logits = self._teacher_forced_logits(batch)
        act_target = batch['action_target'][:, 1:]

        L = self.CrossEntropyLoss(
            act_logits.reshape(-1, act_logits.size(-1)),
            act_target.reshape(-1).long(),
        )

        on_step = stage == 'train'
        self.log(f'{stage}/actor_loss', L, on_step=on_step, on_epoch=not on_step,
                 prog_bar=(stage == 'val'))
        return L

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, 'val')

        b = self.backtest(batch)
        self.log_dict({
            'val/mean_eq': torch.mean(b['end_equity']),
            'val/opt_eq': torch.mean(b['opt_end_equity']),
            'val/bh_eq': torch.mean(b['bh_end_equity']),
        }, on_step=False, on_epoch=True, prog_bar=True)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _equity_bundle(self, actions, Act_out, Ret_raw):
        """Equity for the policy, the oracle and buy-and-hold on one return path.

        Note this uses the RAW returns, not the vol-normalised ones the model
        conditions on -- equity has to be in real units.
        """
        B = actions.size(0)
        act0 = torch.zeros((B, 1), device=actions.device, dtype=actions.dtype)
        p = Ret_raw.squeeze(-1)

        E, e = equity(torch.cat([act0, actions.squeeze(-1)], dim=1), p, c=self.commission)
        t_E, t_e = equity(torch.cat([act0, Act_out.squeeze(-1)], dim=1), p, c=self.commission)

        # Buy-and-hold: fully long from the first step, so it pays commission once.
        bh = torch.ones_like(actions.squeeze(-1))
        bh_E, bh_e = equity(torch.cat([act0, bh], dim=1), p, c=self.commission)

        return (E, e), (t_E, t_e), (bh_E, bh_e)

    def decode_actions(self, act_probs, decode='expected'):
        """Turn a distribution over allocation bins into an allocation.

        'expected' is the default and is used for validation: allocation is
        continuous, so the probability-weighted mean is both a sensible decoder
        and a deterministic one. Sampling would put multinomial noise directly
        on val/mean_eq, which is the early-stopping and checkpoint monitor.
        """
        if decode == 'expected':
            actions = (act_probs * self.action_bins).sum(dim=-1, keepdim=True)
            idx = act_probs.argmax(dim=-1, keepdim=True)
            return actions, idx
        if decode == 'argmax':
            idx = act_probs.argmax(dim=-1, keepdim=True)
            return self.action_bins[idx.squeeze(-1)].unsqueeze(-1), idx
        if decode == 'sample':
            B, Seq, d = act_probs.shape
            idx = torch.multinomial(act_probs.reshape(B * Seq, d), num_samples=1).reshape(B, Seq, 1)
            return self.action_bins[idx.squeeze(-1)].unsqueeze(-1), idx
        raise ValueError(f'unknown decode mode: {decode}')

    def backtest(self, batch, act_temp=1.0, decode='expected'):
        _, Zp1, Ret_probs = self.world_state(batch)

        Act_out = batch['action'][:, 1:]
        Act_out_idx = batch['action_target'][:, 1:]
        Ret_raw = batch['return_raw'][:, 1:]

        cond = torch.cat((Ret_probs, batch['action'][:, :-1]), dim=-1)
        act_logits = self(Zp1, cond) / act_temp
        act_probs = torch.softmax(act_logits, dim=-1)

        actions, act_idx = self.decode_actions(act_probs, decode)

        (E, e), (t_E, t_e), (bh_E, bh_e) = self._equity_bundle(actions, Act_out, Ret_raw)

        return {
            'action': actions,
            'action_prob': act_probs,
            'action_idx': act_idx,
            'return': batch['return'][:, 1:],
            'return_raw': Ret_raw,
            'return_prob': Ret_probs,
            'equity': E,
            'end_equity': e,
            'opt_action': Act_out,
            'opt_action_idx': Act_out_idx,
            'opt_equity': t_E,
            'opt_end_equity': t_e,
            'bh_equity': bh_E,
            'bh_end_equity': bh_e,
        }

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            (p for p in self.parameters() if p.requires_grad),
            lr=self.lr, weight_decay=self.weight_decay,
        )
        total_steps = self.sched_steps or self.trainer.estimated_stepping_batches

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=min(self.warmup_steps, max(total_steps - 1, 1)),
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


class ActorAR(Actor):
    """Autoregressive fine-tune: the policy conditions on its OWN past
    allocations instead of the oracle's.

    This closes the train/test gap left by behaviour cloning -- at deployment
    there is no oracle action to condition on, and small allocation errors
    compound through the turnover term.
    """

    def __init__(self, cfg, jepa):
        super().__init__(cfg, jepa)
        self.ctx_len = cfg['actor']['ar']['ctx_len']
        self.pred_steps = cfg['actor']['ar']['pred_steps']

    def rollout(self, batch, ctx_len, pred_steps, act_temp=1.0, decode='expected'):
        """Roll forward on real market data, feeding back own actions.

        The latents and returns are the true ones (this is a policy rollout, not
        a dream); only the action history is self-generated.
        """
        X, Ret = batch['sample'], batch['return']
        with torch.no_grad():
            Z = self.jepa.encode(X)

        B, S, _ = Z.shape
        pred_steps = min(pred_steps, S - 1 - ctx_len)

        Z_p = Z[:, :ctx_len]
        Ret_p = Ret[:, :ctx_len]
        Act_p = torch.zeros_like(Ret_p)

        action_logits, actions = [], []

        for t in range(pred_steps):
            with torch.no_grad():
                Zp1, Ret_logits = self.jepa.predict(Z_p, Ret_p)
                Ret_probs = torch.softmax(Ret_logits, dim=-1)

            cond = torch.cat((Ret_probs, Act_p), dim=-1)
            logits = self(Zp1, cond)[:, -1:, :]

            # Detached: the fed-back action is a conditioning input, not a path
            # gradients should flow along. new_act is [B, 1, 1] so it can be
            # concatenated onto the time axis of Act_p.
            probs = torch.softmax(logits.detach() / act_temp, dim=-1)
            new_act, _ = self.decode_actions(probs, decode)

            Z_p = truncate(torch.cat([Z_p, Z[:, ctx_len + t:ctx_len + t + 1]], dim=1), self.backbone.max_len)
            Ret_p = truncate(torch.cat([Ret_p, Ret[:, ctx_len + t:ctx_len + t + 1]], dim=1), self.backbone.max_len)
            Act_p = truncate(torch.cat([Act_p, new_act], dim=1), self.backbone.max_len)

            action_logits.append(logits)
            actions.append(new_act)

        return torch.cat(action_logits, dim=1), torch.cat(actions, dim=1), pred_steps

    def _shared_step(self, batch, stage):
        logits, _, steps = self.rollout(batch, self.ctx_len, self.pred_steps)
        act_target = batch['action_target'][:, self.ctx_len:self.ctx_len + steps]

        L = self.CrossEntropyLoss(
            logits.reshape(-1, logits.size(-1)),
            act_target.reshape(-1).long(),
        )

        on_step = stage == 'train'
        self.log(f'{stage}/actor_loss', L, on_step=on_step, on_epoch=not on_step,
                 prog_bar=(stage == 'val'))
        return L

    def backtest(self, batch, act_temp=1.0, decode='expected'):
        _, _, Ret_probs = self.world_state(batch)
        logits, actions, steps = self.rollout(
            batch, self.ctx_len, self.pred_steps, act_temp=act_temp, decode=decode
        )

        lo, hi = self.ctx_len, self.ctx_len + steps
        Act_out = batch['action'][:, lo:hi]
        Ret_raw = batch['return_raw'][:, lo:hi]

        (E, e), (t_E, t_e), (bh_E, bh_e) = self._equity_bundle(actions, Act_out, Ret_raw)

        return {
            'action': actions,
            'action_prob': torch.softmax(logits / act_temp, dim=-1),
            'return': batch['return'][:, lo:hi],
            'return_raw': Ret_raw,
            'return_prob': Ret_probs[:, lo - 1:hi - 1],
            'equity': E,
            'end_equity': e,
            'opt_action': Act_out,
            'opt_equity': t_E,
            'opt_end_equity': t_e,
            'bh_equity': bh_E,
            'bh_end_equity': bh_e,
        }
