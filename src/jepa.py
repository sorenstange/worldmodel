import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L

from transformers import get_cosine_schedule_with_warmup

from modules import *
from util import *


class JEPA(L.LightningModule):
    """Latent world model over 15-minute windows of 1m candles.

    Training combines four terms:
      * `state`   -- one-step MSE between the predicted and encoded next latent.
      * `return`  -- cross-entropy on the next window's binned (vol-normalised)
                     return.
      * `sigreg`  -- LeJEPA Gaussianity penalty. Without it the state term has a
                     trivial constant solution; see CLAUDE.md.
      * `rollout` -- the same two losses but measured along a multi-step
                     rollout that feeds the model its own predictions back.
                     Teacher forcing alone leaves the predictor unprepared for
                     `dream()`, which is exactly where the RL stage will live.
    """

    def __init__(self, cfg):
        super().__init__()
        # Capture `cfg` as a single hparam. Passing the dict positionally would
        # splat its top-level keys into hparams, and one of those keys is
        # 'jepa' -- which then collides with the `jepa` argument Actor takes,
        # making load_from_checkpoint raise on a duplicate keyword.
        self.save_hyperparameters()

        jcfg = cfg['jepa']
        d_model = jcfg['d_model']
        head_drop = jcfg['return_head']['dropout']

        self.encoder = Encoder(
            input_dim=jcfg['input_dim'],
            d_model=d_model,
            num_layers=jcfg['encoder']['num_layers'],
            num_heads=jcfg['encoder']['num_heads'],
            max_len=jcfg['encoder']['max_len'],
            dropout=jcfg['encoder']['dropout'],
            ff_mult=jcfg['encoder'].get('ff_mult', 4),
        )

        self.predictor = Predictor(
            input_dim=d_model,
            d_model=d_model,
            num_layers=jcfg['predictor']['num_layers'],
            num_heads=jcfg['predictor']['num_heads'],
            max_len=jcfg['predictor']['max_len'],
            dropout=jcfg['predictor']['dropout'],
            ff_mult=jcfg['predictor'].get('ff_mult', 4),
        )

        self.return_head = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.LayerNorm(2 * d_model),
            nn.SiLU(),
            nn.Dropout(head_drop),
            nn.Linear(2 * d_model, 2 * d_model),
            nn.LayerNorm(2 * d_model),
            nn.SiLU(),
            nn.Dropout(head_drop),
            nn.Linear(2 * d_model, jcfg['return_head']['num_bins']),
        )

        self.register_buffer(
            'return_bins',
            bin_centers(
                jcfg['return_head']['min_value'],
                jcfg['return_head']['max_value'],
                jcfg['return_head']['num_bins'],
            ),
        )

        self.sigreg = SIGReg(
            knots=jcfg['sigreg']['knots'],
            num_proj=jcfg['sigreg']['num_proj'],
        )

        self.MSELoss = nn.MSELoss()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

        self.lam_state = jcfg['lam_state']
        self.lam_ce = jcfg['lam_CE']
        self.lam_sigreg = jcfg['lam_SIGReg']
        self.lam_rollout = jcfg['lam_rollout']

        self.rollout_steps = jcfg['training']['rollout_steps']
        self.rollout_batch = jcfg['training']['rollout_batch']

        self.lr = jcfg['training']['lr']
        self.weight_decay = jcfg['training']['weight_decay']
        self.warmup_steps = jcfg['training']['warmup_steps']
        self.sched_steps = jcfg['training'].get('sched_steps', None)

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def encode(self, X):
        if X.dim() == 4:
            B, Seq, Win, D = X.shape
        elif X.dim() == 3:
            B = 1
            Seq, Win, D = X.shape

        X = X.reshape(B * Seq, Win, -1)
        Z = self.encoder(X)
        return Z.view(B, Seq, -1)

    def predict(self, Z, Ret):
        Zp1 = self.predictor(Z, Ret)
        ret_logits = self.return_head(Zp1)
        return Zp1, ret_logits

    def forward(self, X, Ret):
        Z = self.encode(X)
        Zp1, ret_logits = self.predict(Z, Ret)
        return Z, Zp1, ret_logits

    def expected_return(self, ret_logits):
        """Differentiable point estimate from the head's distribution.

        Used to close the loop during rollout training: sampling would cut the
        gradient, and argmax is not differentiable either.
        """
        probs = torch.softmax(ret_logits, dim=-1)
        return (probs * self.return_bins).sum(dim=-1, keepdim=True)

    def rollout(self, Z_ctx, Ret_ctx, steps):
        """Roll `steps` ahead feeding predicted latents and returns back in.

        Gradients flow through the whole rollout, which is what makes this train
        the predictor for its own error distribution rather than for the
        teacher-forced one.
        """
        Z_seq, Ret_seq = Z_ctx, Ret_ctx
        pred_Z, pred_logits = [], []

        for _ in range(steps):
            Zp1, ret_logits = self.predict(Z_seq, Ret_seq)

            z_next = Zp1[:, -1:, :]
            lg_next = ret_logits[:, -1:, :]
            r_next = self.expected_return(lg_next)

            Z_seq = truncate(torch.cat([Z_seq, z_next], dim=1), self.predictor.max_len)
            Ret_seq = truncate(torch.cat([Ret_seq, r_next], dim=1), self.predictor.max_len)

            pred_Z.append(z_next)
            pred_logits.append(lg_next)

        return torch.cat(pred_Z, dim=1), torch.cat(pred_logits, dim=1)

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def _shared_step(self, batch, stage):
        X = batch['sample']
        Ret, Ret_target = batch['return'], batch['return_target']

        Z = self.encode(X)

        Z_hat, ret_logits = self.predict(Z[:, :-1], Ret[:, :-1])
        Z_target = Z[:, 1:]
        ret_target = Ret_target[:, 1:]

        L_state = self.MSELoss(Z_hat, Z_target)
        L_ret = self._ce(ret_logits, ret_target)
        L_sigreg = self.sigreg(Z)

        L = self.lam_state * L_state + self.lam_ce * L_ret + self.lam_sigreg * L_sigreg

        L_rollout = torch.zeros((), device=Z.device)
        if self.lam_rollout > 0 and self.rollout_steps > 0:
            L_rollout = self._rollout_loss(Z, Ret, Ret_target)
            L = L + self.lam_rollout * L_rollout

        on_step = stage == 'train'
        self.log_dict({
            f'{stage}/state_loss': L_state,
            f'{stage}/return_loss': L_ret,
            f'{stage}/sigreg_loss': L_sigreg,
            f'{stage}/rollout_loss': L_rollout,
            f'{stage}/jepa_loss': L,
        }, on_step=on_step, on_epoch=not on_step, prog_bar=(stage == 'val'))

        with torch.no_grad():
            diag = latent_diagnostics(Z)
            acc = (ret_logits.argmax(-1) == ret_target.squeeze(-1)).float().mean()
            self.log_dict({
                f'{stage}/latent_std': diag['latent_std'],
                f'{stage}/latent_erank': diag['latent_erank'],
                f'{stage}/return_acc': acc,
            }, on_step=on_step, on_epoch=not on_step)

        return L

    def _ce(self, logits, target):
        return self.CrossEntropyLoss(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1).long(),
        )

    def _rollout_loss(self, Z, Ret, Ret_target):
        B, S, _ = Z.shape
        k = min(self.rollout_steps, S - 1)

        # Cap the rollout batch: k sequential predictor passes are all retained
        # for backprop, so this term dominates memory if left at full batch.
        b = min(self.rollout_batch, B) if self.rollout_batch else B
        Zb, Retb, Tb = Z[:b], Ret[:b], Ret_target[:b]

        # Random context length so the model sees rollouts from everywhere in the
        # sequence rather than always from one offset.
        c = int(torch.randint(low=1, high=max(S - k, 2), size=(1,)).item())

        # Detached targets: the teacher-forced term plus SIGReg already shape the
        # encoder. Letting the rollout pull on the encoder too would add a second
        # route to collapse.
        pred_Z, pred_logits = self.rollout(Zb[:, :c].detach(), Retb[:, :c], k)

        L_state = self.MSELoss(pred_Z, Zb[:, c:c + k].detach())
        L_ret = self._ce(pred_logits, Tb[:, c:c + k])
        return L_state + self.lam_ce * L_ret

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, 'val')

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def dream(self, Z_prompt, Ret_prompt, horizon=15, ret_temperature=1.0):
        """Open-loop rollout with sampled returns. Values are in the model's
        vol-normalised units -- multiply by the window's `vol` to get a real
        return."""
        dream_states, dream_ret, dream_ret_probs, dream_ret_sampled_idx = [], [], [], []

        Z_prompt = truncate(Z_prompt, self.predictor.max_len)
        Ret_prompt = truncate(Ret_prompt, self.predictor.max_len)

        for _ in range(horizon):
            Z_next, ret_logits = self.predict(Z_prompt, Ret_prompt)

            new_Z = Z_next[:, -1:, :]
            ret_logits = ret_logits[:, -1:, :].squeeze(1) / ret_temperature

            ret_probs = torch.softmax(ret_logits, dim=-1)
            ret_sampled_idx = torch.multinomial(ret_probs, num_samples=1)
            new_ret = self.return_bins[ret_sampled_idx]

            Z_prompt = truncate(torch.cat([Z_prompt, new_Z.detach()], dim=1), self.predictor.max_len)
            Ret_prompt = truncate(torch.cat([Ret_prompt, new_ret.detach().unsqueeze(1)], dim=1), self.predictor.max_len)

            dream_states.append(new_Z)
            dream_ret.append(new_ret.detach().unsqueeze(1))
            dream_ret_probs.append(ret_probs.unsqueeze(1))
            dream_ret_sampled_idx.append(ret_sampled_idx.detach().unsqueeze(1))

        return (
            torch.concatenate(dream_states, dim=1),
            torch.concatenate(dream_ret, dim=1),
            torch.concatenate(dream_ret_probs, dim=1),
            torch.concatenate(dream_ret_sampled_idx, dim=1),
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # The cosine schedule must span the steps training will ACTUALLY run.
        # Deriving it from max_epochs meant early stopping cut it off in its
        # warmup plateau, so the LR never decayed. See CLAUDE.md.
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
