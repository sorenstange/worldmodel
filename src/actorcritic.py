import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lightning as L

from transformers import get_cosine_schedule_with_warmup

from modules import *

class ActorCritic(L.LightningModule):
    def __init__(self, jepa_model, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['jepa_model'])
        
        # 1. Fastlås din færdigtrænede verdensmodel
        self.jepa = jepa_model
        self.jepa.freeze()
        
        d_model = cfg['jepa']['d_model']
        self.action_bins = cfg['actorcritic']['action_bins']
        
        self.backbone = Backbone(
            d_model=d_model,
            num_layers=cfg['actorcritic']['num_layers'],
            num_heads=cfg['actorcritic']['num_heads'],
            max_len=cfg['actorcritic']['max_len'],
            condition_dim=1 + self.action_bins,
            dropout=cfg['actorcritic']['dropout']
        )
        
        # 3. Policy (Actor) og Value (Critic) hoveder
        self.actor_head = nn.Sequential(
            nn.Linear(d_model, 2*d_model),
            nn.LayerNorm(2*d_model),
            nn.SiLU(),
            nn.Linear(2*d_model, self.action_bins)
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(d_model, 2*d_model),
            nn.LayerNorm(2*d_model),
            nn.SiLU(),
            nn.Linear(2*d_model, 1)
        )
        
        # Registerer de reelle trading-positioner [-1.0 til 1.0]
        self.register_buffer("positions", torch.linspace(-1.0, 1.0, self.action_bins))
        
        # PPO Hyperparametre
        self.lr = cfg['actorcritic']['lr']
        self.gamma = cfg['actorcritic']['gamma']       
        self.gae_lambda = cfg['actorcritic']['gae_lambda'] 
        self.ppo_clip = cfg['actorcritic']['ppo_clip']    
        self.ent_coef = cfg['actorcritic']['ent_coef']   
        self.vf_coef = cfg['actorcritic']['vf_coef']       
        self.ppo_epochs = cfg['actorcritic']['ppo_epochs']   
        self.dream_horizon = cfg['actorcritic']['dream_horizon'] 
        
        # SLA MANUEL OPTIMERING TIL (Nødvendigt til PPO rollouts)
        self.automatic_optimization = False

    def forward(self, Z, last_returns, last_action_bins):
        """
        Z: [B, Seq, d_model]
        last_returns: [B, Seq, 1]
        last_action_bins: [B, Seq]
        """
        last_actions_one_hot = F.one_hot(last_action_bins.long(), num_classes=self.action_bins).float()
        condition = torch.cat([last_returns, last_actions_one_hot], dim=-1)
        
        features = self.backbone(Z, condition)
        last_step_features = features[:, -1, :] # Fokusér på den nuværende tilstand
        
        action_logits = self.actor_head(last_step_features)
        value = self.critic_head(last_step_features)
        
        return action_logits, value

    def training_step(self, batch, batch_idx):
        # Hent din fælles optimizer
        opt = self.optimizers()
        
        X, y, Ret = batch['sample'], batch['target'], batch['return']
        B, Seq, _ = Ret.shape
        
        # --- SKRIDT 1: GENERER PROMPT FRA REEL DATA ---
        with torch.no_grad():
            Z_start = self.jepa.encode(X) # [B, Seq, d_model]
            
        # Vi tager udgangspunkt i den historiske sekvens før vores drømme-horisont
        start_t = Seq - self.dream_horizon - 1
        Z_history = Z_start[:, :start_t+1, :]
        Ret_history = Ret[:, :start_t+1, :]
        Action_history = y[:, :start_t+1, :].squeeze(-1) # [B, T_hist]

        # Databeholdere til vores PPO Rollout (Drømme-buffer)
        dream_states = []
        dream_returns = []
        dream_actions = []
        dream_action_log_probs = []
        dream_values = []
        dream_rewards = []

        # --- SKRIDT 2: DRIFT UD I DRØMME (ROLLOUT FASEN) ---
        for t in range(self.dream_horizon):
            # Forudsig handlinger og værdier baseret på den opbyggede drømmehistorik
            action_logits, value = self(Z_history, Ret_history, Action_history)
            
            # Gør logits til sandsynligheder og træk en handling (Udforskning/Sampling)
            probs = F.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample() # [B]
            log_prob = dist.log_prob(action) # [B]

            # Få JEPA verdensmodellen til at drømme om det næste latente skridt og pris-logits
            with torch.no_grad():
                # Vi bruger de seneste afkast i historikken som input til verdensmodellens predictor
                Z_next_pred, logits_pred = self.jepa.predict(Z_history, Ret_history)
                new_Z = Z_next_pred[:, -1:, :]
                
                # Omsæt verdensmodellens pris-forudsigelse til et numerisk krypto-afkast (float)
                pred_bin = torch.argmax(logits_pred[:, -1:, :], dim=-1)
                new_market_return = self.jepa.bin_edges[pred_bin].unsqueeze(-1) # [B, 1, 1]

            # BEREGN TRADING REWARD (P&L)
            # Agentens valgte krypto-position (-1.0 til 1.0)
            reel_position = self.positions[action].unsqueeze(-1).unsqueeze(-1) # [B, 1, 1]
            # Reward = Markedets afkast * Agentens position (Gevinst ved short hvis markedet falder!)
            reward = (new_market_return * reel_position).squeeze(-1).squeeze(-1) # [B]

            # Gem data til PPO-opdateringen
            dream_states.append(Z_history[:, -1, :]) # Nuværende latente tilstand
            dream_returns.append(Ret_history[:, -1, :])
            dream_actions.append(action)
            dream_action_log_probs.append(log_prob)
            dream_values.append(value.squeeze(-1))
            dream_rewards.append(reward)

            # Opdater drømmehistorikken autoregressivt til næste skridt
            Z_history = torch.cat([Z_history, new_Z], dim=1)
            Ret_history = torch.cat([Ret_history, new_market_return], dim=1)
            Action_history = torch.cat([Action_history, action.unsqueeze(-1)], dim=1)

        # Stable arrays fra vores drømme [Dream_Horizon, B] -> [B, Dream_Horizon]
        dream_states = torch.stack(dream_states, dim=1)
        dream_returns = torch.stack(dream_returns, dim=1)
        dream_actions = torch.stack(dream_actions, dim=1)
        dream_old_log_probs = torch.stack(dream_action_log_probs, dim=1).detach()
        dream_values = torch.stack(dream_values, dim=1)
        dream_rewards = torch.stack(dream_rewards, dim=1)

        # --- SKRIDT 3: BEREGN RETURNAFKAST OG ADVANTAGES (GAE) ---
        advantages = torch.zeros_like(dream_rewards)
        last_gae_lam = 0
        with torch.no_grad():
            # Vi estimerer værdien af det absolut sidste skridt for at lukke GAE-kæden
            _, next_value = self(Z_history, Ret_history, Action_history)
            next_value = next_value.squeeze(-1)
            
            for t in reversed(range(self.dream_horizon)):
                if t == self.dream_horizon - 1:
                    next_non_terminal = 1.0
                    next_values = next_value
                else:
                    next_non_terminal = 1.0
                    next_values = dream_values[:, t + 1]
                
                # TD Error (Temporal Difference)
                delta = dream_rewards[:, t] + self.gamma * next_values * next_non_terminal - dream_values[:, t]
                advantages[:, t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            
            returns = advantages + dream_values

        # --- SKRIDT 4: PPO OPTIMERINGS-LØKKE (GENBRUG AF DRØMME) ---
        # Vi genbruger drømme-trajektorien for at klemme mest muligt læring ud af dataen
        for ppo_epoch in range(self.ppo_epochs):
            
            # Kør drømmene igennem netværket igen for at få NYE logits og værdier
            # (Da din backbone forventer en sekvens, genskaber vi formatet midlertidigt)
            action_logits, new_values = self(dream_states, dream_returns, dream_actions)
            new_values = new_values.squeeze(-1)

            # Beregn ny sandsynlighedsfordeling og entropi
            probs = F.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            
            # Vi trækker log_probs for de handlinger, som blev taget i drømmen
            new_log_probs = dist.log_prob(dream_actions)
            entropy = dist.entropy().mean()

            # PPO Ratio: r_t(theta) = pi_new(a|s) / pi_old(a|s)
            ratios = torch.exp(new_log_probs - dream_old_log_probs)

            # Normaliser advantages for at stabilisere opdateringen
            norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Surrogat tab 1 og 2 (PPO Clipping mekanismen)
            surr1 = ratios * norm_advantages
            surr2 = torch.clamp(ratios, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * norm_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value Function Loss (Critic) clipped
            critic_loss = F.mse_loss(new_values, returns)

            # SAMLET DELT LOSS (Opdaterer din shared backbone synkront!)
            total_loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy

            # MANUEL BACKWARD & STEP (Siden automatic_optimization=False)
            opt.zero_grad()
            self.manual_backward(total_loss)
            
            # Valgfrit: Clipper gradienter manuelt for maksimal stabilitet
            self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            opt.step()

        # Log de vigtigste PPO-metrikker til dit WandB dashboard
        self.log('ppo/total_loss', total_loss)
        self.log('ppo/actor_loss', actor_loss)
        self.log('ppo/critic_loss', critic_loss)
        self.log('ppo/entropy', entropy)
        self.log('ppo/mean_reward', dream_rewards.mean(), prog_bar=True)
        self.log('ppo/mean_value', dream_values.mean())
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        total_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(total_steps * 0.1) 
        
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
