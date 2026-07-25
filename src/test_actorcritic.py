import torch
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
import torch.nn.functional as F

from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader

from jepa import JEPA
from actorcritic import ActorCritic
from data import CryptoDataset
from util import set_logger

def run_trading_backtest(jepa_ckpt, ac_ckpt, cfg, horizon=60):
    logger = logging.getLogger(cfg['experiment_name'])
    logger.info("Starting Actor-Critic Backtest on Test Data...")

    # 1. Omgå PyTorch 2.6+ pickler-sikkerhed for OmegaConf
    torch.serialization.add_safe_globals([DictConfig])

    # 2. Indlæs den frosne JEPA (verdensmodel)
    logger.info(f"Loading JEPA from: {jepa_ckpt}")
    jepa_model = JEPA.load_from_checkpoint(jepa_ckpt, cfg=cfg, weights_only=False)
    jepa_model.eval()
    jepa_model.cuda()

    # 3. Indlæs din trænede Actor-Critic model
    logger.info(f"Loading Actor-Critic from: {ac_ckpt}")
    model = ActorCritic.load_from_checkpoint(ac_ckpt, jepa_model=jepa_model, cfg=cfg, weights_only=False)
    model.eval()
    model.cuda()

    # 4. Klargør test-loader (shuffle=False for kronologisk rækkefølge)
    test_dataset = CryptoDataset(cfg, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Vi tager én batch (en fuld sekvens) ud til analyse
    batch = next(iter(test_loader))
    X = batch['sample'].cuda()       # [1, Seq, Win, D]
    y_true = batch['target'].cuda()  # [1, Seq, 1]
    ret_true = batch['return'].cuda() # [1, Seq, 1]

    bin_edges = jepa_model.bin_edges
    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2.0).cpu().numpy()

    with torch.no_grad():
        Z_true = model.jepa.encode(X) # Generer de sande latente tilstande
        
        # Vi lader agenten handle over de sidste 'horizon' antal minutter i sekvensen
        start_t = Z_true.size(1) - horizon - 1
        
        Z_history = Z_true[:, :start_t+1, :]
        Ret_history = ret_true[:, :start_t+1, :]
        
        cash_bin_idx = model.action_bins // 2
        Action_history = torch.full((1, start_t + 1), cash_bin_idx, dtype=torch.long, device=X.device)

        # Beholdere til opsamling af data til plots
        actual_market_returns = []
        agent_chosen_positions = []
        agent_rewards_logged = []
        all_predicted_probs = []

        old_position = None
        com_v = cfg['actorcritic']['commission_value']

        # --- BACKTEST LØKKE ---
        for t in range(horizon):
            # Agenten analyserer historikken (Z, afkast, og egne tidligere positioner via AdaLN)
            action_logits, _ = model(Z_history, Ret_history, Action_history)
            
            # Under test tager vi altid den mest sikre handling (argmax)
            chosen_bin = torch.argmax(action_logits, dim=-1) # [1]
            
            # Find den reelle positionværdi (-1.0 til 1.0)
            reel_position = model.positions[chosen_bin].unsqueeze(-1).unsqueeze(-1) # [1, 1, 1]

            # Hent det FAKTISKE markedsafkast for det næste minut
            target_t = start_t + 1 + t
            new_market_return = ret_true[:, target_t:target_t+1, :] # [1, 1, 1]
            market_ret_val = new_market_return.item()

            # Beregn kurtageomkostning ved skift af position
            commission = (reel_position - old_position) * com_v if old_position is not None else reel_position * com_v
            old_position = reel_position

            # Beregn agentens afkast (Markedsafkast * Position - Kurtage)
            reward_val = market_ret_val * reel_position.item() - commission.item()

            # Hent predictor-sandsynlighederne for heatmappet
            _, logits_pred = model.jepa.predict(Z_history, Ret_history)
            probs_t = F.softmax(logits_pred[:, -1, :], dim=-1).cpu().numpy().flatten()

            # Gem data til plotting
            actual_market_returns.append(market_ret_val)
            agent_chosen_positions.append(reel_position.item())
            agent_rewards_logged.append(reward_val)
            all_predicted_probs.append(probs_t)

            # Opdater historikken med de FAKTISKE markedsbevægelser (Klassisk Backtest)
            new_Z = Z_true[:, target_t:target_t+1, :]
            Z_history = torch.cat([Z_history, new_Z], dim=1)
            Ret_history = torch.cat([Ret_history, new_market_return], dim=1)
            Action_history = torch.cat([Action_history, chosen_bin.unsqueeze(-1)], dim=1)

    logger.info("Backtest complete! Preparing plots...")
    
    return (np.array(actual_market_returns), 
            np.array(agent_chosen_positions), 
            np.array(agent_rewards_logged), 
            np.array(all_predicted_probs), 
            y_true[0, start_t+1:start_t+1+horizon].cpu().numpy().flatten(),
            bin_centers)

if __name__ == '__main__':
    # Opsæt dine checkpoints (Erstat med dine præcise run-navne)
    JEPA_CHECKPOINT = "./models/jepa/charmed-violet-1/last.ckpt"
    RL_CHECKPOINT = "./models/actorcritic/autotrader-v1/last.ckpt" 
    CONFIG_PATH = "./config.yaml"
    HORIZON = 60 # Test agenten over en hel time (60 minutter)

    cfg = OmegaConf.load(CONFIG_PATH)
    logger = set_logger(cfg)

    # Kør backtesten
    market_ret, agent_pos, agent_rew, pred_probs, true_bins, bin_centers = run_trading_backtest(
        JEPA_CHECKPOINT, RL_CHECKPOINT, cfg, horizon=HORIZON
    )

    # 1. Beregn Kumulative afkast (Equity Curves)
    market_equity = np.cumprod(1 + market_ret)
    agent_equity = np.cumprod(1 + agent_rew)

    # ==================== GRID PLOTTING DELEN ====================
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    t_steps = np.arange(1, HORIZON + 1)

    # Subplot 1: Equity Curves
    ax1.plot(t_steps, market_equity, label="Buy & Hold (Markedet)", color="black", linewidth=2)
    ax1.plot(t_steps, agent_equity, label="Unified Actor-Critic Agent", color="forestgreen", linewidth=2.5)
    ax1.set_title("1. Porteføljeudvikling (Equity Curve) inkl. Handelsomkostninger", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Vækstfaktor (1.0 = Start)")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Agentens Markedsposition
    # Vi farver områderne for at gøre det ekstremt nemt at se Long/Short/Cash
    ax2.plot(t_steps, agent_pos, color="darkblue", drawstyle="steps-mid", linewidth=2)
    ax2.fill_between(t_steps, agent_pos, 0, where=(agent_pos > 0), facecolor='green', alpha=0.2, label="Long")
    ax2.fill_between(t_steps, agent_pos, 0, where=(agent_pos < 0), facecolor='red', alpha=0.2, label="Short")
    ax2.set_title("2. Agentens Aktive Markedsposition over tid (-1 = Short, 0 = Cash, 1 = Long)", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Position / Gearing")
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Subplot 3: Heatmap over Verdensmodellens sandsynligheder med sande afkast værdier
    min_ret, max_ret = bin_centers[0], bin_centers[-1]
    im = ax3.imshow(pred_probs.T, aspect='auto', cmap='viridis', origin='lower',
                    extent=[0.5, HORIZON + 0.5, min_ret, max_ret])
    
    # Marker den reelle afkast-placering som røde prikker oven på heatmappet
    true_return_vals = bin_centers[true_bins]
    ax3.scatter(t_steps, true_return_vals, color='red', edgecolor='white', s=35, label='Reelt Markedsafkast', zorder=5)
    
    # Formater y-aksen pænt som procenter
    import matplotlib.ticker as mtick
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=2))
    ax3.set_title("3. Verdensmodellens Sandsynlighedssky vs. Virkeligheden", fontsize=13, fontweight='bold')
    ax3.set_xlabel("Tid (Minutter i test-perioden)", fontsize=11)
    ax3.set_ylabel("Afkast pr. minut")
    ax3.legend(loc="upper left")

    # Tilføj farveskala til bunden
    cbar = fig.colorbar(im, ax=ax3, orientation='horizontal', pad=0.15, shrink=0.6)
    cbar.set_label('Verdensmodellens Sandsynlighedsfordeling')

    plt.tight_layout()
    os.makedirs("figs", exist_ok=True)
    plot_path = "figs/actorcritic_backtest_results.png"
    plt.savefig(plot_path, dpi=300)
    
    logger.info(f"Backtest-diagrammet er gemt succesfuldt i '{plot_path}'!")
