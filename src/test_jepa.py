import torch
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import logging
import os

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from jepa import JEPA
from data import CryptoDataset
from util import set_logger

def run_trajectory(checkpoint_path, cfg, horizon=15):
    logger = logging.getLogger(cfg['experiment_name'])

    logger.info(f'Loading model from: {checkpoint_path}')

    model = JEPA.load_from_checkpoint(checkpoint_path, cfg=cfg, weights_only=False)
    model.eval()
    model.cuda()

    test_dataset = CryptoDataset(cfg, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    batch = next(iter(test_loader))
    X = batch['sample'].cuda()      # [1, Seq, Win, D]
    y_true = batch['target'].cuda() # [1, Seq, bins]

    with torch.no_grad():
        Z_true = model.encode(X)
        start_idx = Z_true.size(1) - horizon - 1
        Z_history = Z_true[:, :start_idx+1, :]

        predicted_trajectory = []
        predicted_logits = []

        for t in range(horizon):
            Z_next_pred, logits = model.predict(Z_history)

            new_Z_pred = Z_next_pred[:, -1:, :]
            new_logits = logits[:, -1:, :]

            predicted_trajectory.append(new_Z_pred.cpu().numpy())
            predicted_logits.append(new_logits.cpu().numpy())

            Z_history = torch.cat([Z_history, new_Z_pred], dim=1)
    
    Z_true_segments = Z_true[:, start_idx+1:start_idx+1+horizon, :].cpu().numpy()
    y_true_segments = y_true[:, start_idx+1:start_idx+1+horizon, :].cpu().numpy()
    
    predicted_trajectory = np.concatenate(predicted_trajectory, axis=1) 
    predicted_logits = np.concatenate(predicted_logits, axis=1)         

    logger.info(f'Inferens complete!')
    return Z_true_segments[0], predicted_trajectory[0], y_true_segments[0], predicted_logits[0], test_dataset

if __name__ == '__main__':
    CHECKPOINT = "./models/swift-bush-43/last.ckpt" 
    CONFIG = "./config.yaml"
    horizon = 15
    
    # --- TEMPERATURE CONFIGURATION ---
    temperature = 0.5  # Juster her: >1.0 for mere tilfældighed, <1.0 for mere sikkerhed
    # ---------------------------------

    cfg = OmegaConf.load(CONFIG)
    logger = set_logger(cfg)

    Z_true, Z_pred, Y_true, Y_pred_logits, test_dataset = run_trajectory(CHECKPOINT, cfg, horizon)

    MSE = np.mean((Z_true - Z_pred) ** 2)
    logger.info(f'MSE in the latent space over {horizon} steps: {MSE:.4f}')

    # 1. Omregn logits til rigtige sandsynligheder MED temperaturstyring
    # Vi dividerer rå logits med temperaturen før softmax
    scaled_logits = Y_pred_logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
    Y_pred_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # 2. Definer afkast (returns) for hver enkelt bin ud fra bin_edges
    bin_edges = test_dataset.bin_edges 
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_centers = bin_centers.cpu().numpy()

    # 3. SAMPLING AF OUTCOMES FRA DISTRIBUTIONEN
    # Vi trækker ét bin-index for hvert tidsskridt baseret på de skalerede sandsynligheder
    sampled_bins = []
    num_bins = Y_pred_probs.shape[-1]
    
    for t in range(horizon):
        probs_t = Y_pred_probs[t]
        # Undgå potentielle numeriske flydepunktsfejl (så summen er præcis 1.0)
        probs_t = probs_t / np.sum(probs_t)
        sampled_idx = np.random.choice(np.arange(num_bins), p=probs_t)
        sampled_bins.append(sampled_idx)
        
    sampled_bins = np.array(sampled_bins)

    # 4. Find de tilsvarende returns og beregn cumprod
    pred_returns_argmax = bin_centers[np.argmax(Y_pred_probs, axis=-1)]
    pred_returns_sampled = bin_centers[sampled_bins]
    true_returns = bin_centers[Y_true]

    true_cumprod = np.cumprod(1 + true_returns)
    pred_cumprod_argmax = np.cumprod(1 + pred_returns_argmax)
    pred_cumprod_sampled = np.cumprod(1 + pred_returns_sampled)

    # ==================== PLOTTING DELEN ====================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    t_steps = np.arange(1, horizon + 1)

    # --- ØVERSTE SUBPLOT: Kumulativt Afkast ---
    ax1.plot(t_steps, true_cumprod, label='Faktisk Cumprod (Markedet)', color='black', linewidth=2.5)
    ax1.plot(t_steps, pred_cumprod_argmax, label='Forudsagt Cumprod (Argmax)', color='crimson', linestyle='--', linewidth=2)
    ax1.plot(t_steps, pred_cumprod_sampled, label=f'Samplet Cumprod (Temp={temperature})', color='darkorange', linestyle=':', linewidth=2)
    ax1.set_title(f'JEPA Verdensmodel: Forudsagt, Samplet og Faktisk Kumulativt Afkast', fontsize=14)
    ax1.set_ylabel('Kumulativ Værdi (1.0 = Start)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- NEDERSTE SUBPLOT: Heatmap over sandsynligheder ---
    im = ax2.imshow(Y_pred_probs.T, aspect='auto', cmap='viridis', origin='lower',
                    extent=[0.5, horizon + 0.5, 0, len(bin_centers) - 1])
    
    # Vis både sande og udtrukne bin-placeringer oven på distributionen
    ax2.scatter(t_steps, Y_true, color='red', edgecolor='white', s=45, label='Sand Bin-placering', zorder=5)
    ax2.scatter(t_steps, sampled_bins, color='cyan', marker='x', s=55, linewidths=2, label='Samplet Bin-placering', zorder=6)

    ax2.set_title(f'Forudsagte Sandsynligheder Heatmap (Temp={temperature})', fontsize=14)
    ax2.set_xlabel('Tidsskridt frem i tiden (Minutter)', fontsize=12)
    ax2.set_ylabel('Pris-Bin Index (0-255)', fontsize=12)
    ax2.set_xticks(t_steps)
    ax2.legend(fontsize=11, loc='upper left')

    cbar = fig.colorbar(im, ax=ax2, orientation='vertical', pad=0.02)
    cbar.set_label('Temperatur-skaleret Sandsynlighed', fontsize=11)

    plt.tight_layout()
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/jepa_trajectory_diagnostic.png", dpi=300)
    logger.info("Det avancerede diagnosediagram med temperatur-sampling er gemt som 'figs/jepa_trajectory_diagnostic.png'")
