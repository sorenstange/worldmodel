import torch
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import logging

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
        # Vi starter med de første (Seq - horizon) skridt som historik, 
        # så vi har nok sande fremtidige skridt at sammenligne med
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

            # Rettet variabelnavn her fra nyeste_Z_pred til new_Z_pred
            Z_history = torch.cat([Z_history, new_Z_pred], dim=1)
    
    # Uddrag de sande segmenter for de tilsvarende fremtidige tidsskridt
    Z_true_segments = Z_true[:, start_idx+1:start_idx+1+horizon, :].cpu().numpy()
    y_true_segments = y_true[:, start_idx+1:start_idx+1+horizon, :].cpu().numpy()
    
    predicted_trajectory = np.concatenate(predicted_trajectory, axis=1) # [1, horizon, d_model]
    predicted_logits = np.concatenate(predicted_logits, axis=1)         # [1, horizon, bins]

    logger.info(f'Inferens complete!')
    return Z_true_segments[0], predicted_trajectory[0], y_true_segments[0], predicted_logits[0], test_dataset

if __name__ == '__main__':
    CHECKPOINT = "./models/glad-donkey-38/jepa.ckpt" 
    CONFIG = "./config.yaml"
    horizon = 15

    cfg = OmegaConf.load(CONFIG)
    logger = set_logger(cfg)

    Z_true, Z_pred, Y_true, Y_pred_logits, test_dataset = run_trajectory(CHECKPOINT, cfg, horizon)

    # 1. Beregn MSE i det latente rum
    MSE = np.mean((Z_true - Z_pred) ** 2)
    logger.info(f'MSE in the latent space over {horizon} steps: {MSE:.4f}')

    # 2. Omregn logits til rigtige sandsynligheder (Softmax)
    # Y_pred_logits har formen [horizon, bins]
    exp_logits = np.exp(Y_pred_logits - np.max(Y_pred_logits, axis=-1, keepdims=True))
    Y_pred_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # 3. Definer afkast (returns) for hver enkelt bin ud fra bin_edges
    bin_edges = test_dataset.bin_edges # Formodes at have længden (num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_centers = bin_centers.cpu().numpy()

    pred_returns = np.argmax(Y_pred_probs, axis=-1)
    pred_returns = bin_centers[pred_returns]

    true_returns = bin_centers[Y_true]

    true_cumprod = np.cumprod(1 + true_returns)
    pred_cumprod = np.cumprod(1 + pred_returns)

    # ==================== PLOTTING DELEN ====================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    t_steps = np.arange(1, horizon + 1)

    # --- ØVERSTE SUBPLOT: Kumulativt Afkast ---
    ax1.plot(t_steps, true_cumprod, label='Faktisk Cumprod (Markedet)', color='black', linewidth=2.5)
    ax1.plot(t_steps, pred_cumprod, label='Forudsagt Cumprod (JEPA)', color='crimson', linestyle='--', linewidth=2)
    ax1.set_title('JEPA Verdensmodel: Forudsagt vs. Faktisk Kumulativt Afkast', fontsize=14)
    ax1.set_ylabel('Kumulativ Værdi (1.0 = Start)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- NEDERSTE SUBPLOT: Heatmap over sandsynligheder ---
    # Vi transponerer for at få tid på x-aksen og bins på y-aksen
    # 'origin=lower' sikrer at lavere bin-index (negative afkast) er i bunden
    im = ax2.imshow(Y_pred_probs.T, aspect='auto', cmap='viridis', origin='lower',
                    extent=[0.5, horizon + 0.5, 0, len(bin_centers) - 1])
    
    # Marker den korrekte (sande) bin med røde prikker hen over heatmappet
    ax2.scatter(t_steps, true_returns, color='red', edgecolor='white', s=35, label='Sand Bin-placering', zorder=5)

    ax2.set_title('Forudsagte Sandsynligheder (Heatmap) vs. Sande Pris-bins', fontsize=14)
    ax2.set_xlabel('Tidsskridt frem i tiden (Minutter)', fontsize=12)
    ax2.set_ylabel('Pris-Bin Index (0-255)', fontsize=12)
    ax2.set_xticks(t_steps)
    ax2.legend(fontsize=11, loc='upper left')

    # Tilføj en colorbar for at vise sandsynlighedsskalaen
    cbar = fig.colorbar(im, ax=ax2, orientation='vertical', pad=0.02)
    cbar.set_label('Sandsynlighed', fontsize=11)

    plt.tight_layout()
    plt.savefig("figs/jepa_trajectory_diagnostic.png", dpi=300)
    logger.info("Det avancerede diagnosediagram er gemt som 'figs/jepa_trajectory_diagnostic.png'")
