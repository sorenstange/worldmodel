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

def run_trajectory(checkpoint_path, cfg, horizon=15, temperature=0.5):
    logger = logging.getLogger(cfg['experiment_name'])

    logger.info(f'Loading model from: {checkpoint_path}')

    model = JEPA.load_from_checkpoint(checkpoint_path, cfg=cfg, weights_only=False)
    model.eval()
    model.cuda()

    test_dataset = CryptoDataset(cfg, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    batch = next(iter(test_loader))
    X = batch['sample'].cuda()       # [1, Seq, Win, D]
    y_true = batch['target'].cuda()  # [1, Seq, 1] (Fra det nye data-setup)
    ret_true = batch['return'].cuda() # [1, Seq, 1] (De sande float-afkast)

    # Definer afkast (returns) for hver enkelt bin ud fra bin_edges til autoregressiv brug
    bin_edges = model.bin_edges # Henter direkte fra modellens registrerede buffer
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    with torch.no_grad():
        Z_true = model.encode(X)
        start_idx = Z_true.size(1) - horizon - 1
        
        # Initialiser historikken for både det latente rum og afkastene
        Z_history = Z_true[:, :start_idx+1, :]          # [1, T_hist, d_model]
        Ret_history = ret_true[:, :start_idx+1, :]      # [1, T_hist, 1]

        predicted_trajectory = []
        predicted_logits = []

        for t in range(horizon):
            # NYT: Vi sender nu Ret_history med ind til dine AdaLN lag!
            Z_next_pred, logits = model.predict(Z_history, Ret_history)

            new_Z_pred = Z_next_pred[:, -1:, :]
            new_logits = logits[:, -1:, :] # [1, 1, num_bins]

            predicted_trajectory.append(new_Z_pred.cpu().numpy())
            predicted_logits.append(new_logits.cpu().numpy())

            # --- AUTOREGRESSIV BETINGELSE (AdaLN-FEEDBACK) ---
            # Vi overskalerer logits med temperaturen lokalt i løkken for at simulere den sande drøm
            scaled_logits_t = new_logits.squeeze(0).squeeze(0) / temperature
            probs_t = torch.softmax(scaled_logits_t, dim=-1)
            
            # Træk næste skridts afkast (vi bruger en vægtet sampling pr. default til AdaLN feedbacken)
            sampled_idx = torch.multinomial(probs_t, num_samples=1) # [1]
            new_ret_val = bin_centers[sampled_idx].unsqueeze(0).unsqueeze(0) # [1, 1, 1]

            # Opdater historikken til næste skridt med modellens eget genererede afkast
            Z_history = torch.cat([Z_history, new_Z_pred], dim=1)
            Ret_history = torch.cat([Ret_history, new_ret_val], dim=1)
    
    # Uddrag de korrekte sande segmenter
    Z_true_segments = Z_true[:, start_idx+1:start_idx+1+horizon, :].cpu().numpy()
    y_true_segments = y_true[:, start_idx+1:start_idx+1+horizon, :].squeeze(-1).cpu().numpy()
    
    predicted_trajectory = np.concatenate(predicted_trajectory, axis=1) 
    predicted_logits = np.concatenate(predicted_logits, axis=1).squeeze(0) # Flad ud til [horizon, bins]

    logger.info(f'Inferens complete!')
    return Z_true_segments[0], predicted_trajectory[0], y_true_segments[0], predicted_logits, test_dataset, bin_centers.cpu().numpy()

if __name__ == '__main__':
    CHECKPOINT = "./models/jepa/charmed-violet-1/last.ckpt" 
    CONFIG = "./config.yaml"
    horizon = 15
    
    # --- TEMPERATURE CONFIGURATION ---
    temperature = 0.5  # Juster her: >1.0 for mere støj, <1.0 for mere deterministisk adfærd
    # ---------------------------------

    cfg = OmegaConf.load(CONFIG)
    logger = set_logger(cfg)

    # Kør den opdaterede banegenerering
    Z_true, Z_pred, Y_true, Y_pred_logits, test_dataset, bin_centers = run_trajectory(CHECKPOINT, cfg, horizon, temperature)

    MSE = np.mean((Z_true - Z_pred) ** 2)
    logger.info(f'MSE in the latent space over {horizon} steps: {MSE:.4f}')

    # 1. Omregn samlede logits til sandsynligheder med temperatur
    scaled_logits = Y_pred_logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
    Y_pred_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # 2. Sampling af outcomes baseret på den udtrukne distribution til plottet
    sampled_bins = []
    num_bins = Y_pred_probs.shape[-1]
    
    for t in range(horizon):
        probs_t = Y_pred_probs[t]
        probs_t = probs_t / np.sum(probs_t)
        sampled_idx = np.random.choice(np.arange(num_bins), p=probs_t)
        sampled_bins.append(sampled_idx)
        
    sampled_bins = np.array(sampled_bins)

    # 3. Find de tilsvarende numeriske returns og beregn kumulativ vækst (cumprod)
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
    ax1.set_title(f'JEPA Betinget Verdensmodel (AdaLN): Forudsagt, Samplet og Faktisk Cumprod', fontsize=14)
    ax1.set_ylabel('Kumulativ Værdi (1.0 = Start)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- NEDERSTE SUBPLOT: Heatmap over sandsynligheder ---
    # Hent min og max værdier direkte fra dine bin_centers
    min_return = bin_centers[0]
    max_return = bin_centers[-1]

    # FIX 1: Vi ændrer 'extent' i bunden og toppen til de reelle return-værdier (floats)
    im = ax2.imshow(Y_pred_probs.T, aspect='auto', cmap='viridis', origin='lower',
                    extent=[0.5, horizon + 0.5, min_return, max_return])
    
    # FIX 2: Da y-aksen nu er numeriske returns, skal vi mappe y-værdierne til bin_centers i stedet for indekser
    true_returns_vals = bin_centers[Y_true]
    sampled_returns_vals = bin_centers[sampled_bins]

    # Plot punkterne ved de korrekte float-værdier på y-aksen
    ax2.scatter(t_steps, true_returns_vals, color='red', edgecolor='white', s=45, label='Sand afkast-placering', zorder=5)
    ax2.scatter(t_steps, sampled_returns_vals, color='cyan', marker='x', s=55, linewidths=2, label='Samplet AdaLN-Feedback afkast', zorder=6)

    ax2.set_title(f'Betingede Sandsynligheder Heatmap (Temp={temperature})', fontsize=14)
    ax2.set_xlabel('Tidsskridt frem i tiden (Minutter)', fontsize=12)
    ax2.set_ylabel('Afkast pr. minut (Returns)', fontsize=12)
    ax2.set_xticks(t_steps)
    
    # FIX 3: Formater y-aksens labels pænt som procenter
    import matplotlib.ticker as mtick
    # Viser værdier med procenttegn og 2 decimaler (f.eks. 0.05 bliver til 5.00%)
    # Da dine returns er rå procenter (f.eks. 0.001 for 0.1%), ganger vi ticks med 100
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=2)) 
    
    ax2.legend(fontsize=11, loc='upper left')

    cbar = fig.colorbar(im, ax=ax2, orientation='vertical', pad=0.02)
    cbar.set_label('Temperatur-skaleret Sandsynlighed', fontsize=11)

    plt.tight_layout()
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/jepa_trajectory_diagnostic.png", dpi=300)
    logger.info("Det avancerede diagnosediagram for AdaLN-arkitekturen er gemt i 'figs/jepa_trajectory_diagnostic.png'")
