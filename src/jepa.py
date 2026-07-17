import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lightning as L

from transformers import get_cosine_schedule_with_warmup

from modules import *

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
            num_bins = cfg['jepa']['predictor']['num_bins'],
            dropout = cfg['jepa']['predictor']['dropout']
        )

        self.MSELoss = nn.MSELoss()
        self.SIGRegLoss = SIGReg()
        self.KLDLoss = nn.KLDivLoss(reduction='batchmean')

        self.lam_SIGReg = cfg['jepa']['lam_SIGReg']
        self.lam_KLD = cfg['jepa']['lam_KLD']

        self.lr = cfg['jepa']['lr']

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

    def predict(self, Z):
        Zp1, Ret = self.predictor(Z)
        return Zp1, Ret

    def forward(self, X):
        Z = self.encode(X)
        return self.predict(Z)

    def training_step(self, batch, batch_idx):
        X, y = batch['sample'], batch['target']
        Z = self.encode(X)

        Z_in = Z[:, :-1]
        Z_target = Z[:, 1:]

        Z_hat, logits = self.predict(Z_in)  # logits shape: [B, Seq-1, bins]

        log_probs = F.log_softmax(logits, dim=-1)

        y_target = y[:, 1:]  # Form: [B, Seq-1, bins]

        log_probs_flat = log_probs.reshape(-1, log_probs.size(-1))
        y_target_flat = y_target.reshape(-1, y_target.size(-1))

        L_state = self.MSELoss(Z_hat, Z_target)
        L_kld = self.KLDLoss(log_probs_flat, y_target_flat) # Bruger nu de fladtrykte versioner
        L_sigreg = self.SIGRegLoss(Z.permute(1, 0, 2))

        L = L_state + self.lam_KLD * L_kld + self.lam_SIGReg * L_sigreg

        self.log('train_state_loss', L_state)
        self.log('train_kld_loss', L_kld)
        self.log('train_sigreg_loss', L_sigreg)
        self.log('train_loss', L)
        return L

    
    def validation_step(self, batch, batch_idx):
        X, y = batch['sample'], batch['target']
        Z = self.encode(X)

        Z_in = Z[:, :-1]
        Z_target = Z[:, 1:]

        Z_hat, logits = self.predict(Z_in)
        log_probs = F.log_softmax(logits, dim=-1)

        y_target = y[:, 1:]  # Form: [B, Seq-1, bins]

        log_probs_flat = log_probs.reshape(-1, log_probs.size(-1))
        y_target_flat = y_target.reshape(-1, y_target.size(-1))

        L_state = self.MSELoss(Z_hat, Z_target)
        L_kld = self.KLDLoss(log_probs_flat, y_target_flat) # Bruger nu de fladtrykte versioner
        L_sigreg = self.SIGRegLoss(Z.permute(1, 0, 2))

        L = L_state + self.lam_KLD * L_kld + self.lam_SIGReg * L_sigreg

        self.log('val_state_loss', L_state, on_step=False, on_epoch=True)
        self.log('val_kld_loss', L_kld, on_step=False, on_epoch=True)
        self.log('val_sigreg_loss', L_sigreg, on_step=False, on_epoch=True)
        self.log('val_loss', L, on_step=False, on_epoch=True, prog_bar=True)

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
    
if __name__ == '__main__':
    import wandb
    from omegaconf import OmegaConf
    from lightning.pytorch.loggers import WandbLogger
    from torch.utils.data import DataLoader
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from dotenv import load_dotenv

    from data import CryptoDataset
    from util import set_logger

    load_dotenv()
    wandb.login()

    cfg = OmegaConf.load('./config.yaml')

    logger = set_logger(cfg)
    logger.info('Starting JEPA training')

    train_dataset = CryptoDataset(cfg, mode = 'training')
    val_dataset = CryptoDataset(cfg, mode = 'validation')

    train_loader = DataLoader(train_dataset, 
                              batch_size = cfg['jepa']['training']['batch_size'],
                              shuffle = True,
                              num_workers = 3)
    val_loader = DataLoader(val_dataset, 
                              batch_size = cfg['jepa']['training']['batch_size'],
                              shuffle = False,
                              num_workers = 3)
    
    model = JEPA(cfg)

    wandb_logger = WandbLogger(
        entity='rudyhuy',
        project='worldmodel' 
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./models/{wandb_logger.experiment.name}/", # Gemmer i ./models/{run_name}/
        filename="jepa",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        check_on_train_epoch_end=False, # Vent altid til valideringen er HELT færdig
        verbose=True
    )

    trainer = L.Trainer(
        max_epochs = cfg['jepa']['training']['epochs'],
        accelerator = "auto", 
        devices = "auto",
        accumulate_grad_batches = 4,
        logger = wandb_logger,
        callbacks = [checkpoint_callback, early_stop_callback],
        log_every_n_steps = cfg['jepa']['training']['log_every_n_steps']
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


    
