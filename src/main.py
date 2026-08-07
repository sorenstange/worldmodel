import lightning as L
import wandb
import argparse
from omegaconf import OmegaConf
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from dotenv import load_dotenv

from jepa import JEPA, JEPA_AR
from data import CryptoDataset
from util import set_logger

def pre_train():
    load_dotenv()
    wandb.login()

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = OmegaConf.load('./config.yaml')
    cfg = OmegaConf.to_container(cfg, resolve=True)

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
    logger.info(f'Model architecture: {model}')

    wandb_logger = WandbLogger(
            entity='rudyhuy',
            project='jepa',
            name=cfg['jepa']['name'],
        )    

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./models/jepa/{cfg['jepa']['name']}/", 
        filename="jepa",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = L.Trainer(
        max_epochs = cfg['jepa']['training']['epochs'],
        accelerator = "auto", 
        devices = "auto",
        gradient_clip_val = 1.0,
        logger = wandb_logger,
        callbacks = [checkpoint_callback, lr_monitor],
        log_every_n_steps = cfg['jepa']['training']['log_every_n_steps']
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

def ar_train():
    load_dotenv()
    wandb.login()

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = OmegaConf.load('./config.yaml')
    cfg = OmegaConf.to_container(cfg, resolve=True)

    logger = set_logger(cfg)
    logger.info('Starting AR-JEPA training')

    train_dataset = CryptoDataset(cfg, mode = 'training')
    val_dataset = CryptoDataset(cfg, mode = 'validation')

    train_loader = DataLoader(train_dataset, 
                              batch_size = cfg['jepa']['ar_training']['batch_size'],
                              shuffle = True,
                              num_workers = 3)
    val_loader = DataLoader(val_dataset, 
                              batch_size = cfg['jepa']['ar_training']['batch_size'],
                              shuffle = False,
                              num_workers = 3)

    checkpoint_path = f'./models/jepa/{cfg['jepa']['name']}/last.ckpt'
    model = JEPA_AR.load_from_checkpoint(checkpoint_path)
    logger.info(f'Loaded model from: {checkpoint_path}')
    logger.info(f'Model architecture: {model}')

    wandb_logger = WandbLogger(
            entity='rudyhuy',
            project='jepa',
            name=f'{cfg['jepa']['name']}-AR',
        )    

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./models/jepa/{cfg['jepa']['name']}/", 
        filename="jepa-ar",
        monitor="val/mean_equity",
        mode="max",
        save_top_k=1
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = L.Trainer(
        max_epochs = cfg['jepa']['training']['epochs'],
        accelerator = "auto", 
        devices = "auto",
        gradient_clip_val = 1.0,
        logger = wandb_logger,
        callbacks = [checkpoint_callback, lr_monitor],
        log_every_n_steps = cfg['jepa']['training']['log_every_n_steps']
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

def main():
    parser = argparse.ArgumentParser(
        description="Simpel CLI til at skifte mellem pre-train og ar-train."
    )
    parser.add_argument(
        "mode",
        choices=["pre-train", "ar-train"],
        help="Vælg 'pre-train' for initial træning eller 'ar-train' for post-træning.",
    )

    args = parser.parse_args()

    if args.mode == "pre-train":
        pre_train()

    elif args.mode == "ar-train":
        ar_train()


if __name__ == "__main__":
    main()