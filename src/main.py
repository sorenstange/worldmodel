import lightning as L
import wandb
import argparse
import logging
import os
from omegaconf import OmegaConf
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from dotenv import load_dotenv

from jepa import JEPA, JEPA_AR, JEPA_RL
from data import CryptoDataset
from util import set_logger

def pre_train(cfg, resume=False):
    load_dotenv()
    wandb.login()

    logger = logging.getLogger(cfg['experiment_name'])
    logger.info('Starting JEPA training')

    train_dataset = CryptoDataset(cfg, mode='training')
    val_dataset = CryptoDataset(cfg, mode='validation')

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['jepa']['training']['batch_size'],
        shuffle=True,
        num_workers=3,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['jepa']['training']['batch_size'],
        shuffle=False,
        num_workers=3,
        persistent_workers=True,
    )

    model = JEPA(cfg)
    logger.info(f'Model architecture: {model}')

    wandb_logger = WandbLogger(
        entity='rudyhuy',
        project='jepa',
        name=cfg['jepa']['name'],
    )

    checkpoint_dir = f"./models/jepa/{cfg['jepa']['name']}/"

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="jepa",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Resume from the last checkpoint if requested
    ckpt_path = None

    if resume:
        ckpt_path = f"{checkpoint_dir}last.ckpt"

        if not os.path.exists(ckpt_path):
            logger.warning(
                f"Resume requested, but checkpoint does not exist: {ckpt_path}"
            )
            ckpt_path = None
            wandb_logger = WandbLogger(
                entity='rudyhuy',
                project='jepa',
                name=cfg['jepa']['name'],
            )
        else:
            logger.info(f"Resuming training from: {ckpt_path}")
            wandb_logger = WandbLogger(
                entity='rudyhuy',
                project='jepa',
                name=cfg['jepa']['name'],
                id='ze8315oy',
                resume='must'
            )

    trainer = L.Trainer(
        max_epochs=cfg['jepa']['training']['epochs'],
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=cfg['jepa']['training']['log_every_n_steps'],
    )
    
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )


def ar_train(cfg):
    load_dotenv()
    wandb.login()

    logger = logging.getLogger(cfg['experiment_name'])
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
        max_epochs = cfg['jepa']['ar_training']['epochs'],
        accelerator = "auto", 
        devices = "auto",
        gradient_clip_val = 1.0,
        logger = wandb_logger,
        callbacks = [checkpoint_callback, lr_monitor],
        log_every_n_steps = cfg['jepa']['training']['log_every_n_steps']
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

def rl_train(cfg):
    load_dotenv()
    wandb.login()

    logger = logging.getLogger(cfg['experiment_name'])
    logger.info('Starting RL-JEPA training')

    train_dataset = CryptoDataset(cfg, mode = 'training')
    val_dataset = CryptoDataset(cfg, mode = 'validation')

    train_loader = DataLoader(train_dataset, 
                              batch_size = 24,
                              shuffle = True,
                              num_workers = 3)
    val_loader = DataLoader(val_dataset, 
                              batch_size = 24,
                              shuffle = False,
                              num_workers = 3)

    checkpoint_path = f'./models/jepa/{cfg['jepa']['name']}/last.ckpt'
    model = JEPA_RL.load_from_checkpoint(checkpoint_path)
    logger.info(f'Loaded model from: {checkpoint_path}')
    logger.info(f'Model architecture: {model}')

    wandb_logger = WandbLogger(
            entity='rudyhuy',
            project='jepa',
            name=f'{cfg['jepa']['name']}-RL',
        )    

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./models/jepa/{cfg['jepa']['name']}/", 
        filename="jepa-rl",
        monitor="val/mean_equity",
        mode="max",
        save_top_k=1
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = L.Trainer(
        max_epochs = 1000,
        accelerator = "auto", 
        devices = "auto",
        gradient_clip_val = 1.0,
        logger = wandb_logger,
        callbacks = [checkpoint_callback, lr_monitor],
        log_every_n_steps = 10
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

def main():
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = OmegaConf.load('./config.yaml')
    cfg = OmegaConf.to_container(cfg, resolve=True)

    logger = set_logger(cfg)

    parser = argparse.ArgumentParser(
        description="Simpel CLI til at skifte mellem pre-train og ar-train."
    )
    parser.add_argument(
        "mode",
        choices=["pre-train", "ar-train", 'rl-train'],
        help="Vælg 'pre-train' for initial træning eller 'ar-train' for post-træning.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint.",
    )

    args = parser.parse_args()

    if args.mode == "pre-train":
        pre_train(cfg, resume=args.resume)

    elif args.mode == "ar-train":
        ar_train(cfg)

    elif args.mode == "rl-train":
        rl_train(cfg)

if __name__ == "__main__":
    main()