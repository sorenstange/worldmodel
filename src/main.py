import lightning as L
import wandb
import argparse
import logging
import os
import json

from omegaconf import OmegaConf
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from dotenv import load_dotenv

from jepa import JEPA
from actor import Actor, ActorAR
from data import CryptoDataset
from util import set_logger

def train_jepa(cfg, resume=False):
    load_dotenv()
    wandb.login()

    logger = logging.getLogger(cfg['experiment_name'])
    logger.info('Starting JEPA training')

    train_dataset = CryptoDataset(cfg, mode='training', make_action=False)
    val_dataset = CryptoDataset(cfg, mode='validation', make_action=False)

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

    checkpoint_dir = f"./models/{cfg['jepa']['name']}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor="val/jepa_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val/jepa_loss",
        mode="min",
        patience=cfg['jepa']['training']['patience'],
        min_delta=0.0,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    ckpt_path = None
    if resume:
        ckpt_path = f"{checkpoint_dir}/last.ckpt"

        if not os.path.exists(ckpt_path):
            logger.warning(
                f"Resume requested, but checkpoint does not exist: {ckpt_path}"
            )
            ckpt_path = None
            wandb_logger = WandbLogger(
                entity='rudyhuy',
                project=cfg['experiment_name'],
                name=cfg['jepa']['name'],
            )
        else:
            logger.info(f"Resuming training from: {ckpt_path}")
            wandb_logger = WandbLogger(
                entity='rudyhuy',
                project=cfg['experiment_name'],
                name=cfg['jepa']['name'],
                id='81bnjzdm',
                resume='must'
            )
    else:
        ckpt_path = None
        wandb_logger = WandbLogger(
            entity='rudyhuy',
            project=cfg['experiment_name'],
            name=cfg['jepa']['name'],
        )

    trainer = L.Trainer(
        max_epochs=cfg['jepa']['training']['epochs'],
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        log_every_n_steps=cfg['jepa']['training']['log_every_n_steps'],
    )
    
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )

def train_actor(cfg, resume=False):
    load_dotenv()
    wandb.login()

    logger = logging.getLogger(cfg['experiment_name'])
    logger.info('Starting Actor training')

    train_dataset = CryptoDataset(cfg, mode='training', make_action=True)
    val_dataset = CryptoDataset(cfg, mode='validation', make_action=True)

    jepa = JEPA.load_from_checkpoint(f'./models/{cfg['jepa']['name']}/best.ckpt')

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['actor']['training']['batch_size'],
        shuffle=True,
        num_workers=3,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['actor']['training']['batch_size'],
        shuffle=False,
        num_workers=3,
        persistent_workers=True,
    )

    model = Actor(cfg, jepa)
    logger.info(f'Model architecture: {model}')

    checkpoint_dir = f"./models/{cfg['actor']['name']}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor="val/mean_eq",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val/actor_loss",
        mode="min",
        patience=cfg['actor']['training']['patience'],
        min_delta=0.0,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    ckpt_path = None
    if resume:
        ckpt_path = f"{checkpoint_dir}/last.ckpt"

        if not os.path.exists(ckpt_path):
            logger.warning(
                f"Resume requested, but checkpoint does not exist: {ckpt_path}"
            )
            ckpt_path = None
            wandb_logger = WandbLogger(
                entity='rudyhuy',
                project=cfg['experiment_name'],
                name=cfg['actor']['name'],
            )
        else:
            logger.info(f"Resuming training from: {ckpt_path}")
            wandb_logger = WandbLogger(
                entity='rudyhuy',
                project=cfg['experiment_name'],
                name=cfg['actor']['name'],
                id='81bnjzdm',
                resume='must'
            )
    else:
        ckpt_path = None
        wandb_logger = WandbLogger(
            entity='rudyhuy',
            project=cfg['experiment_name'],
            name=cfg['actor']['name'],
        )

    trainer = L.Trainer(
        max_epochs=cfg['actor']['training']['epochs'],
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        log_every_n_steps=cfg['actor']['training']['log_every_n_steps'],
    )
    
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )

def train_actor_ar(cfg):
    load_dotenv()
    wandb.login()

    logger = logging.getLogger(cfg['experiment_name'])
    logger.info('Starting Actor AR training')

    train_dataset = CryptoDataset(cfg, mode='training', make_action=True)
    val_dataset = CryptoDataset(cfg, mode='validation', make_action=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=3,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=3,
        persistent_workers=True,
    )

    jepa = JEPA.load_from_checkpoint(f'./models/{cfg['jepa']['name']}/best.ckpt')
    model = ActorAR.load_from_checkpoint(f'./models/{cfg['actor']['name']}/best.ckpt', cfg=cfg, jepa=jepa)
    logger.info(f'Model architecture: {model}')

    checkpoint_dir = f"./models/{cfg['actor']['name']}-AR"

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor="val/actor_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val/actor_loss",
        mode="min",
        patience=cfg['actor']['training']['patience'],
        min_delta=0.0,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    ckpt_path = None
    wandb_logger = WandbLogger(
        entity='rudyhuy',
        project=cfg['experiment_name'],
        name=cfg['actor']['name'] + '-AR',
    )

    trainer = L.Trainer(
        max_epochs=cfg['actor']['training']['epochs'],
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        log_every_n_steps=cfg['actor']['training']['log_every_n_steps'],
    )
    
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )


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
        choices=['jepa', 'actor', 'both', 'actor-ar'],
        help="Choose 'jepa' for jepa training or 'actor' for actor training",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint.",
    )

    args = parser.parse_args()

    if args.mode == 'jepa':
        train_jepa(cfg, resume=args.resume)

    elif args.mode == "actor":
        train_actor(cfg, resume=args.resume)
    
    elif args.mode == 'both':
        train_jepa(cfg, resume=args.resume)
        train_actor(cfg, resume=args.resume)
    
    elif args.mode == 'actor-ar':
        train_actor_ar(cfg)


if __name__ == "__main__":
    main()