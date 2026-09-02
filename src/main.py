import argparse
import logging
import os

import lightning as L
import wandb
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from actor import Actor, ActorAR
from data import CryptoDataset
from jepa import JEPA
from util import set_logger

WANDB_ENTITY = 'rudyhuy'
NUM_WORKERS = 3


def make_loaders(cfg, batch_size, make_action):
    train_dataset = CryptoDataset(cfg, mode='training', make_action=make_action)
    val_dataset = CryptoDataset(cfg, mode='validation', make_action=make_action)

    common = dict(num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0,
                  pin_memory=True)
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **common),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **common),
    )


def resolve_run_id(checkpoint_dir, resume):
    """Persist the wandb run id next to the checkpoint.

    The previous version hardcoded a single run id into both the JEPA and actor
    resume paths, so every resumed run wrote into the same wandb run.
    """
    id_path = os.path.join(checkpoint_dir, 'wandb_id.txt')

    if resume and os.path.exists(id_path):
        with open(id_path) as f:
            return f.read().strip(), 'must'

    run_id = wandb.util.generate_id()
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(id_path, 'w') as f:
        f.write(run_id)
    return run_id, 'allow'


def build_trainer(cfg, tcfg, name, checkpoint_dir, monitor, mode, resume):
    logger = logging.getLogger(cfg['experiment_name'])

    ckpt_path = None
    if resume:
        candidate = os.path.join(checkpoint_dir, 'last.ckpt')
        if os.path.exists(candidate):
            logger.info(f'Resuming training from: {candidate}')
            ckpt_path = candidate
        else:
            logger.warning(f'Resume requested, but checkpoint does not exist: {candidate}')

    run_id, resume_mode = resolve_run_id(checkpoint_dir, resume and ckpt_path is not None)

    wandb_logger = WandbLogger(
        entity=WANDB_ENTITY,
        project=cfg['experiment_name'],
        name=name,
        id=run_id,
        resume=resume_mode,
    )

    # Checkpointing and early stopping watch the SAME metric. Previously the
    # actor saved on val/mean_eq but stopped on val/actor_loss, so it could halt
    # while equity was still improving.
    callbacks = [
        ModelCheckpoint(dirpath=checkpoint_dir, filename='best', monitor=monitor,
                        mode=mode, save_top_k=1, save_last=True),
        EarlyStopping(monitor=monitor, mode=mode, patience=tcfg['patience'], min_delta=0.0),
        LearningRateMonitor(logging_interval='step'),
    ]

    trainer = L.Trainer(
        max_epochs=tcfg['epochs'],
        accelerator='auto',
        devices='auto',
        gradient_clip_val=1.0,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=tcfg['log_every_n_steps'],
    )
    return trainer, ckpt_path


def train_jepa(cfg, resume=False):
    logger = logging.getLogger(cfg['experiment_name'])
    logger.info('Starting JEPA training')

    tcfg = cfg['jepa']['training']
    train_loader, val_loader = make_loaders(cfg, tcfg['batch_size'], make_action=False)

    model = JEPA(cfg)
    logger.info(f'JEPA parameters: {sum(p.numel() for p in model.parameters()):,}')

    trainer, ckpt_path = build_trainer(
        cfg, tcfg, cfg['jepa']['name'], f"./models/{cfg['jepa']['name']}",
        monitor='val/jepa_loss', mode='min', resume=resume,
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


def train_actor(cfg, resume=False):
    logger = logging.getLogger(cfg['experiment_name'])
    logger.info('Starting Actor training')

    tcfg = cfg['actor']['training']
    train_loader, val_loader = make_loaders(cfg, tcfg['batch_size'], make_action=True)

    jepa = JEPA.load_from_checkpoint(f"./models/{cfg['jepa']['name']}/best.ckpt", cfg=cfg)
    model = Actor(cfg, jepa)
    logger.info(f'Actor parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    trainer, ckpt_path = build_trainer(
        cfg, tcfg, cfg['actor']['name'], f"./models/{cfg['actor']['name']}",
        monitor='val/mean_eq', mode='max', resume=resume,
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


def train_actor_ar(cfg, resume=False):
    logger = logging.getLogger(cfg['experiment_name'])
    logger.info('Starting Actor AR fine-tune')

    tcfg = cfg['actor']['training']
    # The AR rollout holds `pred_steps` sequential backbone passes for backprop,
    # so it needs a much smaller batch than the teacher-forced stage.
    train_loader, val_loader = make_loaders(cfg, cfg['actor']['ar']['batch_size'], make_action=True)

    jepa = JEPA.load_from_checkpoint(f"./models/{cfg['jepa']['name']}/best.ckpt", cfg=cfg)
    model = ActorAR.load_from_checkpoint(
        f"./models/{cfg['actor']['name']}/best.ckpt", cfg=cfg, jepa=jepa,
    )

    trainer, ckpt_path = build_trainer(
        cfg, tcfg, cfg['actor']['name'] + '-AR', f"./models/{cfg['actor']['name']}-AR",
        monitor='val/mean_eq', mode='max', resume=resume,
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


def main():
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = OmegaConf.load('./config.yaml')
    cfg = OmegaConf.to_container(cfg, resolve=True)

    set_logger(cfg)
    load_dotenv()
    wandb.login()

    parser = argparse.ArgumentParser(description='Train the world model or the actor.')
    parser.add_argument(
        'mode',
        choices=['jepa', 'actor', 'both', 'actor-ar'],
        help="'jepa' for world-model training, 'actor' for the policy, "
             "'actor-ar' for the autoregressive fine-tune.",
    )
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from the last checkpoint.')

    args = parser.parse_args()

    if args.mode == 'jepa':
        train_jepa(cfg, resume=args.resume)
    elif args.mode == 'actor':
        train_actor(cfg, resume=args.resume)
    elif args.mode == 'both':
        train_jepa(cfg, resume=args.resume)
        train_actor(cfg, resume=args.resume)
    elif args.mode == 'actor-ar':
        train_actor_ar(cfg, resume=args.resume)


if __name__ == '__main__':
    main()
