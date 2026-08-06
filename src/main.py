import wandb
from omegaconf import OmegaConf
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from dotenv import load_dotenv

from jepa import JEPA
from data import CryptoDataset
from util import set_logger

if __name__ == '__main__':
    CONTINUE = False

    load_dotenv()
    wandb.login()

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    
    cfg = OmegaConf.load('./config.yaml')
    cfg = OmegaConf.to_container(cfg, resolve=True)
    
    checkpoint_path = f"./models/jepa/{cfg['jepa']['name']}/last.ckpt"

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

    if CONTINUE:
        original_load = torch.load
        torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})

        wandb_logger = WandbLogger(
            entity='rudyhuy',
            project='jepa',
            name=cfg['jepa']['name'],
            id='yf3e2m64',    
            resume='must'          
        )
        fit_ckpt_path = checkpoint_path
    else:
        wandb_logger = WandbLogger(
            entity='rudyhuy',
            project='jepa',
            name=cfg['jepa']['name'],
        )    
        fit_ckpt_path = None

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

    # 2. Send din betingede fit_ckpt_path med her
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=fit_ckpt_path)
