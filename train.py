import boto3
        
import os
import torch
import torch.nn as nn

import math
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path

from diffusionpokemon.models  import (
    Block,
    UpSample, 
    DownSample, 
    ResidualBlock, 
    AttentionBlock, 
    Swish, 
    TimeEmbedding,
    DDPMUNet
)

from diffusionpokemon.datasets  import (
    BaseDataset,
    CartoonPretrainDataset,
    Scaler
)

from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, OneCycleLR

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, Callback, ModelCheckpoint

torch.manual_seed(314)
torch.cuda.manual_seed_all(314)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")


# Define hyperparameters
from dotenv import load_dotenv
load_dotenv()
wandb_api_key = os.environ["wandb_api"]
os.environ["AWS_ACCESS_KEY_ID"] = os.environ["s3_aws_access_key"]
os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["s3_aws_secret_access_key"]

import wandb

batch_size = 16
use_constant_lr = False
lr = 2e-4
pokemon_num_epoch = 500
overfit_batch = 0
weight_decay = 0
model_type = 'complex'
infinite_patience = True
verbose = False
limit_data_size = None
dropout = 0.1
channels_mult = [1, 4, 6]
is_attn = [True, True, True]
n_blocks = 1
n_steps = 1_000
log_wandb = True
ema_decay_factor = None
use_existing_run = "5h05i3q7"
init_new_wandb_run = False

# start a new wandb run to track this script
wandb.login(key=wandb_api_key)

if "run" not in globals() and log_wandb:
    run = wandb.init(
        project="diffusion-pokemon-lightning",
        id=use_existing_run if (use_existing_run and not init_new_wandb_run) else None,
        resume="must" if (use_existing_run and not init_new_wandb_run) else None,
        config={
            "batch_size": batch_size,
            "limit_data_size": limit_data_size,
            "use_constant_lr": use_constant_lr,
            "infinite_patience": infinite_patience,
            "pokemon_num_epoch": pokemon_num_epoch,
            "dropout": dropout,
            "model_type": model_type,
            "channels_mult": channels_mult,
            "is_attn": is_attn,
            "n_steps": n_steps,
            "ema_decay_factor": ema_decay_factor,
            "lr": lr,
            "n_blocks": n_blocks,
            "weight_decay": weight_decay,
            "overfit_batch": overfit_batch
        }
    )
    assert run is not None

# some common things
to_pil = transforms.ToPILImage()


# # Get data
# 
# ## Pokemon dataset


pokemon_dataset = BaseDataset()

pokemon_dataset.load(
    Path('/kaggle/input/pokemon-image-dataset/images'),
    batch_size=batch_size,
    limit_data_size=limit_data_size
)

pokemon_dataset.visualise()


# split train valid
generator1 = torch.Generator().manual_seed(42)

pokemon_train_dataset, pokemon_valid_dataset = torch.utils.data.random_split(
    pokemon_dataset.dataset, 
    [0.8, 0.2], 
    generator=generator1
)

print(f"Train: {len(pokemon_train_dataset)}, Valid: {len(pokemon_valid_dataset)}")

pokemon_train_loader = torch.utils.data.DataLoader(
    pokemon_train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

pokemon_valid_loader = torch.utils.data.DataLoader(
    pokemon_valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

pokemon_img_size = pokemon_train_dataset[0][0].shape[2]
print(f"Pokemon image size: {pokemon_img_size}")


# # Define models


def ema_avg_fn_factory(decay_factor):
    ema_fn = lambda averaged_model_parameter, model_parameter, num_averaged:\
            (1 - decay_factor) * averaged_model_parameter + decay_factor * model_parameter
    return ema_fn


class DiffusionModel(LightningModule):
    
    def __init__(self, n_steps: int=1_000, **kwargs):
        super().__init__()
    
        self.n_steps = n_steps
        self.loss = nn.MSELoss()
        self.input_size = pokemon_img_size
        
        self.register_buffer("beta", torch.linspace(1e-4, 0.02, self.n_steps, device=self.device))
        self.register_buffer("alpha", 1 - self.beta)
        self.register_buffer("alpha_bar", torch.cumprod(self.alpha, dim=0))
        
        self.eps_model = DDPMUNet(**kwargs)
        
        if ema_decay_factor:
            self.eps_averaged_model = DDPMUNet(**kwargs)
            self.eps_averaged_model.load_state_dict(self.eps_model.state_dict())
            
        self.loss_list = []
            
    def training_step(self, batch, batch_index):
        # x is x0 so (bs, 3, w, h)       
        x, _ = batch
        
        # we want to sample a random x_t -> x_t-1 time      
        t = torch.randint(low=0, high=self.n_steps, size=(x.size(0),), dtype=torch.long, device=self.device)
        
        # compute an original noise from gaussian
        true_noise_e = torch.randn_like(x, device=self.device)
        
        # noised sample at time t is
        noised_x_t = self.noise_sample_at_timestep(x, t, true_noise_e)
        
        # we want our model to be able to predict what the noise at time step t is using 
        # a more noised version of the data. If the model can do so accurately, it
        # has captured the dynamics of the reverse process on the input data
        pred_noise = self.eps_model(noised_x_t, t)
        loss = self.loss(pred_noise, true_noise_e)
        
        self.log('train_loss', loss)
        optimizer = self.optimizers()
        self.log("lr", optimizer.param_groups[0]["lr"])
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        t = torch.randint(low=0, high=self.n_steps, size=(x.size(0),), dtype=torch.long, device=self.device)
        true_noise_e = torch.randn_like(x, device=self.device)
        noised_x_t = self.noise_sample_at_timestep(x, t, true_noise_e)
        pred_noise = self.eps_model(noised_x_t, t)
        loss = self.loss(pred_noise, true_noise_e)
        self.loss_list.append(loss)
        
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.loss_list).mean()
        self.log("valid_loss_epoch", avg_loss)
        self.loss_list.clear()

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        
        if ema_decay_factor:
            with torch.no_grad():
                for (avg_n, avg_p), (n, p) in zip(
                    self.eps_averaged_model.named_parameters(), 
                    self.eps_model.named_parameters()
                ):
                    avg_p -= ema_decay_factor * (avg_p - p)
                    p -= p - avg_p
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        if use_constant_lr:
            return optimizer
        
        # StepLR may lead to too small lr
        scheduler = OneCycleLR(
            optimizer, max_lr=lr, epochs=pokemon_num_epoch, steps_per_epoch=len(pokemon_train_loader)
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                "scheduler": scheduler,
                "frequency": 1,
                "interval": "step"
            }
        }
        
    def noise_sample_at_timestep(self, x_zero, timestep_t, noise_e):
        # using the gaussian reparamerisation trick, compute the cumulative variance up to t
        a_t = torch.gather(self.alpha_bar, 0, timestep_t).reshape(-1, 1, 1, 1)
        return torch.sqrt(a_t) * x_zero + torch.sqrt(1 - a_t) * noise_e
    
    def sample_one_step(self, prev_x: torch.Tensor, prev_t: torch.Tensor):
        pred_noise = self.eps_model(prev_x, prev_t)
        beta = torch.gather(self.beta, 0, prev_t).reshape(-1, 1, 1, 1)
        alpha = 1. - beta
        alpha_bar = torch.gather(self.alpha_bar, 0, prev_t).reshape(-1, 1, 1, 1)
        eps_coef = beta / torch.sqrt(1 - alpha_bar)
        mean = (1 / torch.sqrt(alpha)) * (prev_x - eps_coef * pred_noise)
        var = torch.gather(self.beta, 0, prev_t).reshape(-1, 1, 1, 1)
        z = torch.randn(prev_x.shape, device=prev_x.device)
        z *= torch.where(prev_t == 0, 0, 1).reshape(-1, 1, 1, 1)
        sampled = z * (torch.sqrt(var)) + mean
        return sampled
    
    def sample(self):
        with torch.no_grad():
            x = torch.randn((16, 3, self.input_size, self.input_size), device=self.device)
            for i in tqdm(range(self.n_steps - 1, -1, -1)):
                t = i * torch.ones((16,), device=self.device, dtype=torch.long)
                x = self.sample_one_step(x, t)
        
        return x.cpu()
    
class SampleCallback(Callback):
    def __init__(self, freq: int=10):
        super().__init__()
        self.freq = freq

    def on_train_epoch_end(self, trainer: Trainer, pl_module: DiffusionModel) -> None:
        if ((trainer.current_epoch + 1) % self.freq == 0) or (trainer.current_epoch == trainer.max_epochs - 1):
            img_tensor = pl_module.sample()

            if log_wandb:
                wandb.log({
                    "generated_time_0": [
                        wandb.Image(to_pil((img_tensor[j] + 1) / 2))
                        for j in range(img_tensor.shape[0])]
                })


class S3SyncCallback(Callback):
    """
    Synchronise checkpoint folder with bucket
    """
    def __init__(self, local_dir: Path) -> None:
        self.s3 = boto3.resource("s3")
        self.bucket_name = 'longdang-deep-learning-personal-projects'
        self.bucket = self.s3.Bucket(self.bucket_name)
        self.local_dir = local_dir
        self.s3_key = str(self.local_dir.absolute().relative_to(Path('.').resolve()))
        
    def on_train_epoch_end(self, trainer: Trainer, pl_module: DiffusionModel) -> None:
        self.upload_files_to_s3()
        
    def download_files_from_s3(self):
        for file in self.bucket.objects.filter(Prefix=self.s3_key):
            filename = file.key.split("/")[-1]
            self.bucket.download_file(file.key, self.local_dir / filename)
            
    def upload_files_to_s3(self):
        self.delete_folder_on_s3()
        for file in self.local_dir.rglob("*"):
            key = str(file.absolute().relative_to(Path(".").resolve()))
            self.bucket.upload_file(file, key)
            
    def delete_folder_on_s3(self):
        for file in self.bucket.objects.filter(Prefix=self.s3_key):
            self.s3.Object(self.bucket_name, file.key).delete()


# # Init model


model = DiffusionModel(
    n_steps=n_steps, # the more the better for the naive sampling method
    n_blocks=n_blocks, 
    channels_mult=channels_mult, 
    is_attn=is_attn,
    res_block_dropout=dropout, # disable dropout
).to(device)
print("Number of params: ", sum(p.numel() for p in model.eps_model.parameters()))


# # Train


# pretraining_logger = WandbLogger(
#     project="diffusion-pokemon-lightning",
#     prefix="pretrain"
# )

# try:
#     pretraining_logger.watch(model)
# except ValueError as e:
#     if "You can only call `wandb.watch` once per model." not in str(e):
#         raise e

# trainer = Trainer(
#     accelerator="gpu" if device == "cuda" else "cpu",
#     devices=1, 
#     max_epochs=cartoon_num_epoch, 
#     log_every_n_steps=1,
#     precision=32,
# #     accumulate_grad_batches=4,
#     logger=pretraining_logger
# )
# trainer.fit(model, cartoon_loader)


pokemon_logger = WandbLogger(
    project="diffusion-pokemon-lightning",
    prefix="pokemon"
)

try:
    pokemon_logger.watch(model)
except ValueError as e:
    if "You can only call `wandb.watch` once per model." not in str(e):
        raise e

sample_callback = SampleCallback(freq=16)

ckpt_local_dir = Path(f"./diffusion-pokemon-lightning/{run.id}/checkpoints")
if not ckpt_local_dir.exists():
    ckpt_local_dir.mkdir(parents=True)

checkpoint_callback = ModelCheckpoint(
    dirpath=ckpt_local_dir,
    filename="epoch={epoch}-step={step}",
    save_last=True
)

s3_sync_callback = S3SyncCallback(ckpt_local_dir)
if use_existing_run:
    s3_sync_callback.download_files_from_s3() # download existing ckpts
        
trainer = Trainer(
    accelerator="gpu" if device == "cuda" else "cpu", 
    devices=1, 
    max_epochs=pokemon_num_epoch,
    log_every_n_steps=20,
    precision=32,
    logger=pokemon_logger if log_wandb else None,
    callbacks=[sample_callback, checkpoint_callback],
    accumulate_grad_batches=8,
    overfit_batches=overfit_batch,
    max_time="00:10:00:00" # timeout before kaggle
)
trainer.fit(
    model, 
    train_dataloaders=pokemon_train_loader,
    val_dataloaders=pokemon_valid_loader, 
    ckpt_path=(ckpt_local_dir/"last.ckpt") if use_existing_run else None
)


# save to s3
s3_sync_callback.upload_files_to_s3()


wandb.finish()


