import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet50, ResNet50_Weights
from pytorch_lightning import LightningModule

from typing import Optional, Tuple

class Block(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.c = channel
        self.act = nn.SiLU()
        
        self.conv1 = nn.Conv2d(self.c, self.c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.c)
        self.conv2 = nn.Conv2d(self.c, self.c, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm2d(self.c)
        
    def forward(self, x):
        z = self.conv1(x)
        z = self.act(self.bn1(z))
        z = self.conv2(z)
        z = self.act(self.bn2(z))
        return x + z


class Resnet50Decoder(nn.Module):
    def __init__(
        self, in_latent_shape: Tuple[int, int, int]=(64, 7, 7), dropout_rate: float=0.1, num_blocks: int=1, num_upsamples: int=5, output_img_size: int=224):
        super().__init__()
        
        in_latent_dim, in_latent_size, _ = in_latent_shape

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.intermediate_sizes = list(np.linspace(in_latent_size, output_img_size, num_upsamples + 1).round().astype(int))[1:]
        self.intermediate_channels = list(np.linspace(2 * in_latent_dim, 3, num_upsamples + 2).round().astype(int))[1:-1]

        self.upsamples = nn.ModuleList([
            nn.Upsample(size=size, mode="nearest") for size in self.intermediate_sizes
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(c) for c in self.intermediate_channels
        ])
        self.down_channel_convs = nn.ModuleList([
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
            for in_c, out_c in
            zip([in_latent_dim] + self.intermediate_channels, self.intermediate_channels + [3])
        ])
        self.blocks = nn.ModuleList([
            nn.Sequential(*[Block(out_c) for _ in range(num_blocks)])
            for out_c in self.intermediate_channels
        ])


    def forward(self, x):
        for i in range(len(self.intermediate_channels)):
            x = self.upsamples[i](x)
            x = self.down_channel_convs[i](x)
            x = self.blocks[i](self.bns[i](x))
            x = self.dropout(x)

        x = self.down_channel_convs[-1](x)
        return x


class ResnetVAE(LightningModule):

    ENCODER_LATENT_SPACE_LOOKUP = {
        "resnet50": (2048, 7, 7)
    }

    def __init__(
            self, 
            latent_space_dim: int=64, 
            resnet_ver: str="resnet50", 
            reconstruction_resize_shape: Optional[int]=None, 
            decoder_kwargs: dict={},
            optimizer_kwargs: dict={},
        ):
        super().__init__()
        self.encoder_latent_img_shape = self.ENCODER_LATENT_SPACE_LOOKUP[resnet_ver]
        self.decoder_latent_img_shape = (latent_space_dim, *self.encoder_latent_img_shape[1:])
        self.resnet_ver = resnet_ver
        self.reconstruction_resize_shape = nn.Identity() if reconstruction_resize_shape is None else transforms.Resize(reconstruction_resize_shape, antialias=True)

        self.encoder = self.prepare_frozen_encoder()
        self.decoder = self.prepare_decoder(**decoder_kwargs)

        encoder_latent__channels = self.encoder_latent_img_shape[0]
        self.mu = nn.Conv2d(in_channels=encoder_latent__channels, out_channels=latent_space_dim, kernel_size=1, stride=1, padding=0)
        self.log_var = nn.Conv2d(in_channels=encoder_latent__channels, out_channels=latent_space_dim, kernel_size=1, stride=1, padding=0)
        self.optimizer_kwargs = optimizer_kwargs

    def prepare_decoder(self, **kwargs):
        return Resnet50Decoder(in_latent_shape=self.decoder_latent_img_shape, **kwargs)

    def prepare_frozen_encoder(self):
        if self.resnet_ver == "resnet50":
            encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
            encoder.avgpool = nn.Identity()
            encoder.fc = nn.Identity()
        else:
            raise ValueError(f"{self.resnet_ver} is not a recognised resnet model")

        for param in encoder.parameters():
            param.requires_grad = False

        return encoder

    def forward(self, x):
        x = self.encoder(x).reshape((-1, *self.encoder_latent_img_shape))        
        mu = self.mu(x)
        log_var = self.log_var(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        reconstructed = self.decoder(z)
        return reconstructed, mu, log_var

    def vae_loss_function(self, reconstructed, original, mu, log_var, beta=1.0):
        batch_size = original.size(0)
        reconstruction_loss = F.mse_loss(reconstructed, original, reduction='sum') / batch_size
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
        total_loss = reconstruction_loss + beta * kl_divergence
        return total_loss, reconstruction_loss, kl_divergence

    def training_step(self, batch, batch_index):
        x, _ = batch
        x_recon_resized = self.reconstruction_resize_shape(x)
        reconstructed, mu, log_var = self(x)
        loss, reconstruction_loss, kl_divergence = self.vae_loss_function(reconstructed, x_recon_resized, mu, log_var)
        self.log('train_step__loss', loss)
        self.log('train_step__kl_loss', kl_divergence)
        self.log('train_step__reconstruction_loss', reconstruction_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_recon_resized = self.reconstruction_resize_shape(x)
        reconstructed, mu, log_var = self(x)
        loss, reconstruction_loss, kl_divergence = self.vae_loss_function(reconstructed, x_recon_resized, mu, log_var)
        self.log('valid_epoch__loss', loss, on_step=False, on_epoch=True)
        self.log('valid_epoch__kl_loss', kl_divergence, on_step=False, on_epoch=True)
        self.log('valid_epoch__reconstruction_loss', reconstruction_loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer_kwargs["lr"], weight_decay=self.optimizer_kwargs["weight_decay"])
        if self.optimizer_kwargs["use_constant_lr"]:
            return optimizer

        scheduler = ReduceLROnPlateau(optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                "scheduler": scheduler,
                "frequency": 100,
                "interval": "step",
                "monitor": "train_step__loss",
            }
        }
