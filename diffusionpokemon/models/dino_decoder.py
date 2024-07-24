import torch
import torch.nn as nn

from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule

from typing import Callable


class LinearDecoder(LightningModule):
    "A simple linear decoder that is a single linear layer to project embeddings to image using only patch embeds"

    def __init__(self, inv_transform: Callable, model_kwargs: dict = {}, optimizer_kwargs: dict = {}):
        super().__init__()
        
        self.save_hyperparameters()
        self.patch_size = 14
            
        self.backbone = torch.hub.load('facebookresearch/dinov2', model=model_kwargs["dinov2_backbone"])
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.inv_transform = inv_transform
        
        self.decoder = nn.Sequential(
            nn.Linear(self.get_hidden_dim(model_kwargs["dinov2_backbone"]), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )
        
        self.resizer = nn.Upsample(
            (model_kwargs["recon_image_size"], model_kwargs["recon_image_size"]), 
            align_corners=True, 
            mode="bilinear"
        )
        self.l1_loss = nn.L1Loss(reduction="mean")
        self.optimizer_kwargs = optimizer_kwargs
    
    def get_hidden_dim(self, dinov2_backbone: str):
        size_initial = dinov2_backbone[10]
        if size_initial == "s": return 384 
        if size_initial == "b": return 768
        if size_initial == "l": return 1024
        if size_initial == "g": return 1536
        
    def forward_patch_tokens(self, x):
        batch_size, dim, w, h = x.shape
        features = self.backbone.forward_features(x)
        patch_tokens = torch.reshape(
            torch.movedim(features["x_norm_patchtokens"], 1, -1), 
            (batch_size, -1, w // self.patch_size, h // self.patch_size)
        )
        return patch_tokens
    
    def forward(self, x):
        patch_tokens = self.forward_patch_tokens(x)
        recon = torch.movedim(self.decoder(torch.movedim(self.resizer(patch_tokens), 1, -1)), -1, 1)
        return recon
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon_x_ref = self.resizer(self.inv_transform(x))
        img_tensor_out = self(x)
        loss = self.l1_loss(recon_x_ref, img_tensor_out)
        self.log("train_l1__step", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        img_tensor_out = self(x)
        recon_x_ref = self.resizer(self.inv_transform(x))
        loss = self.l1_loss(recon_x_ref, img_tensor_out)
        self.log("valid_l1__epoch", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer_kwargs["lr"], weight_decay=self.optimizer_kwargs["weight_decay"])
        if self.optimizer_kwargs["use_constant_lr"]:
            return optimizer

        scheduler = ReduceLROnPlateau(optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                "scheduler": scheduler,
                "frequency": self.optimizer_kwargs["lr_sched_freq__step"],
                "interval": "step",
                "monitor": "train_l1__step",
            }
        }

    def on_save_checkpoint(self, checkpoint):
        # Do not save backbone model to save space and speed things up
        to_remove_keys = [
            key for key in checkpoint["state_dict"].keys() if key.startswith("backbone")
        ]

        for key in to_remove_keys:
            checkpoint["state_dict"].pop(key)
