import torch

from torch import nn
from .autoencoder_blocks import (
    Block,
    DownSample,
    Swish,
    UpSample,
)

from typing import List

# Define the Variational Autoencoder model
class VAE(nn.Module):
    def __init__(
        self,
        input_channels: int=3, 
        n_channels: int=64, 
        n_blocks: int=2,
        channels_mult: List[int]=[1, 2, 2, 4], 
        is_attn: List[bool]=[False, False, True, False],
        res_block_dropout: float=0.1,
        model_type: str="complex",
    ):
        super(VAE, self).__init__()
        assert channels_mult[0] == 1
        assert len(is_attn) == len(channels_mult)

        self.channels_mult = channels_mult
        self.n_channels = n_channels
        self.is_attn = is_attn

        self.image_proj_in = nn.Conv2d(input_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))
        self.image_proj_out = nn.Conv2d(n_channels, input_channels, kernel_size=(3, 3), padding=(1, 1))
        self.norm_proj_out = nn.GroupNorm(num_groups=32, num_channels=n_channels)

        self.act = Swish()
        
        self.down = nn.ModuleList([self.image_proj_in])
        self.up = nn.ModuleList()

        in_channels = n_channels
        out_channels = n_channels * channels_mult[0]
        for i in range(len(channels_mult)):
            for _ in range(n_blocks):
                self.down.append(Block(
                    in_channels, out_channels, None, 
                    use_attention=is_attn[i],
                    res_block_dropout=res_block_dropout
                ))
                in_channels = out_channels
                
            if i < len(channels_mult) - 1:
                self.down.append(DownSample(out_channels, model_type=model_type))
                out_channels = n_channels * channels_mult[i + 1]

                if i == len(channels_mult) - 2:
                    out_channels = out_channels * 2 # both mean and var

        in_channels = in_channels // 2 # don't need both mean and var
        out_channels = out_channels // 2
        for i in reversed(range(len(channels_mult))):
            for _ in range(n_blocks):
                self.up.append(Block(
                    in_channels, out_channels, None,
                    use_attention=is_attn[i],
                    res_block_dropout=res_block_dropout
                ))
                in_channels = out_channels
            
            if i > 0:
                self.up.append(UpSample(out_channels, model_type=model_type))
                out_channels = n_channels * channels_mult[i - 1]
        self.up.extend([self.norm_proj_out, self.image_proj_out])
    
    def encode(self, x):
        h = self.down(x)
        mu, log_var = torch.chunk(h, 2, dim=1)  # Split into mean and log variance
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        return self.up(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var
    
    def loss_function(self, x, x_recon, mu, log_var):
        recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kl_divergence
