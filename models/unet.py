import torch

from torch import nn
from diffusionpokemon.models.diffusion_autoencoder_blocks import (
    AttentionBlock,
    Block,
    DownSample,
    ResidualBlock,
    Swish,
    TimeEmbedding,
    UpSample,
)

from typing import List

class DDPMUNet(nn.Module):
    def __init__(
        self, 
        n_channels: int=64, 
        n_blocks: int=2,
        channels_mult: List[int]=[1, 2, 2, 4], 
        is_attn: List[bool]=[False, False, True, False],
        res_block_dropout: float=0.1
    ):
        super().__init__()
        assert channels_mult[0] == 1
        assert len(is_attn) == len(channels_mult)
        time_channels = n_channels * 4
        self.image_proj_in = nn.Conv2d(3, n_channels, kernel_size=(3, 3), padding=(1, 1))
        self.image_proj_out = nn.Conv2d(n_channels, 3, kernel_size=(3, 3), padding=(1, 1))
        self.norm_proj_out = nn.GroupNorm(n_groups=32, n_channels=n_channels)
        self.time_embed = TimeEmbedding(time_channels)
        self.act = Swish()
        
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()

        in_channels = n_channels
        out_channels = n_channels * channels_mult[0]
        intermediary_channels = [in_channels]
        for i in range(len(channels_mult)):
            for _ in range(n_blocks):
                self.down.append(Block(
                    in_channels, out_channels, time_channels, 
                    use_attention=is_attn[i],
                    res_block_dropout=res_block_dropout
                ))
                in_channels = out_channels
                intermediary_channels.append(out_channels)
                
            if i < len(channels_mult) - 1:
                self.down.append(DownSample(out_channels))
                intermediary_channels.append(out_channels)
                out_channels = n_channels * channels_mult[i + 1]

        self.middle =  nn.ModuleList([
            ResidualBlock(out_channels, out_channels, time_channels, dropout=res_block_dropout),
            AttentionBlock(out_channels),
            ResidualBlock(out_channels, out_channels, time_channels, dropout=res_block_dropout),
        ])

        for i in reversed(range(len(channels_mult))):
            for _ in range(n_blocks + 1):
                self.up.append(Block(
                    in_channels + intermediary_channels.pop(),
                    out_channels, time_channels,
                    use_attention=is_attn[i],
                    res_block_dropout=res_block_dropout
                ))
                in_channels = out_channels
            
            if i > 0:
                self.up.append(UpSample(out_channels))
                out_channels = n_channels * channels_mult[i - 1]


    def forward(self, x, t):
        t_embed = self.time_embed(t)
        x = self.image_proj_in(x)
        
        intermediates = [x]
        for layer in enumerate(self.down):
            if isinstance(layer, DownSample):
                x = layer(x)
            else:
                x = layer(x, t_embed)
            intermediates.append(x)
                
        x = self.middle(x, t_embed)
            
        for layer in enumerate(self.up[::-1]):
            if isinstance(layer, UpSample):
                x = layer(x)
            else:
                h = intermediates.pop()
                x = layer(
                    torch.concat((h, x), dim=1),
                    t_embed
                )
            
        x = self.image_proj_out(self.act(self.norm_proj_out(x)))
        return x
