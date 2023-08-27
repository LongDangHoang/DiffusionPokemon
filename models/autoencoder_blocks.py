import torch
import torch.nn as nn
import math

from typing import Optional

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

    
class TimeEmbedding(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        
        self.n_channels = n_channels
        
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        log_denom = (
            -math.log(10_000) 
            / (half_dim - 1)
            * torch.arange(half_dim, device=t.device)
        )
        emb = t[:, None] * torch.exp(log_denom)[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        
        return emb

    
class ResidualBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        time_channels: Optional[int]=None, 
        n_groups: int=32, 
        dropout: float=0.1,
    ):
        super().__init__()
        self.act = Swish()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.dropout = nn.Dropout(dropout)
            
        if time_channels:
            self.time_emb = nn.Linear(time_channels, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor]=None):
        h = self.conv1(self.act(self.norm1(x)))
        
        if t is not None and hasattr(self, "time_emb"):
            h += self.time_emb(self.act(t))[:, :, None, None]
            
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + self.shortcut(x)
    

class AttentionBlock(nn.Module):
    def __init__(self, n_channels: int, n_heads: int=1, d_k: int=None, n_groups: int=32):
        super().__init__()
        
        if not d_k:
            d_k = n_channels
        
        self.d_k = d_k
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.project = nn.Linear(n_channels, n_heads * d_k * 3)
        self.group_norm = nn.GroupNorm(n_groups, n_channels)
        self.out = nn.Linear(n_heads * d_k, n_channels)
        self.act = Swish()
        
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor]=None):
        batch_size, n_channels, height, width = x.shape
        x = self.group_norm(x)
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        
        qkv = self.project(x).view(batch_size, -1, self.n_heads, self.d_k * 3)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn: torch.Tensor = torch.einsum('bihk,bjhk->bijh', q, k) * self.d_k ** 0.5
        attn = attn.softmax(dim=2)
        h: torch.Tensor = torch.einsum('bjhk,bijh->bihk', v, attn)
        h = h.view(batch_size, -1, self.n_heads * self.d_k)
        h = self.act(self.out(h))
        h += x
        h = h.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return h
    
    
class Block(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int, 
        time_channels: Optional[int]=None, 
        use_attention: bool=True,
        res_block_dropout: float=0.1,
    ):
        super().__init__()
        
        self.res_block = ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            time_channels=time_channels,
            dropout=res_block_dropout
        )
        self.attn_block = (
            AttentionBlock(
                n_channels=out_channels
            )
            if use_attention
            else nn.Identity()
        )
        
        
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor]=None):
        x = self.res_block(x, t)
        x = self.attn_block(x)
        return x
    
    
class UpSample(nn.Module):
    def __init__(self, n_channels: int, model_type: str="complex"):
        super().__init__()

        if model_type == "complex":
            self.upsample = nn.ConvTranspose2d(
                n_channels, n_channels,
                (4, 4), (2, 2), (1, 1)
            )
        else:
            self.upsample = nn.Upsample(scale_factor=2)
    
    def forward(self, x):
        return self.upsample(x)
    

class DownSample(nn.Module):
    def __init__(self, n_channels: int, model_type: str="complex"):
        super().__init__()

        if model_type == "complex":
            self.downsample = nn.Conv2d(
                n_channels, n_channels,
                (3, 3), (2, 2), (1, 1)
            )
        else:
            self.downsample = nn.AvgPool2d(
                kernel_size=2, stride=2, padding=0
            )
    
    def forward(self, x):
        return self.downsample(x)
