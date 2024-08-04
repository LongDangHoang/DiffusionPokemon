import torch
import torch.nn as nn

from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

from diffusionpokemon.models.unet import UNet

from tqdm import tqdm
from typing import Tuple

class DDPMModel(LightningModule):
    
    def __init__(
            self,
            n_steps: int=1_000,
            input_size: int=64,
            eps_model_kwargs: dict={},
            optimizers_kwarg: dict={},
            is_finetune: bool=False
        ):
        super().__init__()
    
        self.n_steps = n_steps
        self.loss = nn.MSELoss()
        self.input_size = input_size 
        self.optimizer_kwargs = optimizers_kwarg
        self.is_finetune = is_finetune
        
        self.register_buffer("beta", torch.linspace(1e-4, 0.02, self.n_steps, device=self.device))
        self.register_buffer("alpha", 1 - self.beta)
        self.register_buffer("alpha_bar", torch.cumprod(self.alpha, dim=0))
        
        self.eps_model = self.get_eps_model(**eps_model_kwargs)
        self.validation_loss_list = []

        if (self.is_finetune):
            for layer in [
                self.eps_model.image_proj_in,
                self.eps_model.time_embed,
                self.eps_model.down
            ]:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def get_eps_model(self, eps_model_kwargs):
        raise NotImplementedError
            
    def training_step(self, batch, batch_index):
        # x is x0 so (bs, dim, w, h)       
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
        
        self.log('train_loss__step', loss, on_step=True, on_epoch=False, logger=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        t = torch.randint(low=0, high=self.n_steps, size=(x.size(0),), dtype=torch.long, device=self.device)
        true_noise_e = torch.randn_like(x, device=self.device)
        noised_x_t = self.noise_sample_at_timestep(x, t, true_noise_e)
        pred_noise = self.eps_model(noised_x_t, t)
        loss = self.loss(pred_noise, true_noise_e)
        self.log("valid_loss__epoch", loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
    
    def configure_optimizers(self):
        lr = self.optimizer_kwargs["lr"] if "lr" in self.optimizer_kwargs else 2e-4

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=self.optimizer_kwargs["weight_decay"] if "weight_decay" in self.optimizer_kwargs else 0
        )
        
        if (
            "use_constant_lr" not in self.optimizer_kwargs 
            or self.optimizer_kwargs["use_constant_lr"]
        ):
            return optimizer
        
        scheduler = ReduceLROnPlateau(optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                "scheduler": scheduler,
                "frequency": self.optimizer_kwargs["lr_sched_freq__step"],
                "interval": "step",
                "monitor": "train_loss__step",
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
    
    def sample(self, batch_size: int=4):
        with torch.no_grad():
            x = torch.randn((batch_size, 3, self.input_size, self.input_size), device=self.device)
            for i in tqdm(range(self.n_steps - 1, -1, -1)):
                t = i * torch.ones((batch_size,), device=self.device, dtype=torch.long)
                x = self.sample_one_step(x, t)
        
        return x
