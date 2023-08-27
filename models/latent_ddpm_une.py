import torch
import torch.nn as nn

from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import OneCycleLR

from .unet import UNet
from .vae import VAE

from tqdm import tqdm


class LatentDDPMUNet(LightningModule):
    
    def __init__(
            self,
            n_steps: int=1_000,
            input_size: int=64,
            unet_kwargs: dict={},
            vae_kwargs: dict={},
            optimizers_kwargs: dict={},
            is_finetune: bool=False,
            is_training_vae: bool=False,
        ):
        super().__init__()
    
        self.n_steps = n_steps
        self.loss = nn.MSELoss()
        self.input_size = input_size 
        self.optimizers_kwargs = optimizers_kwargs
        self.is_finetune = is_finetune
        self.is_training_vae = is_training_vae
        
        self.register_buffer("beta", torch.linspace(1e-4, 0.02, self.n_steps, device=self.device))
        self.register_buffer("alpha", 1 - self.beta)
        self.register_buffer("alpha_bar", torch.cumprod(self.alpha, dim=0))
        
        self.vae_model = VAE(**vae_kwargs)
        # calculate input channels for UNet
        unet_kwargs["input_channels"] = self.vae_model.n_channels * self.vae_model.channels_mult[-1]
        self.eps_model = UNet(**unet_kwargs)

        self.validation_loss_list = []

        # check which type of training
        self.set_finetune(is_finetune)
        self.set_is_training_vae(is_training_vae)

    def set_finetune(self, is_finetune: bool):
        self.is_finetune = is_finetune

        if (self.is_finetune):
            for layer in [
                self.eps_model.image_proj_in,
                self.eps_model.time_embed,
                self.eps_model.down,
                self.vae_model,
            ]:
                for param in layer.parameters():
                    param.requires_grad = False
        
    def set_is_training_vae(self, is_training_vae: bool):
        self.is_training_vae = is_training_vae
        if (self.is_training_vae and not self.is_finetune):
            for param in self.eps_model.parameters():
                param.requires_grad = False
            
    def training_step(self, batch, batch_index):
        if self.is_training_vae:
            raise NotImplementedError
        else:
            self.latent_unet_training_step(batch, batch_index)
        
    def vae_training_step(self, batch, batch_index):
        x, _ = batch # (bs, 3, w, h)

        x_recon, mu, logvar = self.vae_model.forward(x)


    def latent_unet_training_step(self, batch, batch_index):
        # x is image so (bs, 3, w, h)       
        x, _ = batch

        with torch.no_grad():
            x, latent_logvar = self.vae_model.encode(x) # x is latent_mu
        
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
        
        self.log('train_loss_latent_unet', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        t = torch.randint(low=0, high=self.n_steps, size=(x.size(0),), dtype=torch.long, device=self.device)
        true_noise_e = torch.randn_like(x, device=self.device)
        noised_x_t = self.noise_sample_at_timestep(x, t, true_noise_e)
        pred_noise = self.eps_model(noised_x_t, t)
        loss = self.loss(pred_noise, true_noise_e)
        self.validation_loss_list.append(loss)
        
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_loss_list).mean()
        self.log("valid_loss_epoch", avg_loss)
        self.validation_loss_list.clear()
    
    def configure_optimizers(self):

        lr = self.optimizers_kwargs["lr"] if "lr" in self.optimizers_kwargs else 2e-4

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=self.optimizers_kwargs["weight_decay"] if "weight_decay" in self.optimizers_kwargs else 0
        )
        
        if (
            "use_constant_lr" not in self.optimizers_kwargs 
            or self.optimizers_kwargs["use_constant_lr"]
        ):
            return optimizer
        
        # StepLR may lead to too small lr
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=self.optimizers_kwargs["num_epochs"],
            steps_per_epoch=self.optimizers_kwargs["steps_per_epoch"]
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
