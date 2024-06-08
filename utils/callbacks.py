import boto3
import wandb
import torch

from torchvision.transforms import ToPILImage

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import Logger
from pytorch_lightning.callbacks import Callback
from matplotlib import pyplot as plt

from models.ddpm_unet import DDPMUNet

from pathlib import Path
from typing import List, Optional


class SampleCallback(Callback):
    def __init__(self, logger: Logger, freq: int=10, mode: Optional[str]=None):
        super().__init__()
        self.freq = freq
        self.to_pil = ToPILImage(mode=mode)

        assert logger is not None
        self.logger = logger

    def on_train_epoch_end(self, trainer: Trainer, pl_module: DDPMUNet) -> None:
        if ((trainer.current_epoch + 1) % self.freq == 0) or (trainer.current_epoch == trainer.max_epochs - 1):
            img_tensor = pl_module.sample()
            self.logger.log_image(
                key="generated_time_0",
                images=[
                    self.to_pil((img_tensor[j] + 1) / 2)
                    for j in range(img_tensor.shape[0])
                ]
            )


class DenoiseMidwaySampleCallback(Callback):
    def __init__(self, logger: Logger, seed_img_transformed: torch.Tensor, noise_at_ts: List[int], freq: int=10, pil_mode: Optional[str]=None):
        super().__init__()
        self.freq = freq
        self.to_pil = ToPILImage(mode=pil_mode)
        self.seed_img_transformed = seed_img_transformed
        self.seed_img_shape = self.seed_img_transformed.shape
        
        assert len(noise_at_ts) >= 1
        self.noise_at_ts = noise_at_ts

        assert logger is not None
        self.logger = logger

    def on_train_epoch_end(self, trainer: Trainer, pl_module: DDPMUNet) -> None:
        if not ( ((trainer.current_epoch + 1) % self.freq == 0) or (trainer.current_epoch == trainer.max_epochs - 1) ):
            return
        
        original_image = self.to_pil((self.seed_img_transformed+1)/2)
        denoised_imgs = []
        noised_imgs = []
        with torch.no_grad():
            for t in self.noise_at_ts:
                x = torch.unsqueeze(self.seed_img_transformed, 0).to(pl_module.device)
                true_noise_e = torch.randn_like(x).to(pl_module.device)
                t = torch.as_tensor([t], dtype=torch.long).to(pl_module.device)
                noised_x = pl_module.noise_sample_at_timestep(x, t, true_noise_e)

                x = noised_x
                for i in range(t, -1, -1):
                    x = pl_module.sample_one_step(x, t)
                    t -= 1
                
                denoised_imgs.append(self.to_pil(((x[0]+1)/2).cpu()))
                noised_imgs.append(self.to_pil(((noised_x[0]+1)/2).cpu()))

        fig, axs = plt.subplots(nrows=len(self.noise_at_ts), ncols=3, figsize=(6, 2*len(self.noise_at_ts)))
        
        for row, t in enumerate(self.noise_at_ts):
            axs[row][0].imshow(original_image)
            axs[row][1].imshow(noised_imgs[row])
            axs[row][2].imshow(denoised_imgs[row])
            
            axs[row][0].set_title("Original")
            axs[row][1].set_title(f"Noised a t={t}")
            axs[row][2].set_title("Denoised")

        for ax in axs.ravel():
            ax.axis(False)
        
        self.logger.log_image(
            key="denoise_from_seed_img",
            images=[wandb.Image(fig).image]
        )
        
        plt.close()
