import wandb
import torch
import torchvision.transforms as transforms

from torchvision.transforms import ToPILImage

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from matplotlib import pyplot as plt

from models.ddpm_unet import DDPMUNet

from typing import List, Optional


class SampleCallback(Callback):
    def __init__(self, logger: WandbLogger, freq: int=10, mode: Optional[str]=None):
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
    def __init__(self, logger: WandbLogger, seed_img_transformed: torch.Tensor, noise_at_ts: List[int], freq: int=10, pil_mode: Optional[str]=None):
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
        axs: List[List[plt.Axes]]
        
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


class SampleReconstruction(Callback):
    def __init__(self, logger: WandbLogger, sample_input: torch.Tensor, every_n_epochs: int=100):
        super().__init__()
        self.sample_input = sample_input
        assert len(self.sample_input.shape) == 4, "Please ensure to keep the batch dimension"

        self.logger = logger
        self.inv_normaliser = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
        self.to_pil = transforms.ToPILImage()

        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch % self.every_n_epochs == 0) or (trainer.current_epoch == trainer.max_epochs - 1):
            with torch.no_grad():
                x = self.sample_input.to(pl_module.device)
                reconstructed, mu, log_var = pl_module(x)
                reconstructed = reconstructed.cpu()

            n_images = x.shape[0]
            fig, axs = plt.subplots(nrows=n_images, ncols=2, figsize=(2, n_images))
            for i in range(x.shape[0]):
                orig_img = self.to_pil(self.inv_normaliser(self.sample_input[i]))
                reconstructed_img = self.to_pil(self.inv_normaliser(reconstructed[i]))
                axs[i, 0].imshow(orig_img)
                axs[i, 1].imshow(reconstructed_img)
            for ax in axs.ravel():
                ax.axis(False)

            self.logger.log_image(
                key="sample_reconstruction",
                images=[wandb.Image(fig).image]
            )

            plt.close()
