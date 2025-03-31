import wandb
import torch
import torchvision.transforms as transforms

from torchvision.transforms import ToPILImage

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from matplotlib import pyplot as plt

from diffusionpokemon.models.ddpm_unet import DDPMModel

from typing import Callable, List, Optional


class SampleCallback(Callback):
    def __init__(
        self, 
        logger: WandbLogger, 
        inv_normaliser: Callable, 
        every_n_epochs: Optional[int]=None, 
        every_n_steps: Optional[int]=100,
        mode: Optional[str]=None, 
        batch_size: int=4,
        input_channels: int=3,
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.every_n_steps = every_n_steps
        self.last_log_at = None
        self.freq_type = "steps"

        if self.every_n_steps is None:
            assert self.every_n_epochs is not None
            self.freq_type == "epochs"
        
        self.to_pil = ToPILImage(mode=mode)
        self.inv_normaliser = inv_normaliser
        self.batch_size = batch_size
        self.input_channels = input_channels

        assert logger is not None
        self.logger = logger

    def sample_and_log_image(self, pl_module: DDPMModel):
        img_tensor = pl_module.sample(batch_size=self.batch_size, input_channels=self.input_channels).cpu()
        self.logger.log_image(
            key="generated_time_0",
            images=[
                self.to_pil(self.inv_normaliser(img_tensor[j]))
                for j in range(img_tensor.shape[0])
            ]
        )

    def on_train_batch_end(self, trainer, pl_module: DDPMModel, outputs, batch, batch_idx) -> None:
        if self.freq_type == "epochs":
            return

        if (
            ((trainer.global_step + 1) % self.every_n_steps == 0)
            and (self.last_log_at != trainer.global_step)
        ):
            self.sample_and_log_image(pl_module)
            self.last_log_at = trainer.global_step
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: DDPMModel) -> None:
        if self.freq_type != "epochs":
            return
            
        if (
            ((trainer.current_epoch + 1) % self.every_n_epochs == 0)
            and (self.last_log_at != trainer.current_epoch)
        ):
            self.sample_and_log_image(pl_module)
            self.last_log_at = trainer.current_epoch

    def on_train_end(self, trainer: Trainer, pl_module: DDPMModel) -> None:
        self.sample_and_log_image(pl_module)


class DenoiseMidwaySampleCallback(Callback):
    def __init__(
        self, 
        logger: WandbLogger, 
        seed_img_transformed: torch.Tensor, 
        noise_at_ts: List[int], 
        inv_normaliser: Callable, 
        every_n_epochs: Optional[int]=None,
        every_n_steps: Optional[int]=100,
        pil_mode: Optional[str]=None
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.every_n_steps = every_n_steps
        self.freq_type = "steps"
        self.last_log_at = None

        if self.every_n_steps is None:
            assert self.every_n_epochs is not None
            self.freq_type == "epochs"
            
        self.to_pil = ToPILImage(mode=pil_mode)
        self.seed_img_transformed = seed_img_transformed
        self.seed_img_shape = self.seed_img_transformed.shape
        self.inv_normaliser = inv_normaliser
        
        assert len(noise_at_ts) >= 1
        self.noise_at_ts = noise_at_ts

        assert logger is not None
        self.logger = logger

    def denoise_and_log_image(self, pl_module: DDPMModel):        
        original_image = self.to_pil(self.inv_normaliser(self.seed_img_transformed))
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
                
                denoised_imgs.append(self.to_pil(self.inv_normaliser(x[0].cpu())))
                noised_imgs.append(self.to_pil(self.inv_normaliser(noised_x[0].cpu())))

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

    def on_train_batch_end(self, trainer, pl_module: DDPMModel, outputs, batch, batch_idx) -> None:
        if self.freq_type == "epochs":
            return

        if (
            ((trainer.global_step + 1) % self.every_n_steps == 0)
            and (self.last_log_at != trainer.global_step)
        ):
            self.denoise_and_log_image(pl_module)
            self.last_log_at = trainer.global_step
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: DDPMModel) -> None:
        if self.freq_type != "epochs":
            return
            
        if (
            ((trainer.current_epoch + 1) % self.every_n_epochs == 0)
            and (self.last_log_at != trainer.current_epoch)
        ):
            self.denoise_and_log_image(pl_module)
            self.last_log_at = trainer.current_epoch

    def on_train_end(self, trainer: Trainer, pl_module: DDPMModel) -> None:
        self.denoise_and_log_image(pl_module)
        

class SampleResnetVAEReconstruction(Callback):
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


class SampleDinoDecoderReconstructionLinear(Callback):
    def __init__(self, logger: WandbLogger, sample_input: torch.Tensor,  inv_transform: Callable, every_n_epochs: int=100):
        super().__init__()
        self.sample_input = sample_input
        assert len(self.sample_input.shape) == 4, "Please ensure to keep the batch dimension"

        self.logger = logger
        self.inv_transform = inv_transform
        self.to_pil = transforms.ToPILImage()

        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch % self.every_n_epochs == 0) or (trainer.current_epoch == trainer.max_epochs - 1):
            with torch.no_grad():
                x = self.sample_input.to(pl_module.device)
                reconstructed = pl_module(x)
                reconstructed = reconstructed.cpu()

            n_images = x.shape[0]
            fig, axs = plt.subplots(nrows=n_images, ncols=2, figsize=(4, 2 * n_images))
            for i in range(x.shape[0]):
                orig_img = self.to_pil(self.inv_transform(self.sample_input[i]))
                reconstructed_img = self.to_pil(reconstructed[i])
                axs[i, 0].imshow(orig_img)
                axs[i, 1].imshow(reconstructed_img)
                
            for ax in axs.ravel():
                ax.axis(False)

            self.logger.log_image(
                key="sample_reconstruction",
                images=[wandb.Image(fig).image]
            )

            plt.close()


class DenoiseMidwayLatentSampleCallback(Callback):
    def __init__(
        self, 
        logger: WandbLogger, 
        seed_img_transformed: torch.Tensor, 
        noise_at_ts: List[int], 
        inv_transform: Callable,
        every_n_epochs: int=10, 
        pil_mode: Optional[str]=None,
    ):

        super().__init__()
        self.inv_transform = inv_transform
        self.every_n_epochs = every_n_epochs
        self.to_pil = ToPILImage(mode=pil_mode)
        self.seed_img_transformed = seed_img_transformed
        self.seed_img_shape = self.seed_img_transformed.shape
        
        assert len(noise_at_ts) >= 1
        self.noise_at_ts = noise_at_ts

        assert logger is not None
        self.logger = logger

        raise NotImplementedError

    def on_train_epoch_end(self, trainer: Trainer, pl_module: DDPMModel) -> None:
        if not ( ((trainer.current_epoch + 1) % self.every_n_epochs == 0) or (trainer.current_epoch == trainer.max_epochs - 1) ):
            return
        
        original_image = self.to_pil(self.inv_transform(self.seed_img_transformed))
        # latent = 

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
