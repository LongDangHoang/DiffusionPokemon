import boto3

from torchvision.transforms import ToPILImage

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import Logger
from pytorch_lightning.callbacks import Callback

from ..models.ddpm_unet import DDPMUNet

from pathlib import Path


class SampleCallback(Callback):
    def __init__(self, logger: Logger, freq: int=10):
        super().__init__()
        self.freq = freq
        self.to_pil = ToPILImage()

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

        
class S3SyncCallback(Callback):
    """
    Synchronise checkpoint folder with bucket
    
    Initialises with the checkpoint folder. Path from current working directory to checkpoint folder will be used 
    as key to s3.
    """
    def __init__(self, save_local_dir: Path, load_local_dir: Path=None, every_n_epochs: int=1) -> None:
        self.s3 = boto3.resource("s3")
        self.bucket_name = 'longdang-deep-learning-personal-projects'
        self.bucket = self.s3.Bucket(self.bucket_name)
        
        self.save_local_dir = save_local_dir
        self.load_local_dir = load_local_dir if load_local_dir else save_local_dir
        self.save_s3_key = str(self.save_local_dir.absolute().relative_to(Path('.').resolve()))
        self.load_s3_key = str(self.load_local_dir.absolute().relative_to(Path('.').resolve()))
        
        self.every_n_epochs = every_n_epochs
        self.epoch_counter_state = 0
        
        print("Initialised S3 sync. Saving to:", self.save_s3_key, "and loading from:", self.load_s3_key)
        
    def on_train_epoch_end(self, trainer: Trainer, pl_module: DDPMUNet) -> None:
        self.epoch_counter_state += 1
        if self.epoch_counter_state % self.every_n_epochs == 0:
            self.upload_files_to_s3()
            
    def on_train_end(self, trainer: Trainer, pl_module: DDPMUNet) -> None:
        self.upload_files_to_s3()
        
    def download_files_from_s3(self):
        for file in self.bucket.objects.filter(Prefix=self.load_s3_key):
            filename = file.key.split("/")[-1]
            self.bucket.download_file(file.key, self.load_local_dir / filename)
            
    def upload_files_to_s3(self):
        self.delete_folder_on_s3()
        self.clean_save_local_dir()
        
        for file in self.save_local_dir.rglob("*"):
            key = str(file.absolute().relative_to(Path(".").resolve()))
            self.bucket.upload_file(file, key)
            
    def delete_folder_on_s3(self):
        for file in self.bucket.objects.filter(Prefix=self.save_s3_key):
            self.s3.Object(self.bucket_name, file.key).delete()
    
    def clean_save_local_dir(self):   
        # check if last_v1 is there, make it last if needed
        if (self.save_local_dir / "last-v1.ckpt").exists():
            (self.save_local_dir / "last-v1.ckpt").replace(
                self.save_local_dir / "last.ckpt"
            )    
            
    def download_filename(self, filename: str):
        self.bucket.download_file(self.load_s3_key + "/" + filename, self.load_local_dir / filename)
        
    def upload_filename(self, filename: str):
        self.bucket.upload_file(self.save_local_dir / filename, self.save_s3_key + "/" + filename)
