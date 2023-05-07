import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder

from pathlib import Path
from typing import Any, Optional


class Scaler:
    """Scale [0 - 1] to [-1 - 1]"""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2 - 1 


class BaseDataset:
    def __init__(self):
        # basic transforms that probably won't need to be dynamic
        self.init_transform = transforms.Compose([
            transforms.Resize((64, 64)), # to fit model
            transforms.ToTensor(), 
            Scaler(),
            transforms.RandomHorizontalFlip(p=0.5) # generative model should not care about left/right
        ])
        self.to_pil = transforms.ToPILImage()

        self.dataset: Optional[Dataset] = None
        self.dataloader: Optional[DataLoader] = None

    def load(
            self,
            img_folder_path: Path,
            limit_data_size: Optional[int]=None,
            batch_size: int=64
        ) -> None:
        if self.dataset is not None:
            return

        self.dataset = ImageFolder(
            img_folder_path,
            transform=self.init_transform
        )

        if limit_data_size:
            self.dataset = Subset(
                self.dataset,
                indices=np.random.choice(
                    np.arange(len(self.dataset)),
                    size=limit_data_size,
                    replace=False,
                )
            )

        print("Number of images loaded: ", len(self.dataset))

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True
        )

    def visualise(self):
        img, _ = next(iter(self.dataset))
        plt.imshow(self.to_pil(img))
        plt.show()


class CartoonPretrainDataset(BaseDataset):
    def __init__(self):
        super().__init__()

        # use a different init transform
        self.init_transform = transforms.Compose([
            transforms.RandomCrop(64), # to fit model
            transforms.ToTensor(), 
            Scaler(),
            transforms.RandomHorizontalFlip(p=0.5) # generative model should not care about left/right
        ])

    def load(
            self,
            img_folder_path: Path,
            limit_data_size: Optional[int]=None,
            batch_size: int=64
        ) -> None:
        super().load(img_folder_path, limit_data_size, batch_size)

        # TODO
        # if we are not limiting dataset size, we are likely running into 
        # very similar frames. As such, we random sample within each class
