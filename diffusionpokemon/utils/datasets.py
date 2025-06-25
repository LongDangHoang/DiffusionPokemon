import os

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.datasets.folder import has_file_allowed_extension, IMG_EXTENSIONS
from sklearn.model_selection import train_test_split

def find_all_images(root_dir):
    all_images = []
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                all_images.append(os.path.join(root, fname))
    return all_images


class SimpleImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = read_image(path)
        if self.transform:
            image = self.transform(image)
        return image, path


def create_train_test_datasets_from_path(
    root_dir: str,
    train_ratio: float = 0.8,
    transform=None,
    use_transform_on_validation: bool = False,
    random_seed: int = 2025
):
    """
    Returns (in tuple order):
    - all_dataset: Dataset containing all images
    - train_dataset: Dataset containing training images
    - test_dataset: Dataset containing testing images 
    """
    all_images = find_all_images(root_dir)

    len_all_images = len(all_images)
    train_size = int(train_ratio * len_all_images)
    test_size = len_all_images - train_size
    train_images, test_images = train_test_split(all_images, train_size=train_size, test_size=test_size, random_state=random_seed)


    all_dataset = SimpleImageDataset(all_images, transform=transform)
    train_dataset = SimpleImageDataset(train_images, transform=transform)
    test_dataset = SimpleImageDataset(test_images, transform=transform if use_transform_on_validation else None)

    print(f"Total images: {len_all_images}, Train images: {len(train_images)}, Test images: {len(test_images)}")
    return all_dataset, train_dataset, test_dataset