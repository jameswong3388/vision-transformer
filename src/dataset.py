from pathlib import Path
import torch
import pytorch_lightning as pl
from datasets import load_dataset

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, transforms

import glob
from pandas.core.common import flatten
import random
from PIL import Image

BASE_DIR = Path(__file__).parent.parent


class PatchifyTransform:

    def __init__(self, patch_size):
        """Custom transform that patchifies image on
        patch_size x patch_size flattened patches.

        Args:
            patch_size: the size of patch
        """
        self.patch_size = patch_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Use torch.Tensor.unfold method to add patches
        to new dimension. Flatten new and color dimensions.

        Args:
            img: image tensor to patchify

        Returns:
            Patchified tensor
        """
        res = img.unfold(1, self.patch_size, self.patch_size)  # 3 x 8 x 32 x 4
        res = res.unfold(2, self.patch_size, self.patch_size)  # 3 x 8 x 8 x 4 x 4

        return res.reshape(-1, self.patch_size * self.patch_size * 3)  # -1 x 48 == 64 x 48

class WayangKulitDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, patch_size: int = 4, val_batch_size: int = 16,
                 im_size: int = 32, rotation_degrees: (int, int) = (-30, 30)):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size

        self.train_transform = transforms.Compose(
            [
                transforms.Resize(size=(im_size, im_size)),

                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(size=(im_size, im_size)),
                transforms.RandomRotation(degrees=rotation_degrees),

                transforms.ToTensor(),
                transforms.Normalize((0.63528919, 0.57810118, 0.51988552), (0.33020571, 0.34510824, 0.36673283)),
                PatchifyTransform(patch_size),
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.Resize(size=(im_size, im_size)),

                transforms.ToTensor(),
                transforms.Normalize((0.63528919, 0.57810118, 0.51988552), (0.33020571, 0.34510824, 0.36673283)),
                PatchifyTransform(patch_size),
            ]
        )

        self.ds_train = None
        self.ds_val = None

    def setup(self, stage: str):
        self.ds_train = WayangKulit('dataset/train', transform=self.train_transform)
        self.ds_val = WayangKulit('dataset/val', transform=self.val_transform)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=2, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=2, persistent_workers=True)

    @property
    def classes(self):
        """Returns the amount of WayangKulitDataModule classes"""
        return 14


class WayangKulit(Dataset):
    def __init__(self, dataset_path, transform):
        self.dataset_path = dataset_path
        self.transform = transform
        self.dataset = load_dataset(path=self.dataset_path, name='wayang_kulit')

        self.train_image_paths = []
        self.classes = []

        for data_path in glob.glob(self.dataset_path + '/*'):
            self.classes.append(data_path.split('/')[-1])
            self.train_image_paths.append(glob.glob(data_path + '/*'))

        self.train_image_paths = list(flatten(self.train_image_paths))
        random.shuffle(self.train_image_paths)

    def __len__(self):
        return len(self.dataset['train'])

    def __getitem__(self, idx):
        image_filepath = self.train_image_paths[idx]

        image = Image.open(image_filepath)

        if self.transform:
            image = self.transform(image)

        idx_to_class = {i: j for i, j in enumerate(self.classes)}
        class_to_idx = {value: key for key, value in idx_to_class.items()}

        label = image_filepath.split('/')[-2]
        label = class_to_idx[label]

        return image, label
