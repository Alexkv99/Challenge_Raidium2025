import torch
import random
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, images, masks, augment=False):

        self.images = images
        self.masks = masks
        self.augment = augment

    def transform(self, image, mask):
        """
        Perform data augmentations manually:
        - Random Horizontal Flip
        - Random Rotation (90-degree increments)
        - Add Gaussian Noise (to the image)
        - Adjust Contrast (to the image)
        - Random Crop (only if image size is larger than target)
        """
        # Random horizontal flip
        if random.random() > 0.5:
            image = image.flip(-1)  # Flip along width (HORIZONTAL)
            mask = mask.flip(-1)

        # Random rotation (90 degrees increments)
        if random.random() > 0.5:
            k = random.randint(1, 3)  # Rotate 90, 180, or 270 degrees
            image = image.rot90(k, [1, 2])  # Rotate image
            mask = mask.rot90(k, [0, 1])  # Rotate mask

        # Add Gaussian noise to the image
        if random.random() > 0.5:
            image = self.add_gaussian_noise(image, mean=0, std=0.1)

        # Random contrast adjustment
        if random.random() > 0.5:
            image = self.adjust_contrast(image, gamma=(0.5, 2.0))

        # Random crop (only if the image size is larger than target)
        i, j, h, w = self.random_crop(image, (256, 256))
        image = image[:, i:i+h, j:j+w]
        mask = mask[i:i+h, j:j+w]

        return image, mask

    def add_gaussian_noise(self, image, mean=0, std=0.1):
        """Adds Gaussian noise to an image."""
        noise = torch.randn_like(image) * std + mean
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0, 1)  # Ensure values are in range [0, 1]

    def adjust_contrast(self, image, gamma=(0.5, 2.0)):
        """Adjusts the contrast of an image."""
        gamma_value = random.uniform(gamma[0], gamma[1])
        image = image ** gamma_value
        return torch.clamp(image, 0, 1)  # Ensure values are in range [0, 1]

    def random_crop(self, image, output_size):
        """Generates random crop parameters (i, j, h, w)."""
        _, height, width = image.shape
        target_height, target_width = output_size
        i = random.randint(0, height - target_height)
        j = random.randint(0, width - target_width)
        return i, j, target_height, target_width

    def __getitem__(self, index):
        # Fetch the image and mask tensor
        image = self.images[index]
        mask = self.masks[index]

        # Apply transformation
        if self.augment:
            image, mask = self.transform(image, mask)

        return image, mask

    def __len__(self):
        return len(self.images)


class UnlabeledDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __getitem__(self, index):
        return self.images[index]

    def __len__(self):
        return len(self.images)
