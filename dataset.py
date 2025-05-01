import os
import random
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
import scipy.ndimage

class RoadDataset(Dataset):
    """
    Dataset for road skeletonization.
    Loads raster road images and their skeleton targets, applying augmentations.
    """
    def __init__(self, image_dir, split="train", split_ratio=(0.8, 0.1, 0.1), 
                 augment=False, misalign_max=2):
        """
        image_dir: directory containing image_*.png and target_*.png files.
        split: "train", "val", or "test".
        augment: True for training (apply aug), False for val/test.
        misalign_max: max pixel shift for target misalignment.
        """
        self.image_dir = image_dir
        self.split = split
        self.augment = augment
        self.misalign_max = misalign_max if augment else 0
        # List all image files
        all_images = sorted([f for f in os.listdir(image_dir) 
                              if f.startswith("image_") and f.endswith(".png")])
        total = len(all_images)
        # Deterministic shuffle for consistent train/val/test split
        random.Random(42).shuffle(all_images)
        train_end = int(split_ratio[0] * total)
        val_end = train_end + int(split_ratio[1] * total)
        if split == "train":
            self.files = all_images[:train_end]
        elif split == "val":
            self.files = all_images[train_end:val_end]
        elif split == "test":
            self.files = all_images[val_end:]
        else:
            raise ValueError("Unknown split name")
        # Precompute distance maps for targets
        self.distance_maps = {}
        for fname in self.files:
            target_name = fname.replace("image_", "target_")
            tgt_path = os.path.join(image_dir, target_name)
            tgt_img = Image.open(tgt_path).convert('L')
            tgt_arr = np.array(tgt_img)
            tgt_bin = (tgt_arr > 127).astype(np.uint8)
            dist = scipy.ndimage.distance_transform_edt(1 - tgt_bin)
            dist = np.clip(dist, 0, 10)  # limit distance to 10 px
            self.distance_maps[fname] = dist.astype(np.float32)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        image_path = os.path.join(self.image_dir, fname)
        target_path = os.path.join(self.image_dir, fname.replace("image_", "target_"))
        # Load grayscale image and target
        img = Image.open(image_path).convert('L')
        target = Image.open(target_path).convert('L')
        img_arr = np.array(img)
        tgt_arr = np.array(target)
        img_bin = (img_arr > 127).astype(np.uint8)
        tgt_bin = (tgt_arr > 127).astype(np.uint8)
        if self.augment:
            # Random misalignment of target
            dx = random.randint(-self.misalign_max, self.misalign_max)
            dy = random.randint(-self.misalign_max, self.misalign_max)
            if dx != 0 or dy != 0:
                tgt_bin = np.roll(tgt_bin, shift=dy, axis=0)
                tgt_bin = np.roll(tgt_bin, shift=dx, axis=1)
                # zero out wrap-around artifacts from np.roll
                if dy > 0: tgt_bin[:dy, :] = 0
                elif dy < 0: tgt_bin[dy:, :] = 0
                if dx > 0: tgt_bin[:, :dx] = 0
                elif dx < 0: tgt_bin[:, dx:] = 0
            # Random blur on input
            if random.random() < 0.5:
                radius = random.uniform(0.5, 1.5)
                img = img.filter(ImageFilter.GaussianBlur(radius))
                img_arr = np.array(img)
                img_bin = (img_arr > 127).astype(np.uint8)
            # Pepper noise: remove some road pixels
            # if random.random() < 0.5:
            #     road_pixels = np.argwhere(img_bin == 1)
            #     if len(road_pixels) > 0:
            #         num_pepper = max(1, int(0.005 * len(road_pixels)))
            #         for idx in random.sample(range(len(road_pixels)), num_pepper):
            #             y, x = road_pixels[idx]
            #             img_bin[y, x] = 0
            # # Salt noise: add some spurious road pixels
            # if random.random() < 0.2:
            #     bg_pixels = np.argwhere(img_bin == 0)
            #     if len(bg_pixels) > 0:
            #         num_salt = max(1, int(0.002 * len(bg_pixels)))
            #         for idx in random.sample(range(len(bg_pixels)), num_salt):
            #             y, x = bg_pixels[idx]
            #             img_bin[y, x] = 1
        # Convert to torch tensors (float32, 0-1)
        img_tensor = torch.from_numpy(img_bin.astype(np.float32)).unsqueeze(0)
        tgt_tensor = torch.from_numpy(tgt_bin.astype(np.float32)).unsqueeze(0)
        # Distance map tensor
        dist_map = self.distance_maps.get(fname)
        dist_tensor = torch.from_numpy(dist_map).unsqueeze(0) if dist_map is not None \
                      else torch.zeros_like(img_tensor)
        return img_tensor, tgt_tensor, dist_tensor
