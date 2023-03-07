import os
from glob import glob
from itertools import chain
from typing import Any, Dict, List, Optional

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import skimage.util as util
import torch
from torch.utils.data import DataLoader, Dataset


class CaliforniaDataModule(pl.LightningDataModule):
    def __init__(self, **hparams: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()

        # Create folders sets
        self.hparams["key"] = int(self.hparams["key"])
        val_fold = (self.hparams["key"] + 1) % (4 + 1)

        self.train_transforms = [
            config_to_object("torchvision.transforms", k, v)
            for k, v in self.hparams["train_transform"].items()
        ]

        self.test_transforms = [
            config_to_object("torchvision.transforms", k, v)
            for k, v in self.hparams["test_transform"].items()
        ]

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.generator = torch.Generator().manual_seed(self.hparams["seed"])

        self.batch_size = self.hparams["batch_size"]

        self.assigned_folds = {
            "train": [
                x for x in range(4 + 1) if x not in (self.hparams["key"], val_fold)
            ],
            "val": [val_fold],
            "test": [self.hparams["key"]],
        }
        # Assert assigned_folds values are unique and not overlapping
        folds = list(chain(*self.assigned_folds.values()))
        assert len(set(folds)) == len(folds)

        self.root = "data/california"

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            self.train_dataset = HDF5CaliforniaDataset(
                self.root,
                transforms=self.train_transforms,
                patch_size=self.hparams["patch_size"],
                keep_burned_only=self.hparams["keep_burned_only"],
                mode=self.hparams["mode"],
                fold_list=self.assigned_folds["train"],
                attributes_filter=self.hparams["comments_filter"],
            )

        if stage in ("fit", "validate", None):
            self.val_dataset = HDF5CaliforniaDataset(
                self.root,
                transforms=self.test_transforms,
                patch_size=self.hparams["patch_size"],
                keep_burned_only=self.hparams["keep_burned_only"],
                mode=self.hparams["mode"],
                fold_list=self.assigned_folds["val"],
                attributes_filter=self.hparams["comments_filter"],
            )

        if stage in ("test", None):
            self.test_dataset = HDF5CaliforniaDataset(
                self.root,
                transforms=self.test_transforms,
                patch_size=self.hparams["patch_size"],
                keep_burned_only=self.hparams["keep_burned_only"],
                mode=self.hparams["mode"],
                fold_list=self.assigned_folds["test"],
                attributes_filter=self.hparams["comments_filter"],
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.hparams["num_workers"],
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
            drop_last=False,
        )


class HDF5CaliforniaDataset(Dataset):
    def __init__(
        self,
        hdf5_folder: str,
        patch_size: int = 512,
        fold_list: List[int] = None,
        keep_burned_only: bool = True,
        transforms=None,
        mode: str = "post",
        pre_available: bool = False,
        attributes_filter: List[int] = [],
    ):
        # Assert validity
        if mode not in ["post", "prepost"]:
            raise ValueError("mode must be post or prepost")

        self.transforms = transforms
        self.patches = []
        # No folder list provided
        if fold_list is None or len(fold_list) == 0:
            fold_list = list(range(5))
        fold_list = set([str(x) for x in fold_list])
        # Load all patches
        for dataset_file in glob(f"{hdf5_folder}/*.hdf5"):
            with h5py.File(dataset_file, "r") as dataset:
                for fold in fold_list & set(dataset.keys()):
                    for uid in dataset[fold].keys():
                        matrices = dict(dataset[fold][uid].items())
                        # Filter
                        comments = [
                            int(c)
                            for c in str(matrices["post_fire"].attrs["comments"]).split(
                                "-"
                            )
                            if c.isnumeric()
                        ]
                        if set(comments) & set(attributes_filter):
                            continue
                        if "pre_fire" not in matrices and (
                            pre_available or mode == "prepost"
                        ):
                            continue
                        if mode != "prepost":
                            matrices.pop("pre_fire", None)
                        mask = matrices.pop("mask")[...]
                        # Init
                        img = np.concatenate(
                            list(matrices.values()) + [mask.reshape(*mask.shape, 1)],
                            axis=-1,
                        )
                        img_size = img.shape[0]
                        usable_size = img_size // patch_size * patch_size
                        overlapping_start = img_size - patch_size
                        # Portioning
                        to_cut = img[:usable_size, :usable_size]
                        last_row = img[overlapping_start:, :usable_size]
                        last_column = img[:usable_size, overlapping_start:]
                        last_crop = img[
                            overlapping_start:img_size, overlapping_start:img_size
                        ]
                        # Crop
                        wanted_crop_size = (patch_size, patch_size, img.shape[-1])
                        last_row = util.view_as_blocks(last_row, wanted_crop_size)
                        last_column = util.view_as_blocks(last_column, wanted_crop_size)
                        crops = util.view_as_blocks(to_cut, wanted_crop_size)
                        # Reshaping
                        crops = crops.reshape(
                            crops.shape[0] * crops.shape[1], *wanted_crop_size
                        )
                        last_row = last_row.reshape(
                            last_row.shape[1], *wanted_crop_size
                        )
                        last_column = last_column.reshape(
                            last_column.shape[0], *wanted_crop_size
                        )
                        last_crop = last_crop.reshape(1, *last_crop.shape)
                        # Merge
                        merged = np.concatenate(
                            [crops, last_column, last_row, last_crop]
                        )
                        if keep_burned_only:
                            merged = merged[merged[:, :, :, -1].sum(axis=(1, 2)) > 0]
                        self.patches.append(merged)

        self.patches = np.concatenate(self.patches)
        print(f"Dataset len = {self.patches.shape[0]}")

    def __getitem__(self, item):
        result = {"image": self.patches[item, :, :, :-1]}
        if self.transforms is not None:
            for t in self.transforms:
                result = t(result)
        result["mask"] = torch.from_numpy(self.patches[item, :, :, -1]).unsqueeze(0)
        return result

    def __len__(self):
        return self.patches.shape[0]
