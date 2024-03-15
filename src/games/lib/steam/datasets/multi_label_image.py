import torch
import numpy as np
import PIL.Image
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import v2
from typing import Optional

from steam.utils import calculate_label_smoothing


class MultiLabelImageDataset(Dataset):
    """
    A dataset for multi-label, multi-class image classification. When queried for an item, the dataset performs the following steps:

    #### 1. Load the image

    The image is loaded automatically from the row's `image` column in RGB format. The image is then transformed using the provided
    `transform` (see :func:`~steam.datasets.MultiLabelImageDataset.set_transform`).

    #### 2. Apply label smoothing

    As a data agumentation method, the dataset applies label smoothing (see :func:`~steam.utils.calculate_label_smoothing` for more details).
    This step makes the trained model less confident in their answers, reducing the risk of overfitting.

    #### 3. Return the image and the labels

    Both the image and the labels are returned as PyTorch tensors.
    """

    def __init__(
        self,
        root: Path,
        data: pd.DataFrame,
        num_labels: int,
        label_smoothing: float,
        transform: Optional[v2.Compose] = None,
    ):
        self.root = root
        self.data = data
        self.transform = transform
        self.num_labels = num_labels
        self.label_smoothing = label_smoothing

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        assert self.transform, "You must provide a transform to the dataset"

        item = self.data.iloc[idx]

        # First, get the image
        image_path = self.root / item["image"]
        image = PIL.Image.open(image_path).convert("RGB")

        # Then, prepare the image for the model
        pixel_values = self.transform(image)

        # Get the labels as a PyTorch tensor
        labels = torch.from_numpy(item[4:].values.astype(np.float32))
        assert len(labels) == self.num_labels, "Unexpected number of labels"

        smoothed_labels = calculate_label_smoothing(
            smoothing=self.label_smoothing, num_labels=self.num_labels, value=labels
        )

        return {"pixel_values": pixel_values, "labels": smoothed_labels}

    def load_raw_image(self, idx: int) -> PIL.Image.Image:
        item = self.data.iloc[idx]
        image_path = self.root / item["image"]
        return PIL.Image.open(image_path).convert("RGB")

    def set_transform(self, transform: v2.Compose):
        self.transform = transform
