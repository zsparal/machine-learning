import torch

from typing import TypeVar


T = TypeVar("T", float, torch.Tensor)


def calculate_label_smoothing(smoothing: float, num_labels: int, value: T) -> T:
    return value * (1 - smoothing) + smoothing / num_labels
