from typing import TypedDict, Generic, TypeVar
from torch.utils.data import Dataset

T = TypeVar("T", bound=Dataset)


class SplitDataset(TypedDict, Generic[T]):
    """
    This class provides a generic abstraction over data that is split into training and test sets.
    """

    train: T
    test: T
