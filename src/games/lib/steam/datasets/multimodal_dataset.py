import torch
import PIL.Image
import pandas as pd
import typing

from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import v2
from transformers import PreTrainedTokenizerBase
from typing import Optional


class MultimodalDataset(Dataset):
    """
    A dataset that returns pixel values, input ids, attention masks and labels for a given item in the dataset. The passed-in
    dataframe should have an "image" column containing the image path, and a "text" column containing the expected response from
    the model.
    """

    user_prompt_tokens: Optional[list[int]] = None

    def __init__(
        self,
        root: Path,
        data: pd.DataFrame,
        prompt_template: str,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        transform: Optional[v2.Compose] = None,
    ):
        self.root = root
        self.data = data
        self.transform = transform
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        assert self.transform, "You must provide a transform to the dataset"
        assert self.tokenizer, "You must provide a tokenizer to the dataset"

        # Initialize the user prompt which contains all the tokens that are provided by the user,
        # and not by the model. We can use this later to set appropriate labels and mask certain parts
        # of the attention mask.
        if self.user_prompt_tokens is None:
            user_inputs = self.tokenizer(
                self.prompt_template.replace("<assistant>", "")
            )
            self.user_prompt_tokens = user_inputs.input_ids

        item = self.data.iloc[idx]

        # First, get the image
        image_path = self.root / item["image"]
        image = PIL.Image.open(image_path).convert("RGB")

        # Then, prepare the image for the model
        pixel_values = self.transform(image)

        # Next, we tokenize the text input. This will result in two fields: `input_ids` and `attention_mask`
        text_inputs = typing.cast(
            dict[str, torch.Tensor],
            self.tokenizer(
                self.prompt_template.replace("<assistant>", f" {item['text']}"),
                return_tensors="pt",
            ),
        )

        # We need to generate a "labels" value, that will contain a masking token (-100) for all tokens that we
        # do not want to include in the loss function later. In our case, this means that we want to mask all
        # user-provided tokens
        labels = torch.Tensor(text_inputs["input_ids"].squeeze()).clone()
        labels[: len(self.user_prompt_tokens)] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"].squeeze(),
            "attention_mask": text_inputs["attention_mask"].squeeze(),
            "labels": labels,
        }

    def load_raw_image(self, idx: int) -> PIL.Image.Image:
        item = self.data.iloc[idx]
        image_path = self.root / item["image"]
        return PIL.Image.open(image_path).convert("RGB")

    def set_transform(self, transform: v2.Compose):
        self.transform = transform

    def set_tokenizer(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
