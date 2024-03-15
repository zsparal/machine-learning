import msgspec
import pandas as pd
import torch

from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, TypedDict
from sklearn.model_selection import train_test_split

from steam.game import SteamGame
from steam.datasets import MultiLabelImageDataset, SplitDataset
from steam.utils import calculate_label_smoothing


@dataclass
class GenreClassificationMetadata:
    labels: list[str]
    id2label: dict[int, str]
    label2id: dict[str, int]
    genre_counts: pd.DataFrame
    label_weights: torch.Tensor
    positive_label_threshold: float


class DatasetColumns(TypedDict):
    id: str
    title: str
    image: str
    genres: list[str]


def load(
    steam_games: Path,
    top_n_by_popularity: Optional[int] = None,
    label_smoothing: float = 0.1,
    test_ratio: float = 0.2,
    random_seed: int = 42,
) -> Tuple[SplitDataset[MultiLabelImageDataset], GenreClassificationMetadata]:
    """
    Load, preprocess and create datasets from the Steam games data for a genre classification task based on game images.

    Parameters:
        steam_games: The path to the Steam games JSON file to load.
        top_n_by_popularity: Number of top games by popularity to include. Useful for reducing dataset size, as the top entries on Steam tend to have better and more varied images as well.
        label_smoothing: Label smoothing factor for reducing overfitting.
        test_ratio: Ratio of data to use for testing dataset.
        random_seed: Random seed for reproducibility.

    Returns:
        A tuple containing the training and testing datasets along with metadata for genre classification.
    """

    # We want to exclude certain genres from the Steam library. The main goal of this classification task is to distinguish
    # between games, not utilities or other software that found its way to Steam's library. Any entries that contain any of these
    # genres will be excluded later to improve the quality of the dataset for this task.
    excluded_genres = set(
        [
            "Accounting",
            "Animation & Modeling",
            "Audio Production",
            "Design & Illustration",
            "Education",
            "Game Development",
            "Movie",
            "Photo Editing",
            "Software Training",
            "Utilities",
            "Video Production",
            "Web Publishing",
        ]
    )

    def include_game_in_dataset(game: SteamGame) -> bool:
        # If the game doesn't have any images we can't use it for an image classification task
        if not game.page_information.images:
            return False

        # Only include games that are in the top N by popularity.
        if (
            top_n_by_popularity is not None
            and game.popularity_rank > top_n_by_popularity
        ):
            return False

        # If the game doesn't have genre information then it doesn't make sense to try to classify it
        if not game.page_information.genres:
            return False

        # Remove any games that are in the excluded genres
        return all(
            genre not in excluded_genres for genre in game.page_information.genres
        )

    def to_dataset_items(game: SteamGame) -> Iterable[DatasetColumns]:
        return (
            {
                "id": game.id,
                "title": game.page_information.title,
                "image": image,
                "genres": game.page_information.genres,
            }
            for image in game.page_information.images
        )

    def calculate_label_weights(
        genre_counts: pd.DataFrame, label2id: dict[str, int]
    ) -> torch.Tensor:
        """
        Label weights are calculated based on the number of games in each genre. We can use these weights during training
        in the model's loss function to reduce the effect of an imbalanced dataset.
        """
        genre_counts["genre_order"] = genre_counts["genres"].map(label2id)
        sorted_genres = genre_counts.sort_values("genre_order")
        sorted_genres.drop("genre_order", axis=1, inplace=True)

        dataset_size = sorted_genres["count"].sum()
        genre_counts_tensor = torch.tensor(sorted_genres["count"].tolist())

        return (dataset_size - genre_counts_tensor) / genre_counts_tensor

    def create_metadata(df: pd.DataFrame) -> GenreClassificationMetadata:
        labels = df.columns[4:].to_list()
        genre_counts = df["genres"].explode().value_counts().reset_index()
        id2label = {i: label for i, label in enumerate(labels)}
        label2id = {label: i for i, label in enumerate(labels)}
        positive_label_threshold = calculate_label_smoothing(
            smoothing=label_smoothing, num_labels=len(labels), value=1.0
        )
        return GenreClassificationMetadata(
            labels=labels,
            id2label=id2label,
            label2id=label2id,
            genre_counts=genre_counts,
            label_weights=calculate_label_weights(genre_counts, label2id),
            positive_label_threshold=positive_label_threshold - 0.1,
        )

    # Load the Steam games data
    all_games = msgspec.json.decode(steam_games.read_text(), type=list[SteamGame])

    # Flatten the data by creating one record per image with the associated genres
    base_df = pd.DataFrame.from_records(
        record
        for game in all_games
        if include_game_in_dataset(game)
        for record in to_dataset_items(game)
    )

    # Add the one-hot encoded genres to the dataframe
    one_hot_genres = base_df["genres"].str.join("|").str.get_dummies()
    df = pd.concat([base_df, one_hot_genres], axis=1)

    train_df, eval_df = train_test_split(
        df, test_size=test_ratio, random_state=random_seed
    )

    # Prepare the metadata
    metadata = create_metadata(train_df)

    train = MultiLabelImageDataset(
        root=steam_games.parent,
        data=train_df,
        num_labels=len(metadata.labels),
        label_smoothing=label_smoothing,
    )

    test = MultiLabelImageDataset(
        root=steam_games.parent,
        data=eval_df,
        num_labels=len(metadata.labels),
        label_smoothing=label_smoothing,
    )

    return {"train": train, "test": test}, metadata
