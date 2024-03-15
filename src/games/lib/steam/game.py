import posixpath
from typing import Literal, Optional

import msgspec

SteamUrlType = Literal["app", "sub"]


class SteamReview(msgspec.Struct):
    sentiment: str
    number_of_reviewers: int
    positive_percentage: int


class SteamPageInformation(msgspec.Struct):
    type: Literal["game", "dlc", "package"]

    title: str
    description: Optional[str]
    price: Optional[int | Literal["free"]]
    images: list[str]
    genres: list[str]
    tags: list[str]
    developers: list[str]
    publishers: list[str]
    release_date: Optional[str]
    recent_reviews: Optional[SteamReview]
    all_reviews: Optional[SteamReview]

    def get_image_name(self, index: int) -> str:
        assert index < len(self.images), "That image doesn't exist"
        return posixpath.basename(self.images[index])


class UncrawledSteamGame(msgspec.Struct):
    id: str
    url: str
    raw_name: Optional[str]
    url_type: SteamUrlType
    popularity_rank: int
    page_information: Optional[SteamPageInformation]

    def crawled_page_id(self) -> str:
        return f"{self.id}_{self.raw_name or 'Unknown'}"


class SteamGame(msgspec.Struct):
    id: str
    url: str
    raw_name: Optional[str]
    url_type: SteamUrlType
    popularity_rank: int
    page_information: SteamPageInformation

    def crawled_page_id(self) -> str:
        return f"{self.id}_{self.raw_name or 'Unknown'}"
