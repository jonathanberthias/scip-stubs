from collections.abc import Mapping, Sequence
from typing import TypeVar

_K = TypeVar("_K")
_V = TypeVar("_V")

def multidict(
    D: Mapping[_K, _V] | Mapping[_K, Sequence[_V]],
) -> list[list[_K] | dict[_K, _V]]:
    """
    creates a multidictionary
    """
