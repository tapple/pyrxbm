from dataclasses import dataclass


@dataclass
class UniqueId:
    Time: int
    Index: int
    Random: int
