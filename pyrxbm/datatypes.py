from dataclasses import dataclass
import numpy as np


class CFrame:
    def __init__(self, *, mat: np.ndarray = None):
        if mat is None:
            self._mat = np.identity(4)
        else:
            self._mat = mat

    def GetComponents(self) -> np.ndarray:
        return np.concatenate((self._mat[0:3, 3], self._mat[0:3, 0:3].ravel()))

    def __str__(self):
        return ", ".join(self.GetComponents())


CFrame.Identity = CFrame()


@dataclass
class UniqueId:
    Time: int
    Index: int
    Random: int
