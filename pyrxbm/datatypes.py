from dataclasses import dataclass
import numpy as np


def vector_from_normal_id(normal_id: int) -> np.ndarray:
    coords = np.zeros(3, dtype=np.float32)
    coords[normal_id % 3] = -1 if normal_id > 2 else 1
    return coords


def rotation_matrix_from_orient_id(orient_id: int):
    r0 = vector_from_normal_id(orient_id // 6)
    r1 = vector_from_normal_id(orient_id % 6)
    r2 = np.cross(r0, r1)
    return np.concatenate((r0, r1, r2))


class CFrame:
    def __init__(self, *args, mat=None):
        if mat is None:
            self._mat = np.identity(4, dtype=np.float32)
        else:
            self._mat = mat
        if len(args) == 0:  # identity
            pass
        elif len(args) == 1:  # vector pos
            raise NotImplementedError()
        elif len(args) == 2:  # vector pos, vector lookat
            raise NotImplementedError()
        elif len(args) == 3:  # scalar pos
            self._mat[0:3, 3] = args
        elif len(args) == 7:  # scalar pos, scalar quaternion
            raise NotImplementedError()
        elif len(args) == 12:  # scalar pos, scalar rot
            self._mat[0:3, 3] = args[0:3]
            self._mat[0, 0:3] = args[3:6]
            self._mat[1, 0:3] = args[6:9]
            self._mat[2, 0:3] = args[9:12]
        else:
            raise ValueError(f"Unexpected number of arguments: {len(args)}")

    def GetComponents(self) -> np.ndarray:
        return np.concatenate((self._mat[0:3, 3], self._mat[0:3, 0:3].ravel()))

    def __str__(self):
        return ", ".join(str(c) for c in self.GetComponents())


CFrame.Identity = CFrame()


@dataclass
class UniqueId:
    Time: int
    Index: int
    Random: int
