from __future__ import annotations

from dataclasses import dataclass
import numpy as np

Enum: TypeAlias = int
Double: TypeAlias = float
Int64: TypeAlias = int


def normal_id_to_vector(normal_id: int) -> np.ndarray:
    coords = np.zeros(3, dtype=np.float32)
    coords[normal_id % 3] = -1 if normal_id > 2 else 1
    return coords


_NORMAL_VECTORS = np.row_stack([normal_id_to_vector(i) for i in range(6)])


def orient_id_to_rotation_matrix(orient_id: int):
    """Returns a flattened 3x3 rotation matrix"""
    r0 = _NORMAL_VECTORS[orient_id // 6]
    r1 = _NORMAL_VECTORS[orient_id % 6]
    r2 = np.cross(r0, r1)
    return np.concatenate((r0, r1, r2))


def vector_to_normal_id(v: np.ndarray) -> int | None:
    ids = np.isclose(_NORMAL_VECTORS @ v, 1).nonzero()[0]
    return ids[0] if ids.size == 1 else None


def rotation_matrix_to_orient_id(r: np.ndarray) -> int | None:
    """Converts a flattened 3x3 matrix to orient id"""
    xi = vector_to_normal_id(r[0:3])
    yi = vector_to_normal_id(r[3:6])
    zi = vector_to_normal_id(r[6:9])
    if None in (xi, yi, zi) or len({xi % 3, yi % 3, zi % 3}) != 3:
        return None
    return 6 * xi + yi


class CFrame:
    Identity: CFrame

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
