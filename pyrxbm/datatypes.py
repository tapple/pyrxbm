from __future__ import annotations

from dataclasses import dataclass
import numpy as np

Enum: TypeAlias = int
Double: TypeAlias = float
Int64: TypeAlias = int
SecurityCapabilities: TypeAlias = int


def normal_id_to_vector(normal_id: int) -> np.ndarray:
    coords = np.zeros(3, dtype=np.float32)
    coords[normal_id % 3] = -1 if normal_id > 2 else 1
    return coords


_NORMAL_VECTORS = np.column_stack([normal_id_to_vector(i) for i in range(6)])


def orient_id_to_rotation_matrix(orient_id: int):
    """Returns a flattened 3x3 rotation matrix"""
    r0 = _NORMAL_VECTORS[:, orient_id // 6]
    r1 = _NORMAL_VECTORS[:, orient_id % 6]
    r2 = np.cross(r0, r1)
    return np.concatenate((r0, r1, r2))


def vectors_to_normal_ids(v: np.ndarray, tol: float = 1e-40) -> list[int | None]:
    """
    :param v: a 2d array of vectors to convert to normal ids
    :param tol: How close to aligned the vector should be to count.
        Roblox Studio seems to use 1e-30 at most.
        On my test file, it made a meaningful difference between 1e-5 and 1e-8,
        and negligible difference outside that range.
        A good default would be 2e-7, the middle of that range
    :return: normal id for each vector
    """
    zeros = np.isclose(v, 0, rtol=tol, atol=tol)
    ones = np.isclose(abs(v), 1, rtol=tol, atol=tol)
    is_01s = np.logical_xor(zeros, ones).all(1)
    matchgroups = np.isclose(v @ _NORMAL_VECTORS, 1, rtol=tol, atol=tol)
    idss = (c.nonzero()[0] for c in matchgroups)
    return [
        ids[0] if ids.size == 1 and is_01 else None for ids, is_01 in zip(idss, is_01s)
    ]


def _normal_ids_to_orient_id(normal_ids: list[int | None]) -> int | None:
    if None in normal_ids:
        return None
    elif len({id % 3 for id in normal_ids}) != 3:
        return None
    else:
        return 6 * normal_ids[0] + normal_ids[1]


def rotation_matrices_to_orient_ids(r: np.ndarray) -> list[int | None]:
    """
    :param v: a 2d array of rotation matrices to convert to normal ids.
        Each row is a flattened 3x3 rotation matrix
    :return: orient id for each matrix
    """
    all_normal_ids = vectors_to_normal_ids(r.reshape(-1, 3))
    return [
        _normal_ids_to_orient_id(all_normal_ids[3 * i : 3 * i + 3])
        for i in range(r.size // 9)
    ]


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
    random: int
    time: int
    index: int

    def __str__(self):
        return f"{self.random:016x}{self.time:08x}{self.index:08x}"
