from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class Affine:
    """A 2D affine transformation. Stored as a homogeneous 3x3 matrix."""

    matrix: NDArray

    @classmethod
    def from_parts(
        cls,
        translation: NDArray[np.floating] | None = None,
        rotation: NDArray[np.floating] | None = None,
        scale: float | None = None,
    ) -> Affine:
        """Creates an affine transformation from translation, rotation, scale.

        Args:
            translation: A 2D vector representing the translation.
            rotation: A 2x2 rotation matrix.
            scale: A scaling factor.

        Returns:
            An Affine object representing the transformation.
        """
        if scale is None:
            scale = 1.0
        if translation is None:
            translation = np.zeros(2, dtype=np.float64)
        if rotation is None:
            rotation = np.eye(2, dtype=np.float64)
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2, :2] = rotation * scale
        matrix[:2, 2] = translation
        return cls(matrix)

    def __matmul__(self, other: Affine) -> Affine:
        """Combines two affine transformations."""
        return Affine(self.matrix @ other.matrix)

    def inverse(self) -> Affine:
        """Returns the inverse of this affine transformation."""
        inv_matrix = np.linalg.inv(self.matrix)
        return Affine(inv_matrix)

    def to_homogeneous(self) -> NDArray[np.floating]:
        """Convert into equivalent homogeneous transformation matrix."""
        return self.matrix
