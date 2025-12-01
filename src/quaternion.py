import argparse as arg
import logging
import sys
from argparse import Namespace
from typing import Self

import numpy as np
from numpy import ndarray


class Quaternion:
    """Python quaternion representation."""

    def __init__(self: Self, *x: list) -> None:
        """Initialize the quaternion.

        Args:
            *x (list): If a single argument is given, it should be a
                a list or ndarray. Otherwise there should be four arguments
                representing the four quaternion's components.
        """
        if len(x) == 1:
            self._cmp = x[0] if isinstance(x[0], ndarray) else np.array(x[0])
        elif len(x) == 4:
            self._cmp = np.array(x)
        else:
            raise ValueError("Quaternion needs four components")

    def __add__(lhs: Self, rhs: "Quaternion") -> "Quaternion":
        """Add quaternions.

        Args:
            lhs (Self): This quaternion.
            rhs (Quaternion): An other quaternion.
        Returns:
            Quaternion: Sum of the quaternions.
        """
        return Quaternion(
            lhs.w + rhs.w, lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z
        )

    def __mul__(lhs: Self, rhs: "Quaternion") -> "Quaternion":
        """Multiply quaternions.

        Args:
            lhs (Self): This quaternion.
            rhs (Quaternion): An other quaternion.
        Returns:
            Quaternion: Product of the quaternions.
        """
        if isinstance(rhs, Quaternion):
            return Quaternion(
                lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z,
                lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y,
                lhs.w * rhs.y + lhs.y * rhs.w + lhs.z * rhs.x - lhs.x * rhs.z,
                lhs.w * rhs.z + lhs.z * rhs.w + lhs.x * rhs.y - lhs.y * rhs.x
            )
        return Quaternion(lhs._cmp * rhs)

    @property
    def w(self: Self) -> float:
        """Get the real component of the quaternion."""
        return self._cmp[0]

    @property
    def x(self: Self) -> float:
        """Get the i component of the quaternion."""
        return self._cmp[1]

    @property
    def y(self: Self) -> float:
        """Get the j component of the quaternion."""
        return self._cmp[2]

    @property
    def z(self: Self) -> float:
        """Get the k component of the quaternion."""
        return self._cmp[3]


def get_args(description: str = '') -> Namespace:
    """Manages program arguments.

    Args:
        description (str): is the program helper description.
    Returns:
        Namespace: The arguments.
    """
    av = arg.ArgumentParser(description=description)
    av.add_argument("--debug", action="store_true", help="Traceback mode.")
    return av.parse_args()


def main() -> int:
    """Test main.

    Returns:
        int: return status 0 (success) 1 (error).
    """
    try:
        av = get_args(main.__doc__)
        fmt = "%(asctime)s | %(levelname)s - %(message)s"
        if av.debug:
            logging.basicConfig(level=logging.DEBUG, format=fmt)
        else:
            logging.basicConfig(level=logging.INFO, format=fmt)
        logging.info("This module is not testable alone at the moment.")
        return 0
    except Exception as err:
        debug = "av" in locals() and hasattr(av, "debug") and av.debug
        logging.critical("Fatal error: %s", err, exc_info=debug)
        return 1


if __name__ == "__main__":
    sys.exit(main())
