"""Augmentation program. Output different picture modifications."""
import argparse as arg
import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import Self

import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from numpy import random as rng


class Augmentation:
    """Perform multiple augmentation techniques for an image."""

    def __init__(self: Self, path: str | Path) -> None:
        """Initialize the augmentation object.

        Args:
            path (str): The path of the image.
        """
        self._path = path if isinstance(path, Path) else Path(path)
        self._img = plt.imread(path)
        self._flp = self._flip()
        self._rot = self._rotate()
        plt.imshow(self._rot)

    def show(self: Self) -> None:
        """Show the plot."""
        plt.show()

    def _flip(self: Self) -> ndarray:
        """Randomly flip the loaded image.

        Returns:
            ndarray: The flipped image.
        """
        img = self._img.copy()
        if rng.binomial(1, 0.5) == 0:
            return img[-1::-1]
        else:
            return img[:, -1::-1]

    def _rotate(self: Self) -> None:
        """Randomly rotate the loaded image.

        Returns:
            ndarray: The flipped image.
        """
        img = self._img.copy()
        h, w, rad = img.shape[0], img.shape[1], rng.random_sample() * 2 * np.pi
        mat = np.array([
            [np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]
        ])
        sup_i, sup_j = (h // 2), (w // 2)
        inf_i, inf_j = sup_i - h, sup_j - w
        row = np.repeat(np.arange(inf_i, sup_i)[:, None], w, axis=1)
        vec = np.stack([row, row.T], axis=1)
        trs = np.tile([-inf_i, -inf_j], (w, 1)).T
        trs = np.repeat(trs[None, :, :], h, axis=0)
        vec = (mat @ vec)
        range_i = np.max(vec[:, 0, :]) - np.min(vec[:, 0, :])
        range_j = np.max(vec[:, 1, :]) - np.min(vec[:, 1, :])
        ratio = np.max([(h - 1) / range_i, (w - 1) / range_j])
        vec = np.round(vec * ratio + trs)
        vec[:, 0, :] = np.clip(vec[:, 0, :], 0, h - 1)
        vec[:, 1, :] = np.clip(vec[:, 1, :], 0, w - 1)
        img = img * 0 + 255
        for i in range(h):
            for j in range(w):
                img[*vec.astype(int)[i].T[j]] = self._img[i, j]
        return img


def get_args(description: str = '') -> Namespace:
    """Manages program arguments.

    Args:
        description (str): is the program helper description.
    Returns:
        Namespace: The arguments.
    """
    av = arg.ArgumentParser(description=description)
    av.add_argument("file", help="Image file")
    av.add_argument("--debug", action="store_true", help="Traceback mode.")
    return av.parse_args()


def main() -> int:
    """Produce new pictures based on one, for data augmentation.

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
        Augmentation(av.file).show()
        return 0
    except Exception as err:
        debug = "av" in locals() and hasattr(av, "debug") and av.debug
        logging.critical("Fatal error: %s", err, exc_info=debug)
        return 1


if __name__ == "__main__":
    sys.exit(main())
