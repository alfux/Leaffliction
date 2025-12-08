"""Transformation program. Output different image characteristics."""
import argparse as arg
import logging
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray


class Transformation:
    """Perform different image characteristic extractions."""

    def __init__(self: Self, path: str | Path | list) -> None:
        """Initialize the transformation object.

        Args:
            path (str | Path | list): The path or list of path of image(s).
        """
        if isinstance(path, list):
            self._path = [p if isinstance(p, Path) else Path(p) for p in path]
        else:
            self._path = [path if isinstance(path, Path) else Path(path)]
        self._img = [plt.imread(p) for p in self._path]
        self._hst = [self._color_histogram(i) for i in range(len(self._path))]

    def _color_histogram(self: Self, i: int) -> ndarray:
        """Compute color histogram arrays.

        Args:
            i (int): Image index.
        Returns:
            ndarray: Histogram matrix.
        """
        img = self._img[i]
        r, g, b = self.rgb_color(img)
        by, gm, lightness = self.lab_color(img)
        h, s, v = self.hsv_color(img)

    @staticmethod
    def rgb_color(img: ndarray) -> tuple[ndarray]:
        """Split an RGB image in Red, Green and Blue maps.

        Args:
            img (ndarray): The image.
        Returns:
            tuple[ndarray]: R, G and B maps.
        """

        img[:, :, 0], img[:, :, 1], img[:, :, 2]

    @staticmethod
    def

    @staticmethod
    def hsv_color(img: ndarray) -> tuple[ndarray, ndarray, ndarray]:
        """Convert an RGB image to a Hue, Saturation and Value maps.

        Args:
            img (ndarray): The image.
        Returns:
            tuple[ndarray, ndarray, ndarray]: H, S and V maps.
        """
        img = img.astype(float)
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        v = img.max(axis=2)
        span = v - img.min(axis=2)
        s = np.zeros_like(v, dtype=float)
        np.divide(span, v, out=s, where=(v != 0))
        h = np.zeros_like(v, dtype=float)
        defined = (span != 0)
        mask_r = defined & (v == r)
        mask_g = defined & ~mask_r & (v == g)
        mask_b = defined & ~(mask_r | mask_g) & (v == b)
        h[mask_r] = (60 * (g[mask_r] - b[mask_r]) / span[mask_r]) % 360
        h[mask_g] = 60 * (b[mask_g] - r[mask_g]) / span[mask_g] + 120
        h[mask_b] = 60 * (r[mask_b] - g[mask_b]) / span[mask_b] + 240
        return h, s, v


def get_args(description: str = '') -> Namespace:
    """Manages program arguments.

    Args:
        description (str): is the program helper description.
    Returns:
        Namespace: The arguments.
    """
    av = arg.ArgumentParser(description=description)
    av.add_argument("src", help="Path of file or directory.")
    av.add_argument("--dst", default=None, help="Saving destination")
    av.add_argument("--debug", action="store_true", help="Traceback mode.")
    return av.parse_args()


def main() -> int:
    """Image characteristics extraction.

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
        path = Path(av.src)
        if path.is_dir():
            path = [Path(p) for p in os.listdir(path)]
        transformation = Transformation(path)
        return 0
    except Exception as err:
        debug = "av" in locals() and hasattr(av, "debug") and av.debug
        logging.critical("Fatal error: %s", err, exc_info=debug)
        return 1


if __name__ == "__main__":
    sys.exit(main())
