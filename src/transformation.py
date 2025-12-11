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
from matplotlib.axes import Axes


class Transformation:
    """Perform different image characteristic extractions."""

    def __init__(self: Self, path: str | Path) -> None:
        """Initialize the transformation object.

        Args:
            path (str | Path): The path of the image.
        """
        self._path = path if isinstance(path, Path) else Path(path)
        self._img = plt.imread(path)
        self._rgb = self.to_rgb(self._img)
        self._bgl = self.to_bgl(self._rgb[0], self._rgb[2], self._rgb[2])
        self._hsv = self.to_hsv(self._rgb[0], self._rgb[2], self._rgb[2])

    def show(self: Self) -> None:
        """Show image transformations."""
        # fig = plt.figure(figsize=(16, 9))
        # ax: Axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        img = self._gaussian_blur(self.green)
        img = np.broadcast_to(img[:, :, None], (img.shape[0], img.shape[1], 3))
        plt.imshow(img.astype(int))
        plt.show()

    @property
    def red(self: Self) -> ndarray:
        """Get the red component matrix.

        Returns:
            ndarray: Red component matrix.
        """
        return self._rgb[0]

    @property
    def green(self: Self) -> ndarray:
        """Get the green component matrix.

        Returns:
            ndarray: Green component matrix.
        """
        return self._rgb[1]

    @property
    def blue(self: Self) -> ndarray:
        """Get the blue component matrix.

        Returns:
            ndarray: Blue component matrix.
        """
        return self._rgb[2]

    @property
    def blue_yellow(self: Self) -> ndarray:
        """Get the blue-yellow component matrix.

        Returns:
            ndarray: Blue-yellow component matrix.
        """
        return self._bgl[0]

    @property
    def green_magenta(self: Self) -> ndarray:
        """Get the green-magenta component matrix.

        Returns:
            ndarray: Green-magenta component matrix.
        """
        return self._bgl[1]

    @property
    def lightness(self: Self) -> ndarray:
        """Get the lightness component matrix.

        Returns:
            ndarray: Lightness component matrix.
        """
        return self._bgl[2]

    @property
    def hue(self: Self) -> ndarray:
        """Get the hue component matrix.

        Returns:
            ndarray: Hue component matrix.
        """
        return self._hsv[0]

    @property
    def saturation(self: Self) -> ndarray:
        """Get the saturation component matrix.

        Returns:
            ndarray: Saturation component matrix.
        """
        return self._hsv[1]

    @property
    def value(self: Self) -> ndarray:
        """Get the value component matrix.

        Returns:
            ndarray: Value component matrix.
        """
        return self._hsv[2]

    def _gaussian_blur(
            self: Self, chan: ndarray, *, radius: int = 10
    ) -> ndarray:
        """Gaussian convolution transformation of an array.

        Args:
            chan (ndarray): 2D matrix of values.
            mean (float): Mean of the gaussian.
            std (float): Standard deviation of the gaussian.
        Returns:
            ndarray: The convolution.
        """
        span = np.arange(2 * radius + 1) - radius
        n, m = np.meshgrid(span, span, indexing="ij")
        kernel = np.exp((-n ** 2 - m ** 2) / 2)
        kernel /= kernel.sum()
        padded = np.pad(chan, radius)
        stride = np.lib.stride_tricks.as_strided(padded, ())
        convolution = np.empty_like(chan)
        for i in range(chan.shape[0]):
            for j in range(chan.shape[1]):
                convolution[i, j] = (
                    padded[i:(i + len(span)), j:(j + len(span))] * kernel
                ).sum()
        return convolution.astype(int)

    @staticmethod
    def to_rgb(img: ndarray) -> ndarray:
        """Split an RGB image in Red, Green and Blue maps.

        Args:
            img (ndarray): The image.
        Returns:
            ndarray: (3, i, j) float RGB matrix.
        """
        img = img.astype(float)
        return img.transpose(2, 0, 1)

    @staticmethod
    def to_bgl(r: ndarray, g: ndarray, b: ndarray) -> ndarray:
        """Convert an RGB image to green-magenta, blue-yellow, lightness.

        These are projections of the colors on the chroma plane orthogonal to
        to (1, 1, 1). Basis of the chroma plane is the projection of (0, 0, 1)
        which is blue-yellow axis, and the vectorial product of (1, 1, 1) and
        (1, 0, 0) which gives an orthogonal vector between green-magenta and
        red-cyan axes.
        Args:
            r (ndarray): Red float matrix.
            g (ndarray): Green float matrix.
            b (ndarray): Blue float matrix.
        Returns:
            ndarray: (3, i, j) green-magenta/blue-yellow/lightness matrix.
        """
        y = r + g
        blue_yellow = (y - 2 * b) / np.sqrt(6)
        green_magenta = (r - g) / np.sqrt(2)
        lightness = (y + b) / np.sqrt(3)
        return np.stack((blue_yellow, green_magenta, lightness), axis=0)

    @staticmethod
    def to_hsv(r: ndarray, g: ndarray, b: ndarray) -> ndarray:
        """Convert an RGB image to a Hue, Saturation and Value maps.

        Hue is the chromatic angle, Saturation represents the color intensity
        and Value is the dominant RGB component.
        Args:
            r (ndarray): Red float matrix.
            g (ndarray): Green float matrix.
            b (ndarray): Blue float matrix.
        Returns:
            ndarray: (3, i, j) Hue/Saturation/Value matrix.
        """
        v = np.max([r, g, b], axis=0)
        span = v - np.min([r, g, b], axis=0)
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
        return np.stack((h, s, v), axis=0)


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
        transformation.show()
        return 0
    except Exception as err:
        debug = "av" in locals() and hasattr(av, "debug") and av.debug
        logging.critical("Fatal error: %s", err, exc_info=debug)
        return 1


if __name__ == "__main__":
    sys.exit(main())
