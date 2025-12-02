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

from .quaternion import Quaternion


class Augmentation:
    """Perform multiple augmentation techniques for an image."""

    def __init__(self: Self, path: str | Path) -> None:
        """Initialize the augmentation object.

        Args:
            path (str): The path of the image.
        """
        self._path = path if isinstance(path, Path) else Path(path)
        self._img = plt.imread(path)
        self._h, self._w, self._vec, self._trs = self._get_base_data()
        self._flp = self._flip()
        self._rot = self._rotate()
        self._skw = self._skew()
        self._shr = self._shear()
        self._crp = self._crop()
        self._dst = self._distortion()

    def show(self: Self) -> None:
        """Show the plot."""
        fig, axes = plt.subplots(1, 7, figsize=(16, 9))
        for axis in axes:
            axis.axis("off")
        axes[0].imshow(self._img)
        axes[1].imshow(self._flp)
        axes[2].imshow(self._rot)
        axes[3].imshow(self._skw)
        axes[4].imshow(self._shr)
        axes[5].imshow(self._crp)
        axes[6].imshow(self._dst)
        plt.show()

    def _flip(self: Self) -> ndarray:
        """Randomly flip the loaded image.

        Returns:
            ndarray: The flipped image.
        """
        if rng.binomial(1, 0.5) == 0:
            return self._img.copy()[-1::-1]
        else:
            return self._img.copy()[:, -1::-1]

    def _rotate(self: Self) -> ndarray:
        """Randomly rotate the loaded image.

        Returns:
            ndarray: The flipped image.
        """
        r = 1 - 2 * rng.binomial(1, 0.5)
        r = r * (rng.random_sample() * 0.8 * np.pi + 0.1 * np.pi)
        mat = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
        vec = (mat @ self._vec)
        return self._scale(vec)

    def _skew(self: Self) -> ndarray:
        """Randomly skew the loaded image.

        Retruns:
            ndarray: The skewed image.
        """
        mat = self._rngrotmat()
        i = self._vec[:, 0, :]
        j = self._vec[:, 1, :]
        k = np.zeros((self._h, self._w))
        return self._scale((mat @ np.stack([i, j, k], axis=1))[:, :-1, :])

    def _shear(self: Self) -> ndarray:
        """Randomly shear the loaded image.

        Returns:
            ndarray: The sheared image.
        """
        shr = rng.random_sample() * np.pi / 2 - np.pi / 4
        vec = self._vec.copy()
        tmp = shr * vec[:, 0, :].astype(float)
        vec[:, 0, :] += shr * vec[:, 1, :].astype(float)
        vec[:, 1, :] += tmp
        return self._scale(vec)

    def _crop(self: Self) -> ndarray:
        """Randomly crop the image.

        Returns:
            ndarray: The croped image.
        """
        factor = rng.random_sample() * 0.25 + 0.5
        h, w = np.int16(factor * self._h), np.int16(factor * self._w)
        a = np.int16(rng.random_sample() * (self._h - h))
        b = np.int16(rng.random_sample() * (self._w - w))
        return self._img.copy()[a:(a + h), b:(b + h)]

    def _distortion(self: Self) -> ndarray:
        """Randomly twist the image.

        Returns:
            ndarray: The twisted image.
        """
        i, j = self._vec[:, 0, :], self._vec[:, 1, :]
        norms = np.sqrt(i ** 2 + j ** 2)
        maximum = np.min([self._vec.shape[0] // 2, self._vec.shape[2] // 2])
        intensity = (1 - np.clip(norms / maximum, 0, 1))
        rads = (np.pi / 2) * intensity
        cos, sin = np.cos(rads), np.sin(rads)
        out_i = cos * i - sin * j
        out_j = sin * i + cos * j
        twist = np.stack([out_i, out_j], axis=1).astype(float)
        twist = self._scale(twist, base=self._img.copy())
        sub0 = twist[1:-1, 1:-1]
        sub_og = self._img[1:-1, 1:-1]
        sub1, sub3 = twist[:-2, :-2].astype(float), twist[2:, 2:].astype(float)
        sub2, sub4 = twist[2:, :-2].astype(float), twist[:-2, 2:].astype(float)
        idx = np.where(np.all(sub0 == sub_og, axis=-1))
        sub0[idx] = (sub1[idx] + sub2[idx] + sub3[idx] + sub4[idx]) / 4
        sub_int = np.repeat(intensity[1:-1, 1:-1, None] ** .01, 3, axis=2)[idx]
        sub0[idx] = sub_int * sub0[idx] + (1 - sub_int) * sub_og[idx]
        twist[1:-1, 1:-1] = sub0
        return twist

    def _rngrotmat(self: Self) -> ndarray:
        """Generate a random 3D rotation matrix.

        Retunrs:
            ndarry: Rotation matrix.
        """
        rad = r = 1 - 2 * rng.binomial(1, 0.5)
        r = r * (
            rng.random_sample() * 0.8 * (np.pi / 4) + 0.1 * (np.pi / 4)
        ) / 2
        axis = rng.random_sample(3) + 1e-15
        axis = axis / np.linalg.norm(axis)
        lrq = Quaternion(np.cos(rad), *(np.sin(rad) * axis))
        rrq = Quaternion(np.cos(rad), *(np.sin(rad) * -axis))
        i = lrq * Quaternion(0, 1, 0, 0) * rrq
        j = lrq * Quaternion(0, 0, 1, 0) * rrq
        k = lrq * Quaternion(0, 0, 0, 1) * rrq
        return np.array([[i.x, j.x, k.x], [i.y, j.y, k.y], [i.z, j.z, k.z]])

    def _get_base_data(self: Self) -> tuple:
        """Prepare an img coordinates system to work on.

        Returns:
            tuple: height, width, coordinates, translation to 0.
        """
        h, w, = self._img.shape[0], self._img.shape[1]
        sup_i, sup_j = (h // 2), (w // 2)
        inf_i, inf_j = sup_i - h, sup_j - w
        row_i = np.repeat(np.arange(inf_i, sup_i)[:, None], w, axis=1)
        row_j = np.repeat(np.arange(inf_j, sup_j)[None, :], h, axis=0)
        vec = np.stack([row_i, row_j], axis=1).astype(float)
        trs = np.tile([-inf_i, -inf_j], (w, 1)).T
        trs = np.repeat(trs[None, :, :], h, axis=0)
        return h, w, vec, trs

    def _scale(self: Self, vec: ndarray, base: ndarray = None) -> ndarray:
        """Scale the image to fit the original frame.

        Args:
            vec (ndarray): The new coordinates of each pixel.
        Returns:
            ndarray: Scaled image.
        """
        range_i = np.max(vec[:, 0, :]) - np.min(vec[:, 0, :])
        range_j = np.max(vec[:, 1, :]) - np.min(vec[:, 1, :])
        ratio = np.min([(self._h - 1) / range_i, (self._w - 1) / range_j])
        vec = np.round(vec * ratio + self._trs)
        i = np.clip(vec[:, 0, :], 0, self._h - 1).astype(int)
        j = np.clip(vec[:, 1, :], 0, self._w - 1).astype(int)
        if base is not None:
            img = base
        else:
            img = np.full(self._img.shape, 255, dtype=np.uint8)
        img[i, j] = self._img
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
