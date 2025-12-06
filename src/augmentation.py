"""Augmentation program. Output different picture modifications."""
import argparse as arg
import logging
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Self

import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from numpy import random as rng
from tqdm import tqdm

from .quaternion import Quaternion


class Augmentation:
    """Perform multiple augmentation techniques for an image."""

    def __init__(self: Self, path: str | Path | list) -> None:
        """Initialize the augmentation object.

        Args:
            path (str | Path | list): The path or list of path of image(s).
        """
        if isinstance(path, list):
            self._path = [p if isinstance(p, Path) else Path(p) for p in path]
        else:
            self._path = [path if isinstance(path, Path) else Path(path)]
        self._img = [plt.imread(p) for p in self._path]
        self._h, self._w, self._vec, self._trs = self._get_base_data()
        self._flp = [self._flip(i) for i in range(len(self._path))]
        self._rot = [self._rotate(i) for i in range(len(self._path))]
        self._skw = [self._skew(i) for i in range(len(self._path))]
        self._shr = [self._shear(i) for i in range(len(self._path))]
        self._crp = [self._crop(i) for i in range(len(self._path))]
        self._dst = [self._distortion(i) for i in range(len(self._path))]

    def show(self: Self) -> None:
        """Show the plot."""
        _, axes = plt.subplots(len(self._path), 7, figsize=(16, 9))
        if len(self._path) > 1:
            self._multi_show(axes)
        else:
            self._mono_show(axes)
        plt.show()

    def _mono_show(self: Self, axes: ndarray) -> None:
        """Show multiple plot lines.

        Args:
            axes (ndarray): list of plot axes.
        """
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[0].imshow(self._img[0])
        axes[1].set_title("Flip")
        axes[1].axis("off")
        axes[1].imshow(self._flp[0])
        axes[2].set_title("Rotate")
        axes[2].axis("off")
        axes[2].imshow(self._rot[0])
        axes[3].set_title("Skew")
        axes[3].axis("off")
        axes[3].imshow(self._skw[0])
        axes[4].set_title("Shear")
        axes[4].axis("off")
        axes[4].imshow(self._shr[0])
        axes[5].set_title("Crop")
        axes[5].axis("off")
        axes[5].imshow(self._crp[0])
        axes[6].set_title("Distortion")
        axes[6].axis("off")
        axes[6].imshow(self._dst[0])

    def _multi_show(self: Self, axes: ndarray) -> None:
        """Show multiple plot lines.

        Args:
            axes (ndarray): list of plot axes.
        """
        axes[0, 0].set_title("Original")
        axes[0, 1].set_title("Flip")
        axes[0, 2].set_title("Rotate")
        axes[0, 3].set_title("Skew")
        axes[0, 4].set_title("Shear")
        axes[0, 5].set_title("Crop")
        axes[0, 6].set_title("Distortion")
        for i in range(len(axes)):
            axes[i, 0].axis("off")
            axes[i, 0].imshow(self._img[i])
            axes[i, 1].axis("off")
            axes[i, 1].imshow(self._flp[i])
            axes[i, 2].axis("off")
            axes[i, 2].imshow(self._rot[i])
            axes[i, 3].axis("off")
            axes[i, 3].imshow(self._skw[i])
            axes[i, 4].axis("off")
            axes[i, 4].imshow(self._shr[i])
            axes[i, 5].axis("off")
            axes[i, 5].imshow(self._crp[i])
            axes[i, 6].axis("off")
            axes[i, 6].imshow(self._dst[i])

    def save(self: Self, dst: str | Path = None) -> None:
        """Save images to their original directory.

        Args:
            dst (str | Path): Save directory. Default to original's directory.
        """
        if dst is not None:
            dst = dst if isinstance(dst, Path) else Path(dst)
            for i in range(len(self._path)):
                self._save_one(dst / self._path[i].name, i)
        else:
            for i in range(len(self._path)):
                self._save_one(self._path[i], i)

    def _save_one(self: Self, path: Path, i: int) -> None:
        """Save augmentations for image at index i.

        Args:
            path (Path): The saving path.
            i (int): Index of the image
        """
        plt.imsave(path.with_stem(path.stem + "_flip"), self._flp[i])
        plt.imsave(path.with_stem(path.stem + "_rotate"), self._rot[i])
        plt.imsave(path.with_stem(path.stem + "_skew"), self._skw[i])
        plt.imsave(path.with_stem(path.stem + "_shear"), self._shr[i])
        plt.imsave(path.with_stem(path.stem + "_crop"), self._crp[i])
        plt.imsave(path.with_stem(path.stem + "_distortion"), self._dst[i])

    def _flip(self: Self, i: int) -> ndarray:
        """Randomly flip the loaded image.

        Args:
            i (int): Image index.
        Returns:
            ndarray: The flipped image.
        """
        if rng.binomial(1, 0.5) == 0:
            return self._img[i].copy()[-1::-1]
        else:
            return self._img[i].copy()[:, -1::-1]

    def _rotate(self: Self, i: int) -> ndarray:
        """Randomly rotate the loaded image.

        Args:
            i (int): Image index.
        Returns:
            ndarray: The flipped image.
        """
        r = 1 - 2 * rng.binomial(1, 0.5)
        r = r * (rng.random_sample() * 0.8 * np.pi + 0.1 * np.pi)
        mat = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
        vec = (mat @ self._vec[i])
        return self._scale(vec, i)

    def _skew(self: Self, i: int) -> ndarray:
        """Randomly skew the loaded image.

        Args:
            i (int): Image index.
        Retruns:
            ndarray: The skewed image.
        """
        mat = self.rngrotmat()
        a = self._vec[i][:, 0, :]
        b = self._vec[i][:, 1, :]
        c = np.zeros((self._h[i], self._w[i]))
        return self._scale((mat @ np.stack([a, b, c], axis=1))[:, :-1, :], i)

    def _shear(self: Self, i: int) -> ndarray:
        """Randomly shear the loaded image.

        Args:
            i (int): Image index.
        Returns:
            ndarray: The sheared image.
        """
        shr = rng.random_sample() * np.pi / 8 + np.pi / 16
        shr *= 1 - 2 * rng.binomial(1, 0.5)
        vec = self._vec[i].copy()
        tmp = shr * vec[:, 0, :].astype(float)
        vec[:, 0, :] += shr * vec[:, 1, :].astype(float)
        vec[:, 1, :] += tmp
        return self._scale(vec, i)

    def _crop(self: Self, i: int) -> ndarray:
        """Randomly crop the image.

        Args:
            i (int): Image index.
        Returns:
            ndarray: The croped image.
        """
        factor = rng.random_sample() * 0.25 + 0.5
        h, w = np.int16(factor * self._h[i]), np.int16(factor * self._w[i])
        a = np.int16(rng.random_sample() * (self._h[i] - h))
        b = np.int16(rng.random_sample() * (self._w[i] - w))
        crop = self._img[i].copy()[a:(a + h), b:(b + h)]
        idx = np.linspace(0, crop.shape[0] - 1, self._img[i].shape[0])
        idx = np.rint(idx).astype(int)
        jdx = np.linspace(0, crop.shape[1] - 1, self._img[i].shape[1])
        jdx = np.rint(jdx).astype(int)
        return crop[idx[:, None], jdx[None, :]]

    def _distortion(self: Self, i: int) -> ndarray:
        """Randomly twist the image.

        Args:
            i (int): Image index.
        Returns:
            ndarray: The twisted image.
        """
        a, b = self._vec[i][:, 0, :], self._vec[i][:, 1, :]
        norms = np.sqrt(a ** 2 + b ** 2)
        maximum = np.min(
            [self._vec[i].shape[0] // 2, self._vec[i].shape[2] // 2]
        )
        intensity = (1 - np.clip(norms / maximum, 0, 1))
        rads = (np.pi / 4) + rng.random_sample() * (np.pi / 4)
        rads = (1 - 2 * rng.binomial(1, 0.5)) * rads * intensity
        cos, sin = np.cos(rads), np.sin(rads)
        out_i = cos * a - sin * b
        out_j = sin * a + cos * b
        twist = np.stack([out_i, out_j], axis=1).astype(float)
        twist = self._scale(twist, i, base=self._img[i].copy())
        sub0 = twist[1:-1, 1:-1]
        sub_og = self._img[i][1:-1, 1:-1]
        sub1, sub3 = twist[:-2, :-2].astype(float), twist[2:, 2:].astype(float)
        sub2, sub4 = twist[2:, :-2].astype(float), twist[:-2, 2:].astype(float)
        idx = np.where(np.all(sub0 == sub_og, axis=-1))
        sub0[idx] = (sub1[idx] + sub2[idx] + sub3[idx] + sub4[idx]) / 4
        sub_int = np.repeat(intensity[1:-1, 1:-1, None] ** .01, 3, axis=2)[idx]
        sub0[idx] = sub_int * sub0[idx] + (1 - sub_int) * sub_og[idx]
        twist[1:-1, 1:-1] = sub0
        return twist

    def _get_base_data(self: Self) -> tuple:
        """Prepare an img coordinates system to work on.

        Returns:
            tuple: height, width, coordinates, translation to 0.
        """
        h = np.array([img.shape[0] for img in self._img])
        w = np.array([img.shape[1] for img in self._img])
        sup_i = np.array([(v // 2) for v in h])
        sup_j = np.array([(v // 2) for v in w])
        inf_i = sup_i - h
        inf_j = sup_j - w
        row_i = [
            np.repeat(np.arange(inf_i[i], sup_i[i])[:, None], w[i], axis=1)
            for i in range(len(inf_i))
        ]
        row_j = [
            np.repeat(np.arange(inf_j[i], sup_j[i])[None, :], h[i], axis=0)
            for i in range(len(inf_j))
        ]
        vec = [
            np.stack([row_i[i], row_j[i]], axis=1).astype(float)
            for i in range(len(row_i))
        ]
        trs = [np.repeat(
            np.tile([-inf_i[i], -inf_j[i]], (w[i], 1)).T[None, :, :],
            h[i],
            axis=0
        ) for i in range(len(inf_i))]
        return h, w, vec, trs

    def _scale(
            self: Self, vec: ndarray, i: int, base: ndarray = None
    ) -> ndarray:
        """Scale the image to fit the original frame.

        Args:
            vec (ndarray): The new coordinates of each pixel.
            i (int): Image index.
            base (ndarray): Base background.
        Returns:
            ndarray: Scaled image.
        """
        range_i = np.max(vec[:, 0, :]) - np.min(vec[:, 0, :])
        range_j = np.max(vec[:, 1, :]) - np.min(vec[:, 1, :])
        ratio = np.min(
            [(self._h[i] - 1) / range_i, (self._w[i] - 1) / range_j]
        )
        vec = np.round(vec * ratio + self._trs[i])
        idx = np.clip(vec[:, 0, :], 0, self._h[i] - 1).astype(int)
        jdx = np.clip(vec[:, 1, :], 0, self._w[i] - 1).astype(int)
        if base is not None:
            img = base
        else:
            img = np.full(self._img[i].shape, 255, dtype=np.uint8)
        img[idx, jdx] = self._img[i]
        return img

    @staticmethod
    def rngrotmat() -> ndarray:
        """Generate a random 3D rotation matrix.

        Retunrs:
            ndarry: Rotation matrix.
        """
        rad = (1 - 2 * rng.binomial(1, 0.5)) * (
            rng.random_sample() * (np.pi / 4) + np.pi / 8
        ) / 2
        axis = rng.random_sample(3) + 1e-15
        axis *= 1 - 2 * rng.binomial(1, 0.5, 3)
        axis = axis / np.linalg.norm(axis)
        lrq = Quaternion(np.cos(rad), *(np.sin(rad) * axis))
        rrq = Quaternion(np.cos(rad), *(np.sin(rad) * -axis))
        i = lrq * Quaternion(0, 1, 0, 0) * rrq
        j = lrq * Quaternion(0, 0, 1, 0) * rrq
        k = lrq * Quaternion(0, 0, 0, 1) * rrq
        return np.array([[i.x, j.x, k.x], [i.y, j.y, k.y], [i.z, j.z, k.z]])


def get_args(description: str = '') -> Namespace:
    """Manages program arguments.

    Args:
        description (str): is the program helper description.
    Returns:
        Namespace: The arguments.
    """
    av = arg.ArgumentParser(description=description)
    av.add_argument("files", help="Image files or directory")
    av.add_argument("--save-path", default=None, help="Save directory")
    av.add_argument("--no-show", action="store_true", help="Set display off")
    av.add_argument("--no-save", action="store_true", help="Set saving off")
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
        path = Path(av.files)
        if path.is_dir():
            path = [Path(path / p) for p in os.listdir(path)]
        else:
            path = [path]
        for i in tqdm(range(0, len(path), 10)):
            aug = Augmentation(path[i:i + 10])
            if not av.no_show:
                aug.show()
            if not av.no_save:
                aug.save(av.save_path)
        return 0
    except Exception as err:
        debug = "av" in locals() and hasattr(av, "debug") and av.debug
        logging.critical("Fatal error: %s", err, exc_info=debug)
        return 1


if __name__ == "__main__":
    sys.exit(main())
