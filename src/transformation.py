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
