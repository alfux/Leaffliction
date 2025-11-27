"""Distribution program. Detect a folder architecture and plot charts."""
import argparse as arg
from argparse import Namespace
import json
import logging
import os
from pathlib import Path
import sys
from typing import Self, Generator


class Distribution:
    """Exploration class for directory architecture."""

    def __init__(self: Self, path: str | Path) -> None:
        """Initialize the instance.

        Args:
            path (str): The path of the directory to explore.
        """
        self._path = path if isinstance(path, Path) else Path(path)
        self._arch = dict(self._architecture(self._path))
        self._ctgy = dict(self._categories(self._arch))

    def _architecture(self: Self, path: Path) -> Generator:
        """Architecture generator.

        Args:
            path (Path): Path of the directory to explore.
        Yields:
            tuple: (dir_name, content) or ("files", files_list).
        """
        files = []
        for elem in os.listdir(path):
            sub = path / elem
            if sub.is_dir():
                yield ('_' + elem, dict(self._architecture(sub)))
            else:
                files.append(elem)
        yield ("files", files)

    def _categories(self: Self, arch: dict) -> Generator:
        """Categories generator. Count (sub)categories elements.

        Args:
            arch (dict): File architecture.
        Yields:
            tuple: (category, count).
        """
        count = len(arch["files"])
        for key in arch.keys():
            if key != "files":
                sub = dict(self._categories(arch[key]))
                count += sub["count"]
                yield (key, sub)
        yield ("count", count)


def get_args(description: str = '') -> Namespace:
    """Manages program arguments.

    Args:
        description (str): is the program helper description.
    Returns:
        Namespace: The arguments.
    """
    av = arg.ArgumentParser(description=description)
    av.add_argument("path", help="Directory to process.")
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
        Distribution(av.path).explore()
    except Exception as err:
        debug = "av" in locals() and hasattr(av, "debug") and av.debug
        logging.critical("Fatal error: %s", err, exc_info=debug)
        return 1


if __name__ == "__main__":
    sys.exit(main())
