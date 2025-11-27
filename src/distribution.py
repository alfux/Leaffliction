"""Distribution program. Detect a folder architecture and plot charts."""
import argparse as arg
from argparse import Namespace
import logging
import os
from pathlib import Path
import sys
from typing import Self, Generator

from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.patches import Wedge
from matplotlib.text import Text
import matplotlib.pyplot as plt


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
        self._fg: Figure = plt.figure(figsize=(16, 9))
        self._pie: Axes = self._fg.add_axes([0.05, 0.15, 0.4, 0.75])
        self._bar: Axes = self._fg.add_axes([0.55, 0.15, 0.4, 0.75])
        self._set_pie_bar(self._pie, self._bar)

    def show(self: Self) -> None:
        """Show the plot."""
        plt.show()

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

    def _set_pie_bar(
            self: Self, pie: Axes, bar: Axes
    ) -> tuple[tuple[list[Wedge], list[Text], list[Text]], BarContainer]:
        """Set the pie chart.

        Args:
            pie (Axes): The Axes to draw the pie chart on.
            bar (Axes): The Axes to draw the bar chart on.
        Returns:
            tuple: (list of wedges, list of labels).
        """
        counts = [c["count"] for k, c in self._ctgy.items() if k[0] == '_']
        labels = [c[1:] for c in self._ctgy.keys() if c[0] == '_']
        wdg, lbl, pct = pie.pie(counts, labels=labels, autopct="%.1f%%")
        tks = list(range(len(counts)))
        ctn = bar.bar(tks, counts, color=[w.get_facecolor() for w in wdg])
        bar.set_xticks(tks)
        bar.set_xticklabels(labels)
        plt.setp(bar.get_xticklabels(), rotation=-45)
        return ((wdg, lbl, pct), ctn)


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
        Distribution(av.path).show()
    except Exception as err:
        debug = "av" in locals() and hasattr(av, "debug") and av.debug
        logging.critical("Fatal error: %s", err, exc_info=debug)
        return 1


if __name__ == "__main__":
    sys.exit(main())
