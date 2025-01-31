import gc
from typing import Optional

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from numpy import ndarray
from torch import Tensor


class MatPlotLib:
    """!
    A helper class providing useful functions for interacting with matplotlib.
    """

    @staticmethod
    def format_image(img: Tensor, n_channels: int = 3) -> ndarray:
        """!
        Turn a 4d pytorch tensor into a 3d numpy array
        @param img: the 4d tensor
        @param n_channels: the number of channels of the image to keep
        @return the 3d array
        """
        img = torch.squeeze(img)[-n_channels:]
        img = torch.swapaxes(img, 0, 1)
        img = torch.swapaxes(img, 1, 2)
        return img.detach().cpu().numpy()

    @staticmethod
    def save_figure(
        figure_path: str, dpi: int = 300, tight: bool = True, close: bool = True
    ) -> None:
        """!
        Save a matplotlib figure.
        @param figure_path: the name of the file used to save the figure
        @param dpi: the number of dpi
        @param tight: True to use plt.tight_layout() before saving, false otherwise
        @param close: True to close the figure after saving, false otherwise
        """
        if tight is True:
            plt.tight_layout()
        plt.savefig(figure_path, dpi=dpi, transparent=True)
        if close is True:
            MatPlotLib.close()

    @staticmethod
    def close(fig: Optional[Figure] = None) -> None:
        """!
        Close the figure passed as parameter or the current figure.
        @param fig: the figure to close
        """

        # Clear the current axes.
        plt.cla()

        # Clear the current figure.
        plt.clf()

        # Closes all the figure windows.
        plt.close("all")

        # Closes the matplotlib figure
        plt.close(plt.gcf() if fig is None else fig)

        # Forces the garbage collection
        gc.collect()
