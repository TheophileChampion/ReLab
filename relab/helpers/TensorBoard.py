from os.path import join
from typing import Optional

from pandas import DataFrame

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import logging

from relab.helpers.FileSystem import FileSystem


class TensorBoard:
    """!
    Class containing useful functions related to the TensorBoard monitoring framework.
    """

    @staticmethod
    def load_log_file(file : str, metric_name : str):  # TODO -> Tuple[]
        """!
        Load all the data present in the log file.
        @param file: path to tensorflow log file
        @param metric_name: the name of the metric in the tensorboard event file
        @return a dataframe containing the log file information
        """

        try:
            # Load all the data present in the log file.
            size_guidance = {
                "compressedHistograms": 1,
                "images": 1,
                "scalars": 0,
                "histograms": 1,
            }
            events = EventAccumulator(file, size_guidance=size_guidance)
            events.Reload()
            events = events.Scalars(metric_name)
            steps = list(map(lambda x: x.step, events))
            values = list(map(lambda x: x.value, events))
            return steps, values

        except Exception as e:
            # Tell the user that a file could not be loaded.
            logging.error(f"Could not process '{file}': {e}.")
            return [], []

    @staticmethod
    def load_log_directory(directory : str, metric : str) -> Optional[DataFrame]:
        """!
        Load all the event files present in the directory.
        @param directory: the target directory
        @param metric: the name of the scalar entries in the tensorboard event file
        @return a dataframe containing the metric values of all the event files in the directory
        """

        # Iterate over all files in the directory.
        all_steps, all_values = [], []
        for file in FileSystem.files_in(directory):

            # Extract the steps and metric values from the current file.
            steps, values = TensorBoard.load_log_file(join(directory, file), metric)
            all_steps += steps
            all_values += values

        # Return a dataframe containing the steps and associated values.
        df = pd.DataFrame({"step": all_steps, metric: all_values})
        return None if len(df.index) == 0 else df
