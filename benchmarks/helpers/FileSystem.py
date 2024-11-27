from os.path import isfile, join, exists, dirname
import os
from pathlib import Path


class FileSystem:
    """
    A helper class providing useful functions to interact with the filesystem.
    """

    @staticmethod
    def files_in(directory):
        """
        Retrieve the name of the files present within the directory passed as parameters.
        :param directory: the directory whose files must be returned
        :return: the files
        """

        # Iterate over all directory entries.
        files = []
        for entry in os.listdir(directory):

            # Add the current entry, if it is a file.
            if isfile(join(directory, entry)):
                files.append(entry)

        return files

    @staticmethod
    def create_directory_and_file(checkpoint_path):
        """
        Create the directory and file of the checkpoint if they do not already exist.
        :param checkpoint_path: the checkpoint path
        """
        checkpoint_dir = dirname(checkpoint_path)
        if not exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not exists(checkpoint_path):
            file = Path(checkpoint_path)
            file.touch(exist_ok=True)
