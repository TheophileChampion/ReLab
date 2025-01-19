import time
import logging


class Timer:
    """
    A class used for tracking the execution time of a block of code.
    """

    def __init__(self, name=None):
        """
        Create a timer.
        :param name: the name of the block of code whose time is being tracked
        """
        self.name = name

    def __enter__(self):
        """
        Start the timer.
        """
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_instance, traceback):
        """
        Stop the timer and display the time elapsed.
        :param exc_type: exception type (unused)
        :param exc_instance: exception instance (unused)
        :param traceback: traceback object (unused)
        """
        if self.name:
            logging.info("[%s]" % self.name,)
        logging.info("Elapsed: %s" % ((time.time() - self.start_time) * 1000))
