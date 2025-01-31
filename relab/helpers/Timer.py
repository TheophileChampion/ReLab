import time
import logging
from types import TracebackType
from typing import Optional


class Timer:
    """!
    A class used for tracking the execution time of a block of code.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        """!
        Create a timer.
        @param name: the name of the block of code whose time is being tracked
        """

        # @var name
        # The name of the block of code whose time is being tracked.
        self.name = name

        # @var start_time
        # The time when the timer was started.
        self.start_time = 0.0

    def __enter__(self) -> None:
        """!
        Start the timer.
        """
        self.start_time = time.time()

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_instance: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        """!
        Stop the timer and display the time elapsed.
        @param exc_type: exception type (unused)
        @param exc_instance: exception instance (unused)
        @param traceback: traceback object (unused)
        """
        if self.name:
            logging.info(
                "[%s]" % self.name,
            )
        logging.info("Elapsed: %s" % ((time.time() - self.start_time) * 1000))
        return None
