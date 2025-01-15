from benchmarks.agents.memory.cpp import IntDeque


class Deque:
    """
    A double ended queue allowing for storage and retrieval of integers.
    """

    def __init__(self, max_size):
        """
        Create a double ended queue containing integer.
        :param max_size: the maximum number of integers the queue can contain
        """
        self.queue = IntDeque(max_size)

    def append(self, value):
        """
        Add an integer to the end of the queue.
        :param value: the integer to add
        """
        self.queue.append(value)

    def append_left(self, value):
        """
        Add an integer to the front of the queue.
        :param value: the integer to add
        """
        self.queue.append_left(value)

    def pop(self):
        """
        Remove an integer from the end of the queue.
        """
        self.queue.pop()

    def pop_left(self):
        """
        Remove an integer from the front of the queue.
        """
        self.queue.pop_left()

    def __getitem__(self, index):
        """
        Retrieve the element whose indices are passed as parameters.
        :param index: the index of the element to retrieve
        :return: the element
        """
        return self.queue.get(index)

    def __len__(self):
        """
        Retrieve the number of integers stored in the queue.
        :return: the number of integers stored in the queue
        """
        return self.queue.length()

    def clear(self):
        """
        Empty the queue.
        """
        return self.queue.clear()
