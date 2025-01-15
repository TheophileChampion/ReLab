import pytest

from benchmarks.agents.memory.Deque import Deque
from collections import deque


class TestDeque:
    """
    Class testing the double ended queue.
    """

    @staticmethod
    def compare(queue_1, queue_2):
        """
        Compare the queues passed as parameters.
        :param queue_1: the first queue
        :param queue_2: the second queue
        """
        assert len(queue_1) == len(queue_2)
        for i in range(len(queue_1)):
            assert queue_1[i] == queue_2[i]

    @pytest.mark.parametrize("elements, max_size", [
        ([5, 1, 1], 2), ([], 4), ([1, 2, 3, 4], 4), ([10, 2], 4),
    ])
    def test_clear(self, elements, max_size):

        # Create two deque (one from my c++ library and one from python standard library).
        queue = Deque(max_size=max_size)
        python_queue = deque(maxlen=max_size)

        # Add all elements to the end of the double ended queues.
        for element in elements:
            queue.append(element)
            python_queue.append(element)

        # Clear both double ended queues.
        queue.clear()
        python_queue.clear()

        # Check that both queues are identical.
        self.compare(queue, python_queue)

    @pytest.mark.parametrize("elements, max_size", [
        ([5, 1, 1], 2), ([], 4), ([1, 2, 3, 4], 4), ([10, 2], 4),
    ])
    def test_append(self, elements, max_size):

        # Create two deque (one from my c++ library and one from python standard library).
        queue = Deque(max_size=max_size)
        python_queue = deque(maxlen=max_size)

        # Add all elements to the end of the double ended queues.
        for element in elements:
            queue.append(element)
            python_queue.append(element)

        # Check that both queues are identical.
        self.compare(queue, python_queue)

    @pytest.mark.parametrize("elements, max_size", [
        ([5, 1, 1], 2), ([], 4), ([1, 2, 3, 4], 4), ([10, 2], 4),
    ])
    def test_append_left(self, elements, max_size):

        # Create two deque (one from my c++ library and one from python standard library).
        queue = Deque(max_size=max_size)
        python_queue = deque(maxlen=max_size)

        # Add all elements to the end of the double ended queues.
        for element in elements:
            queue.append_left(element)
            python_queue.appendleft(element)

        # Check that both queues are identical.
        self.compare(queue, python_queue)

    @pytest.mark.parametrize("elements, n_pops, max_size", [
        ([5, 1, 1], 1, 2), ([1, 2, 3, 4], 3, 4), ([10, 2], 2, 4),
    ])
    def test_pop_left(self, elements, n_pops, max_size):

        # Create two deque (one from my c++ library and one from python standard library).
        queue = Deque(max_size=max_size)
        python_queue = deque(maxlen=max_size)

        # Add all elements to the end of the double ended queues.
        for element in elements:
            queue.append_left(element)
            python_queue.appendleft(element)

        # Remove elements from the front of the double ended queues.
        for _ in range(n_pops):
            queue.pop_left()
            python_queue.popleft()

        # Check that both queues are identical.
        self.compare(queue, python_queue)

    @pytest.mark.parametrize("elements, n_pops, max_size", [
        ([5, 1, 1], 1, 2), ([1, 2, 3, 4], 3, 4), ([10, 2], 2, 4),
    ])
    def test_pop(self, elements, n_pops, max_size):

        # Create two deque (one from my c++ library and one from python standard library).
        queue = Deque(max_size=max_size)
        python_queue = deque(maxlen=max_size)

        # Add all elements to the end of the double ended queues.
        for element in elements:
            queue.append_left(element)
            python_queue.appendleft(element)

        # Remove elements from the front of the double ended queues.
        for _ in range(n_pops):
            queue.pop()
            python_queue.pop()

        # Check that both queues are identical.
        self.compare(queue, python_queue)

    @pytest.mark.parametrize("elements, max_size, length", [
        ([5, 1, 1], 2, 2), ([1, 2, 3, 4], 4, 4), ([10, 2], 4, 2),
    ])
    def test_len(self, elements, max_size, length):

        # Create two deque (one from my c++ library and one from python standard library).
        queue = Deque(max_size=max_size)
        python_queue = deque(maxlen=max_size)

        # Add all elements to the end of the double ended queues.
        for element in elements:
            queue.append(element)
            python_queue.append(element)

        # Check that both queues have the expected length.
        assert len(queue) == len(python_queue) == length
