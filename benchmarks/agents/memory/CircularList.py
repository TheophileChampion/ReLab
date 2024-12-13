import collections


# Class storing a circular index similar to polar coordinate.
CircularIndex = collections.namedtuple("CircularIndex", field_names=["angle", "radius"])


class CircularList:
    """
    A list that can contain an infinite number of elements where:
     - removal of the first element is O(N / C)
     - addition of an element at the end of the list is amortized O(1)
     - access and modification of elements are O(1)

    N = number of elements in the list
    C = the estimated size of the list (how big we expect the list to be)
    """

    def __init__(self, expected_size):
        """
        Create a circular list.
        :param expected_size: the estimated size of the list (how big we expect the list to be)
        """
        self.expected_size = expected_size
        self.elements = [[] for _ in range(expected_size)]
        self.first_index = 0
        self.current_index = 0

    def append(self, element):
        """
        Add an element at the end of the list.
        :param element: the element to add
        """
        c_index = self.circular_index(self.current_index, first_index=0)
        self.elements[c_index.angle].append(element)
        self.current_index += 1

    def pop(self):
        """
        Remove the first element from the list.
        :return: the removed element
        """
        c_index = self.circular_index(self.first_index, first_index=0)
        element = self.elements[c_index.angle][0]
        self.elements[c_index.angle] = self.elements[c_index.angle][1:]
        self.first_index += 1
        return element

    def __getitem__(self, index):
        """
        Retrieve an element from the list.
        :param index: the index of the element to retrieve
        :return: the element
        """
        if index >= len(self):
            return None
        c_index = self.circular_index(index)
        return self.elements[c_index.angle][c_index.radius]

    def __setitem__(self, index, element):
        """
        Replace an element in the list.
        :param index: the index of the element to replace
        :param element: the new element
        """
        c_index = self.circular_index(index)
        self.elements[c_index.angle][c_index.radius] = element

    def circular_index(self, index, first_index=None):
        """
        Compute the circular index corresponding to the provided index.
        :param index: the index whose circular index must be computed
        :param first_index: the index to consider as the first element in the circular list
        :return: the circular index
        """
        if index < 0:
            index += len(self)
        if first_index is None:
            first_index = self.first_index
        return CircularIndex((index + first_index) % self.expected_size, index // self.expected_size)

    def __len__(self):
        """
        Retrieve the number of elements currently stored in the list.
        :return: the number of elements currently stored in the list
        """
        return self.current_index - self.first_index

    def clear(self):
        """
        Clear the content of the circular list.
        """
        self.elements = [[] for _ in range(self.expected_size)]
        self.current_index = 0
        self.first_index = 0
