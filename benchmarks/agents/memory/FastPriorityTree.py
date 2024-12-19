from benchmarks.agents.memory.cpp import PriorityTree


class FastPriorityTree:
    """
    A class storing the experience priorities.
    """

    def __init__(self, capacity, initial_priority, n_children):
        """
        Create a priority tree.
        :param capacity: the tree's capacity
        :param initial_priority: the initial priority given to first elements
        :param n_children: the number of children each node has
        """
        self.priority_tree = PriorityTree(
            capacity=capacity, initial_priority=initial_priority, n_children=n_children
        )

    def sum(self):
        """
        Compute the sum of all priorities.
        :return: the sum of all priorities
        """
        return self.priority_tree.sum()

    def max(self):
        """
        Find the largest priority.
        :return: the largest priority
        """
        return self.priority_tree.max()

    def clear(self):
        """
        Empty the priority tree.
        """
        self.priority_tree.clear()

    def __len__(self):
        """
        Retrieve the number of experiences stored in the buffer.
        :return: the number of experiences stored in the buffer
        """
        return self.priority_tree.length()

    def append(self, priority):
        """
        Add a priority in the priority tree.
        :param priority: the new priority
        """
        self.priority_tree.append(priority)

    def __getitem__(self, index):
        """
        Retrieve an element from the list.
        :param index: the index of the experience whose priority must be retrieved
        :return: the element
        """
        return self.priority_tree.get(index)

    def __setitem__(self, index, priority):
        """
        Replace an element in the list.
        :param index: the index of the experience whose priority must be replaced
        :param priority: the new priority
        """
        self.priority_tree.set(index, priority)

    def sample_indices(self, n):
        """
        Sample indices of buffer elements proportionally to their priorities.
        :param n: the number of indices to sample
        :return: the sampled indices
        """
        return self.priority_tree.sample_indices(n)

    def tower_sampling(self, priority):
        """
        Compute the experience index associated to the sampled priority using inverse transform sampling (tower sampling).
        :param priority: the sampled priority
        :return: the experience index
        """
        return self.priority_tree.tower_sampling(priority)

    def parent_index(self, idx):
        """
        Compute the index of the parent element.
        :param idx: the index of the element whose parent index must be computed
        :return: the parent index
        """
        return self.priority_tree.parent_index(idx)

    def update_sum_tree(self, index, new_priority):
        """
        Update the sum-tree to reflect an element being set to a new priority.
        :param index: the internal index of the element
        :param new_priority: the new priority
        """
        return self.priority_tree.update_sum_tree(index, new_priority)

    def update_max_tree(self, index, new_priority):
        """
        Update the max-tree to reflect an element being set to a new priority.
        :param index: the internal index of the element
        :param new_priority: the new priority
        """
        return self.priority_tree.update_max_tree(index, new_priority)

    def to_str(self, tree):
        """
        Create a string representation of the tree whose name is passed as parameters.
        :param tree: the name of the tree to turn into a string, i.e., either 'sum' or 'max'
        :return: a string representing a tree
        """
        if tree == "max":
            return self.priority_tree.max_tree_to_str()
        return self.priority_tree.sum_tree_to_str()
