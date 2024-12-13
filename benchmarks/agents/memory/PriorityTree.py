from decimal import Decimal
from random import uniform

import math
import torch



class PriorityTree:
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

        # Store the priority tree parameters.
        self.initial_priority = initial_priority
        self.capacity = capacity
        self.n_children = n_children

        # Robust computation of the trees' depth.
        self.depth = math.floor(math.log(self.capacity, n_children))
        if int(math.pow(n_children, self.depth)) < self.capacity:
            self.depth += 1

        # Create a tensor of priorities, an empty sum-tree and an empty max-tree.
        self.priorities = torch.zeros([self.capacity])
        self.sum_tree = self.create_sum_tree(self.depth, n_children)
        self.max_tree = self.create_max_tree(self.depth, n_children)
        self.current_id = 0

    @staticmethod
    def create_sum_tree(depth, n_children):
        """
        Create a sum-tree.
        Importantly, tree elements must be python floats to avoid numerical precision error of pytorch tensors.
        :param depth: the tree's depth
        :param n_children: the number of children each node has
        :return: the tree
        """
        return [
            [0.0 for _ in range(int(math.pow(n_children, i)))]
            for i in reversed(range(depth))
        ]

    @staticmethod
    def create_max_tree(depth, n_children):
        """
        Create a max-tree.
        :param depth: the tree's depth
        :param n_children: the number of children each node has
        :return: the tree
        """
        return [
            torch.zeros([int(math.pow(n_children, i))])
            for i in reversed(range(depth))
        ]

    def sum(self):
        """
        Compute the sum of all priorities.
        :return: the sum of all priorities
        """
        if self.current_id == 0:
            return 0
        return self.sum_tree[-1][0]

    def max(self):
        """
        Find the largest priority.
        :return: the largest priority
        """
        if self.current_id == 0:
            return self.initial_priority
        return self.max_tree[-1][0].item()

    def clear(self):
        """
        Empty the priority tree.
        """
        self.current_id = 0
        self.priorities = torch.zeros([self.capacity])
        self.sum_tree = self.create_sum_tree(self.depth, self.n_children)
        self.max_tree = self.create_max_tree(self.depth, self.n_children)

    def __len__(self):
        """
        Retrieve the number of experiences stored in the buffer.
        :return: the number of experiences stored in the buffer
        """
        return min(self.current_id, self.capacity)

    def append(self, priority):
        """
        Add a priority in the priority tree.
        :param priority: the new priority
        """
        index = self.current_id % self.capacity
        self.update_sum_tree(index, priority)
        self.update_max_tree(index, priority)
        self.priorities[index] = priority
        self.current_id += 1

    def __getitem__(self, index):
        """
        Retrieve an element from the list.
        :param index: the index of the experience whose priority must be retrieved
        :return: the element
        """
        return self.priorities[self.internal_index(index)]

    def __setitem__(self, index, priority):
        """
        Replace an element in the list.
        :param index: the index of the experience whose priority must be replaced
        :param priority: the new priority
        """
        index = self.internal_index(index)
        self.update_sum_tree(index, priority)
        self.update_max_tree(index, priority)
        self.priorities[index] = priority

    def internal_index(self, index):
        """
        Transform an experience index to its internal index
        :param index: the experience index
        :return: the internal index
        """
        if self.current_id >= self.capacity:
           index += self.current_id
        return index % self.capacity

    def external_index(self, index):
        """
        Transform an internal index to its experience index
        :param index: the internal index
        :return: the experience index
        """
        if self.current_id >= self.capacity:
            index -= (self.current_id % self.capacity)
        return index % self.capacity

    def sample_indices(self, n):
        """
        Sample indices of buffer elements proportionally to their priorities.
        :param n: the number of indices to sample
        :return: the sampled indices
        """

        # Sample 'n' indices with a probability proportional to their priorities.
        indices = []
        for i in range(n):

            # Sample a priority between zero and the sum of priorities.
            sampled_priority = uniform(0, self.sum())

            # Compute the element index associated to the sampled priority using inverse transform sampling.
            indices.append(self.tower_sampling(sampled_priority))
        return indices

    def tower_sampling(self, priority):
        """
        Compute the experience index associated to the sampled priority using inverse transform sampling (tower sampling).
        :param priority: the sampled priority
        :return: the experience index
        """

        # If the priority is larger than the sum of priorities, return the index of the last element.
        if priority > self.sum():
            return self.external_index(len(self) - 1)

        # Go down the sum-tree until the leaf node is reached.
        index = 0
        for level in reversed(range(-1, self.depth - 1)):

            # Iterate over the children of the current node, keeping track of the sum of priorities.
            total = 0
            for i in range(self.n_children):

                # Get the priority of the next child.
                child_index = self.n_children * index + i
                new_priority = self.priorities[child_index] if level == -1 else self.sum_tree[level][child_index]

                # If the priority is about to be superior to the total, stop iterating over the children.
                if priority <= total + new_priority:
                    index = child_index
                    priority -= total
                    break

                # Otherwise, increase the sum of priorities.
                total += new_priority

        # Return the element index corresponding to the sampled priority.
        return self.external_index(index)

    def parent_index(self, idx):
        """
        Compute the index of the parent element.
        :param idx: the index of the element whose parent index must be computed
        :return: the parent index
        """
        return idx if idx < 0 else idx // self.n_children

    def update_sum_tree(self, index, new_priority):
        """
        Update the sum-tree to reflect an element being set to a new priority.
        :param index: the internal index of the element
        :param new_priority: the new priority
        """

        # Compute the parent index.
        parent_index = self.parent_index(index)

        # Go up the tree until the root node is reached.
        depth = 0
        priority = self.priorities[index].item()
        while depth < self.depth:

            # Update the sums in the sum-tree.
            diff = Decimal(self.sum_tree[depth][parent_index]) + Decimal(new_priority) - Decimal(priority)
            self.sum_tree[depth][parent_index] = float(diff)

            # Update parent indices and tree depth.
            depth += 1
            parent_index = self.parent_index(parent_index)

    def update_max_tree(self, index, new_priority):
        """
        Update the max-tree to reflect an element being set to a new priority.
        :param index: the internal index of the element
        :param new_priority: the new priority
        """

        # Compute the parent index and the old priority.
        parent_index = self.parent_index(index)
        old_priority = self.priorities[index].item()

        # Go up the tree until the root node is reached.
        depth = 0
        while depth < self.depth:

            # Update the maximum values in the max-tree.
            parent_value = self.max_tree[depth][parent_index].item()
            if parent_value == old_priority:
                self.max_tree[depth][parent_index] = self.max_child_value(depth, parent_index, index, old_priority, new_priority)
            elif parent_value < new_priority:
                self.max_tree[depth][parent_index] = new_priority
            else:
                break

            # Update parent indices and tree depth.
            depth += 1
            parent_index = self.parent_index(parent_index)

    def max_child_value(self, depth, parent_index, index, old_priority, new_priority):
        """
        Compute the maximum value among the child nodes.
        :param depth: the depth of the parent node
        :param parent_index: the internal index of the parent node
        :param index: the internal index of the node whose value is being set to a new priority
        :param old_priority: the old priority
        :param new_priority: the new priority
        :return: the maximum value
        """
        first_child = self.n_children * parent_index
        if depth == 0:
            children = self.priorities[first_child: first_child + self.n_children]
            children[index - first_child] = new_priority
            max_value = children.max()
            children[index - first_child] = old_priority
        else:
            max_value = self.max_tree[depth - 1][first_child: first_child + self.n_children].max()
        return max_value

    @staticmethod
    def to_str(tree):
        """
        Create a string representation of the tree passed as parameters.
        :param tree: the tree
        :return: a string representing a tree
        """
        tree_str = []
        for row in tree:
            tree_str.append(row.tolist())
        return f"{tree_str}"
