import collections
import math
from random import uniform

import numpy as np
from torch import FloatTensor, BoolTensor, IntTensor, cat, unsqueeze
import torch

import benchmarks


class Experience:
    """
    Class storing an experience.
    """

    def __init__(self, obs, action, reward, done, next_obs):
        """
        Create an experience.
        :param obs: the observation at time t
        :param action: the action at time t
        :param reward: the reward received by the agent after taking the action at time t
        :param done: True if the episode ended, False otherwise
        :param next_obs: the observation at time t + 1
        """
        self.obs = obs
        self.action = action
        self.reward = reward
        self.done = done
        self.next_obs = next_obs


class BufferElement:
    """
    Class storing an element of the buffer used to optimize memory requirement.
    """

    def __init__(self, index, frame, action, reward, done, priority=None, heap_index=None):
        """
        Create a replay buffer element.
        :param index: a unique index associated to the element
        :param frame: a frame (grayscale image from the environment)
        :param action: the action taken by the agent
        :param reward: the reward received by the agent
        :param done: True if the episode ended, False otherwise
        :param priority: the priority of the element
        :param heap_index: the index of the element in the heap
        """
        self.index = index
        self.frame = frame
        self.action = action
        self.reward = reward
        self.done = done
        self.priority = priority
        self.heap_index = heap_index


class ReplayBuffer:
    """
    Class implementing a replay buffer with support for prioritization [1] and multistep Q-learning [2] from:

    [1] Tom Schaul. Prioritized experience replay. arXiv preprint arXiv:1511.05952, 2015.
    [2] Richard S Sutton. Learning to predict by the methods of temporal differences. Machine learning, 3:9–44, 1988.
    """

    def __init__(
        self, capacity=10000, initial_priority=None, n_steps=1, gamma=0.99, omega=1.0, omega_is=1.0, n_children=10,
        stack_size=None
    ):
        """
        Constructor.
        :param capacity: the number of experience the buffer can store
        :param initial_priority: the maximum experience priority given to new transitions, None for no prioritization
        :param n_steps: the number of steps for which rewards are accumulated in multistep Q-learning
        :param gamma: the discount factor
        :param omega: the prioritization exponent
        :param omega_is: the important sampling exponent
        :param n_children: the maximum number of children each node in the priority tree can have
        :param stack_size: the number of stacked frame in each observation, if None use the configuration
        """

        # Store the buffer's parameters, and the device on which to store experiences.
        self.capacity = capacity
        self.initial_priority = initial_priority
        self.n_steps = n_steps
        self.gamma = gamma
        self.omega = omega
        self.omega_is = omega_is
        self.stack_size = benchmarks.config("stack_size") if stack_size is None else stack_size
        self.device = benchmarks.device()

        # The experience replay buffer.
        self.buffer = collections.deque(maxlen=capacity + self.stack_size)
        self.current_id = 0

        # The last few returns used for multistep Q-learning.
        self.last_returns = collections.deque(maxlen=n_steps)
        self.last_actions = collections.deque(maxlen=n_steps)

        # The priorities associated with all experiences in the replay buffer.
        self.priorities = PriorityTree(buffer=self, n_children=n_children)

        # The indices of the last sampled experiences.
        self.indices = []

    def append(self, exp):
        """
        Add a new experience to the buffer
        :param exp: the experience to add
        """

        # Get the multistep action and cumulated return.
        exp.action, exp.reward = self.get_multistep_action_and_return(exp)

        # Keep track of whether the buffer is full.
        buffer_is_full = (len(self) == self.capacity)

        # Add the experience to the replay buffer.
        if len(self.buffer) == 0:
            for i in range(self.stack_size):
                self.buffer.append(self.create_buffer_element(exp, i))
        self.buffer.append(self.create_buffer_element(exp, -1))

        # Check whether a priority must be associated to the last frame of the replay buffer.
        if exp.reward is None or not self.prioritized():
            return

        # Initialize the experience priority.
        if buffer_is_full:
            old_element = self.buffer[self.stack_size + self.n_steps - 2]
            self.buffer[-1].heap_index = self.priorities.replace(old_element, self.buffer[-1])
        else:
            self.buffer[-1].heap_index = self.priorities.push(self.buffer[-1])

    def sample(self, batch_size):
        """
        Sample a batch from the replay buffer.
        :param batch_size: the size of the batch to sample
        :return: observations, actions, rewards, done, next_observations
        where:
        - observations: the batch of observations
        - actions: the actions performed
        - rewards: the rewards received
        - done: whether the environment stop after performing the actions
        - next_observations: the observations received after performing the actions
        """

        # Sample a batch from the replay buffer.
        if self.prioritized():
            self.indices = self.priorities.sample_indices(batch_size)
        else:
            self.indices = np.random.choice(len(self), batch_size)

        obs, actions, rewards, done, next_obs = zip(*[self.get_experience(idx) for idx in self.indices])

        # Convert the batch into a torch tensor stored on the proper device.
        return self.list_to_tensor(obs).to(self.device), \
            IntTensor(actions).to(self.device), \
            FloatTensor(rewards).to(self.device), \
            BoolTensor(done).to(self.device), \
            self.list_to_tensor(next_obs).to(self.device)

    def report(self, loss):
        """
        Report the loss associated with all the transitions of the previous batch.
        :param loss: the loss of all previous transitions
        :return: the loss (weighted by the importance sampling weights, if prioritization is needed)
        """

        # If no prioritization is required, simply return.
        if not self.prioritized():
            return loss

        # Add a small positive constant to avoid zero probabilities.
        loss += 1e-5

        # Raise the loss to the power of the prioritization exponent.
        if self.omega != 1.0:
            loss = loss.pow(self.omega)

        # Update the priorities and compute the importance sampling weights.
        sum_priorities = self.priorities.sum()
        priorities = self.update_priorities(loss)
        weights = len(self) * torch.tensor(priorities).to(self.device) / sum_priorities
        weights = torch.pow(weights, -self.omega_is)
        return loss * weights / weights.max()

    def clear(self):
        """
        Empty the replay buffer.
        """
        self.buffer.clear()
        self.priorities.clear()
        self.indices.clear()
        self.last_returns.clear()

    def prioritized(self):
        """
        Check whether the buffer is prioritized.
        :return: True for prioritized replay buffer, False otherwise
        """
        return self.initial_priority is not None

    def __len__(self):
        """
        Retrieve the number of elements in the buffer.
        :return: the number of elements contained in the replay buffer
        """
        return max(len(self.buffer) - self.stack_size, 0)

    @staticmethod
    def list_to_tensor(tensor_list):
        """
        Transform a list of n dimensional tensors into a tensor with n+1 dimensions.
        :param tensor_list: the list of tensors
        :return: the output tensor
        """
        return cat([unsqueeze(tensor, 0) for tensor in tensor_list])

    def create_buffer_element(self, exp, f_index):
        """
        Create a buffer element corresponding to the i-th frame of an observation.
        :param exp: the experience whose observation is turned into a frame
        :param f_index: index of the extracted frame (if negative the frame comes from exp.next_obs, otherwise it comes from exp.obs)
        :return: the buffer element
        """
        frame = exp.next_obs[f_index, :, :] if f_index < 0 else exp.obs[f_index, :, :]
        priority = self.priorities.max() if self.prioritized() else None
        element = BufferElement(
            self.current_id, frame.detach().clone().unsqueeze(dim=0), exp.action, exp.reward, exp.done, priority
        )
        self.current_id += 1
        return element

    def get_experience(self, idx):
        """
        Retrieve the experience corresponding to the index.
        :param idx: the index
        :return: the experience
        """

        # Retrieve the element containing the experience's reward, action and done flag.
        element = self.buffer[idx + self.stack_size + self.n_steps - 1]

        # Collect the frames of the experience's observations.
        frames = []
        for i in range(self.stack_size + self.n_steps):
            frames.append(self.buffer[idx + i].frame)
        obs = torch.concat(frames[0:self.stack_size])
        next_obs = torch.concat(frames[self.n_steps:self.stack_size + self.n_steps])

        # Return a tuple representing the experience.
        return obs, element.action, element.reward, element.done, next_obs

    def get_experience_index(self, element_index):
        """
        Compute the index of the experience from the index of the buffer element.
        :param element_index: the index of the buffer element
        :return: the index of the experience
        """
        return element_index - self.buffer[0].index

    def get_multistep_action_and_return(self, exp):
        """
        Compute the multistep action and cumulated return of (past) experiences.
        :param exp: the new experience to use for the update of returns
        :return: the multistep action and return to add to the replay buffer, (None, None) if no return should be added
        """

        # Update the returns of last experiences.
        for i in range(len(self.last_returns)):
            self.last_returns[i] += math.pow(self.gamma, i + 1) * exp.reward

        # Add the current reward to the queue of last returns.
        self.last_returns.appendleft(exp.reward)
        self.last_actions.appendleft(exp.action)

        # Return the reward that must be added to the replay buffer.
        if len(self.last_returns) < self.n_steps:
            return None, None
        else:
            return self.last_actions.pop(), self.last_returns.pop()

    def update_priorities(self, loss):
        """
        Update the priorities of the batch elements.
        :param loss: the loss of the batch elements
        :return: the old priorities of the batch elements
        """

        # Iterate over the indices corresponding to the previous batch.
        priorities = []
        for i, idx in enumerate(self.indices):

            # Keep track of old priorities.
            priorities.append(self.buffer[idx + self.stack_size].priority)

            # Update the priorities of the batch elements.
            priority = loss[i].item()
            if math.isinf(priority) or math.isnan(priority):
                priority = self.priorities.max()
            self.priorities.set(self.buffer[idx + self.stack_size + self.n_steps - 1], priority)

        # Return the old priorities.
        return priorities


class PriorityTree:
    """
    A class implementing the priority tree of a replay buffer.

    A priority tree is a custom data structure composed of:
     - a max-heap storing references to the replay buffer experiences, and sorting them by priority
     - a sum-tree whose nodes contain the sum of the descendant (experience) priorities

    Importantly, when priorities are swapped in the heap:
     - the sums in the sum-tree are updated accordingly
     - the heap indices of the experiences in the replay buffer are updated accordingly
    """

    def __init__(self, buffer, n_children):
        """
        Create an empty priority tree.
        :param buffer: the replay buffer whose priorities are handled by the priority tree
        :param n_children: the maximum number of children each node in the priority tree can have
        """
        self.buffer = buffer
        self.n_children = n_children
        self.priorities = []
        self.tree = SumTree(self.buffer.capacity, self.n_children, self)

    def sum(self):
        """
        Compute the sum of all priorities.
        :return: the sum of all priorities
        """
        if len(self.buffer) == 0:
            return 0
        return self.tree.sum_tree[-1][0].item()

    def max(self):
        """
        Find the largest priority.
        :return: the largest priority
        """
        if len(self.buffer) == 0:
            return self.buffer.initial_priority
        return self.priorities[0].priority

    def clear(self):
        """
        Empty the priority tree.
        """
        self.priorities.clear()
        self.tree = SumTree(self.buffer.capacity, self.n_children, self)

    def push(self, element):
        """
        Add a new buffer element to the max-heap.
        :param element: the buffer element
        :return: the heap index of the new buffer element
        """
        return MaxHeap.push(self.priorities, element, self.tree.update)

    def replace(self, element, new_element):
        """
        Replace an element of the heap of priority.
        :param element: the element to be replaced
        :param new_element: the new element
        :return: the heap index of the new buffer element
        """

        # Update the sum-tree.
        self.tree.set(element.heap_index, new_element.priority)

        # Replace the old element by the new element.
        idx = element.heap_index
        self.priorities[idx] = new_element
        new_element.heap_index = idx

        # Ensure that the heap property is satisfied.
        return MaxHeap.ensure_consistency(self.priorities, idx, self.tree.update)

    def set(self, element, new_priority):
        """
        Replace the priority of an element in the heap.
        :param element: the element whose priority must be replaced
        :param new_priority: the new priority
        :return: the new heap index of the element
        """

        # Update the sum-tree.
        self.tree.set(element.heap_index, new_priority)

        # Replace the element's priority by the new priority.
        element.priority = new_priority
        idx = element.heap_index

        # Ensure that the heap property is satisfied.
        return MaxHeap.ensure_consistency(self.priorities, idx, self.tree.update)

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
            index = self.tree.tower_sampling(sampled_priority)

            # Convert the element index into the corresponding experience index.
            indices.append(self.buffer.get_experience_index(index - self.buffer.stack_size))
        return indices


class SumTree:
    """
    A class implementing a sum-tree.
    """

    def __init__(self, capacity, n_children, priority_tree):
        """
        Create a sum-tree filled with zeros.
        :param capacity: the replay buffer capacity
        :param n_children: the maximum number of children of all nodes in the tree
        :param priority_tree: the priority tree
        :return: the created sum-tree
        """

        # Store the sum-tree parameters.
        self.priority_tree = priority_tree
        self.n_children = n_children

        # Robust computation of the tree's depth.
        self.depth = math.floor(math.log(capacity, n_children))
        if int(math.pow(n_children, self.depth)) < capacity:
            self.depth += 1

        # Create an empty sum-tree.
        self.sum_tree = [
            torch.zeros([int(math.pow(n_children, i))]) for i in reversed(range(self.depth))
        ]
        for i, element in enumerate(priority_tree.priorities):
            self.set(i, element.priority, initialize=True)

    def parent_index(self, idx):
        """
        Compute the index of the parent element.
        :param idx: the index of the element whose parent index must be computed
        :return: the parent index
        """
        return idx if idx < 0 else idx // self.n_children

    def update(self, index_1, index_2=-1):
        """
        Update the sum-tree to reflect the swap of the priorities whose indices are provided as parameters.
        If the second index is negative, update the sum-tree to reflect the addition of a new priority.
        :param index_1: index of the first priority
        :param index_2: index of the second priority
        """

        # Compute the parent indices.
        parent_1 = self.parent_index(index_1)
        parent_2 = self.parent_index(index_2)

        # Go up the tree until both indices are from the same subtree.
        depth = 0
        priority_1 = self.priority_tree.priorities[index_1].priority
        priority_2 = self.priority_tree.priorities[index_2].priority if index_2 >= -1 else 0
        while parent_1 != parent_2 and depth < self.depth:

            # Update the sums in the sum-tree.
            if parent_2 >= 0:
                self.sum_tree[depth][parent_1] += priority_2 - priority_1
                self.sum_tree[depth][parent_2] += priority_1 - priority_2
            else:
                self.sum_tree[depth][parent_1] += priority_1

            # Update parent indices and tree depth.
            depth += 1
            parent_1 = self.parent_index(parent_1)
            parent_2 = self.parent_index(parent_2)

    def set(self, index, new_priority, initialize=False):
        """
        Update the sum-tree to reflect an element been set to a new priority.
        :param index: the index of the element
        :param new_priority: the new priority
        :param initialize: True, if the function is called to initialize the tree, False otherwise
        """

        # Compute the parent index.
        parent_index = self.parent_index(index)

        # Go up the tree until the root node is reached.
        depth = 0
        priority = self.priority_tree.priorities[index].priority
        while depth < self.depth:

            # Update the sums in the sum-tree.
            self.sum_tree[depth][parent_index] += new_priority - (0 if initialize is True else priority)

            # Update parent indices and tree depth.
            depth += 1
            parent_index = self.parent_index(parent_index)

    def tower_sampling(self, priority):
        """
        Compute the element index associated to the sampled priority using inverse transform sampling (tower sampling).
        :param priority: the sampled priority
        :return: the element index
        """

        # If the priority is larger than the sum of priorities, return the index of the last element.
        if priority > self.sum_tree[-1][0]:
            return self.priority_tree.priorities[-1].index

        # Go down the sum-tree until the leaf node is reached.
        index = 0
        for level in reversed(range(-1, self.depth - 1)):

            # Iterate over the children of the current node, keeping track of the sum of priorities.
            total = 0
            for i in range(self.n_children):

                # Get the priority of the next child.
                if level == -1:
                    new_priority = self.priority_tree.priorities[self.n_children * index + i].priority
                else:
                    new_priority = self.sum_tree[level][self.n_children * index + i]

                # If the priority is about to be superior to the total, stop iterating over the children.
                if priority <= total + new_priority:
                    index = self.n_children * index + i
                    priority -= total
                    break

                # Otherwise, increase the sum of priorities.
                total += new_priority

        # Return the element index corresponding to the sampled priority.
        return self.priority_tree.priorities[index].index

    def __str__(self):
        """
        Create a string representation of the tree.
        :return: a string representing a tree
        """
        tree = []
        for row in self.sum_tree:
            tree.append(row.tolist())
        return f"{tree}"


class MaxHeap:
    """
    A class implementing a max-heap.
    """

    @staticmethod
    def is_higher(x1, x2):
        """
        Check if the first argument goes higher in the heap than the second.
        :param x1: the first element
        :param x2:  the second element
        :return: True if the first argument goes higher in the heap than the second, False otherwise
        """
        return x1.priority > x2.priority

    @staticmethod
    def parent_index(idx):
        """
        Compute the index of the parent element.
        :param idx: the index of the element whose parent index must be computed
        :return: the parent index
        """
        return (idx - 1) // 2

    @staticmethod
    def children_indices(idx):
        """
        Compute the index of the child elements.
        :param idx: the index of the element whose children indices must be computed
        :return: the children index
        """
        child_index = idx * 2 + 1
        return child_index, child_index + 1

    @staticmethod
    def swap(array, index_1, index_2, update_fc):
        """
        Swap two elements in the heap.
        :param array: the list containing the heap elements
        :param index_1: index of the first element
        :param index_2: index of the second element
        :param update_fc: the function that update the sum-tree to reflect changes to the heap
        """

        # Update the sum-tree.
        update_fc(index_1, index_2)

        # Update the heap indices.
        array[index_2].heap_index = index_1
        array[index_1].heap_index = index_2

        # Swap the elements in the heap.
        tmp = array[index_2]
        array[index_2] = array[index_1]
        array[index_1] = tmp

    @staticmethod
    def push(array, element, update_fc):
        """
        Add an element to the heap.
        :param array: the list containing the heap elements
        :param element: the new element
        :param update_fc: the function that update the sum-tree to reflect changes to the heap
        :return: the index in the heap where the new element was added
        """

        # Add the value to the heap.
        idx = len(array)
        array.append(element)
        update_fc(idx)
        if idx == 0:
            return idx

        # Ensure the new value satisfies the heap property.
        parent_index = MaxHeap.parent_index(idx)
        while MaxHeap.is_higher(element, array[parent_index]):
            MaxHeap.swap(array, idx, parent_index, update_fc)
            idx = parent_index
            if idx == 0:
                break
            parent_index = MaxHeap.parent_index(idx)
        return idx

    @staticmethod
    def ensure_children_consistency(array, idx, update_fc):
        """
        Ensure that the heap property is satisfied, by pulling the new element down as many times as necessary.
        :param array: the list containing the heap elements
        :param idx: the index of the new element in the heap
        :param update_fc: the function that update the sum-tree to reflect changes to the heap
        :return: the index of the new element in the heap
        """

        while True:

            # Compute the children indices.
            id_child_1, id_child_2 = MaxHeap.children_indices(idx)

            # Check if the new element reached the end of the heap.
            heap_size = len(array)
            if id_child_1 >= heap_size:
                break

            # Retrieve the index of the child that should be highest in the heap.
            if id_child_2 >= heap_size:
                higher_child = id_child_1
            elif MaxHeap.is_higher(array[id_child_1], array[id_child_2]):
                higher_child = id_child_1
            else:
                higher_child = id_child_2

            # Check if the new element should be pulled down, if not, stop the loop iteration.
            if MaxHeap.is_higher(array[higher_child], array[idx]):
                MaxHeap.swap(array, idx, higher_child, update_fc)
                idx = higher_child
            else:
                break

        # Return the index of the new element node.
        return idx

    @staticmethod
    def ensure_consistency(array, idx, update_fc):
        """
        Ensure that the heap property is satisfied, by pulling the new element up or down as many times as necessary.
        :param array: the list containing the heap elements
        :param idx: the index of the new element in the heap
        :param update_fc: the function that update the sum-tree to reflect changes to the heap
        :return: the index of the new element in the heap
        """

        # Ensure that the heap property is satisfied, by pulling the new element down as many times as necessary.
        idx = MaxHeap.ensure_children_consistency(array, idx, update_fc)

        # Ensure that the heap property is satisfied, by pulling the new element up as many times as necessary.
        return MaxHeap.ensure_parent_consistency(array, idx, update_fc)

    @staticmethod
    def ensure_parent_consistency(array, idx, update_fc):
        """
        Ensure that the heap property is satisfied, by pulling the new element up as many times as necessary.
        :param array: the list containing the heap elements
        :param idx: the index of the new element in the heap
        :param update_fc: the function that update the sum-tree to reflect changes to the heap
        :return: the index of the new element in the heap
        """

        while True:

            # Compute the parent index.
            parent_idx = MaxHeap.parent_index(idx)

            # Check if the new element reached the start of the heap.
            if idx == 0:
                break

            # Check if the new element should be pulled up, if not, stop the loop iteration.
            if MaxHeap.is_higher(array[idx], array[parent_idx]):
                MaxHeap.swap(array, idx, parent_idx, update_fc)
                idx = parent_idx
            else:
                break

        # Return the index of the new element node.
        return idx

    @staticmethod
    def to_str(array, key_fc=None):
        """
        Create a string representation of the tree.
        :param array: the list containing the heap elements
        :param key_fc: a function returning the element's key to be displayed
        :return: a string representing a tree
        """
        if key_fc is None:
            key_fc = lambda x: x
        heap_string = MaxHeap.sub_heap_to_str(array, 0, key_fc)
        return "[]" if heap_string is None else f"[{heap_string}]"

    @staticmethod
    def sub_heap_to_str(array, idx, key_fc):
        """
        Create a sting representing the sub-heap.
        :param array: the list containing the heap elements
        :param idx: the index of the root node of the sub-heap
        :param key_fc: a function returning the element's key to be displayed
        :return: the string representing the sub-heap
        """

        # If the index is outside the heap, return None.
        if idx >= len(array):
            return None

        # Retrieve the representation of the sub-heap of each child.
        index_child_1, index_child_2 = MaxHeap.children_indices(idx)
        sub_heap_1 = MaxHeap.sub_heap_to_str(array, index_child_1, key_fc)
        sub_heap_2 = MaxHeap.sub_heap_to_str(array, index_child_2, key_fc)

        # Return the representation of the sub-heap corresponding to the index specified as parameters.
        if sub_heap_1 is None and sub_heap_2 is None:
            return f"{key_fc(array[idx])}"
        elif sub_heap_2 is None:
            return f"{key_fc(array[idx])} -> [{sub_heap_1}]"
        elif sub_heap_1 is None:
            return f"{key_fc(array[idx])} -> [{sub_heap_2}]"
        else:
            return f"{key_fc(array[idx])} -> [{sub_heap_1}, {sub_heap_2}]"
