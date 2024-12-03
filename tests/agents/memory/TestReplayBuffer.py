import math

import pytest
import torch

import benchmarks
from benchmarks.agents.memory.ReplayBuffer import ReplayBuffer, Experience, MaxHeap, SumTree, PriorityTree


class MockElement:
    """
    A mock buffer element.
    """

    def __init__(self, index=None, frame=None, action=None, reward=None, done=None, priority=None, heap_index=None):
        self.index = index
        self.frame = frame
        self.action = action
        self.reward = reward
        self.done = done
        self.priority = priority
        self.index = index
        self.heap_index = heap_index


class MockSumTree:
    """
    A mock sum-tree.
    """

    def __init__(self):
        self.updates = []

    def update(self, index_1, index_2=None):
        self.updates.append(index_1)
        self.updates.append(index_2)


class MockReplayBuffer:
    """
    A mock replay buffer.
    """

    def __init__(self, capacity=4, initial_priority=10, length=4):
        self.stack_size = 4
        self.capacity = capacity
        self.initial_priority = initial_priority
        self.length = length

    def __len__(self):
        return self.length

    @staticmethod
    def get_experience_index(index):
        return index


class MockPriorityTree:
    """
    A mock priority tree.
    """

    def __init__(self, priorities):
        self.priorities = [
            MockElement(priority=priority, heap_index=i, index=i + 10)
            for i, priority in enumerate(priorities)
        ]


class TestReplayBuffer:
    """
    Class testing the replay buffer with support for prioritization and multistep Q-learning.
    """

    @pytest.fixture
    def experiences_10(self):
        # Create a list of ten experiences.
        return [
            Experience(
                obs=torch.arange(0, 4).unsqueeze(dim=1).unsqueeze(dim=2).repeat(1, 10, 10) + index,
                action=index, reward=index, done=False, next_obs=torch.zeros([4, 10, 10]) + index + 4
            ) for index in range(10)
        ]

    @pytest.mark.parametrize("initial_priority, result", [(None, False), (10, True)])
    def test_prioritized(self, initial_priority, result):

        # Arrange.
        buffer = ReplayBuffer(initial_priority=initial_priority)

        # Assert.
        assert buffer.prioritized() == result

    @pytest.mark.parametrize("experiences, n_experiences, n_elements", [
        ([], 0, 0),
        ([Experience(torch.zeros([4, 10, 10]), 0, 0.0, False, torch.zeros([4, 10, 10]))], 1, 5),
        ([Experience(torch.zeros([4, 10, 10]), 0, 0.0, False, torch.zeros([4, 10, 10])) for _ in range(2)], 2, 6),
        ([Experience(torch.zeros([4, 10, 10]), 0, 0.0, False, torch.zeros([4, 10, 10])) for _ in range(3)], 3, 7),
        ([Experience(torch.zeros([4, 10, 10]), 0, 0.0, False, torch.zeros([4, 10, 10])) for _ in range(10)], 10, 14),
    ])
    def test_len(self, experiences, n_experiences, n_elements):

        # Arrange.
        buffer = ReplayBuffer(stack_size=4)

        # Act.
        for experience in experiences:
            buffer.append(experience)

        # Assert.
        assert len(buffer) == n_experiences
        assert len(buffer.buffer) == n_elements

    @pytest.mark.parametrize("experiences", [
        ([]),
        ([Experience(torch.zeros([4, 10, 10]), 0, 0.0, False, torch.zeros([4, 10, 10]))]),
        ([Experience(torch.zeros([4, 10, 10]), 0, 0.0, False, torch.zeros([4, 10, 10])) for _ in range(2)]),
        ([Experience(torch.zeros([4, 10, 10]), 0, 0.0, False, torch.zeros([4, 10, 10])) for _ in range(3)]),
        ([Experience(torch.zeros([4, 10, 10]), 0, 0.0, False, torch.zeros([4, 10, 10])) for _ in range(10)]),
    ])
    def test_clear(self, experiences):

        # Arrange.
        buffer = ReplayBuffer(stack_size=4)

        # Act.
        for experience in experiences:
            buffer.append(experience)
        buffer.clear()

        # Assert.
        assert len(buffer) == 0
        assert len(buffer.buffer) == 0

    @pytest.mark.parametrize("index, experience", [(
        index, Experience(
            obs=torch.arange(0, 4).unsqueeze(dim=1).unsqueeze(dim=2).repeat(1, 10, 10) + index * 4,
            action=0, reward=0.0, done=False, next_obs=torch.ones([4, 10, 10]) * -1
        )
    ) for index in range(10)])
    def test_create_buffer_element(self, index, experience):

        # Arrange.
        buffer = ReplayBuffer(stack_size=4)

        # Acts.
        elements = [buffer.create_buffer_element(experience, f_index) for f_index in range(-1, 4)]

        # Assert.
        current_id = 0
        for f_index, element in enumerate(elements):
            if f_index > 0:
                assert element.frame[0][0][0] == (f_index - 1) + index * 4
            else:
                assert element.frame[0][0][0] == -1
            assert current_id == element.index
            current_id += 1

    @pytest.mark.parametrize("n_steps", [1, 3])
    def test_get_experience(self, experiences_10, n_steps):

        # Arrange.
        buffer = ReplayBuffer(stack_size=4, n_steps=n_steps, gamma=1)

        # Acts.
        for experience in experiences_10:
            buffer.append(experience)

        # Assert.
        for index in range(11 - n_steps):
            obs, action, reward, done, next_obs = buffer.get_experience(index)
            for f_index in range(0, 4):
                assert obs[f_index][0][0] == index + f_index
                assert next_obs[f_index][0][0] == index + f_index + n_steps
            assert action == index

            # Cumulated reward over n steps, assuming reward_t=t and gamma=1:
            #   reward_t^n = sum_i=1^n reward_t + i - 1
            #              = n * [reward_t + (n - 1) / 2]
            assert reward == n_steps * (index + (n_steps - 1) / 2)

    def test_get_experience_index(self, experiences_10):

        # Arrange.
        buffer = ReplayBuffer(capacity=5, stack_size=4)

        # Acts.
        for experience in experiences_10:
            buffer.append(experience)

        # Assert.
        for experience_index, element_index in enumerate(range(5, 10)):
            assert buffer.get_experience_index(element_index) == experience_index

    @pytest.mark.parametrize("n_steps", [1, 3])
    def test_append(self, experiences_10, n_steps):

        # Arrange.
        buffer = ReplayBuffer(stack_size=4, n_steps=n_steps, gamma=1)

        # Acts.
        for experience in experiences_10:
            buffer.append(experience)

        # Assert.
        for index in range(8):
            obs, action, reward, done, next_obs = buffer.get_experience(index)
            for f_index in range(0, 4):
                assert obs[f_index][0][0] == index + f_index
                assert next_obs[f_index][0][0] == index + f_index + n_steps
            assert action == index

            # Cumulated reward over n steps, assuming reward_t=t and gamma=1:
            #   reward_t^n = sum_i=1^n reward_t + i - 1
            #              = n * [reward_t + (n - 1) / 2]
            assert reward == n_steps * (index + (n_steps - 1) / 2)

    def test_report(self, experiences_10):

        # Arrange.
        buffer = ReplayBuffer(capacity=4, initial_priority=1, omega_is=0.5)

        # Acts.
        for experience in experiences_10[0:buffer.capacity]:
            buffer.append(experience)
        buffer.sample(2)
        loss = 2 * torch.ones([2]).to(benchmarks.device())
        loss = buffer.report(loss)

        # Assert.
        for i in range(2):
            assert abs(loss[i].item() - 2.0) < 0.0001

    def test_update_priorities(self, experiences_10):

        # Arrange.
        buffer = ReplayBuffer(capacity=4, initial_priority=1)

        # Acts.
        for experience in experiences_10[0:buffer.capacity]:
            buffer.append(experience)
        buffer.sample(2)
        indices = buffer.indices
        loss = torch.zeros([2]).to(benchmarks.device())
        buffer.update_priorities(loss)

        # Assert.
        for i in range(buffer.capacity):
            element_idx = i + buffer.stack_size + buffer.n_steps - 1
            if i in indices:
                assert buffer.buffer[element_idx].priority == 0
            else:
                assert buffer.buffer[element_idx].priority == 1


class TestPriorityTree:
    """
    A class testing the priority tree of a replay buffer.
    """

    @pytest.mark.parametrize("elements, result", [
        ([MockElement(priority=0), MockElement(priority=0), MockElement(priority=0), MockElement(priority=0)], 0),
        ([MockElement(priority=0), MockElement(priority=1)], 1),
        ([MockElement(priority=10), MockElement(priority=0), MockElement(priority=0), MockElement(priority=0)], 10),
        ([MockElement(priority=1), MockElement(priority=10), MockElement(priority=100), MockElement(priority=1000)], 1111),
    ])
    def test_sum(self, elements, result):

        # Arrange.
        buffer = MockReplayBuffer(capacity=4, length=len(elements))
        priority_tree = PriorityTree(buffer, n_children=2)

        # Act.
        for element in elements:
            element.heap_index = priority_tree.push(element)

        # Assert.
        assert priority_tree.sum() == result

    @pytest.mark.parametrize("elements, result", [
        ([MockElement(priority=0), MockElement(priority=0), MockElement(priority=0), MockElement(priority=0)], 0),
        ([MockElement(priority=0), MockElement(priority=1)], 1),
        ([MockElement(priority=10), MockElement(priority=0), MockElement(priority=0), MockElement(priority=0)], 10),
        ([MockElement(priority=1), MockElement(priority=10), MockElement(priority=100), MockElement(priority=1000)], 1000),
    ])
    def test_max(self, elements, result):

        # Arrange.
        buffer = MockReplayBuffer(capacity=4, length=len(elements))
        priority_tree = PriorityTree(buffer, n_children=2)

        # Act.
        for element in elements:
            element.heap_index = priority_tree.push(element)

        # Assert.
        assert priority_tree.max() == result

    def test_clear(self):

        # Arrange.
        elements = [MockElement(priority=0), MockElement(priority=0), MockElement(priority=0), MockElement(priority=0)]
        buffer = MockReplayBuffer(capacity=4, length=len(elements))
        priority_tree = PriorityTree(buffer, n_children=2)

        # Act.
        for element in elements:
            element.heap_index = priority_tree.push(element)
        priority_tree.clear()

        # Assert.
        assert len(priority_tree.priorities) == 0
        assert str(priority_tree.tree) == "[[0.0, 0.0], [0.0]]"

    def test_push(self):

        # Arrange.
        elements = [MockElement(priority=0), MockElement(priority=1), MockElement(priority=2), MockElement(priority=3)]
        buffer = MockReplayBuffer(capacity=4, length=len(elements))
        priority_tree = PriorityTree(buffer, n_children=2)

        # Act.
        for element in elements:
            element.heap_index = priority_tree.push(element)

        # Assert.
        assert len(priority_tree.priorities) == 4
        for i, priority in enumerate([3, 2, 1, 0]):
            assert priority_tree.priorities[i].priority == priority
        assert str(priority_tree.tree) == "[[5.0, 1.0], [6.0]]"

    def test_replace(self):

        # Arrange.
        elements = [MockElement(priority=0), MockElement(priority=1), MockElement(priority=2), MockElement(priority=3)]
        buffer = MockReplayBuffer(capacity=4, length=len(elements))
        priority_tree = PriorityTree(buffer, n_children=2)

        # Act.
        for element in elements:
            element.heap_index = priority_tree.push(element)
        priority_tree.replace(priority_tree.priorities[0], MockElement(priority=-1))

        # Assert.
        assert len(priority_tree.priorities) == 4
        for i, priority in enumerate([2, 0, 1, -1]):
            assert priority_tree.priorities[i].priority == priority
        assert str(priority_tree.tree) == "[[2.0, 0.0], [2.0]]"

    def test_set(self):

        # Arrange.
        elements = [MockElement(priority=0), MockElement(priority=1), MockElement(priority=2), MockElement(priority=3)]
        buffer = MockReplayBuffer(capacity=4, length=len(elements))
        priority_tree = PriorityTree(buffer, n_children=2)

        # Act.
        for element in elements:
            element.heap_index = priority_tree.push(element)
        priority_tree.set(priority_tree.priorities[0], -1)

        # Assert.
        assert len(priority_tree.priorities) == 4
        for i, priority in enumerate([2, 0, 1, -1]):
            assert priority_tree.priorities[i].priority == priority
        assert str(priority_tree.tree) == "[[2.0, 0.0], [2.0]]"

    def test_sample_indices(self):

        # Arrange.
        elements = [MockElement(index=4, priority=0), MockElement(index=5, priority=1), MockElement(index=6, priority=2), MockElement(index=7, priority=0)]
        buffer = MockReplayBuffer(capacity=4, length=len(elements))
        priority_tree = PriorityTree(buffer, n_children=2)

        # Act.
        for element in elements:
            element.heap_index = priority_tree.push(element)
        indices = priority_tree.sample_indices(10000)

        # Assert.
        probs = []
        for i in range(4):
            probs.append(indices.count(i))
        total = sum(probs)
        for i in range(4):
            probs[i] /= total

        for prob, result in zip(probs, [0.0, 1/3, 2/3, 0.0]):
            assert abs(prob - result) < 0.01


class TestSumTree:
    """
    A class testing the sum-tree of a replay buffer.
    """

    @pytest.mark.parametrize("index, n_children, result", [
        (0, 10, 0), (9, 10, 0), (10, 10, 1), (19, 10, 1), (20, 10, 2),
        (0, 5, 0), (4, 5, 0), (5, 5, 1), (9, 5, 1), (10, 5, 2),
    ])
    def test_parent_index(self, index, n_children, result):

        # Arrange.
        tree = SumTree(capacity=100, n_children=n_children, priority_tree=None)

        # Assert.
        assert tree.parent_index(index) == result

    @pytest.mark.parametrize("priorities, indices_1, indices_2, result", [
        ([5.0, 0, 0, 0], [0], [0], "[[5.0, 0.0], [5.0]]"),
        ([5.0, 0, 0, 0], [0], [2], "[[0.0, 5.0], [5.0]]"),
        ([5.0, 0, 0], [3], [-1], "[[5.0, 1.0], [6.0]]"),
        ([1.0, 2.0, 3.0, 4.0], [0, 1], [2, 3], "[[7.0, 3.0], [10.0]]"),
    ])
    def test_update(self, priorities, indices_1, indices_2, result):

        # Arrange.
        priority_tree = MockPriorityTree(priorities)
        capacity = 4
        tree = SumTree(capacity=capacity, n_children=2, priority_tree=priority_tree)

        # Act.
        while len(priority_tree.priorities) != capacity:
            priority_tree.priorities.append(MockElement(priority=1.0, heap_index=len(priority_tree.priorities)))
        for index_1, index_2 in zip(indices_1, indices_2):
            tree.update(index_1, index_2)

        # Assert.
        assert str(tree) == result

    @pytest.mark.parametrize("priorities, indices, new_priorities, result", [
        ([0, 0, 0, 0], [0], [5], "[[5.0, 0.0], [5.0]]"),
        ([0, 0, 0, 0], [3], [5], "[[0.0, 5.0], [5.0]]"),
        ([1, 0, 0, 0], [0], [5], "[[5.0, 0.0], [5.0]]"),
        ([1, 0, 0, 0], [1], [5], "[[6.0, 0.0], [6.0]]"),
        ([1, 0, 0, 0], [1, 2, 3], [5, 3, 10], "[[6.0, 13.0], [19.0]]"),
    ])
    def test_set(self, priorities, indices, new_priorities, result):

        # Arrange.
        priority_tree = MockPriorityTree(priorities)
        tree = SumTree(capacity=4, n_children=2, priority_tree=priority_tree)

        # Act.
        for index, new_priority in zip(indices, new_priorities):
            tree.set(index, new_priority)

        # Assert.
        assert str(tree) == result

    @pytest.mark.parametrize("priorities, priority, result", [
        ([1, 1, 1, 1], 0.0, 10),
        ([1, 1, 1, 1], 0.2, 10),
        ([1, 1, 1, 1], 1.5, 11),
        ([1, 1, 1, 1], 2.3, 12),
        ([1, 1, 1, 1], 3.9, 13),
        ([1, 1, 1, 1], 4.0, 13),
        ([1, 1, 1, 1], 10.0, 13),
    ])
    def test_tower_sampling(self, priorities, priority, result):

        # Arrange.
        priority_tree = MockPriorityTree(priorities)
        tree = SumTree(capacity=4, n_children=2, priority_tree=priority_tree)

        # Assert.
        assert tree.tower_sampling(priority) == result


class TestMaxHeap:
    """
    A class testing the max-heap of a replay buffer.
    """

    @pytest.mark.parametrize("x1, x2, result", [
        (MockElement(priority=0), MockElement(priority=1), False),
        (MockElement(priority=0), MockElement(priority=0), False),
        (MockElement(priority=1), MockElement(priority=0), True),
    ])
    def test_is_higher(self, x1, x2, result):

        # Assert.
        assert MaxHeap.is_higher(x1, x2) == result

    @pytest.mark.parametrize("idx, result", [
        (0, -1), (1, 0), (2, 0), (3, 1), (4, 1), (5, 2), (6, 2), (7, 3), (8, 3)
    ])
    def test_parent_index(self, idx, result):

        # Assert.
        assert MaxHeap.parent_index(idx) == result

    @pytest.mark.parametrize("idx, result", [
        (0, (1, 2)), (1, (3, 4)), (2, (5, 6)), (3, (7, 8))
    ])
    def test_children_indices(self, idx, result):

        # Assert.
        assert MaxHeap.children_indices(idx) == result

    @pytest.mark.parametrize("array, index_1, index_2, result_str, result_updates", [
        ([MockElement(priority=42)], 0, 0, "[42]", [0, 0]),
        ([MockElement(priority=20), MockElement(priority=19), MockElement(priority=18), MockElement(priority=17), MockElement(priority=16), MockElement(priority=15), MockElement(priority=14)], 0, 6, "[14 -> [19 -> [17, 16], 18 -> [15, 20]]]", [0, 6]),
        ([MockElement(priority=20), MockElement(priority=19), MockElement(priority=18), MockElement(priority=17), MockElement(priority=16), MockElement(priority=15), MockElement(priority=14)], 2, 5, "[20 -> [19 -> [17, 16], 15 -> [18, 14]]]", [2, 5]),
        ([MockElement(priority=100), MockElement(priority=19), MockElement(priority=36), MockElement(priority=17), MockElement(priority=3), MockElement(priority=25), MockElement(priority=1), MockElement(priority=2), MockElement(priority=7)], 1, 7, "[100 -> [2 -> [17 -> [19, 7], 3], 36 -> [25, 1]]]", [1, 7])
    ])
    def test_swap(self, array, index_1, index_2, result_str, result_updates):

        # Arrange.
        tree = MockSumTree()

        # Act.
        MaxHeap.swap(array, index_1, index_2, tree.update)

        # Assert.
        assert MaxHeap.to_str(array, key_fc=lambda x: x.priority) == result_str
        assert len(tree.updates) == len(result_updates)
        for idx, result_idx in zip(tree.updates, result_updates):
            assert idx == result_idx

    @pytest.mark.parametrize("array, element, result_str, result_index, result_updates", [
        ([], MockElement(priority=0), "[0]", 0, [0, None]),
        ([MockElement(priority=42)], MockElement(priority=0), "[42 -> [0]]", 1, [1, None]),
        ([MockElement(priority=42)], MockElement(priority=100), "[100 -> [42]]", 0, [1, None, 1, 0]),
        ([MockElement(priority=20), MockElement(priority=19), MockElement(priority=18), MockElement(priority=17), MockElement(priority=16), MockElement(priority=15), MockElement(priority=14)], MockElement(priority=0), "[20 -> [19 -> [17 -> [0], 16], 18 -> [15, 14]]]", 7, [7, None]),
        ([MockElement(priority=100), MockElement(priority=19), MockElement(priority=36), MockElement(priority=17), MockElement(priority=3), MockElement(priority=25), MockElement(priority=1), MockElement(priority=2), MockElement(priority=7)], MockElement(priority=200), "[200 -> [100 -> [17 -> [2, 7], 19 -> [3]], 36 -> [25, 1]]]", 0, [9, None, 9, 4, 4, 1, 1, 0])
    ])
    def test_push(self, array, element, result_str, result_index, result_updates):

        # Arrange.
        tree = MockSumTree()

        # Act.
        index = MaxHeap.push(array, element, tree.update)

        # Assert.
        assert index == result_index
        assert MaxHeap.to_str(array, key_fc=lambda x: x.priority) == result_str
        assert len(tree.updates) == len(result_updates)
        for idx, result_idx in zip(tree.updates, result_updates):
            assert idx == result_idx

    @pytest.mark.parametrize("array, index, result_str, result_index, result_updates", [
        ([MockElement(priority=42)], 0, "[42]", 0, []),
        ([MockElement(priority=20), MockElement(priority=19), MockElement(priority=18), MockElement(priority=17), MockElement(priority=16), MockElement(priority=15), MockElement(priority=14)], 0, "[20 -> [19 -> [17, 16], 18 -> [15, 14]]]", 0, []),
        ([MockElement(priority=0), MockElement(priority=19), MockElement(priority=36), MockElement(priority=17), MockElement(priority=3), MockElement(priority=25), MockElement(priority=1), MockElement(priority=2), MockElement(priority=7)], 0, "[36 -> [19 -> [17 -> [2, 7], 3], 25 -> [0, 1]]]", 5, [0, 2, 2, 5])
    ])
    def test_ensure_children_consistency(self, array, index, result_str, result_index, result_updates):

        # Arrange.
        tree = MockSumTree()

        # Act.
        index = MaxHeap.ensure_children_consistency(array, index, tree.update)

        # Assert.
        assert index == result_index
        assert MaxHeap.to_str(array, key_fc=lambda x: x.priority) == result_str
        assert len(tree.updates) == len(result_updates)
        for idx, result_idx in zip(tree.updates, result_updates):
            assert idx == result_idx

    @pytest.mark.parametrize("array, index, result_str, result_index, result_updates", [
        ([MockElement(priority=42)], 0, "[42]", 0, []),
        ([MockElement(priority=20), MockElement(priority=19), MockElement(priority=18), MockElement(priority=17), MockElement(priority=16), MockElement(priority=15), MockElement(priority=14)], 6, "[20 -> [19 -> [17, 16], 18 -> [15, 14]]]", 6, []),
        ([MockElement(priority=100), MockElement(priority=19), MockElement(priority=36), MockElement(priority=17), MockElement(priority=3), MockElement(priority=25), MockElement(priority=200), MockElement(priority=2), MockElement(priority=7)], 6, "[200 -> [19 -> [17 -> [2, 7], 3], 100 -> [25, 36]]]", 0, [6, 2, 2, 0])
    ])
    def test_ensure_parent_consistency(self, array, index, result_str, result_index, result_updates):

        # Arrange.
        tree = MockSumTree()

        # Act.
        index = MaxHeap.ensure_parent_consistency(array, index, tree.update)

        # Assert.
        assert index == result_index
        assert MaxHeap.to_str(array, key_fc=lambda x: x.priority) == result_str
        assert len(tree.updates) == len(result_updates)
        for idx, result_idx in zip(tree.updates, result_updates):
            assert idx == result_idx

    @pytest.mark.parametrize("array, result", [
        ([], "[]"),
        ([42], "[42]"),
        ([20, 19, 18, 17, 16, 15, 14], "[20 -> [19 -> [17, 16], 18 -> [15, 14]]]"),
        ([100, 19, 36, 17, 3, 25, 1, 2, 7], "[100 -> [19 -> [17 -> [2, 7], 3], 36 -> [25, 1]]]")
    ])
    def test_to_str(self, array, result):

        # Assert.
        assert MaxHeap.to_str(array) == result
