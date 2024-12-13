import pytest

from benchmarks.agents.memory.PriorityTree import PriorityTree


class TestPriorityTree:
    """
    A class testing the priority tree of a replay buffer.
    """

    @pytest.mark.parametrize("elements, result", [
        ([0, 0, 0, 0], 0),
        ([0, 1], 1),
        ([10, 0, 0, 0], 10),
        ([1, 10, 100, 1000], 1111),
        ([1, 10, 100, 1000, 2], 1112),
    ])
    def test_sum(self, elements, result):

        # Arrange.
        priority_tree = PriorityTree(capacity=4, initial_priority=1.0, n_children=2)

        # Act.
        for element in elements:
            priority_tree.append(element)

        # Assert.
        assert priority_tree.sum() == result

    @pytest.mark.parametrize("elements, result", [
        ([0, 0, 0, 0], 0),
        ([0, 1], 1),
        ([10, 0, 0, 0], 10),
        ([1, 10, 100, 1000], 1000),
        ([1000, 10, 100, 1, 999], 999),
    ])
    def test_max(self, elements, result):

        # Arrange.
        priority_tree = PriorityTree(capacity=4, initial_priority=1.0, n_children=2)

        # Act.
        for element in elements:
            priority_tree.append(element)

        # Assert.
        assert priority_tree.max() == result

    @pytest.mark.parametrize("elements", [
        ([0, 0, 0, 0]),
        ([0, 1]),
        ([10, 0, 0, 0]),
        ([1000, 10, 100, 1, 999]),
    ])
    def test_clear(self, elements):

        # Arrange.
        priority_tree = PriorityTree(capacity=4, initial_priority=1.0, n_children=2)

        # Act.
        for element in elements:
            priority_tree.append(element)
        priority_tree.clear()

        # Assert.
        assert len(priority_tree) == 0
        assert priority_tree.to_str(priority_tree.max_tree) == "[[0.0, 0.0], [0.0]]"
        assert priority_tree.to_str(priority_tree.sum_tree) == "[[0.0, 0.0], [0.0]]"

    @pytest.mark.parametrize("elements, sum_result, max_result, length_result", [
        ([0, 0, 0, 0], "[[0.0, 0.0], [0.0]]", "[[0.0, 0.0], [0.0]]", 4),
        ([0, 1, 2, 3], "[[1.0, 5.0], [6.0]]", "[[1.0, 3.0], [3.0]]", 4),
        ([0, 1], "[[1.0, 0.0], [1.0]]", "[[1.0, 0.0], [1.0]]", 2),
        ([10, 0, 0, 0], "[[10.0, 0.0], [10.0]]", "[[10.0, 0.0], [10.0]]", 4),
        ([1, 10, 100, 1000], "[[11.0, 1100.0], [1111.0]]", "[[10.0, 1000.0], [1000.0]]", 4),
        ([1000, 10, 100, 1, 999], "[[1009.0, 101.0], [1110.0]]", "[[999.0, 100.0], [999.0]]", 4),
    ])
    def test_append(self, elements, sum_result, max_result, length_result):

        # Arrange.
        priority_tree = PriorityTree(capacity=4, initial_priority=1.0, n_children=2)

        # Act.
        for element in elements:
            priority_tree.append(element)

        # Assert.
        assert len(priority_tree) == length_result
        assert priority_tree.to_str(priority_tree.sum_tree) == sum_result
        assert priority_tree.to_str(priority_tree.max_tree) == max_result

    @pytest.mark.parametrize("elements, set_indices, set_values, sum_result, max_result, length_result", [
        ([0, 0, 0, 0], [2, 3], [10, 5], "[[0.0, 15.0], [15.0]]", "[[0.0, 10.0], [10.0]]", 4),
        ([0, 1, 2, 3], [1, 2], [3, 0], "[[3.0, 3.0], [6.0]]", "[[3.0, 3.0], [3.0]]", 4),
        ([0, 1], [0, 1], [2, -1], "[[1.0, 0.0], [1.0]]", "[[2.0, 0.0], [2.0]]", 2),
        ([10, 0, 0, 0], [0, 0], [0, -1], "[[-1.0, 0.0], [-1.0]]", "[[0.0, 0.0], [0.0]]", 4),
        ([1, 10, 100, 1000], [3], [-1], "[[11.0, 99.0], [110.0]]", "[[10.0, 100.0], [100.0]]", 4),
        ([1000, 10, 100, 1, 999], [0, 1, 2], [0, 0, 0], "[[999.0, 0.0], [999.0]]", "[[999.0, 0.0], [999.0]]", 4),
    ])
    def test_setitem(self, elements, set_indices, set_values, sum_result, max_result, length_result):

        # Arrange.
        priority_tree = PriorityTree(capacity=4, initial_priority=1.0, n_children=2)

        # Act.
        for element in elements:
            priority_tree.append(element)
        for index, value in zip(set_indices, set_values):
            priority_tree[index] = value

        # Assert.
        assert len(priority_tree) == length_result
        assert priority_tree.to_str(priority_tree.sum_tree) == sum_result
        assert priority_tree.to_str(priority_tree.max_tree) == max_result

    @pytest.mark.parametrize("elements, results, length_result", [
        ([0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], 4),
        ([0, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 3], 4),
        ([0, 1], [0, 1, 0, 0, 0, 1, 0, 0], 2),
        ([10, 0, 0, 0], [10, 0, 0, 0, 10, 0, 0, 0], 4),
        ([1, 10, 100, 1000], [1, 10, 100, 1000, 1, 10, 100, 1000], 4),
        ([1000, 10, 100, 1, 999], [10, 100, 1, 999, 10, 100, 1, 999], 4),
    ])
    def test_getitem(self, elements, results, length_result):

        # Arrange.
        priority_tree = PriorityTree(capacity=4, initial_priority=1.0, n_children=2)

        # Act.
        for element in elements:
            priority_tree.append(element)

        # Assert.
        assert len(priority_tree) == length_result
        for i, priority in enumerate(results):
            assert priority_tree[i - 4] == priority

    @pytest.mark.parametrize("index, n_children, result", [
        (0, 10, 0), (9, 10, 0), (10, 10, 1), (19, 10, 1), (20, 10, 2),
        (0, 5, 0), (4, 5, 0), (5, 5, 1), (9, 5, 1), (10, 5, 2),
    ])
    def test_parent_index(self, index, n_children, result):

        # Arrange.
        priority_tree = PriorityTree(capacity=100, initial_priority=1, n_children=n_children)

        # Assert.
        assert priority_tree.parent_index(index) == result

    @pytest.mark.parametrize("elements, indices, new_values, result", [
        ([1] * 8, [0, 1, 2, 3, 4, 5, 6, 7], [0] * 8, 0),
        ([1] * 8, [0, 1, 2, 3, 4, 5, 6, 7], [0.5] * 8, 4),
        ([1] * 6, [0, 1, 2, 3, 4, 5], [0.5] * 6, 3),
        ([1e9] * 6, [0, 1, 2, 3, 4, 5], [0.1232] * 6, 0.1232 * 6),
    ])
    def test_update_sum_tree(self, elements, indices, new_values, result):

        # Arrange.
        priority_tree = PriorityTree(capacity=8, initial_priority=1, n_children=2)

        # Act.
        for element in elements:
            priority_tree.append(element)
        for index, value in zip(indices, new_values):
            priority_tree.update_sum_tree(index, value)
            priority_tree.update_max_tree(index, value)
            priority_tree.priorities[index] = value

        # Assert.
        assert abs(priority_tree.sum() - result) < 0.001

    def test_sample_indices(self):

        # Arrange.
        elements = [0, 1, 2, 0]
        priority_tree = PriorityTree(capacity=4, initial_priority=1.0, n_children=2)

        # Act.
        for element in elements:
            priority_tree.append(element)
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

    @pytest.mark.parametrize("capacity, n_children, elements, priority, result", [
        (4, 2, [1, 1, 1, 1], 0.0, 0),
        (4, 2, [1, 1, 1, 1], 0.2, 0),
        (4, 2, [1, 1, 1, 1], 1.5, 1),
        (4, 2, [1, 1, 1, 1], 2.3, 2),
        (4, 2, [1, 1, 1, 1], 3.9, 3),
        (4, 2, [1, 1, 1, 1], 4.0, 3),
        (4, 2, [1, 1, 1, 1], 10.0, 3),
        (4, 2, [1, 1, 1, 1, 1], 0.0, 3),
        (4, 2, [1, 1, 1, 1, 1], 0.2, 3),
        (4, 2, [1, 1, 1, 1, 1], 1.5, 0),
        (4, 2, [1, 1, 1, 1, 1], 2.3, 1),
        (4, 2, [1, 1, 1, 1, 1], 3.9, 2),
        (4, 2, [1, 1, 1, 1, 1], 4.0, 2),
        (4, 2, [1, 1, 1, 1, 1], 10.0, 2),
        (8, 2, [1, 2, 1, 2, 1, 2, 1, 2], 0.0, 0),
        (8, 2, [1, 2, 1, 2, 1, 2, 1, 2], 0.2, 0),
        (8, 2, [1, 2, 1, 2, 1, 2, 1, 2], 1.5, 1),
        (8, 2, [1, 2, 1, 2, 1, 2, 1, 2], 2.3, 1),
        (8, 2, [1, 2, 1, 2, 1, 2, 1, 2], 3.9, 2),
        (8, 2, [1, 2, 1, 2, 1, 2, 1, 2], 7.1, 5),
        (8, 2, [1, 2, 1, 2, 1, 2, 1, 2], 15.0, 7),
        (4, 3, [1, 1, 1, 1], 0.0, 0),
        (4, 3, [1, 1, 1, 1], 0.2, 0),
        (4, 3, [1, 1, 1, 1], 1.5, 1),
        (4, 3, [1, 1, 1, 1], 2.3, 2),
        (4, 3, [1, 1, 1, 1], 3.9, 3),
        (4, 3, [1, 1, 1, 1], 4.0, 3),
        (4, 3, [1, 1, 1, 1], 10.0, 3),
        (4, 5, [1, 1, 1, 1, 1], 0.0, 3),
        (4, 3, [1, 1, 1, 1, 1], 0.2, 3),
        (4, 3, [1, 1, 1, 1, 1], 1.5, 0),
        (4, 3, [1, 1, 1, 1, 1], 2.3, 1),
        (4, 7, [1, 1, 1, 1, 1], 3.9, 2),
        (4, 3, [1, 1, 1, 1, 1], 4.0, 2),
        (4, 4, [1, 1, 1, 1, 1], 10.0, 2),
        (8, 3, [1, 2, 1, 2, 1, 2, 1, 2], 0.0, 0),
        (8, 3, [1, 2, 1, 2, 1, 2, 1, 2], 0.2, 0),
        (8, 3, [1, 2, 1, 2, 1, 2, 1, 2], 1.5, 1),
        (8, 3, [1, 2, 1, 2, 1, 2, 1, 2], 2.3, 1),
        (8, 5, [1, 2, 1, 2, 1, 2, 1, 2], 3.9, 2),
        (8, 3, [1, 2, 1, 2, 1, 2, 1, 2], 7.1, 5),
        (8, 3, [1, 2, 1, 2, 1, 2, 1, 2], 15.0, 7),
        (8, 2, [1, 3, 3, 1, 1, 1], 0.0, 0),
        (8, 2, [1, 3, 3, 1, 1, 1], 0.2, 0),
        (8, 2, [1, 3, 3, 1, 1, 1], 1.5, 1),
        (8, 2, [1, 3, 3, 1, 1, 1], 2.3, 1),
        (8, 2, [1, 3, 3, 1, 1, 1], 3.9, 1),
        (8, 2, [1, 3, 3, 1, 1, 1], 4.2, 2),
        (8, 2, [1, 3, 3, 1, 1, 1], 5.7, 2),
        (8, 2, [1, 3, 3, 1, 1, 1], 6.8, 2),
        (8, 2, [1, 3, 3, 1, 1, 1], 11.0, 5),
    ])
    def test_tower_sampling(self, capacity, n_children, elements, priority, result):

        # Arrange.
        priority_tree = PriorityTree(capacity=capacity, initial_priority=1, n_children=n_children)

        # Act.
        for element in elements:
            priority_tree.append(element)

        # Assert.
        assert priority_tree.tower_sampling(priority) == result
