// Copyright 2025 Theophile Champion. No Rights Reserved.

#include "agents/memory/test_priority_tree.hpp"
#include <torch/extension.h>

#include <memory>
#include <string>
#include <vector>

#include "relab_test.hpp"

using namespace relab::agents::memory;

namespace relab::test::agents::memory {

PriorityTreeParameters::PriorityTreeParameters(const std::initializer_list<float> &elements, float result) :
    elements(elements), result(result) {}

PriorityTreeParameters::PriorityTreeParameters() : PriorityTreeParameters({}, 0) {}

void TestPriorityTree::SetUp() {
  // Create a priority tree.
  this->params = GetParam();
  int capacity = 4;
  int initial_priority = 1.0;
  int n_children = 2;
  this->priority_tree = std::make_unique<PriorityTree>(capacity, initial_priority, n_children);
}

TEST_P(TestPriorityTree, TestSum) {  // cppcheck-suppress[syntaxError]
  // Act.
  for (auto element : params.elements) {
    priority_tree->append(element);
  }

  // Assert.
  EXPECT_EQ(priority_tree->sum(), params.result);
}

TEST_P(TestPriorityTree, TestClear) {  // cppcheck-suppress[syntaxError]
  // Act.
  for (auto element : params.elements) {
    priority_tree->append(element);
  }
  priority_tree->clear();

  // Assert.
  EXPECT_EQ(priority_tree->size(), 0);
  EXPECT_EQ(priority_tree->maxTreeToStr(), "[[0.0, 0.0], [0.0]]");
  EXPECT_EQ(priority_tree->sumTreeToStr(), "[[0.0, 0.0], [0.0]]");
}

TEST_P(TestPriorityTree, TestSaveAndLoad) {  // cppcheck-suppress[syntaxError]
  // Add all elements to the priority tree.
  for (auto element : params.elements) {
    priority_tree->append(element);
  }

  // Save the priority tree.
  std::stringstream ss;
  priority_tree->save(ss);

  // Load the priority tree.
  auto loaded_priority_tree = PriorityTree(10, 10, 10);
  loaded_priority_tree.load(ss);

  // Check that the saved and loaded priority trees are identical.
  EXPECT_EQ(*priority_tree, loaded_priority_tree);
}

INSTANTIATE_TEST_SUITE_P(
    UnitTests, TestPriorityTree,
    testing::Values(
        PriorityTreeParameters({0, 0, 0, 0}, 0), PriorityTreeParameters({0, 1}, 1),
        PriorityTreeParameters({10, 0, 0, 0}, 10), PriorityTreeParameters({1, 10, 100, 1000}, 1111),
        PriorityTreeParameters({1, 10, 100, 1000, 2}, 1112)
    )
);

TEST_P(TestPriorityTree2, TestMax) {  // cppcheck-suppress[syntaxError]
  // Arrange.
  auto params = GetParam();
  int capacity = 4;
  int initial_priority = 1.0;
  int n_children = 2;
  auto priority_tree = PriorityTree(capacity, initial_priority, n_children);

  // Act.
  for (auto element : params.elements) {
    priority_tree.append(element);
  }

  // Assert.
  EXPECT_EQ(priority_tree.max(), params.result);
}

INSTANTIATE_TEST_SUITE_P(
    UnitTests, TestPriorityTree2,
    testing::Values(
        PriorityTreeParameters({0, 0, 0, 0}, 0), PriorityTreeParameters({0, 1}, 1),
        PriorityTreeParameters({10, 0, 0, 0}, 10), PriorityTreeParameters({1, 10, 100, 1000}, 1000),
        PriorityTreeParameters({1000, 10, 100, 1, 999}, 999)
    )
);

PriorityTreeParameters4::PriorityTreeParameters4(
    const std::initializer_list<float> &elements, const std::string &sum_result, const std::string &max_result,
    int length_result
) : elements(elements), sum_result(sum_result), max_result(max_result), length_result(length_result) {}

TEST_P(TestPriorityTree4, TestAppend) {  // cppcheck-suppress[syntaxError]
  // Arrange.
  auto params = GetParam();
  int capacity = 4;
  int initial_priority = 1.0;
  int n_children = 2;
  auto priority_tree = PriorityTree(capacity, initial_priority, n_children);

  // Act.
  for (auto element : params.elements) {
    priority_tree.append(element);
  }

  // Assert.
  EXPECT_EQ(priority_tree.size(), params.length_result);
  EXPECT_EQ(priority_tree.sumTreeToStr(), params.sum_result);
  EXPECT_EQ(priority_tree.maxTreeToStr(), params.max_result);
}

INSTANTIATE_TEST_SUITE_P(
    UnitTests, TestPriorityTree4,
    testing::Values(
        PriorityTreeParameters4({0, 0, 0, 0}, "[[0.0, 0.0], [0.0]]", "[[0.0, 0.0], [0.0]]", 4),
        PriorityTreeParameters4({0, 1, 2, 3}, "[[1.0, 5.0], [6.0]]", "[[1.0, 3.0], [3.0]]", 4),
        PriorityTreeParameters4({0, 1}, "[[1.0, 0.0], [1.0]]", "[[1.0, 0.0], [1.0]]", 2),
        PriorityTreeParameters4({10, 0, 0, 0}, "[[10.0, 0.0], [10.0]]", "[[10.0, 0.0], [10.0]]", 4),
        PriorityTreeParameters4({1, 10, 100, 1000}, "[[11.0, 1100.0], [1111.0]]", "[[10.0, 1000.0], [1000.0]]", 4),
        PriorityTreeParameters4({1000, 10, 100, 1, 999}, "[[1009.0, 101.0], [1110.0]]", "[[999.0, 100.0], [999.0]]", 4)
    )
);

PriorityTreeParameters5::PriorityTreeParameters5(
    const std::initializer_list<float> &elements, const std::initializer_list<int> &set_indices,
    const std::initializer_list<float> &set_values, const std::string &sum_result, const std::string &max_result,
    int length_result
) :
    elements(elements), set_indices(set_indices), set_values(set_values), sum_result(sum_result),
    max_result(max_result), length_result(length_result) {}

TEST_P(TestPriorityTree5, TestSetItem) {  // cppcheck-suppress[syntaxError]
  // Arrange.
  auto params = GetParam();
  int capacity = 4;
  int initial_priority = 1.0;
  int n_children = 2;
  auto priority_tree = PriorityTree(capacity, initial_priority, n_children);

  // Act.
  for (auto element : params.elements) {
    priority_tree.append(element);
  }
  for (size_t i = 0; i < params.set_values.size(); i++) {
    priority_tree.set(params.set_indices[i], params.set_values[i]);
  }

  // Assert.
  EXPECT_EQ(priority_tree.size(), params.length_result);
  EXPECT_EQ(priority_tree.sumTreeToStr(), params.sum_result);
  EXPECT_EQ(priority_tree.maxTreeToStr(), params.max_result);
}

INSTANTIATE_TEST_SUITE_P(
    UnitTests, TestPriorityTree5,
    testing::Values(
        PriorityTreeParameters5({0, 0, 0, 0}, {2, 3}, {10, 5}, "[[0.0, 15.0], [15.0]]", "[[0.0, 10.0], [10.0]]", 4),
        PriorityTreeParameters5({0, 1, 2, 3}, {1, 2}, {3, 0}, "[[3.0, 3.0], [6.0]]", "[[3.0, 3.0], [3.0]]", 4),
        PriorityTreeParameters5({0, 1}, {0, 1}, {2, -1}, "[[1.0, 0.0], [1.0]]", "[[2.0, 0.0], [2.0]]", 2),
        PriorityTreeParameters5({10, 0, 0, 0}, {0, 0}, {0, -1}, "[[-1.0, 0.0], [-1.0]]", "[[0.0, 0.0], [0.0]]", 4),
        PriorityTreeParameters5(
            {1, 10, 100, 1000}, {3}, {-1}, "[[11.0, 99.0], [110.0]]", "[[10.0, 100.0], [100.0]]", 4
        ),
        PriorityTreeParameters5(
            {1000, 10, 100, 1, 999}, {0, 1, 2}, {0, 0, 0}, "[[999.0, 0.0], [999.0]]", "[[999.0, 0.0], [999.0]]", 4
        )
    )
);

PriorityTreeParameters6::PriorityTreeParameters6(
    const std::initializer_list<float> &elements, const std::initializer_list<float> &results, int length_result
) : elements(elements), results(results), length_result(length_result) {}

TEST_P(TestPriorityTree6, TestGetItem) {  // cppcheck-suppress[syntaxError]
  // Arrange.
  auto params = GetParam();
  int capacity = 4;
  int initial_priority = 1.0;
  int n_children = 2;
  auto priority_tree = PriorityTree(capacity, initial_priority, n_children);

  // Act.
  for (auto element : params.elements) {
    priority_tree.append(element);
  }

  // Assert.
  EXPECT_EQ(priority_tree.size(), params.length_result);
  for (size_t i = 0; i < params.results.size(); i++) {
    EXPECT_EQ(priority_tree.get(i - params.length_result), params.results[i]);
  }
}

INSTANTIATE_TEST_SUITE_P(
    UnitTests, TestPriorityTree6,
    testing::Values(
        PriorityTreeParameters6({0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, 4),
        PriorityTreeParameters6({0, 1, 2, 3}, {0, 1, 2, 3, 0, 1, 2, 3}, 4),
        PriorityTreeParameters6({0, 1}, {0, 1, 0, 1}, 2),
        PriorityTreeParameters6({10, 0, 0, 0}, {10, 0, 0, 0, 10, 0, 0, 0}, 4),
        PriorityTreeParameters6({1, 10, 100, 1000}, {1, 10, 100, 1000, 1, 10, 100, 1000}, 4),
        PriorityTreeParameters6({1000, 10, 100, 1, 999}, {10, 100, 1, 999, 10, 100, 1, 999}, 4)
    )
);

PriorityTreeParameters7::PriorityTreeParameters7(int index, int n_children, int result) :
    index(index), n_children(n_children), result(result) {}

TEST_P(TestPriorityTree7, TestParentIndex) {  // cppcheck-suppress[syntaxError]
  // Arrange.
  auto params = GetParam();
  int capacity = 100;
  int initial_priority = 1.0;
  auto priority_tree = PriorityTree(capacity, initial_priority, params.n_children);

  // Assert.
  EXPECT_EQ(priority_tree.parentIndex(params.index), params.result);
}

INSTANTIATE_TEST_SUITE_P(
    UnitTests, TestPriorityTree7,
    testing::Values(
        PriorityTreeParameters7(0, 10, 0), PriorityTreeParameters7(9, 10, 0), PriorityTreeParameters7(10, 10, 1),
        PriorityTreeParameters7(19, 10, 1), PriorityTreeParameters7(20, 10, 2), PriorityTreeParameters7(0, 5, 0),
        PriorityTreeParameters7(4, 5, 0), PriorityTreeParameters7(5, 5, 1), PriorityTreeParameters7(9, 5, 1),
        PriorityTreeParameters7(10, 5, 2)
    )
);

PriorityTreeParameters8::PriorityTreeParameters8(
    const std::vector<float> &elements, int max_index, const std::vector<float> &new_values, float result
) : elements(elements), new_values(new_values), result(result) {
  this->indices.resize(max_index);
  std::iota(this->indices.begin(), this->indices.end(), 0);
}

TEST_P(TestPriorityTree8, TestUpdateSumTree) {  // cppcheck-suppress[syntaxError]
  // Arrange.
  auto params = GetParam();
  int capacity = 8;
  int initial_priority = params.elements[0];
  int n_children = 2;
  auto priority_tree = PriorityTree(capacity, initial_priority, n_children);

  // Act.
  for (auto element : params.elements) {
    priority_tree.append(element);
  }
  for (size_t i = 0; i < params.new_values.size(); i++) {
    priority_tree.set(params.indices[i], params.new_values[i]);
  }

  // Assert.
  EXPECT_TRUE(abs(priority_tree.sum() - params.result) < TEST_EPSILON);
}

INSTANTIATE_TEST_SUITE_P(
    UnitTests, TestPriorityTree8,
    testing::Values(
        PriorityTreeParameters8(repeat<float>(1, 8), 8, repeat<float>(0, 8), 0),
        PriorityTreeParameters8(repeat<float>(1, 8), 8, repeat<float>(0.5, 8), 4),
        PriorityTreeParameters8(repeat<float>(1, 6), 6, repeat<float>(0.5, 6), 3),
        PriorityTreeParameters8(repeat<float>(MAX_PRIORITY, 6), 6, repeat<float>(0.1232, 6), 0.1232 * 6)
    )
);

PriorityTreeParameters9::PriorityTreeParameters9(
    int capacity, int n_children, const std::vector<float> &elements, float priority, float result
) : capacity(capacity), n_children(n_children), elements(elements), priority(priority), result(result) {}

TEST(TestPriorityTree, TestSampleIndices) {  // cppcheck-suppress[syntaxError]
  // Arrange.
  int capacity = 4;
  int initial_priority = 1.0;
  int n_children = 2;
  auto priority_tree = PriorityTree(capacity, initial_priority, n_children);

  // Act.
  for (auto element : {0, 1, 2, 0}) {
    priority_tree.append(element);
  }
  auto indices = priority_tree.sampleIndices(50000);

  // Assert.
  torch::Tensor probs = indices.bincount(torch::ones_like(indices), capacity).to(torch::kFloat32);
  auto total = probs.sum();
  for (int i = 0; i < capacity; i++) {
    probs[i] /= total;
  }
  std::vector<float> results = {0.0, 1.0 / 3.0, 2.0 / 3.0, 0.0};
  for (int i = 0; i < capacity; i++) {
    EXPECT_TRUE(abs(probs[i].item<float>() - results[i]) < 0.01);
  }
}

TEST_P(TestPriorityTree9, TestTowerSampling) {  // cppcheck-suppress[syntaxError]
  // Arrange.
  auto params = GetParam();
  int initial_priority = 1.0;
  auto priority_tree = PriorityTree(params.capacity, initial_priority, params.n_children);

  // Act.
  for (auto element : params.elements) {
    priority_tree.append(element);
  }

  // Assert.
  EXPECT_EQ(priority_tree.towerSampling(params.priority), params.result);
}

INSTANTIATE_TEST_SUITE_P(
    UnitTests, TestPriorityTree9,
    testing::Values(
        PriorityTreeParameters9(4, 2, {1, 1, 1, 1}, 0.0, 0), PriorityTreeParameters9(4, 2, {1, 1, 1, 1}, 0.2, 0),
        PriorityTreeParameters9(4, 2, {1, 1, 1, 1}, 1.5, 1), PriorityTreeParameters9(4, 2, {1, 1, 1, 1}, 2.3, 2),
        PriorityTreeParameters9(4, 2, {1, 1, 1, 1}, 3.9, 3), PriorityTreeParameters9(4, 2, {1, 1, 1, 1}, 4.0, 3),
        PriorityTreeParameters9(4, 2, {1, 1, 1, 1}, 10.0, 3), PriorityTreeParameters9(4, 2, {1, 1, 1, 1, 1}, 0.0, 3),
        PriorityTreeParameters9(4, 2, {1, 1, 1, 1, 1}, 0.2, 3), PriorityTreeParameters9(4, 2, {1, 1, 1, 1, 1}, 1.5, 0),
        PriorityTreeParameters9(4, 2, {1, 1, 1, 1, 1}, 2.3, 1), PriorityTreeParameters9(4, 2, {1, 1, 1, 1, 1}, 3.9, 2),
        PriorityTreeParameters9(4, 2, {1, 1, 1, 1, 1}, 4.0, 2), PriorityTreeParameters9(4, 2, {1, 1, 1, 1, 1}, 10.0, 2),
        PriorityTreeParameters9(8, 2, {1, 2, 1, 2, 1, 2, 1, 2}, 0.0, 0),
        PriorityTreeParameters9(8, 2, {1, 2, 1, 2, 1, 2, 1, 2}, 0.2, 0),
        PriorityTreeParameters9(8, 2, {1, 2, 1, 2, 1, 2, 1, 2}, 1.5, 1),
        PriorityTreeParameters9(8, 2, {1, 2, 1, 2, 1, 2, 1, 2}, 2.3, 1),
        PriorityTreeParameters9(8, 2, {1, 2, 1, 2, 1, 2, 1, 2}, 3.9, 2),
        PriorityTreeParameters9(8, 2, {1, 2, 1, 2, 1, 2, 1, 2}, 7.1, 5),
        PriorityTreeParameters9(8, 2, {1, 2, 1, 2, 1, 2, 1, 2}, 15.0, 7),
        PriorityTreeParameters9(4, 3, {1, 1, 1, 1}, 0.0, 0), PriorityTreeParameters9(4, 3, {1, 1, 1, 1}, 0.2, 0),
        PriorityTreeParameters9(4, 3, {1, 1, 1, 1}, 1.5, 1), PriorityTreeParameters9(4, 3, {1, 1, 1, 1}, 2.3, 2),
        PriorityTreeParameters9(4, 3, {1, 1, 1, 1}, 3.9, 3), PriorityTreeParameters9(4, 3, {1, 1, 1, 1}, 4.0, 3),
        PriorityTreeParameters9(4, 3, {1, 1, 1, 1}, 10.0, 3), PriorityTreeParameters9(4, 5, {1, 1, 1, 1, 1}, 0.0, 3),
        PriorityTreeParameters9(4, 3, {1, 1, 1, 1, 1}, 0.2, 3), PriorityTreeParameters9(4, 3, {1, 1, 1, 1, 1}, 1.5, 0),
        PriorityTreeParameters9(4, 3, {1, 1, 1, 1, 1}, 2.3, 1), PriorityTreeParameters9(4, 7, {1, 1, 1, 1, 1}, 3.9, 2),
        PriorityTreeParameters9(4, 3, {1, 1, 1, 1, 1}, 4.0, 2), PriorityTreeParameters9(4, 4, {1, 1, 1, 1, 1}, 10.0, 2),
        PriorityTreeParameters9(8, 3, {1, 2, 1, 2, 1, 2, 1, 2}, 0.0, 0),
        PriorityTreeParameters9(8, 3, {1, 2, 1, 2, 1, 2, 1, 2}, 0.2, 0),
        PriorityTreeParameters9(8, 3, {1, 2, 1, 2, 1, 2, 1, 2}, 1.5, 1),
        PriorityTreeParameters9(8, 3, {1, 2, 1, 2, 1, 2, 1, 2}, 2.3, 1),
        PriorityTreeParameters9(8, 5, {1, 2, 1, 2, 1, 2, 1, 2}, 3.9, 2),
        PriorityTreeParameters9(8, 3, {1, 2, 1, 2, 1, 2, 1, 2}, 7.1, 5),
        PriorityTreeParameters9(8, 3, {1, 2, 1, 2, 1, 2, 1, 2}, 15.0, 7),
        PriorityTreeParameters9(8, 2, {1, 3, 3, 1, 1, 1}, 0.0, 0),
        PriorityTreeParameters9(8, 2, {1, 3, 3, 1, 1, 1}, 0.2, 0),
        PriorityTreeParameters9(8, 2, {1, 3, 3, 1, 1, 1}, 1.5, 1),
        PriorityTreeParameters9(8, 2, {1, 3, 3, 1, 1, 1}, 2.3, 1),
        PriorityTreeParameters9(8, 2, {1, 3, 3, 1, 1, 1}, 3.9, 1),
        PriorityTreeParameters9(8, 2, {1, 3, 3, 1, 1, 1}, 4.2, 2),
        PriorityTreeParameters9(8, 2, {1, 3, 3, 1, 1, 1}, 5.7, 2),
        PriorityTreeParameters9(8, 2, {1, 3, 3, 1, 1, 1}, 6.8, 2),
        PriorityTreeParameters9(8, 2, {1, 3, 3, 1, 1, 1}, 11.0, 5)
    )
);
}  // namespace relab::test::agents::memory
