// Copyright 2025 Theophile Champion. No Rights Reserved.

#ifndef TESTS_INC_AGENTS_MEMORY_TEST_PRIORITY_TREE_HPP_
#define TESTS_INC_AGENTS_MEMORY_TEST_PRIORITY_TREE_HPP_

#include <memory>
#include <string>
#include <vector>

#include "agents/memory/priority_tree.hpp"
#include <gtest/gtest.h>

namespace relab::test::agents::memory::impl {

using namespace relab::agents::memory;

/**
 * A class storing the parameters of the priority tree tests.
 */
class PriorityTreeParameters {
 public:
  std::vector<float> elements;
  float result;

 public:
  /**
   * Create a structure storing the parameters of the priority tree tests.
   * @param elements the priorities to append to the queue
   * @param result the test result
   */
  PriorityTreeParameters(const std::initializer_list<float> &elements, float result = 0);

  /**
   * Create a structure storing the parameters of the priority tree tests.
   */
  PriorityTreeParameters();
};

/**
 * A fixture class for testing the priority tree.
 */
class TestPriorityTree : public testing::TestWithParam<PriorityTreeParameters> {
 public:
  PriorityTreeParameters params;
  std::unique_ptr<PriorityTree> priority_tree;

 public:
  /**
   * Setup of th fixture class before calling a unit test.
   */
  void SetUp();
};

/**
 * A fixture class for testing the priority tree.
 */
class TestPriorityTree2 : public testing::TestWithParam<PriorityTreeParameters> {};

/**
 * A class storing the parameters of the priority tree tests.
 */
class PriorityTreeParameters4 {
 public:
  std::vector<float> elements;
  std::string sum_result;
  std::string max_result;
  float length_result;

 public:
  /**
   * Create a structure storing the parameters of the priority tree tests.
   * @param elements the priorities to append to the queue
   * @param sum_result a string representing the sum-tree of the priority tree
   * @param max_result a string representing the max-tree of the priority tree
   * @param length_result the expected size of the priority tree
   */
  PriorityTreeParameters4(
      const std::initializer_list<float> &elements, const std::string &sum_result,
      const std::string &max_result, int length_result
  );
};

/**
 * A fixture class for testing the priority tree.
 */
class TestPriorityTree4 : public testing::TestWithParam<PriorityTreeParameters4> {};

/**
 * A class storing the parameters of the priority tree tests.
 */
class PriorityTreeParameters5 {
 public:
  std::vector<float> elements;
  std::vector<int> set_indices;
  std::vector<float> set_values;
  std::string sum_result;
  std::string max_result;
  float length_result;

 public:
  /**
   * Create a structure storing the parameters of the priority tree tests.
   * @param elements the priorities to append to the queue
   * @param set_indices the indices of priorities whose value must be changed
   * @param set_values the new values of the priorities that must be changed
   * @param sum_result a string representing the sum-tree of the priority tree
   * @param max_result a string representing the max-tree of the priority tree
   * @param length_result the expected size of the priority tree
   */
  PriorityTreeParameters5(
      const std::initializer_list<float> &elements,
      const std::initializer_list<int> &set_indices,
      const std::initializer_list<float> &set_values, const std::string &sum_result,
      const std::string &max_result, int length_result
  );
};

/**
 * A fixture class for testing the priority tree.
 */
class TestPriorityTree5 : public testing::TestWithParam<PriorityTreeParameters5> {};

/**
 * A class storing the parameters of the priority tree tests.
 */
class PriorityTreeParameters6 {
 public:
  std::vector<float> elements;
  std::vector<float> results;
  float length_result;

 public:
  /**
   * Create a structure storing the parameters of the priority tree tests.
   * @param elements the priorities to append to the queue
   * @param results the priorities that must be contained in the priority tree
   * @param length_result the expected size of the priority tree
   */
  PriorityTreeParameters6(
      const std::initializer_list<float> &elements,
      const std::initializer_list<float> &results, int length_result
  );
};

/**
 * A fixture class for testing the priority tree.
 */
class TestPriorityTree6 : public testing::TestWithParam<PriorityTreeParameters6> {};

/**
 * A class storing the parameters of the priority tree tests.
 */
class PriorityTreeParameters7 {
 public:
  int index;
  int n_children;
  int result;

 public:
  /**
   * Create a structure storing the parameters of the priority tree tests.
   * @param index the index of the node whose parent index must be computed
   * @param n_children the number of children each node of the priority tree has
   * @param result the test result
   */
  PriorityTreeParameters7(int index, int n_children, int result);
};

/**
 * A fixture class for testing the priority tree.
 */
class TestPriorityTree7 : public testing::TestWithParam<PriorityTreeParameters7> {};

/**
 * A class storing the parameters of the priority tree tests.
 */
class PriorityTreeParameters8 {
 public:
  std::vector<float> elements;
  std::vector<int> indices;
  std::vector<float> new_values;
  float result;

 public:
  /**
   * Create a structure storing the parameters of the priority tree tests.
   * @param elements the priorities to append to the queue
   * @param max_index the largest index of the priorities whose value must be
   * changed
   * @param new_values the new values of the priorities that must be set
   * @param result the test result
   */
  PriorityTreeParameters8(
      const std::vector<float> &elements, int max_index,
      const std::vector<float> &new_values, float result
  );
};

/**
 * A fixture class for testing the priority tree.
 */
class TestPriorityTree8 : public testing::TestWithParam<PriorityTreeParameters8> {};

/**
 * A class storing the parameters of the priority tree tests.
 */
class PriorityTreeParameters9 {
 public:
  int capacity;
  int n_children;
  std::vector<float> elements;
  float priority;
  float result;

 public:
  /**
   * Create a structure storing the parameters of the priority tree tests.
   * @param capacity the capacity of the priority tree
   * @param n_children the number of children each node of the priority tree has
   * @param elements the priorities to append to the queue
   * @param priority the priority whose corresponding index must be computed
   * using tower sampling
   * @param result the test result
   */
  PriorityTreeParameters9(
      int capacity, int n_children, const std::vector<float> &elements, float priority,
      float result
  );
};

/**
 * A fixture class for testing the priority tree.
 */
class TestPriorityTree9 : public testing::TestWithParam<PriorityTreeParameters9> {};
}  // namespace relab::test::agents::memory::impl

namespace relab::test::agents::memory {
using impl::PriorityTreeParameters;
using impl::PriorityTreeParameters4;
using impl::PriorityTreeParameters5;
using impl::PriorityTreeParameters6;
using impl::PriorityTreeParameters7;
using impl::PriorityTreeParameters8;
using impl::PriorityTreeParameters9;
using impl::TestPriorityTree;
using impl::TestPriorityTree2;
using impl::TestPriorityTree4;
using impl::TestPriorityTree5;
using impl::TestPriorityTree6;
using impl::TestPriorityTree7;
using impl::TestPriorityTree8;
using impl::TestPriorityTree9;
}  // namespace relab::test::agents::memory

#endif  // TESTS_INC_AGENTS_MEMORY_TEST_PRIORITY_TREE_HPP_
