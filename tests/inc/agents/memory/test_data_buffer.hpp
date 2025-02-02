// Copyright 2025 Theophile Champion. No Rights Reserved.

#ifndef TESTS_INC_AGENTS_MEMORY_TEST_DATA_BUFFER_HPP_
#define TESTS_INC_AGENTS_MEMORY_TEST_DATA_BUFFER_HPP_

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "agents/memory/data_buffer.hpp"
#include "agents/memory/experience.hpp"

namespace relab::test::agents::memory::impl {

using relab::agents::memory::DataBuffer;

/**
 * A class storing the parameters of the data buffer tests.
 */
class DataBufferParameters {
 public:
  int capacity;
  int n_steps;
  float gamma;
  float initial_priority;
  int n_children;

 public:
  /**
   * Create a structure storing the parameters of the data buffer tests.
   * @param capacity the number of experiences the buffer can store
   * @param n_steps the number of steps for which rewards are accumulated in
   * multistep Q-learning
   * @param gamma the discount factor
   * @param initial_priority the initial priority given to first elements
   * @param n_children the number of children each node has
   */
  DataBufferParameters(int capacity, int n_steps, float gamma, float initial_priority = 1, int n_children = 10);

  /**
   * Create a structure storing the parameters of the data buffer tests.
   */
  DataBufferParameters();
};

/**
 * A fixture class for testing the data buffer.
 */
class TestDataBuffer : public testing::TestWithParam<DataBufferParameters> {
 public:
  DataBufferParameters params;
  std::unique_ptr<DataBuffer> buffer;
  std::vector<torch::Tensor> observations;

 public:
  /**
   * Setup of th fixture class before calling a unit test.
   */
  void SetUp();
};
}  // namespace relab::test::agents::memory::impl

namespace relab::test::agents::memory {
using impl::DataBufferParameters;
using impl::TestDataBuffer;
}  // namespace relab::test::agents::memory

#endif  // TESTS_INC_AGENTS_MEMORY_TEST_DATA_BUFFER_HPP_
