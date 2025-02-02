// Copyright 2025 Theophile Champion. No Rights Reserved.

#ifndef TESTS_INC_AGENTS_MEMORY_TEST_FRAME_BUFFER_HPP_
#define TESTS_INC_AGENTS_MEMORY_TEST_FRAME_BUFFER_HPP_

#include <memory>

#include "agents/memory/experience.hpp"
#include "agents/memory/frame_buffer.hpp"
#include <gtest/gtest.h>

namespace relab::test::agents::memory::impl {

using namespace relab::agents::memory;

/**
 * A class storing the parameters of the frame buffer tests.
 */
class FrameBufferParameters {
public:
  int capacity;
  int frame_skip;
  int n_steps;
  int stack_size;
  float gamma;

public:
  /**
   * Create a structure storing the parameters of the frame buffer tests.
   * @param capacity the number of experiences the buffer can store
   * @param frame_skip the number of times each action is repeated in the
   * environment
   * @param n_steps the number of steps for which rewards are accumulated in
   * multistep Q-learning
   * @param stack_size the number of frames per observation
   */
  FrameBufferParameters(int capacity, int frame_skip, int n_steps, int stack_size);

  /**
   * Create a structure storing the parameters of the frame buffer tests.
   */
  FrameBufferParameters();
};

/**
 * A fixture class for testing the frame buffer.
 */
class TestFrameBuffer : public testing::TestWithParam<FrameBufferParameters> {
public:
  FrameBufferParameters params;
  std::unique_ptr<FrameBuffer> buffer;
  std::vector<torch::Tensor> observations;

public:
  /**
   * Setup of th fixture class before calling a unit test.
   */
  void SetUp();
};
}  // namespace relab::test::agents::memory::impl

namespace relab::test::agents::memory {
using impl::FrameBufferParameters;
using impl::TestFrameBuffer;
}  // namespace relab::test::agents::memory

#endif  // TESTS_INC_AGENTS_MEMORY_TEST_FRAME_BUFFER_HPP_
