// Copyright 2025 Theophile Champion. No Rights Reserved.

#ifndef TESTS_INC_AGENTS_MEMORY_TEST_REPLAY_BUFFER_HPP_
#define TESTS_INC_AGENTS_MEMORY_TEST_REPLAY_BUFFER_HPP_

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "agents/memory/compressors.hpp"
#include "agents/memory/experience.hpp"
#include "agents/memory/replay_buffer.hpp"

namespace relab::test::agents::memory::impl {

using relab::agents::memory::ReplayBuffer;
using relab::agents::memory::CompressorType;
using relab::agents::memory::Experience;

/**
 * A class storing the parameters of the replay buffer tests.
 */
class ReplayBufferParameters {
 public:
  int prioritized;
  int capacity;
  int batch_size;
  int frame_skip;
  int stack_size;
  int screen_size;
  int n_steps;
  float gamma;
  CompressorType comp_type;
  std::map<std::string, float> args;

 public:
  /**
   * Create a structure storing the parameters of the replay buffer tests.
   * @param capacity the number of experiences the buffer can store
   * @param n_steps the number of steps for which rewards are accumulated in
   * multistep Q-learning
   * @param gamma the discount factor
   */
  ReplayBufferParameters(int capacity, int n_steps, float gamma);

  /**
   * Create a structure storing the parameters of the replay buffer tests.
   * @param prioritized true if the buffer is prioritized, false otherwise
   * @param batch_size the size of batches to sample
   */
  explicit ReplayBufferParameters(bool prioritized, int batch_size = 32);

  /**
   * Create a structure storing the parameters of the replay buffer tests.
   */
  ReplayBufferParameters();
};

/**
 * A fixture class for testing the replay buffer.
 */
class TestReplayBuffer : public testing::TestWithParam<ReplayBufferParameters> {
 public:
  ReplayBufferParameters params;
  std::unique_ptr<ReplayBuffer> buffer;
  std::vector<torch::Tensor> observations;

 public:
  /**
   * Setup of th fixture class before calling a unit test.
   */
  void SetUp();
};

/**
 * A fixture class for testing the replay buffer.
 */
class TestReplayBuffer2 : public testing::TestWithParam<ReplayBufferParameters> {};

/**
 * A fixture class for testing the replay buffer.
 */
class TestReplayBuffer3 : public testing::TestWithParam<int> {
 public:
  std::unique_ptr<ReplayBuffer> buffer;
  std::vector<Experience> experiences;

 public:
  /**
   * Setup of th fixture class before calling a unit test.
   */
  void SetUp();
};
}  // namespace relab::test::agents::memory::impl

namespace relab::test::agents::memory {
using impl::ReplayBufferParameters;
using impl::TestReplayBuffer;
using impl::TestReplayBuffer2;
using impl::TestReplayBuffer3;
}  // namespace relab::test::agents::memory

#endif  // TESTS_INC_AGENTS_MEMORY_TEST_REPLAY_BUFFER_HPP_
