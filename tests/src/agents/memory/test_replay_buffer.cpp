// Copyright 2025 Theophile Champion. No Rights Reserved.

#include "agents/memory/test_replay_buffer.hpp"
#include "agents/memory/compressors.hpp"
#include "helpers/torch.hpp"
#include "relab_test.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <torch/extension.h>

using namespace relab::agents::memory;
using namespace relab::helpers;

namespace relab::test::agents::memory {

/**
 * Implementation of the TestReplayBuffer test suite.
 */

ReplayBufferParameters::ReplayBufferParameters(int capacity, int n_steps,
                                               float gamma)
    : prioritized(false), capacity(capacity), batch_size(32), frame_skip(1),
      stack_size(4), screen_size(84), n_steps(n_steps), gamma(gamma),
      comp_type(CompressorType::ZLIB) {
  this->args["n_steps"] = n_steps;
  this->args["gamma"] = gamma;
}

ReplayBufferParameters::ReplayBufferParameters(bool prioritized, int batch_size)
    : ReplayBufferParameters(4, 1, 1) {
  this->batch_size = batch_size;
  this->prioritized = prioritized;
  if (prioritized == true) {
    this->args["initial_priority"] = 1;
    this->args["omega_is"] = 0.5;
  }
}

ReplayBufferParameters::ReplayBufferParameters()
    : ReplayBufferParameters(false) {}

void TestReplayBuffer::SetUp() {
  // Create the replay buffer.
  this->params = GetParam();
  this->buffer = std::make_unique<ReplayBuffer>(
      this->params.capacity, this->params.batch_size, this->params.frame_skip,
      this->params.stack_size, this->params.screen_size, this->params.comp_type,
      this->params.args);

  // Create the observations at time t.
  int n_observations = 2 * this->params.capacity + this->params.n_steps;
  this->observations = getObservations(n_observations, this->params.frame_skip,
                                       this->params.stack_size);
}

TEST_P(TestReplayBuffer, TestStoringAndRetrievalMultipleEpisodes) {
  // Create the experiences at time t.
  auto experiences =
      getExperiences(observations, 2 * params.capacity - 1, params.capacity);

  // Create the multistep experiences at time t (experiences expected to be
  // returned by the frame buffer).
  auto results =
      getResultExperiences(observations, params.gamma, params.n_steps,
                           2 * params.capacity, params.capacity);

  // Fill the buffer with experiences.
  for (int t = 0; t < params.capacity; t++) {
    buffer->append(experiences[t]);
  }

  // Check that experiences in the frame buffer are as expected.
  auto indices = torch::arange(params.capacity);
  auto batch = buffer->getExperiences(indices);
  compareExperiences(batch, results.begin(), params.capacity);

  // Keep pushing experiences to the buffer, effectively replacing all
  // experiences in the frame buffer.
  for (int t = 0; t < params.capacity - 1; t++) {
    buffer->append(experiences[params.capacity + t]);
  }

  // Check that the new experiences in the frame buffer are as expected.
  batch = buffer->getExperiences(indices);
  int n_experiences = params.capacity - params.n_steps + 1;
  compareExperiences(batch, results.begin() + n_experiences - 1, n_experiences);
}

TEST_P(TestReplayBuffer, TestStoringAndRetrieval) {
  // Create the experiences at time t.
  auto experiences = getExperiences(observations, observations.size() - 1);

  // Create the multistep experiences at time t (experiences expected to be
  // returned by the replay buffer).
  auto results = getResultExperiences(observations, params.gamma,
                                      params.n_steps, 2 * params.capacity);

  // Fill the buffer with experiences.
  int n_experiences = params.capacity + params.args["n_steps"] - 1;
  for (int t = 0; t < n_experiences; t++) {
    buffer->append(experiences[t]);
  }

  // Check that experiences in the frame buffer are as expected.
  auto indices = torch::arange(params.capacity);
  auto batch = buffer->getExperiences(indices);
  compareExperiences(batch, results.begin(), params.capacity);

  // Keep pushing experiences to the buffer, effectively replacing all
  // experiences in the frame buffer.
  for (int t = 0; t < params.capacity; t++) {
    buffer->append(experiences[t + n_experiences]);
  }

  // Check that the new experiences in the frame buffer are as expected.
  batch = buffer->getExperiences(indices);
  compareExperiences(batch, results.begin() + params.capacity, params.capacity);
}

TEST_P(TestReplayBuffer, TestSaveAndLoad) {
  // Create the experiences at time t.
  auto experiences = getExperiences(observations, observations.size() - 1);

  // Fill the buffer with experiences.
  int n_experiences = params.capacity + params.n_steps - 1;
  for (int t = 0; t < n_experiences; t++) {
    buffer->append(experiences[t]);
  }

  // Save the frame buffer.
  std::stringstream ss;
  buffer->saveToFile(ss);

  // Load the frame buffer.
  auto loaded_buffer = ReplayBuffer();
  loaded_buffer.loadFromFile(ss);

  // Check that the saved and loaded frame buffers are identical.
  EXPECT_EQ(*buffer, loaded_buffer);
}

INSTANTIATE_TEST_SUITE_P(UnitTests, TestReplayBuffer,
                         testing::Values(ReplayBufferParameters(5, 1, 1),
                                         ReplayBufferParameters(5, 1, 0.9),
                                         ReplayBufferParameters(5, 2, 1),
                                         ReplayBufferParameters(5, 2, 0.99),
                                         ReplayBufferParameters(6, 2, 0.95),
                                         ReplayBufferParameters(7, 1, 0.5),
                                         ReplayBufferParameters(8, 3, 0.75),
                                         ReplayBufferParameters(9, 2, 0.8),
                                         ReplayBufferParameters(5, 1, 0.98),
                                         ReplayBufferParameters(5, 1, 0.999),
                                         ReplayBufferParameters(9, 8, 0.1)));

TEST(TestReplayBuffer, TestReport) {
  // Arrange.
  auto params = ReplayBufferParameters(true, 2);
  auto buffer = ReplayBuffer(params.capacity, params.batch_size,
                             params.frame_skip, params.stack_size,
                             params.screen_size, params.comp_type, params.args);
  auto observations = getObservations(11, params.frame_skip, params.stack_size);
  auto experiences = getExperiences(observations, 10);

  // Acts.
  for (int t = 0; t < params.capacity; t++) {
    buffer.append(experiences[t]);
  }
  buffer.sample();
  auto loss = 2 * torch::ones({2}).to(getDevice());
  loss = buffer.report(loss);

  // Assert.
  for (int i = 0; i < params.capacity; i++) {
    if (torch::isin(i, buffer.getLastIndices()).item<bool>()) {
      EXPECT_TRUE(std::abs(buffer.getPriority(i) - 2.0) < 0.0001);
    } else {
      EXPECT_EQ(buffer.getPriority(i), 1.0);
    }
  }
  for (int i = 0; i < 2; i++) {
    EXPECT_TRUE(abs(loss[i].item<float>() - 2.0) < 0.0001);
  }
}

/**
 * Implementation of the TestReplayBuffer2 test suite.
 */

TEST_P(TestReplayBuffer2, TestGetPrioritized) {
  // Arrange.
  auto params = GetParam();
  auto buffer = ReplayBuffer(params.capacity, params.batch_size,
                             params.frame_skip, params.stack_size,
                             params.screen_size, params.comp_type, params.args);

  // Assert.
  EXPECT_EQ(buffer.getPrioritized(), params.prioritized);
}

INSTANTIATE_TEST_SUITE_P(UnitTests, TestReplayBuffer2,
                         testing::Values(ReplayBufferParameters(true),
                                         ReplayBufferParameters(false)));

/**
 * Implementation of the TestReplayBuffer3 test suite.
 */

void TestReplayBuffer3::SetUp() {
  // Arrange.
  auto n_experiences = GetParam();
  this->buffer = std::make_unique<ReplayBuffer>();
  auto observations = getObservations(n_experiences + 1);
  this->experiences = getExperiences(observations, n_experiences);
}

TEST_P(TestReplayBuffer3, TestSize) {
  // Act.
  for (auto experience : experiences) {
    buffer->append(experience);
  }

  // Assert.
  EXPECT_EQ(buffer->size(), experiences.size());
}

TEST_P(TestReplayBuffer3, TestClear) {
  // Act.
  for (auto experience : experiences) {
    buffer->append(experience);
  }
  buffer->clear();

  // Assert.
  EXPECT_EQ(buffer->size(), 0);
}

INSTANTIATE_TEST_SUITE_P(UnitTests, TestReplayBuffer3,
                         testing::Values(0, 1, 2, 3, 10));
} // namespace relab::test::agents::memory
