// Copyright 2025 Theophile Champion. No Rights Reserved.

#include "agents/memory/test_data_buffer.hpp"
#include "agents/memory/data_buffer.hpp"
#include "relab_test.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <torch/extension.h>

using namespace relab::agents::memory;

namespace relab::test::agents::memory {

DataBufferParameters::DataBufferParameters(int capacity, int n_steps,
                                           float gamma, float initial_priority,
                                           int n_children)
    : capacity(capacity), n_steps(n_steps), gamma(gamma),
      initial_priority(initial_priority), n_children(n_children) {}

DataBufferParameters::DataBufferParameters()
    : DataBufferParameters(0, 0, 0, 0, 0) {}

void TestDataBuffer::SetUp() {

  // Create the replay buffer.
  this->params = GetParam();
  this->buffer = std::make_unique<DataBuffer>(
      this->params.capacity, this->params.n_steps, this->params.gamma,
      this->params.initial_priority, this->params.n_children);

  // Create the observations at time t.
  int n_observations = 2 * this->params.capacity + this->params.n_steps;
  this->observations = getObservations(n_observations, 1, 4);
}

TEST_P(TestDataBuffer, TestStoringAndRetrievalMultipleEpisodes) {

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
  auto [actions, rewards, dones] = (*buffer)[indices];
  for (int t = 0; t < params.capacity; t++) {
    EXPECT_EQ(actions[t].item<int>(), results[t].action);
    EXPECT_TRUE(std::abs(rewards[t].item<float>() - results[t].reward) <
                TEST_EPSILON);
    EXPECT_EQ(dones[t].item<bool>(), results[t].done);
  }

  // Keep pushing experiences to the buffer, effectively replacing all
  // experiences in the frame buffer.
  for (int t = 0; t < params.capacity - 1; t++) {
    buffer->append(experiences[params.capacity + t]);
  }

  // Check that the new experiences in the frame buffer are as expected.
  std::tie(actions, rewards, dones) = (*buffer)[indices];
  for (int t = 0; t < params.capacity - params.n_steps + 1; t++) {
    EXPECT_EQ(actions[t].item<int>(),
              results[params.capacity - params.n_steps + t].action);
    EXPECT_TRUE(std::abs(rewards[t].item<float>() -
                         results[params.capacity - params.n_steps + t].reward) <
                TEST_EPSILON);
    EXPECT_EQ(dones[t].item<bool>(),
              results[params.capacity - params.n_steps + t].done);
  }
}

TEST_P(TestDataBuffer, TestStoringAndRetrieval) {

  // Create the experiences at time t.
  auto experiences = getExperiences(observations, observations.size() - 1);

  // Create the multistep experiences at time t (experiences expected to be
  // returned by the replay buffer).
  auto results = getResultExperiences(observations, params.gamma,
                                      params.n_steps, 2 * params.capacity);

  // Fill the buffer with experiences.
  int n_experiences = params.capacity + params.n_steps - 1;
  for (int t = 0; t < n_experiences; t++) {
    buffer->append(experiences[t]);
  }

  // Check that experiences in the frame buffer are as expected.
  auto indices = torch::arange(params.capacity);
  auto [actions, rewards, dones] = (*buffer)[indices];
  for (int t = 0; t < params.capacity; t++) {
    EXPECT_EQ(actions[t].item<int>(), results[t].action);
    EXPECT_TRUE(std::abs(rewards[t].item<float>() - results[t].reward) <
                TEST_EPSILON);
    EXPECT_EQ(dones[t].item<bool>(), results[t].done);
  }

  // Keep pushing experiences to the buffer, effectively replacing all
  // experiences in the frame buffer.
  for (int t = 0; t < params.capacity; t++) {
    buffer->append(experiences[t + n_experiences]);
  }

  // Check that the new experiences in the frame buffer are as expected.
  std::tie(actions, rewards, dones) = (*buffer)[indices];
  for (int t = 0; t < params.capacity; t++) {
    EXPECT_EQ(actions[t].item<int>(), results[t + params.capacity].action);
    EXPECT_TRUE(std::abs(rewards[t].item<float>() -
                         results[t + params.capacity].reward) < TEST_EPSILON);
    EXPECT_EQ(dones[t].item<bool>(), results[t + params.capacity].done);
  }
}

TEST_P(TestDataBuffer, TestSaveAndLoad) {

  // Create the experiences at time t.
  auto experiences = getExperiences(observations, observations.size() - 1);

  // Fill the buffer with experiences.
  int n_experiences = params.capacity + params.n_steps - 1;
  for (int t = 0; t < n_experiences; t++) {
    buffer->append(experiences[t]);
  }

  // Save the data buffer.
  std::stringstream ss;
  buffer->save(ss);

  // Load the data buffer.
  auto loaded_buffer = DataBuffer(10, 10, 10, 10, 10);
  loaded_buffer.load(ss);

  // Check that the saved and loaded data buffers are identical.
  EXPECT_EQ(*buffer, loaded_buffer);
}

INSTANTIATE_TEST_SUITE_P(
    UnitTests, TestDataBuffer,
    testing::Values(
        DataBufferParameters(5, 1, 1), DataBufferParameters(5, 1, 0.9),
        DataBufferParameters(5, 2, 1), DataBufferParameters(5, 2, 0.99),
        DataBufferParameters(6, 2, 0.95), DataBufferParameters(7, 1, 0.5),
        DataBufferParameters(8, 3, 0.75), DataBufferParameters(9, 2, 0.8),
        DataBufferParameters(5, 6, 0.979), DataBufferParameters(5, 1, 0.98),
        DataBufferParameters(5, 1, 0.999), DataBufferParameters(9, 8, 0.1)));
} // namespace relab::test::agents::memory
