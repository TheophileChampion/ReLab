// Copyright 2025 Theophile Champion. No Rights Reserved.

#include "agents/memory/test_frame_buffer.hpp"
#include "relab_test.hpp"
#include <memory>
#include <torch/extension.h>

using namespace relab::agents::memory;

namespace relab::test::agents::memory {

/**
 * Implementation of the TestReplayBuffer test suite.
 */

void TestFrameBuffer::SetUp() {

  // Create the frame buffer.
  this->params = GetParam();
  this->buffer = std::make_unique<FrameBuffer>(
      this->params.capacity, this->params.frame_skip, this->params.n_steps,
      this->params.stack_size);

  // Create the observations at time t.
  int n_observations = 2 * this->params.capacity + this->params.n_steps;
  this->observations = getObservations(n_observations, this->params.frame_skip,
                                       this->params.stack_size);
}

FrameBufferParameters::FrameBufferParameters(int capacity, int frame_skip,
                                             int n_steps, int stack_size)
    : capacity(capacity), frame_skip(frame_skip), n_steps(n_steps),
      stack_size(stack_size), gamma(1) {}

FrameBufferParameters::FrameBufferParameters()
    : FrameBufferParameters(0, 0, 0, 0) {}

TEST_P(TestFrameBuffer, TestStoringAndRetrievalMultipleEpisodes) {

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
  auto [obs_t, obs_tn] = (*buffer)[indices];
  for (int t = 0; t < params.capacity; t++) {
    EXPECT_EQ_TENSOR(results[t].obs, obs_t[t]);
    EXPECT_EQ_TENSOR(results[t].next_obs, obs_tn[t]);
  }

  // Keep pushing experiences to the buffer, effectively replacing all
  // experiences in the frame buffer.
  for (int t = 0; t < params.capacity - 1; t++) {
    buffer->append(experiences[params.capacity + t]);
  }

  // Check that the new experiences in the frame buffer are as expected.
  std::tie(obs_t, obs_tn) = (*buffer)[indices];
  for (int t = 0; t < params.capacity - params.n_steps + 1; t++) {
    EXPECT_EQ_TENSOR(results[params.capacity - params.n_steps + t].obs,
                     obs_t[t]);
    EXPECT_EQ_TENSOR(results[params.capacity - params.n_steps + t].next_obs,
                     obs_tn[t]);
  }
}

TEST_P(TestFrameBuffer, TestStoringAndRetrieval) {

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
  auto [obs_t, obs_tn] = (*buffer)[indices];
  for (int t = 0; t < params.capacity; t++) {
    EXPECT_EQ_TENSOR(results[t].obs, obs_t[t]);
    EXPECT_EQ_TENSOR(results[t].next_obs, obs_tn[t]);
  }

  // Keep pushing experiences to the buffer, effectively replacing all
  // experiences in the frame buffer.
  for (int t = 0; t < params.capacity; t++) {
    buffer->append(experiences[t + n_experiences]);
  }

  // Check that the new experiences in the frame buffer are as expected.
  std::tie(obs_t, obs_tn) = (*buffer)[indices];
  for (int t = 0; t < params.capacity; t++) {
    EXPECT_EQ_TENSOR(results[params.capacity + t].obs, obs_t[t]);
    EXPECT_EQ_TENSOR(results[params.capacity + t].next_obs, obs_tn[t]);
  }
}

TEST_P(TestFrameBuffer, TestSaveAndLoad) {

  // Create the experiences at time t.
  auto experiences = getExperiences(observations, observations.size() - 1);

  // Fill the buffer with experiences.
  int n_experiences = params.capacity + params.n_steps - 1;
  for (int t = 0; t < n_experiences; t++) {
    buffer->append(experiences[t]);
  }

  // Save the frame buffer.
  std::stringstream ss;
  buffer->save(ss);

  // Load the frame buffer.
  auto loaded_buffer = FrameBuffer(10, 10, 10, 10);
  loaded_buffer.load(ss);

  // Check that the saved and loaded frame buffers are identical.
  EXPECT_EQ(*buffer, loaded_buffer);
}

INSTANTIATE_TEST_SUITE_P(
    UnitTests, TestFrameBuffer,
    testing::Values(
        FrameBufferParameters(5, 1, 1, 2), FrameBufferParameters(5, 1, 1, 4),
        FrameBufferParameters(5, 1, 2, 4), FrameBufferParameters(5, 2, 2, 4),
        FrameBufferParameters(6, 2, 2, 4), FrameBufferParameters(7, 3, 1, 2),
        FrameBufferParameters(8, 1, 3, 4), FrameBufferParameters(9, 1, 2, 1),
        FrameBufferParameters(5, 1, 6, 1), FrameBufferParameters(5, 6, 1, 1),
        FrameBufferParameters(5, 1, 1, 6), FrameBufferParameters(9, 9, 8, 9)));

TEST(TestFrameBuffer, TestEncodingAndDecoding) {

  // Create a frame buffer.
  auto buffer = FrameBuffer(8, 1, 1, 4);

  for (int i = 0; i < 256; i++) {

    // Create the i-th frame to encode and decode.
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto frame = i / 255 * torch::ones({84, 84}, options);

    // Encode and decode the i-th frame.
    auto encoded_frame = buffer.encode(frame);
    auto decoded_frame = buffer.decode(encoded_frame);

    // Check that the initial and decoded frames are identical.
    EXPECT_EQ_TENSOR(frame, decoded_frame);
  }
}
} // namespace relab::test::agents::memory
