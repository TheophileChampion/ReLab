// Copyright 2025 Theophile Champion. No Rights Reserved.

#include "relab_test.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>
#include <torch/extension.h>

namespace relab::test::impl {

float getMultiStepReward(float r_0, int increment, float gamma, int n) {
  float r = 0;
  for (auto i = 0; i < n; i++) {
    r += std::pow(gamma, i) * (r_0 + i * increment);
  }
  return r;
}

torch::Tensor getObservation(int t, int frame_skip, int stack_size) {
  auto o_0 = torch::arange(0, stack_size)
                 .unsqueeze(1)
                 .unsqueeze(2)
                 .repeat({1, 84, 84});
  return (o_0 + t * frame_skip) / 255;
}

std::vector<torch::Tensor>
getObservations(int n, int frame_skip, int stack_size) {
  std::vector<torch::Tensor> observations;
  for (int t = 0; t < n; t++) {
    observations.push_back(getObservation(t, frame_skip, stack_size));
  }
  return observations;
}

std::vector<Experience> getExperiences(
    const std::vector<torch::Tensor> &observations, int n, int episode_length
) {
  // Compute the number of episodes to generate.
  int length = (episode_length == -1) ? n : episode_length;
  int n_episodes = std::ceil(static_cast<float>(n) / length);

  // Create the experiences.
  std::vector<Experience> experiences;
  for (int i = 0; i < n_episodes; i++) {
    for (int t = 0; t < length; t++) {

      // Add an experiment.
      bool done = (episode_length == -1) ? false : (t == length - 1);
      experiences.push_back(
          Experience(observations[t], t, t, done, observations[t + 1])
      );

      // Check whether to stop adding experiments.
      if (static_cast<int>(experiences.size()) >= n) {
        break;
      }
    }
  }
  return experiences;
}

std::vector<Experience> getResultExperiences(
    const std::vector<torch::Tensor> &observations, float gamma, int n_steps,
    int n, int episode_length
) {
  // Compute the number of episodes to generate.
  int length = (episode_length == -1) ? n : episode_length;
  int n_episodes = std::ceil(static_cast<float>(n) / length);

  // Create the experiences.
  std::vector<Experience> experiences;
  for (int i = 0; i < n_episodes; i++) {
    for (int t = 0; t < length; t++) {

      // Add an experiment.
      int tn = t + n_steps;
      if (episode_length != -1) {
        tn = std::min(tn, length);
      }
      float rn = getMultiStepReward(t, 1, gamma, tn - t);
      bool done = (episode_length == -1) ? false : (tn == length);
      experiences.push_back(
          Experience(observations[t], t, rn, done, observations[tn])
      );

      // Check whether to stop adding experiments.
      if (static_cast<int>(experiences.size()) >= n) {
        break;
      }
    }
  }
  return experiences;
}

void compareExperiences(
    Batch batch, std::vector<Experience>::iterator experiences_it,
    int n_experiences
) {
  auto [obs, action, reward, done, next_obs] = batch;
  for (int t = 0; t < n_experiences; t++) {
    EXPECT_EQ_TENSOR(obs[t], experiences_it->obs);
    EXPECT_EQ(action[t].cpu().item<int>(), experiences_it->action);
    EXPECT_TRUE(
        abs(reward[t].cpu().item<float>() - experiences_it->reward) <
        TEST_EPSILON
    );
    EXPECT_EQ(done[t].cpu().item<bool>(), experiences_it->done);
    EXPECT_EQ_TENSOR(next_obs[t], experiences_it->next_obs);
    experiences_it++;
  }
}

template <class T> std::vector<T> repeat(T value, int n) {
  std::vector<T> vector(n, value);
  return vector;
}

// Explicit instantiation.
template std::vector<float> repeat(float value, int n);

}  // namespace relab::test::impl
