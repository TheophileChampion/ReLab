// Copyright 2025 Theophile Champion. No Rights Reserved.

#ifndef TESTS_INC_RELAB_TEST_HPP_
#define TESTS_INC_RELAB_TEST_HPP_

#include <vector>

#include "agents/memory/experience.hpp"

#define TEST_EPSILON 1e-5
#define MAX_PRIORITY 1e9

#define EXPECT_EQ_TENSOR(x, y) EXPECT_TRUE(torch::all(torch::eq(x.cpu(), y.cpu())).item<bool>())

namespace relab::test::impl {

using relab::agents::memory::Batch;
using relab::agents::memory::Experience;

/**
 * Compute the multistep reward.
 * @param r_0 the immediate reward
 * @param increment the increment by which the reward increases at each time
 * step
 * @param gamma the discount factor
 * @param n the number of steps for which the multistep reward is computed
 * @return the multistep reward
 */
float getMultiStepReward(float r_0, int increment, float gamma, int n);

/**
 * Retrieve the observation at time step t.
 * @param t the time step
 * @param frame_skip the number of times each action is repeated in the
 * environment
 * @param stack_size the number of frames per observation
 */
torch::Tensor getObservation(int t, int frame_skip = 1, int stack_size = 4);

/**
 * Retrieve the first n observations.
 * @param n the number of observations
 * @param frame_skip the number of times each action is repeated in the
 * environment
 * @param stack_size the number of frames per observation
 */
std::vector<torch::Tensor> getObservations(int n, int frame_skip = 1, int stack_size = 4);

/**
 * Retrieve the first n experiences.
 * @param observations the list of observations used to create the experiences
 * @param n the number of experiences
 * @param episode_length the length of an episode
 */
std::vector<Experience> getExperiences(const std::vector<torch::Tensor> &observations, int n, int episode_length = -1);

/**
 * Retrieve the multistep experiences at time t (experiences expected to be
 * returned by the replay buffer).
 * @param observations the list of observations used to create the experiences
 * @param gamma the discount factor
 * @param n the number of steps for which the multistep reward is computed
 * @param n the number of experiences
 * @param episode_length the length of an episode
 */
std::vector<Experience> getResultExperiences(
    const std::vector<torch::Tensor> &observations, float gamma, int n_steps, int n, int episode_length = -1
);

/**
 * Compare the batch experiences to the expected experiences.
 * @param batch the batch whose experiences must be compared
 * @param experiences_it an iterator pointing to the first expected experience
 * @param n_experiences the number of experiences to compare
 */
void compareExperiences(Batch batch, std::vector<Experience>::iterator experiences_it, int n_experiences);

/**
 * Create a vector filled with a specific value.
 * @param value the value used to fill the vector
 * @param n the vector's size
 * @return the vector
 */
template <class T> std::vector<T> repeat(T value, int n);
}  // namespace relab::test::impl

namespace relab::test {
using impl::compareExperiences;
using impl::getExperiences;
using impl::getMultiStepReward;
using impl::getObservation;
using impl::getObservations;
using impl::getResultExperiences;
using impl::repeat;
}  // namespace relab::test

#endif  // TESTS_INC_RELAB_TEST_HPP_
