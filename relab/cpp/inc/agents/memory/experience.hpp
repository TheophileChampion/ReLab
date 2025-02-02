// Copyright 2025 Theophile Champion. No Rights Reserved.
/**
 * @file experience.hpp
 * @brief Declaration of the experience class.
 */

#ifndef RELAB_CPP_INC_AGENTS_MEMORY_EXPERIENCE_HPP_
#define RELAB_CPP_INC_AGENTS_MEMORY_EXPERIENCE_HPP_

#include <torch/extension.h>
#include <tuple>

namespace relab::agents::memory::impl {

using torch::Tensor;

// Alias for a batch of experiences.
using Batch = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>;

/**
 * @brief Class storing an experience.
 */
class Experience {
public:
  /// @var obs
  /// The observation tensor at time t.
  Tensor obs;

  /// @var action
  /// The action taken at time t.
  int action;

  /// @var reward
  /// The reward received at time t + 1.
  float reward;

  /// @var done
  /// Flag indicating if the episode ended after this transition.
  bool done;

  /// @var next_obs
  /// The observation tensor at time t + 1.
  Tensor next_obs;

public:
  /**
   * Create an experience.
   * @param obs the observation at time t
   * @param action the action at time t
   * @param reward the reward at time t + 1
   * @param done true if episode ended, false otherwise
   * @param next_obs the observation at time t + 1
   */
  Experience(Tensor obs, int action, float reward, bool done, Tensor next_obs);
};
} // namespace relab::agents::memory::impl

namespace relab::agents::memory {
using impl::Batch;
using impl::Experience;
} // namespace relab::agents::memory

#endif // RELAB_CPP_INC_AGENTS_MEMORY_EXPERIENCE_HPP_
