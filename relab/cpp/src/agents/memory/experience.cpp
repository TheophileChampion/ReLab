// Copyright 2025 Theophile Champion. No Rights Reserved.

#include "agents/memory/experience.hpp"

namespace relab::agents::memory {

Experience::Experience(torch::Tensor obs, int action, float reward, bool done, torch::Tensor next_obs) {
  this->obs = obs;
  this->action = action;
  this->reward = reward;
  this->done = done;
  this->next_obs = next_obs;
}
}  // namespace relab::agents::memory
