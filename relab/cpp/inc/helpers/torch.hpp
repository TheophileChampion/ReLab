// Copyright 2025 Theophile Champion. No Rights Reserved.
/**
 * @file torch.hpp
 * @brief Helper functions for interacting with libtorch.
 */

#ifndef RELAB_CPP_INC_HELPERS_TORCH_HPP_
#define RELAB_CPP_INC_HELPERS_TORCH_HPP_

#include <vector>

#include <torch/extension.h>

namespace relab::helpers {

/**
 * Retrieves the device on which the computation should be performed.
 * @return the device
 */
torch::Device getDevice();

/**
 * Check if two tensors are equal.
 * @param tensor_1 the first tensor to compare
 * @param tensor_2 the second tensor to compare
 * @return true if the tensors are equal, false otherwise
 */
bool tensorsAreEqual(
    const torch::Tensor tensor_1, const torch::Tensor tensor_2
);
}  // namespace relab::helpers

#endif  // RELAB_CPP_INC_HELPERS_TORCH_HPP_
