// Copyright 2025 Theophile Champion. No Rights Reserved.

#include "helpers/debug.hpp"

namespace relab::helpers {

torch::Device getDevice() {
  bool use_cuda =
      (torch::cuda::is_available() and torch::cuda::device_count() >= 1);
  return torch::Device((use_cuda == true) ? torch::kCUDA : torch::kCPU);
}

bool tensorsAreEqual(
    const torch::Tensor tensor_1, const torch::Tensor tensor_2
) {

  // If both tensors are empty they are equal.
  if (tensor_1.numel() == 0 && tensor_2.numel() == 0) {
    return true;
  }

  // Otherwise check whether the tensors have the same number of elements, same
  // shapes and same elements.
  if (tensor_1.numel() != tensor_2.numel() ||
      tensor_1.sizes() != tensor_2.sizes()) {
    return false;
  }
  return torch::all(torch::eq(tensor_1, tensor_2)).item<bool>();
}
}  // namespace relab::helpers
