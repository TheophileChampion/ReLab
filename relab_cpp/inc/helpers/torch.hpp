#ifndef TORCH_HPP
#define TORCH_HPP

#include <torch/extension.h>
#include <vector>

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
    bool tensorsAreEqual(const torch::Tensor tensor_1, const torch::Tensor tensor_2);
}

#endif //TORCH_HPP
