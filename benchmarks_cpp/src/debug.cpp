#include "debug.hpp"

template<class T>
void debug::print_tensor(const torch::Tensor &tensor, int max_n_elements, bool new_line) {

    // Display the most important information about the tensor.
    std::cout << "Tensor(type: " << tensor.dtype() << ", shape: " << tensor.sizes() << ", values: [";

    // Retrieve the number of elements that needs to be displayed.
    if (max_n_elements == -1) {
        max_n_elements = tensor.numel();
    }
    max_n_elements = std::min(max_n_elements, static_cast<int>(tensor.numel()));

    // Display the tensor's values, if needed.
    if (max_n_elements != 0) {
        torch::Tensor tensor_cpu = (tensor.is_cuda()) ? tensor.clone().cpu() : tensor;
        T *ptr = tensor_cpu.data_ptr<T>();
        std::vector<T> vector{ptr, ptr + max_n_elements};
        for (auto i = 0; i < max_n_elements; i++) {
            if (i != 0)
                std::cout << " ";
            std::cout << vector[i];
        }
    }
    debug::print_ellipse(max_n_elements, tensor.numel());
    std::cout << "])";
    if (new_line == true) {
        std::cout << std::endl;
    }
}

template<>
void debug::print_tensor<bool>(const torch::Tensor &tensor, int max_n_elements, bool new_line) {

    // Display the most important information about the tensor.
    std::cout << "Tensor(type: " << tensor.dtype() << ", shape: " << tensor.sizes() << ", values: [";

    // Retrieve the number of elements that needs to be displayed.
    if (max_n_elements == -1) {
        max_n_elements = tensor.numel();
    }
    max_n_elements = std::min(max_n_elements, static_cast<int>(tensor.numel()));

    // Display the tensor's elements.
    if (max_n_elements != 0) {
        torch::Tensor tensor_cpu = (tensor.is_cuda()) ? tensor.clone().cpu() : tensor;
        char *ptr = (char *) tensor_cpu.data_ptr();
        std::vector<char> vector{ptr, ptr + max_n_elements};
        for (auto i = 0; i < max_n_elements; i++) {
            if (i != 0)
                std::cout << " ";
            debug::print_bool(vector[i]);
        }
    }
    debug::print_ellipse(max_n_elements, tensor.numel());
    std::cout << "])";
    if (new_line == true) {
        std::cout << std::endl;
    }
}

template<class T>
void debug::print_vector(const std::vector<T> &vector, int max_n_elements) {

    // Display the most important information about the tensor.
    int size = static_cast<int>(vector.size());
    std::cout << "std::vector(type: " << torch::CppTypeToScalarType<T>() << ", size: " << size << ", values: [";

    // Retrieve the number of elements that needs to be displayed.
    if (max_n_elements == -1) {
        max_n_elements = size;
    }
    max_n_elements = std::min(max_n_elements, size);

    // Display the tensor's values, if needed.
    if (max_n_elements != 0) {
        for (auto i = 0; i < max_n_elements; i++) {
            if (i != 0)
                std::cout << " ";
            std::cout << vector[i];
        }
    }
    debug::print_ellipse(max_n_elements, size);
    std::cout << "])" << std::endl;
}

template<class TensorType, class DataType>
void debug::print_vector(const std::vector<TensorType> &vector, int start, int max_n_elements) {

    // Display the most important information about the tensor.
    int size = static_cast<int>(vector.size());
    std::cout << "std::vector(size: " << size << ", values: [";

    // Retrieve the number of elements that needs to be displayed.
    if (max_n_elements == -1) {
        max_n_elements = size;
    }
    max_n_elements = std::min(max_n_elements, size);

    // Display the tensor's values, if needed.
    if (max_n_elements != 0) {
        for (auto i = 0; i < max_n_elements; i++) {
            if (i != 0)
                std::cout << " ";
            debug::print_tensor<DataType>(vector[i], max_n_elements, false);
        }
    }
    debug::print_ellipse(max_n_elements, size);
    std::cout << "])" << std::endl;
}

void debug::print_bool(bool value) {
    std::cout << ((value == true) ? "true" : "false");
}

void debug::print_ellipse(int max_n_elements, int size) {
    if (max_n_elements != size) {
        std::cout << ((max_n_elements != 0) ? " ..." : "...");
    }
}

// Explicit instantiations.
template void debug::print_tensor<int>(const torch::Tensor &tensor, int max_n_elements, bool new_line);
template void debug::print_tensor<long>(const torch::Tensor &tensor, int max_n_elements, bool new_line);
template void debug::print_tensor<bool>(const torch::Tensor &tensor, int max_n_elements, bool new_line);
template void debug::print_tensor<float>(const torch::Tensor &tensor, int max_n_elements, bool new_line);

template void debug::print_vector<int>(const std::vector<int> &vector, int max_n_elements);

template void debug::print_vector<torch::Tensor, float>(const std::vector<torch::Tensor> &vector, int start, int max_n_elements);
