// Copyright 2025 Theophile Champion. No Rights Reserved.

#include "helpers/debug.hpp"

namespace relab::helpers {

template<class T>
void print_tensor(const torch::Tensor &tensor, int max_n_elements, bool new_line) {
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
    print_ellipse(max_n_elements, tensor.numel());
    std::cout << "])";
    if (new_line == true) {
        std::cout << std::endl;
    }
}

template<>
void print_tensor<bool>(const torch::Tensor &tensor, int max_n_elements, bool new_line) {
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
            print_bool(vector[i]);
        }
    }
    print_ellipse(max_n_elements, tensor.numel());
    std::cout << "])";
    if (new_line == true) {
        std::cout << std::endl;
    }
}

template<class T>
void print_vector(const std::vector<T> &vector, int max_n_elements) {
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
    print_ellipse(max_n_elements, size);
    std::cout << "])" << std::endl;
}

template<class TensorType, class DataType>
void print_vector(const std::vector<TensorType> &vector, int start, int max_n_elements) {
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
            print_tensor<DataType>(vector[i], max_n_elements, false);
        }
    }
    print_ellipse(max_n_elements, size);
    std::cout << "])" << std::endl;
}

void print_bool(bool value) {
    std::cout << ((value == true) ? "true" : "false");
}

void print_ellipse(int max_n_elements, int size) {
    if (max_n_elements != size) {
        std::cout << ((max_n_elements != 0) ? " ..." : "...");
    }
}

std::string Logger::levelToString(LogLevel level) {
    switch (level) {
    case DEBUG:
        return "DEBUG";
    case INFO:
        return "INFO";
    case WARNING:
        return "WARNING";
    case ERROR:
        return "ERROR";
    case CRITICAL:
        return "CRITICAL";
    default:
        return "UNKNOWN";
    }
}

Logger::Logger(LogLevel level, const std::string &logger_name)
    : level(level), logger_name(logger_name) {}

void Logger::debug(const std::string &message) {
    this->log(DEBUG, message);
}

void Logger::info(const std::string &message) {
    this->log(INFO, message);
}

void Logger::warning(const std::string &message) {
    this->log(WARNING, message);
}

void Logger::critical(const std::string &message) {
    this->log(CRITICAL, message);
}

void Logger::log(LogLevel level, const std::string &message) {
    if (level >= this->level) {
        std::cout << this->levelToString(level) << ":" << this->logger_name << ":" << message << std::endl;
    }
}

// Explicit instantiations.
template void print_tensor<int>(const torch::Tensor &tensor, int max_n_elements, bool new_line);
template void print_tensor<long>(const torch::Tensor &tensor, int max_n_elements, bool new_line);
template void print_tensor<bool>(const torch::Tensor &tensor, int max_n_elements, bool new_line);
template void print_tensor<float>(const torch::Tensor &tensor, int max_n_elements, bool new_line);

template void print_vector<int>(const std::vector<int> &vector, int max_n_elements);

template void print_vector<torch::Tensor, float>(const std::vector<torch::Tensor> &vector, int start, int max_n_elements);
}  // namespace relab::helpers
