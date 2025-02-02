// Copyright 2025 Theophile Champion. No Rights Reserved.

#include "helpers/serialize.hpp"

#include <vector>

namespace relab::helpers {

template <class T> std::vector<T> load_vector(std::istream &checkpoint) {

  // Create the variables required for loading the vector.
  int capacity = load_value<int>(checkpoint);
  std::vector<T> vector;
  vector.reserve(capacity);

  // Load the vector.
  int size = load_value<int>(checkpoint);
  for (auto i = 0; i < size; i++) {
    vector.push_back(load_value<T>(checkpoint));
  }
  return vector;
}

template <class T>
void save_vector(const std::vector<T> &vector, std::ostream &checkpoint) {

  // Save the vector.
  int capacity = static_cast<int>(vector.capacity());
  save_value(capacity, checkpoint);
  int size = static_cast<int>(vector.size());
  save_value(size, checkpoint);
  for (auto i = 0; i < size; i++) {
    save_value<T>(vector[i], checkpoint);
  }
}

template <class TensorType, class DataType>
std::vector<TensorType> load_vector(std::istream &checkpoint) {

  // Create the variables required for loading the vector.
  int capacity = load_value<int>(checkpoint);
  std::vector<TensorType> vector;
  vector.reserve(capacity);

  // Load the vector.
  int size = load_value<int>(checkpoint);
  for (auto i = 0; i < size; i++) {
    vector.push_back(load_tensor<DataType>(checkpoint));
  }
  return vector;
}

template <class TensorType, class DataType>
void save_vector(
    const std::vector<TensorType> &vector, std::ostream &checkpoint
) {

  // Save the vector of tensors.
  int capacity = static_cast<int>(vector.capacity());
  save_value(capacity, checkpoint);
  int size = static_cast<int>(vector.size());
  save_value(size, checkpoint);
  for (auto i = 0; i < size; i++) {
    save_tensor<DataType>(vector[i], checkpoint);
  }
}

template <class T> T load_value(std::istream &checkpoint) {
  T value;
  checkpoint.read((char *)&value, sizeof(value));
  return value;
}

template <class T> void save_value(const T &value, std::ostream &checkpoint) {
  checkpoint.write((char *)&value, sizeof(value));
}

template <class T> torch::Tensor load_tensor(std::istream &checkpoint) {

  // Load a header describing the tensor's shape.
  int n_dim = load_value<int>(checkpoint);
  long n_elements = 1;
  std::vector<long> shape;
  for (auto i = 0; i < n_dim; i++) {
    long size = load_value<long>(checkpoint);
    n_elements *= size;
    shape.push_back(size);
  }

  // Check if the tensor is empty.
  if (n_elements == 0) {
    return torch::Tensor();
  }

  // Load the tensor.
  bool is_cuda = load_value<bool>(checkpoint);
  auto options = torch::TensorOptions().dtype(torch::CppTypeToScalarType<T>());
  torch::Tensor tensor = torch::zeros(at::IntArrayRef(shape), options);
  checkpoint.read((char *)tensor.data_ptr(), sizeof(T) * n_elements);
  if (is_cuda == true) {
    tensor = tensor.to(torch::Device(torch::kCUDA));
  }
  return tensor;
}

template <class T>
void save_tensor(const torch::Tensor &tensor, std::ostream &checkpoint) {

  // Save a header describing the tensor's shape.
  int n_dim = tensor.dim();
  save_value(n_dim, checkpoint);
  for (auto i = 0; i < n_dim; i++) {
    save_value<long>(tensor.size(i), checkpoint);
  }

  // Check if the tensor is empty.
  if (tensor.numel() == 0) {
    return;
  }

  // Save the tensor.
  bool is_cuda = tensor.is_cuda();
  save_value(is_cuda, checkpoint);
  torch::Tensor tensor_cpu = (is_cuda) ? tensor.clone().cpu() : tensor;
  checkpoint.write((char *)tensor_cpu.data_ptr(), sizeof(T) * tensor.numel());
}

// Explicit instantiations.
template std::vector<int> load_vector<int>(std::istream &checkpoint);
template std::vector<double> load_vector<double>(std::istream &checkpoint);
template void
save_vector<int>(const std::vector<int> &vector, std::ostream &checkpoint);
template void save_vector<double>(
    const std::vector<double> &vector, std::ostream &checkpoint
);

template std::vector<torch::Tensor>
load_vector<torch::Tensor, float>(std::istream &checkpoint);
template void save_vector<torch::Tensor, float>(
    const std::vector<torch::Tensor> &vector, std::ostream &checkpoint
);

template int load_value<int>(std::istream &checkpoint);
template bool load_value<bool>(std::istream &checkpoint);
template long load_value<long>(std::istream &checkpoint);
template float load_value<float>(std::istream &checkpoint);
template double load_value<double>(std::istream &checkpoint);
template void save_value<int>(const int &value, std::ostream &checkpoint);
template void save_value<bool>(const bool &value, std::ostream &checkpoint);
template void save_value<long>(const long &value, std::ostream &checkpoint);
template void save_value<float>(const float &value, std::ostream &checkpoint);
template void save_value<double>(const double &value, std::ostream &checkpoint);

template torch::Tensor load_tensor<int>(std::istream &checkpoint);
template torch::Tensor load_tensor<long>(std::istream &checkpoint);
template torch::Tensor load_tensor<bool>(std::istream &checkpoint);
template torch::Tensor load_tensor<float>(std::istream &checkpoint);
template void
save_tensor<int>(const torch::Tensor &tensor, std::ostream &checkpoint);
template void
save_tensor<long>(const torch::Tensor &tensor, std::ostream &checkpoint);
template void
save_tensor<bool>(const torch::Tensor &tensor, std::ostream &checkpoint);
template void
save_tensor<float>(const torch::Tensor &tensor, std::ostream &checkpoint);
}  // namespace relab::helpers
