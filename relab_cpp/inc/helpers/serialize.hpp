/**
 * @file serialize.hpp
 * @brief Helper functions to save/load variables into/from a checkpoint.
 */

#ifndef SERIALIZE_HPP
#define SERIALIZE_HPP

#include <torch/extension.h>
#include <vector>

namespace relab::helpers {

    /**
     * Load a vector of integers from a stream.
     * @param checkpoint the stream reading from the checkpoint file
     * @return the vector of integers
     */
    template<class T>
    std::vector<T> load_vector(std::istream &checkpoint);

    /**
     * Save a vector of integers into a stream.
     * @param vector the vector of integers
     * @param checkpoint the stream writing into the checkpoint file
     */
    template<class T>
    void save_vector(const std::vector<T> &vector, std::ostream &checkpoint);

    /**
     * Load a vector of tensors from a stream.
     * @param checkpoint the stream reading from the checkpoint file
     * @return the vector of tensors
     */
    template<class TensorType, class DataType>
    std::vector<TensorType> load_vector(std::istream &checkpoint);

    /**
     * Save a vector of tensors into a stream.
     * @param vector the vector of tensors
     * @param checkpoint the stream writing into the checkpoint file
     */
    template<class TensorType, class DataType>
    void save_vector(const std::vector<TensorType> &vector, std::ostream &checkpoint);

    /**
     * Load a value from a stream.
     * @param checkpoint the stream reading from the checkpoint file
     * @return the value
     */
    template<class T>
    T load_value(std::istream &checkpoint);

    /**
     * Save a value into a stream.
     * @param value the value to save
     * @param checkpoint the stream writing into the checkpoint file
     */
    template<class T>
    void save_value(const T &value, std::ostream &checkpoint);

    /**
     * Load a tensor from a stream.
     * @param checkpoint the stream reading from the checkpoint file
     * @return the tensor
     */
    template<class T>
    torch::Tensor load_tensor(std::istream &checkpoint);

    /**
     * Save a tensor into a stream.
     * @param tensor the tensor to save
     * @param checkpoint the stream writing into the checkpoint file
     */
    template<class T>
    void save_tensor(const torch::Tensor &tensor, std::ostream &checkpoint);
}

#endif //SERIALIZE_HPP
