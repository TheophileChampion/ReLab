#ifndef DEBUG_HPP
#define DEBUG_HPP

#include <torch/extension.h>
#include <vector>

namespace relab::helpers {

    /**
     * Print a tensor on the standard output.
     * @param tensor the tensor to display
     * @param max_n_elements the maximum number of tensor elements to display, by default all elements are displayed
     */
    template<class T>
    void print_tensor(const torch::Tensor &tensor, int max_n_elements=-1, bool new_line=true);

    /**
     * Print a vector on the standard output.
     * @param vector the vector to display
     * @param max_n_elements the maximum number of vector elements to display, by default all elements are displayed
     */
    template<class T>
    void print_vector(const std::vector<T> &vector, int max_n_elements=-1);

    /**
     * Print a vector of tensors on the standard output.
     * @param vector the vector of tensors
     * @param start the index corresponding to the first element in the vector
     * @param max_n_elements the maximum number of vector elements to display, by default all elements are displayed
     */
    template<class TensorType, class DataType>
    void print_vector(const std::vector<TensorType> &vector, int start=0, int max_n_elements=-1);

    /**
     * Print a boolean on the standard output.
     * @param value the boolean to display
     */
    void print_bool(bool value);

    /**
     * Print an ellipse on the standard output.
     * @param max_n_elements the maximum number of vector elements to display
     * @param size the size of the container
     */
    void print_ellipse(int max_n_elements, int size);
}

#endif //DEBUG_HPP
