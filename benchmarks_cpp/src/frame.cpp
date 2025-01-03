#include "frame.hpp"

Frame::Frame() : index(-1), data(torch::empty({})) {}

Frame::Frame(int index, torch::Tensor data) : index(index), data(data) {}

Frame &Frame::operator=(const Frame &other) {

    // Check for self assignment.
    if (this == &other) {
        return *this;
    }

    // Copy the attributes.
    this->index = other.index;
    this->data = other.data;
    return *this;
}
