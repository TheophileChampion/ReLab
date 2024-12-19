#include "frame.hpp"

Frame::Frame() : index(-1), data(torch::empty({})) {}

Frame::Frame(int index, torch::Tensor data) : index(index), data(data) {}
