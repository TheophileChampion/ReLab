#include <pybind11/pybind11.h>
#include "replay_buffer.hpp"
#include "priority_tree.hpp"
#include "frame_buffer.hpp"
#include "data_buffer.hpp"
#include "compressors.hpp"
#include "deque.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(cpp, m) {

    m.doc() = "A C++ module providing a fast replay buffer.";

    py::enum_<CompressorType>(m, "CompressorType")
        .value("RAW", CompressorType::RAW)
        .value("ZLIB", CompressorType::ZLIB);

    py::class_<ReplayBuffer>(m, "FastReplayBuffer")
        .def(py::init<int, int, int, int, int, CompressorType>(), "capacity"_a = 10000, "batch_size"_a = 32, "frame_skip"_a = 1, "stack_size"_a = 4, "screen_size"_a = 84, "type"_a = CompressorType::ZLIB)
        .def(py::init<int, int, int, int, int, CompressorType, std::map<std::string, float>>(), "capacity"_a = 10000, "batch_size"_a = 32, "frame_skip"_a = 1, "stack_size"_a = 4, "screen_size"_a = 84, "type"_a = CompressorType::ZLIB, "m_args"_a)
        .def(py::init<int, int, int, int, int, CompressorType, std::map<std::string, float>>(), "capacity"_a = 10000, "batch_size"_a = 32, "frame_skip"_a = 1, "stack_size"_a = 4, "screen_size"_a = 84, "type"_a = CompressorType::ZLIB, "p_args"_a)
        .def(py::init<int, int, int, int, int, CompressorType, std::map<std::string, float>, std::map<std::string, float>>(), "capacity"_a = 10000, "batch_size"_a = 32, "frame_skip"_a = 1, "stack_size"_a = 4, "screen_size"_a = 84, "type"_a = CompressorType::ZLIB, "p_args"_a, "m_args"_a)
        .def("append", &ReplayBuffer::append, "Add an experience to the replay buffer.")
        .def("sample", &ReplayBuffer::sample, "Sample a batch from the replay buffer.")
        .def("report", &ReplayBuffer::report, "Report the loss associated with all the transitions of the previous batch.")
        .def("get_experiences", &ReplayBuffer::getExperiences, "Retrieve the experiences whose indices are passed as parameters.")
        .def("clear", &ReplayBuffer::clear, "Empty the replay buffer.")
        .def("length", &ReplayBuffer::size, "Retrieve the number of elements in the buffer.")
        .def("is_prioritized", &ReplayBuffer::getPrioritized, "Retrieve a boolean indicating whether the replay buffer is prioritized.")
        .def("get_last_indices", &ReplayBuffer::getLastIndices, "Retrieves the last sampled indices.")
        .def("get_priority", &ReplayBuffer::getPriority, "Retrieves the priority at the provided index.");

    py::class_<PriorityTree>(m, "FastPriorityTree")
        .def(py::init<int, float, int>(), "capacity"_a, "initial_priority"_a, "n_children"_a)
        .def("append", &PriorityTree::append, "Add a priority in the priority tree.")
        .def("sum", &PriorityTree::sum, "Compute the sum of all priorities.")
        .def("max", &PriorityTree::max, "Find the largest priority.")
        .def("clear", &PriorityTree::clear, "Empty the priority tree.")
        .def("length", &PriorityTree::size, "Retrieve the number of priorities stored in the priority tree.")
        .def("get", &PriorityTree::get, "Retrieve a priority from the priority tree.")
        .def("set", &PriorityTree::set, "Replace a priority in the priority tree.")
        .def("parent_index", &PriorityTree::parentIndex, "Compute the index of the parent element.")
        .def("sample_indices", &PriorityTree::sampleIndices, "Sample indices of buffer elements proportionally to their priorities.")
        .def("tower_sampling", &PriorityTree::towerSampling, "Compute the experience index associated to the sampled priority using inverse transform sampling.")
        .def("sum_tree_to_str", &PriorityTree::sumTreeToStr, "Create a string representation of the sum-tree.")
        .def("max_tree_to_str", &PriorityTree::maxTreeToStr, "Create a string representation of the max-tree.")
        .def("update_sum_tree", &PriorityTree::updateSumTree, "Update the sum-tree to reflect an element being set to a new priority.")
        .def("update_max_tree", &PriorityTree::updateMaxTree, "Update the max-tree to reflect an element being set to a new priority.");

    py::class_<DataBuffer>(m, "FastDataBuffer")
        .def(py::init<int, int, float, float, int>(), "capacity"_a, "n_steps"_a, "gamma"_a, "initial_priority"_a, "n_children"_a)
        .def("append", &DataBuffer::append, "Add the datum of the next experience to the buffer.")
        .def("get", &DataBuffer::operator[], "Retrieve the data of the experiences whose indices are passed as parameters.")
        .def("length", &DataBuffer::size, "Retrieve the number of experiences stored in the buffer.")
        .def("clear", &DataBuffer::clear, "Empty the data buffer.");

    py::class_<FrameBuffer>(m, "FastFrameBuffer")
        .def(py::init<int, int, int, int, int>(), "capacity"_a, "frame_skip"_a, "n_steps"_a, "stack_size"_a, "screen_size"_a = 84)
        .def("append", &FrameBuffer::append, "Add the frames of the next experience to the buffer.")
        .def("get", &FrameBuffer::operator[], "Retrieve the observations of the experience whose index is passed as parameters.")
        .def("length", &FrameBuffer::size, "Retrieve the number of experiences stored in the buffer.")
        .def("clear", &FrameBuffer::clear, "Empty the frame buffer.")
        .def("encode", &FrameBuffer::encode, "Encode a frame to compress it.")
        .def("decode", &FrameBuffer::decode, "Decode a frame to decompress it.");

    py::class_<Deque<int>>(m, "IntDeque")
        .def(py::init<int>(), "max_size"_a)
        .def("append", &Deque<int>::push_back, "Add an element at the end of the queue.")
        .def("append_left", &Deque<int>::push_front, "Add an element at the front of the queue.")
        .def("get", &Deque<int>::get, "Retrieve the element whose index is passed as parameters.")
        .def("clear", &Deque<int>::clear, "Remove all the elements of the deque.")
        .def("pop", &Deque<int>::pop_back, "Remove an elements from the end of the deque.")
        .def("pop_left", &Deque<int>::pop_front, "Remove an elements from the front of the deque.")
        .def("length", &Deque<int>::size, "Return the size of the deque.");
}
