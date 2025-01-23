#include <pybind11/pybind11.h>
#include "agents/memory/replay_buffer.hpp"
#include "agents/memory/compressors.hpp"
#include "agents/memory/experience.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace relab::agents::memory;

PYBIND11_MODULE(cpp, m) {

    m.doc() = "A C++ module providing a fast replay buffer.";

    py::class_<Experience>(m, "Experience")
        .def(py::init<torch::Tensor, int, float, bool, torch::Tensor>(), "obs"_a, "action"_a, "reward"_a, "done"_a, "next_obs"_a);

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
        .def("load", &ReplayBuffer::load, "Load a replay buffer from the filesystem.")
        .def("save", &ReplayBuffer::save, "Save the replay buffer on the filesystem.")
        .def("clear", &ReplayBuffer::clear, "Empty the replay buffer.")
        .def("length", &ReplayBuffer::size, "Retrieve the number of elements in the buffer.")
        .def("is_prioritized", &ReplayBuffer::getPrioritized, "Retrieve a boolean indicating whether the replay buffer is prioritized.")
        .def("get_last_indices", &ReplayBuffer::getLastIndices, "Retrieves the last sampled indices.")
        .def("get_priority", &ReplayBuffer::getPriority, "Retrieves the priority at the provided index.");
}
