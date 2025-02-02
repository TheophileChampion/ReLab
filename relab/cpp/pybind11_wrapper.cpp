// Copyright 2025 Theophile Champion. No Rights Reserved.

#include <pybind11/pybind11.h>

#include <map>
#include <string>

#include "agents/memory/compressors.hpp"
#include "agents/memory/experience.hpp"
#include "agents/memory/replay_buffer.hpp"

namespace py = pybind11;
using namespace pybind11::literals;  // NOLINT
using relab::agents::memory::CompressorType;
using relab::agents::memory::Experience;
using relab::agents::memory::ReplayBuffer;

PYBIND11_MODULE(cpp, m) {
  m.doc() = "A module providing C++ acceleration for ReLab.";

  auto m_agents = m.def_submodule("agents", "A module containing a C++ implementation of RL agents.");
  auto m_memory = m_agents.def_submodule("memory", "A module containing a C++ implementation of the replay buffer.");

  py::class_<Experience>(m_memory, "Experience")
      .def(
          py::init<torch::Tensor, int, float, bool, torch::Tensor>(), "obs"_a, "action"_a, "reward"_a, "done"_a,
          "next_obs"_a
      );

  py::enum_<CompressorType>(m_memory, "CompressorType")
      .value("RAW", CompressorType::RAW)
      .value("ZLIB", CompressorType::ZLIB);

  py::class_<ReplayBuffer>(m_memory, "FastReplayBuffer")
      .def(
          py::init<int, int, int, int, int, CompressorType>(), "capacity"_a = 10000, "batch_size"_a = 32,
          "frame_skip"_a = 1, "stack_size"_a = 4, "screen_size"_a = 84, "type"_a = CompressorType::ZLIB
      )
      .def(
          py::init<int, int, int, int, int, CompressorType, std::map<std::string, float>>(), "capacity"_a = 10000,
          "batch_size"_a = 32, "frame_skip"_a = 1, "stack_size"_a = 4, "screen_size"_a = 84,
          "type"_a = CompressorType::ZLIB, "args"_a
      )
      .def("append", &ReplayBuffer::append, "Add an experience to the replay buffer.")
      .def("sample", &ReplayBuffer::sample, "Sample a batch from the replay buffer.")
      .def(
          "report", &ReplayBuffer::report,
          "Report the loss associated with all the transitions of the "
          "previous batch."
      )
      .def("load", &ReplayBuffer::load, "Load a replay buffer from the filesystem.")
      .def("save", &ReplayBuffer::save, "Save the replay buffer on the filesystem.")
      .def("clear", &ReplayBuffer::clear, "Empty the replay buffer.")
      .def("length", &ReplayBuffer::size, "Retrieve the number of elements in the buffer.");
}
