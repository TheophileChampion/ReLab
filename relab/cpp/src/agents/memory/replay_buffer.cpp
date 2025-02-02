// Copyright 2025 Theophile Champion. No Rights Reserved.

#include "agents/memory/replay_buffer.hpp"
#include <cmath>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>

#include "helpers/debug.hpp"
#include "helpers/serialize.hpp"
#include "helpers/torch.hpp"

using namespace relab::helpers;
using namespace std::experimental::filesystem;

namespace relab::agents::memory {

ReplayBuffer::ReplayBuffer(
    int capacity, int batch_size, int frame_skip, int stack_size,
    int screen_size, CompressorType type, std::map<std::string, float> args
) : device(getDevice()) {
  // Keep in mind whether the replay buffer is prioritized.
  this->prioritized = false;
  for (auto key : {"initial_priority", "omega", "omega_is", "n_children"}) {
    if (args.find(key) != args.end()) {
      this->prioritized = true;
      break;
    }
  }

  // Default values of the prioritization and multistep arguments.
  std::map<std::string, float> default_args = {
      {"initial_priority", 1.0}, {"omega", 1.0},   {"omega_is", 1.0},
      {"n_children", 10},        {"n_steps", 1.0}, {"gamma", 0.99}
  };

  // Complete arguments with default values.
  args.insert(default_args.begin(), default_args.end());

  // Store the buffer parameters.
  this->capacity = capacity;
  this->batch_size = batch_size;
  this->stack_size = stack_size;
  this->frame_skip = frame_skip;
  this->gamma = args["gamma"];
  this->n_steps = static_cast<int>(args["n_steps"]);
  this->initial_priority = args["initial_priority"];
  this->n_children = static_cast<int>(args["n_children"]);
  this->omega = args["omega"];
  this->omega_is = args["omega_is"];

  // The buffer storing the frames of all experiences.
  int n_threads = std::min(
      static_cast<int>(std::thread::hardware_concurrency()), batch_size
  );
  this->observations = std::make_unique<FrameBuffer>(
      this->capacity, this->frame_skip, this->n_steps, this->stack_size,
      screen_size, type, n_threads
  );

  // The buffer storing the data (i.e., actions, rewards, dones and priorities)
  // of all experiences.
  this->data = std::make_unique<DataBuffer>(
      this->capacity, this->n_steps, this->gamma, this->initial_priority,
      this->n_children
  );
}

void ReplayBuffer::append(const Experience &experience) {
  this->observations->append(experience);
  this->data->append(experience);
}

Batch ReplayBuffer::sample() {
  // Sample a batch from the replay buffer.
  if (this->prioritized == true) {
    this->indices =
        this->data->getPriorities()->sampleIndices(this->batch_size);
  } else {
    this->indices = torch::randint(0, this->size(), {this->batch_size});
  }

  // Retrieve the batch corresponding to the sampled indices.
  return this->getExperiences(this->indices);
}

torch::Tensor ReplayBuffer::report(torch::Tensor &loss) {
  // If the buffer is not prioritized, don't update the priorities.
  if (this->prioritized == false) {
    return loss;
  }

  // Add a small positive constant to avoid zero probabilities.
  loss += 1e-5;

  // Raise the loss to the power of the prioritization exponent.
  if (this->omega != 1.0) {
    loss = loss.pow(this->omega);
  }

  // Collect the old priorities.
  torch::Tensor priorities = torch::zeros({this->batch_size}, at::kFloat);
  for (int i = 0; i < this->batch_size; i++) {
    int idx = this->indices[i].item<int>();
    priorities[i] = this->data->getPriorities()->get(idx);
  }

  // Update the priorities.
  float sum_priorities = this->data->getPriorities()->sum();
  for (int i = 0; i < this->batch_size; i++) {
    int idx = this->indices[i].item<int>();
    float priority = loss[i].item<float>();
    if (std::isfinite(priority) == false) {
      priority = this->data->getPriorities()->max();
    }
    this->data->getPriorities()->set(idx, priority);
  }

  // Update the priorities and compute the importance sampling weights.
  torch::Tensor weights =
      this->size() * priorities.to(this->device) / sum_priorities;
  weights = torch::pow(weights, -this->omega_is);
  return loss * weights / weights.max();
}

void ReplayBuffer::load(
    std::string checkpoint_path, std::string checkpoint_name, bool save_all
) {
  // Check that the replay buffer checkpoint exist.
  auto path =
      this->getCheckpointPath(checkpoint_path, checkpoint_name, save_all);
  if (!exists(path) || !path.has_filename()) {
    logging.info("Could not load the replay buffer from: " + path.string());
    return;
  }

  // Open the checkpoint file.
  std::ifstream checkpoint;
  checkpoint.open(path.string());
  this->loadFromFile(checkpoint);
}

void ReplayBuffer::loadFromFile(std::istream &checkpoint) {
  // Read the replay buffer from the checkpoint file.
  this->prioritized = load_value<bool>(checkpoint);
  this->capacity = load_value<int>(checkpoint);
  this->batch_size = load_value<int>(checkpoint);
  this->stack_size = load_value<int>(checkpoint);
  this->frame_skip = load_value<int>(checkpoint);
  this->gamma = load_value<float>(checkpoint);
  this->n_steps = load_value<int>(checkpoint);
  this->initial_priority = load_value<float>(checkpoint);
  this->n_children = load_value<int>(checkpoint);
  this->omega = load_value<float>(checkpoint);
  this->omega_is = load_value<float>(checkpoint);
  this->observations->load(checkpoint);
  this->data->load(checkpoint);
  this->indices = load_tensor<long>(checkpoint);
}

void ReplayBuffer::save(
    std::string checkpoint_path, std::string checkpoint_name, bool save_all
) {
  // Create the replay buffer checkpoint directory and file, if they do not
  // exist.
  auto path =
      this->getCheckpointPath(checkpoint_path, checkpoint_name, save_all);

  auto directory_name = (path.has_filename()) ? path.parent_path() : path;
  if (!exists(directory_name)) {
    create_directory(directory_name);
  }

  // Open the checkpoint file.
  std::ofstream checkpoint;
  checkpoint.open(path.string());
  this->saveToFile(checkpoint);
}

void ReplayBuffer::saveToFile(std::ostream &checkpoint) {
  // Write the replay buffer in the checkpoint file.
  save_value(this->prioritized, checkpoint);
  save_value(this->capacity, checkpoint);
  save_value(this->batch_size, checkpoint);
  save_value(this->stack_size, checkpoint);
  save_value(this->frame_skip, checkpoint);
  save_value(this->gamma, checkpoint);
  save_value(this->n_steps, checkpoint);
  save_value(this->initial_priority, checkpoint);
  save_value(this->n_children, checkpoint);
  save_value(this->omega, checkpoint);
  save_value(this->omega_is, checkpoint);
  this->observations->save(checkpoint);
  this->data->save(checkpoint);
  save_tensor<long>(this->indices, checkpoint);
}

path ReplayBuffer::getCheckpointPath(
    std::string &checkpoint_path, std::string &checkpoint_name, bool save_all
) {
  // If all replay buffer must be saved and checkpoint name was not provided,
  // replace "model" by "buffer" in the checkpoint path.
  if (checkpoint_name == "" && save_all == true) {
    int index = checkpoint_path.find("model_");
    path new_checkpoint_path = checkpoint_path.replace(index, 6, "buffer_");
    return new_checkpoint_path;
  }

  // Concatenate the checkpoint directory with the checkpoint file name.
  path directory(std::getenv("CHECKPOINT_DIRECTORY"));
  path file((checkpoint_name == "") ? "buffer.pt" : checkpoint_name);
  return directory / file;
}

void ReplayBuffer::print(bool verbose) {
  // Display the most important information about the replay buffer.
  std::cout << "ReplayBuffer[prioritized: ";
  print_bool(this->prioritized);
  std::cout << ", capacity: " << this->capacity
            << ", batch_size: " << this->batch_size
            << ", stack_size: " << this->stack_size
            << ", frame_skip: " << this->frame_skip
            << ", gamma: " << this->gamma << ", n_steps: " << this->n_steps
            << ", initial_priority: " << this->initial_priority
            << ", n_children: " << this->n_children
            << ", omega: " << this->omega << ", omega_is: " << this->omega_is
            << "]" << std::endl;

  // Display optional information about the replay buffer.
  if (verbose == true) {
    std::cout << " #-> indices = ";
    print_tensor<long>(this->indices);
    std::cout << " #-> observations: ";
    this->observations->print(verbose, "     ");
    std::cout << " #-> data: ";
    this->data->print(verbose, "     ");
  }
}

Batch ReplayBuffer::getExperiences(torch::Tensor &indices) {
  auto observations = (*this->observations)[indices];
  auto data = (*this->data)[indices];
  return std::make_tuple(
      std::get<0>(observations).to(this->device), std::get<0>(data),
      std::get<1>(data), std::get<2>(data),
      std::get<1>(observations).to(this->device)
  );
}

int ReplayBuffer::size() { return this->observations->size(); }

void ReplayBuffer::clear() {
  this->observations->clear();
  this->data->clear();
  this->indices = torch::Tensor();
}

bool ReplayBuffer::getPrioritized() { return this->prioritized; }

torch::Tensor ReplayBuffer::getLastIndices() { return this->indices; }

float ReplayBuffer::getPriority(int index) {
  return this->data->getPriorities()->get(index);
}

bool operator==(const ReplayBuffer &lhs, const ReplayBuffer &rhs) {
  // Check that all attributes of standard types are identical.
  if (lhs.prioritized != rhs.prioritized || lhs.capacity != rhs.capacity ||
      lhs.batch_size != rhs.batch_size || lhs.stack_size != rhs.stack_size ||
      lhs.frame_skip != rhs.frame_skip || lhs.gamma != rhs.gamma ||
      lhs.n_steps != rhs.n_steps ||
      lhs.initial_priority != rhs.initial_priority ||
      lhs.n_children != rhs.n_children || lhs.omega != rhs.omega ||
      lhs.omega_is != rhs.omega_is) {
    return false;
  }

  // Compare the frame and data buffers are identical.
  if (*lhs.observations != *rhs.observations || *lhs.data != *rhs.data) {
    return false;
  }

  // Compare the indices.
  return tensorsAreEqual(lhs.indices, rhs.indices);
}
}  // namespace relab::agents::memory
