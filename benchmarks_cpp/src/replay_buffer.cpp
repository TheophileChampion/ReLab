#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include "replay_buffer.hpp"

ReplayBuffer::ReplayBuffer(
    int capacity, int batch_size, int frame_skip, int stack_size, int screen_size, CompressorType type,
    std::map<std::string, float> p_args, std::map<std::string, float> m_args
): device(ReplayBuffer::getDevice()) {

    // A map storing all the prioritization and multistep arguments.
    std::map<std::string, float> args;
    args.insert(p_args.begin(), p_args.end());
    args.insert(m_args.begin(), m_args.end());

    // Keep in mind whether the replay buffer is prioritized.
    this->prioritized = false;
    for (auto key : {"initial_priority", "omega", "omega_is", "n_children"}) {
        if (args.find(key) != args.end()) {
          this->prioritized = true;
          break;
        }
    }

    // Default values of the prioritization and multistep arguments.
    std::map<std::string, float> default_p_args = {
        {"initial_priority", 1.0},
        {"omega", 1.0},
        {"omega_is", 1.0},
        {"n_children", 10}
    };
    std::map<std::string, float> default_m_args = {
        {"n_steps", 1.0},
        {"gamma", 0.99}
    };

    // Complete arguments with default values.
    args.insert(default_p_args.begin(), default_p_args.end());
    args.insert(default_m_args.begin(), default_m_args.end());

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
    int n_threads = std::min(static_cast<int>(thread::hardware_concurrency()), batch_size);
    this->observations = std::make_unique<FrameBuffer>(
        this->capacity, this->frame_skip, this->n_steps, this->stack_size, screen_size, type, n_threads
    );

    // The buffer storing the data (i.e., actions, rewards, dones and priorities) of all experiences.
    this->data = std::make_unique<DataBuffer>(
        this->capacity, this->n_steps, this->gamma, this->initial_priority, this->n_children
    );

    // TODO
    std::cout << "capacity: "<< this->capacity << " batch_size: " << this->batch_size << " stack_size: " << this->stack_size
              << " frame_skip: " << this->frame_skip << " gamma: " << this->gamma << " n_steps: " << this->n_steps
              << " initial_priority: " << this->initial_priority << " n_children: " << this->n_children
              << " omega: " << this->omega << " omega_is: " << this->omega_is << " prioritized: " << this->prioritized << std::endl;
}

void ReplayBuffer::append(const ExperienceTuple &experience_tuple) {
    this->observations->append(experience_tuple);
    this->data->append(experience_tuple);
}

Batch ReplayBuffer::sample() {

    // Sample a batch from the replay buffer.
    if (this->prioritized == true) {
        this->indices = this->data->getPriorities()->sampleIndices(this->batch_size);
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
    torch::Tensor weights = this->size() * priorities.to(this->device) / sum_priorities;
    weights = torch::pow(weights, -this->omega_is);
    return loss * weights / weights.max();
}

Batch ReplayBuffer::getExperiences(torch::Tensor &indices) {
    auto observations = (*this->observations)[indices];
    auto data = (*this->data)[indices];
    return std::make_tuple(
        std::get<0>(observations).to(this->device),
        std::get<0>(data), std::get<1>(data), std::get<2>(data),
        std::get<1>(observations).to(this->device)
    );
}

int ReplayBuffer::size() {
    return this->observations->size();
}

void ReplayBuffer::clear() {
    this->observations->clear();
    this->data->clear();
    this->indices = torch::Tensor();
}

bool ReplayBuffer::getPrioritized() {
    return this->prioritized;
}

torch::Device ReplayBuffer::getDevice() {
    bool use_cuda = (torch::cuda::is_available() and torch::cuda::device_count() >= 1);
    return torch::Device((use_cuda == true) ? torch::kCUDA: torch::kCPU);
}

torch::Tensor ReplayBuffer::getLastIndices() {
    return this->indices;
}

float ReplayBuffer::getPriority(int index) {
    return this->data->getPriorities()->get(index);
}
