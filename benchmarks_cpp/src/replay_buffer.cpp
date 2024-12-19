#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include "replay_buffer.hpp"


ReplayBuffer::ReplayBuffer(
    int capacity, int batch_size, int frame_skip, int stack_size,
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
    this->observations = std::make_unique<FrameBuffer>(this->capacity, this->frame_skip, this->n_steps, this->stack_size);

    // The buffer storing the data (i.e., actions, rewards, dones and priorities) of all experiences.
    this->data = std::make_unique<DataBuffer>(this->capacity, this->n_steps, this->gamma, this->initial_priority, this->n_children);

    // TODO
    std::cout << "capacity: "<< this->capacity << " batch_size: " << this->batch_size << " stack_size: " << this->stack_size
              << " frame_skip: " << this->frame_skip << " gamma: " << this->gamma << " n_steps: " << this->n_steps
              << " initial_priority: " << this->initial_priority << " n_children: " << this->n_children
              << " omega: " << this->omega << " omega_is: " << this->omega_is << " prioritized: " << this->prioritized << std::endl;
}

void ReplayBuffer::append(ExperienceTuple experience_tuple) {
    this->observations->append(experience_tuple);
    this->data->append(experience_tuple);

    // TODO
    // TODO "obs: {" << experience.obs << "} " <<
    // TODO << " next_obs: {" << experience.next_obs << "}"
    // TODO std::cout << "action: " << experience.action << " reward: " << experience.reward << " done: " << experience.done << std::endl;
}

Batch ReplayBuffer::sample() {

    // Sample a batch from the replay buffer.
    if (this->prioritized == true) {
        this->indices = this->data->getPriorities()->sampleIndices(this->batch_size);
    } else {
        this->indices = torch::randint(0, this->length(), {this->batch_size});
    }

    // Retrieve the batch corresponding to the sampled indices.
    return this->getExperiences(this->indices);  // TODO ensure the observations are on device
}

torch::Tensor ReplayBuffer::report(torch::Tensor loss) {

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

    // If the buffer is prioritized, update the priorities.
    float sum_priorities = this->data->getPriorities()->sum();
    torch::Tensor priorities = torch::zeros({this->batch_size}, at::kFloat);
    for (int i = 0; i < this->batch_size; i++) {
        int idx = this->indices[i].item<int>();
        float priority = loss[i].item<float>();
        if (std::isfinite(priority) == false) {
            priority = this->data->getPriorities()->max();
        }
        priorities[i] = this->data->getPriorities()->get(idx);
        this->data->getPriorities()->set(idx, priority);
    }

    // Update the priorities and compute the importance sampling weights.
    torch::Tensor weights = this->length() * priorities.to(this->device) / sum_priorities;
    weights = torch::pow(weights, -this->omega_is);
    return loss * weights / weights.max();
}

Batch ReplayBuffer::getExperiences(torch::Tensor indices) {
    std::vector<torch::Tensor> obs;
    std::vector<torch::Tensor> next_obs;

    for (auto i = 0; i < this->batch_size; i++) {
        int index = indices.index({i}).item<int>();
        auto observations = (*this->observations)[index];
        obs.push_back(std::get<0>(observations));
        next_obs.push_back(std::get<1>(observations));
    }
    auto data = (*this->data)[indices];
    return std::make_tuple(
        this->listToTensor(obs).to(this->device),
        std::get<0>(data), std::get<1>(data), std::get<2>(data),
        this->listToTensor(next_obs).to(this->device)
    );
}

int ReplayBuffer::length() {
    return this->observations->length();
}

void ReplayBuffer::clear() {
    this->observations->clear();
    this->data->clear();
    this->indices = torch::Tensor();
}

bool ReplayBuffer::getPrioritized() {
    return this->prioritized;
}

torch::Tensor ReplayBuffer::listToTensor(std::vector<torch::Tensor> &tensor_list) {
    for (auto it = tensor_list.begin(); it != tensor_list.end(); it++) {
        *it = torch::unsqueeze(*it, 0);
    }
    return torch::cat(tensor_list);
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
