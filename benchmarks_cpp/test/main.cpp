#include <cstdlib>
#include "replay_buffer.hpp"
#include <torch/extension.h>
#include <thread>

int main(int argc, char *argv[])
{
    // Create replay buffer.
    auto capacity = 1000000;
    auto buffer = std::make_shared<ReplayBuffer>(capacity=capacity);

    // Append an experience to the replay buffer.
    for (auto i = 0; i < 2 * capacity; i++) {
        if (i == capacity) {
            std::cout << "buffer is full!" << std::endl;
        }
        auto obs = torch::zeros({4, 84, 84});
        auto next_obs = torch::ones({4, 84, 84});
        auto action = 0;
        auto reward = 1.0;
        auto done = false;
        buffer->append(std::make_tuple(obs, action, reward, done, next_obs));
    }
    std::cout << "after append!" << std::endl;

    // Sample a batch from the replay buffer.0
    for (auto i = 0; i < capacity + 10; i++) {
        auto batch = buffer->sample();
    }
    std::cout << "sample!" << std::endl;

    // Report loss of previous batch to the replay buffer.
    auto loss = torch::ones({2});
    loss = buffer->report(loss);
    std::cout << "report!" << std::endl;

    return EXIT_SUCCESS;
}
