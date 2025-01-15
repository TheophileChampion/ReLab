#ifndef EXPERIENCE_HPP
#define EXPERIENCE_HPP

#include <torch/extension.h>


// Alias for a single experience.
using ExperienceTuple = std::tuple<torch::Tensor, int, float, bool, torch::Tensor>;

// Alias for a batch of experiences.
using Batch = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;


/**
 * Class storing an experience.
 */
class Experience {

public:

    torch::Tensor obs;
    int action;
    float reward;
    bool done;
    torch::Tensor next_obs;

public:

    /**
     * Create an experience.
     * @param obs the observation at time t
     * @param action the action at time t
     * @param reward the reward at time t + 1
     * @param done true if episode ended, false otherwise
     * @param next_obs the observation at time t + 1
     */
    Experience(torch::Tensor obs, int action, float reward, bool done, torch::Tensor next_obs);

    /**
     * Create an experience.
     * @param obs the observation at time t
     * @param action the action at time t
     * @param reward the reward at time t + 1
     * @param done true if episode ended, false otherwise
     * @param next_obs the observation at time t + 1
     */
    Experience(const ExperienceTuple &experience);
};

#endif //EXPERIENCE_HPP
