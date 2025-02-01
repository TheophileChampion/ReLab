/**
 * @file experience.hpp
 * @brief Declaration of the experience class.
 */

#ifndef EXPERIENCE_HPP
#define EXPERIENCE_HPP

#include <torch/extension.h>

namespace relab::agents::memory {

    // Alias for a batch of experiences.
    using Batch = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

    /**
     * @brief Class storing an experience.
     */
    class Experience {

    public:

        /// @var obs
        /// The observation tensor at time t.
        torch::Tensor obs;

        /// @var action
        /// The action taken at time t.
        int action;

        /// @var reward
        /// The reward received at time t + 1.
        float reward;

        /// @var done
        /// Flag indicating if the episode ended after this transition.
        bool done;

        /// @var next_obs
        /// The observation tensor at time t + 1.
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
    };
}

#endif //EXPERIENCE_HPP
