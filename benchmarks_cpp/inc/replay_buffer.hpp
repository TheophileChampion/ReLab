#ifndef REPLAY_BUFFER_HPP
#define REPLAY_BUFFER_HPP

#include <torch/extension.h>
#include "frame_buffer.hpp"
#include "data_buffer.hpp"
#include "experience.hpp"


/**
 * Class implementing a replay buffer with support for prioritization [1] and multistep Q-learning [2] from:
 *
 * [1] Tom Schaul. Prioritized experience replay. arXiv preprint arXiv:1511.05952, 2015.
 * [2] Richard S Sutton. Learning to predict by the methods of temporal differences. Machine learning, 3:9–44, 1988.
 */
class ReplayBuffer {

private:

    // Keep in mind whether the replay buffer is prioritized.
    bool prioritized;
    
    // Store the buffer parameters.
    int capacity;
    int batch_size;
    int stack_size;
    int frame_skip;
    float gamma;
    int n_steps;
    float initial_priority;
    int n_children;
    float omega;
    float omega_is;

    // The device on which computation is performed.
    torch::Device device;

    // The buffer storing the frames of all experiences.
    std::unique_ptr<FrameBuffer> observations;
    
    // The buffer storing the data (i.e., actions, rewards, dones and priorities) of all experiences.
    std::unique_ptr<DataBuffer> data;
    
    // The indices of the last sampled experiences.
    torch::Tensor indices;

public:

    /**
     * Create a replay buffer.
     * @param capacity the number of experience the buffer can store
     * @param batch_size the size of the batch to sample
     * @param frame_skip the number of times each action is repeated in the environment, if None use the configuration
     * @param stack_size the number of stacked frame in each observation, if None use the configuration
     * @param p_args the prioritization arguments (None for no prioritization) composed of:
     *     - initial_priority: the maximum experience priority given to new transitions
     *     - omega: the prioritization exponent
     *     - omega_is: the important sampling exponent
     *     - n_children: the maximum number of children each node of the priority-tree can have
     * @param m_args the multistep arguments (None for no multistep) composed of:
     *     - n_steps: the number of steps for which rewards are accumulated in multistep Q-learning
     *     - gamma: the discount factor
     */
    ReplayBuffer(
        int capacity=10000, int batch_size=32, int frame_skip=1, int stack_size=4,
        std::map<std::string, float> p_args={}, std::map<std::string, float> m_args={}
    );

    /**
     * Add a new experience to the buffer.
     * @param experience_tuple the experience to add
     */
    void append(ExperienceTuple experience_tuple);

    /**
     * Sample a batch from the replay buffer.
     * @return (observations, actions, rewards, done, next_observations) where:
     *   - observations: the batch of observations
     *   - actions: the actions performed
     *   - rewards: the rewards received
     *   - done: whether the environment stop after performing the actions
     *   - next_observations: the observations received after performing the actions
     */
    Batch sample();

    /**
     * Report the loss associated with all the transitions of the previous batch.
     * @param loss the loss of all previous transitions
     * @return the new loss
     */
    torch::Tensor report(torch::Tensor loss);

    /**
     * Retrieve the experiences whose indices are passed as parameters.
     * @param indices the experience indices
     * @return the experiences
     */
    Batch getExperiences(torch::Tensor indices);

    /**
     * Retrieve the number of elements in the buffer.
     * @return the number of elements contained in the replay buffer
     */
    int length();

    /**
     * Empty the replay buffer.
     */
    void clear();

    /**
     * Retrieve a boolean indicating whether the replay buffer is prioritized.
     * @return true if the replay buffer is prioritized, false otherwise
     */
    bool getPrioritized();

    /**
     * Transform a list of n dimensional tensors into a tensor with n+1 dimensions.
     * @param tensor_list the list of tensors
     * @return the output tensor
     */
    static torch::Tensor listToTensor(std::vector<torch::Tensor> &tensor_list);

    /**
     * Retrieves the device on which the computation should be performed.
     * @return the device
     */
    static torch::Device getDevice();

    /**
     * Retrieve the last sampled indices.
     * @return the indices
     */
    torch::Tensor getLastIndices();

    /**
     * Retrieve the priority at the provided index.
     * @param index the index
     * @return the priority
     */
    float getPriority(int index);
};

#endif //REPLAY_BUFFER_HPP
