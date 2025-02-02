// Copyright 2025 Theophile Champion. No Rights Reserved.
/**
 * @file replay_buffer.hpp
 * @brief Declaration of the replay buffer class.
 */

#ifndef RELAB_CPP_INC_AGENTS_MEMORY_REPLAY_BUFFER_HPP_
#define RELAB_CPP_INC_AGENTS_MEMORY_REPLAY_BUFFER_HPP_

#include <torch/extension.h>

#include <map>
#include <memory>
#include <string>

#include "agents/memory/compressors.hpp"
#include "agents/memory/data_buffer.hpp"
#include "agents/memory/experience.hpp"
#include "agents/memory/frame_buffer.hpp"
#include <experimental/filesystem>

namespace relab::agents::memory {

/**
 * @brief Class implementing a replay buffer.
 *
 * @details
 * For more information about the original papers, please refer to the
 * documentation of MDQN and PrioritizedDQN.
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

  // The buffer storing the data (i.e., actions, rewards, dones and priorities)
  // of all experiences.
  std::unique_ptr<DataBuffer> data;

  // The indices of the last sampled experiences.
  torch::Tensor indices;

 public:
  /**
   * Create a replay buffer.
   * @param capacity the number of experience the buffer can store
   * @param batch_size the size of the batch to sample
   * @param frame_skip the number of times each action is repeated in the
   * environment, if None use the configuration
   * @param stack_size the number of stacked frame in each observation, if None
   * use the configuration
   * @param screen_size: the size of the images used by the agent to learn
   * @param type the type of compression to use
   * @param args the prioritization and multistep arguments composed of:
   *     - initial_priority: the maximum experience priority given to new
   * transitions
   *     - omega: the prioritization exponent
   *     - omega_is: the important sampling exponent
   *     - n_children: the maximum number of children each node of the
   * priority-tree can have
   *     - n_steps: the number of steps for which rewards are accumulated in
   * multistep Q-learning
   *     - gamma: the discount factor
   */
  ReplayBuffer(
      int capacity = 10000, int batch_size = 32, int frame_skip = 1, int stack_size = 4, int screen_size = 84,
      CompressorType type = CompressorType::ZLIB, std::map<std::string, float> args = {}
  );

  /**
   * Add a new experience to the buffer.
   * @param experience the experience to add
   */
  void append(const Experience &experience);

  /**
   * Sample a batch from the replay buffer.
   * @return (observations, actions, rewards, done, next_observations) where:
   *   - observations: the batch of observations
   *   - actions: the actions performed
   *   - rewards: the rewards received
   *   - done: whether the environment stop after performing the actions
   *   - next_observations: the observations received after performing the
   * actions
   */
  Batch sample();

  /**
   * Report the loss associated with all the transitions of the previous batch.
   * @param loss the loss of all previous transitions
   * @return the new loss
   */
  torch::Tensor report(torch::Tensor &loss);

  /**
   * Load a replay buffer from the filesystem.
   * @param checkpoint_path: the full checkpoint path from which the agent has
   * been loaded
   * @param checkpoint_name: the name of the checkpoint from which the replay
   * buffer must be loaded ("" for default name)
   * @param save_all: true if all replay buffer must be saved, false otherwise
   */
  void load(std::string checkpoint_path, std::string checkpoint_name, bool save_all);

  /**
   * Load a replay buffer from the filesystem.
   * @param checkpoint a stream reading from the checkpoint file
   */
  void loadFromFile(std::istream &checkpoint);

  /**
   * Save the replay buffer on the filesystem.
   * @param checkpoint_path: the full checkpoint path in which the agent has
   * been saved
   * @param checkpoint_name: the name of the checkpoint in which the replay
   * buffer must be saved ("" for default name)
   * @param save_all: true if all replay buffer must be saved, false otherwise
   */
  void save(std::string checkpoint_path, std::string checkpoint_name, bool save_all);

  /**
   * Save the replay buffer on the filesystem.
   * @param checkpoint a stream writing into the checkpoint file
   */
  void saveToFile(std::ostream &checkpoint);

  /**
   * Retrieve the path to the file in which the replay buffer must be saved.
   * @param checkpoint_path: the full checkpoint path in which the agent has
   * been saved
   * @param checkpoint_name: the name of the checkpoint in which the replay
   * buffer must be saved ("" for default name)
   * @param save_all: true if all replay buffer must be saved, false otherwise
   * @return the path to the file in which the replay buffer must be saved
   */
  std::experimental::filesystem::path
  getCheckpointPath(std::string &checkpoint_path, std::string &checkpoint_name, bool save_all);

  /**
   * Print the replay buffer on the standard output.
   * @param verbose true if the full replay buffer should be displayed, false
   * otherwise
   */
  void print(bool verbose = false);

  /**
   * Retrieve the experiences whose indices are passed as parameters.
   * @param indices the experience indices
   * @return the experiences
   */
  Batch getExperiences(torch::Tensor &indices);

  /**
   * Retrieve the number of elements in the buffer.
   * @return the number of elements contained in the replay buffer
   */
  int size();

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

  /**
   * Compare two replay buffers.
   * @param lhs the replay buffer on the left-hand-side of the equal sign
   * @param rhs the replay buffer on the right-hand-side of the equal sign
   * @return true if the replay buffers are identical, false otherwise
   */
  friend bool operator==(const ReplayBuffer &lhs, const ReplayBuffer &rhs);
};
}  // namespace relab::agents::memory

#endif  // RELAB_CPP_INC_AGENTS_MEMORY_REPLAY_BUFFER_HPP_
