// Copyright 2025 Theophile Champion. No Rights Reserved.
/**
 * @file frame_buffer.hpp
 * @brief Declaration of the frame buffer class.
 */

#ifndef RELAB_CPP_INC_AGENTS_MEMORY_FRAME_BUFFER_HPP_
#define RELAB_CPP_INC_AGENTS_MEMORY_FRAME_BUFFER_HPP_

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "agents/memory/compressors.hpp"
#include "agents/memory/experience.hpp"
#include "agents/memory/frame_storage.hpp"
#include "helpers/deque.hpp"
#include "helpers/thread_pool.hpp"

namespace relab::agents::memory::impl {

using namespace relab::helpers;

/**
 * @brief A buffer allowing for storage and retrieval of experience
 * observations.
 */
class FrameBuffer {
  private:
    // The device on which computation is performed.
    torch::Device device;

    // Store the frame buffer's parameters.
    int frame_skip;
    int stack_size;
    int capacity;
    int n_steps;
    int screen_size;

    // A frame storage containing all the buffer's frames.
    FrameStorage frames;

    // Lists storing the observation references of each experience.
    std::vector<int> references_t;
    std::vector<int> references_tn;
    int current_ref;

    // Queue storing recent observations references (for multistep Q-learning).
    Deque<int> past_references;

    // Boolean tracking whether the next experience starts a new episode.
    bool new_episode;

    // A compressor to encode and decode the stored frames.
    std::unique_ptr<Compressor> png;

    // A thread pool to parallelize the decompression.
    ThreadPool pool;

  public:
    /**
     * Create a frame buffer.
     * @param capacity the number of experiences the buffer can store
     * @param frame_skip the number of times each action is repeated in the
     * environment
     * @param n_steps the number of steps for which rewards are accumulated in
     * multistep Q-learning
     * @param stack_size the number of stacked frame in each observation
     * @param screen_size: the size of the images used by the agent to learn
     * @param type the type of compression to use
     * @param n_threads the number of threads to use for speeding up the
     * decompression of tensors
     */
    FrameBuffer(
        int capacity, int frame_skip, int n_steps, int stack_size, int screen_size = 84,
        CompressorType type = CompressorType::ZLIB, int n_threads = 1
    );

    /**
     * Add the frames of the next experience to the buffer.
     * @param experience the experience whose frames must be added to the buffer
     */
    void append(const Experience &experience);

    /**
     * Retrieve the observations of the experience whose index is passed as
     * parameters.
     * @param indices the indices of the experiences whose observations must be
     * retrieved
     * @return the observations at time t and t + n_steps
     */
    std::tuple<torch::Tensor, torch::Tensor> operator[](const torch::Tensor &indices);

    /**
     * Retrieve the number of experiences stored in the buffer.
     * @return the number of experiences stored in the buffer
     */
    int size();

    /**
     * Empty the frame buffer.
     */
    void clear();

    /**
     * Add a frame to the buffer.
     * @param frame the frame
     * @return the unique index of the frame
     */
    int addFrame(const torch::Tensor &frame);

    /**
     * Add an observation references to the buffer.
     * @param t the index of the first reference in the queue of past references
     * @param tn the index of the second reference in the queue of past references
     */
    void addReference(int t, int tn);

    /**
     * Retrieve the index of the first reference of the buffer.
     * @return the index
     */
    int firstReference();

    /**
     * Encode a frame to compress it.
     * @param frame the frame to encode
     * @return the encoded frame
     */
    torch::Tensor encode(const torch::Tensor &frame);

    /**
     * Decode a frame to decompress it.
     * @param frame the encoded frame to decode
     * @return the decoded frame
     */
    torch::Tensor decode(const torch::Tensor &frame);

    /**
     * Load the frame buffer from the checkpoint.
     * @param checkpoint a stream reading from the checkpoint file
     */
    void load(std::istream &checkpoint);

    /**
     * Save the frame buffer in the checkpoint.
     * @param checkpoint a stream writing into the checkpoint file
     */
    void save(std::ostream &checkpoint);

    /**
     * Print the frame buffer on the standard output.
     * @param verbose true if the full frame buffer should be displayed, false
     * otherwise
     * @param prefix the prefix to add an front of the optional information
     */
    void print(bool verbose = false, const std::string &prefix = "");

    /**
     * Check if two frame buffers are identical.
     * @param lhs the frame buffer on the left-hand-side of the equal sign
     * @param rhs the frame buffer on the right-hand-side of the equal sign
     * @return true if the frame buffers are identical, false otherwise
     */
    friend bool operator==(const FrameBuffer &lhs, const FrameBuffer &rhs);

    /**
     * Check if two frame buffers are different.
     * @param lhs the frame buffer on the left-hand-side of the different sign
     * @param rhs the frame buffer on the right-hand-side of the different sign
     * @return true if the frame buffers are different, false otherwise
     */
    friend bool operator!=(const FrameBuffer &lhs, const FrameBuffer &rhs);
};
}  // namespace relab::agents::memory::impl

namespace relab::agents::memory {
using impl::FrameBuffer;
}  // namespace relab::agents::memory

#endif  // RELAB_CPP_INC_AGENTS_MEMORY_FRAME_BUFFER_HPP_
