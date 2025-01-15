#ifndef FRAME_BUFFER_HPP
#define FRAME_BUFFER_HPP

#include <vector>
#include "frame_storage.hpp"
#include "thread_pool.hpp"
#include "compressors.hpp"
#include "experience.hpp"
#include "deque.hpp"


/**
 * A buffer allowing for storage and retrieval of experience observations.
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

    // A queue storing the recent observations references (for multistep Q-learning).
    Deque<int> past_references;

    // A boolean keeping track of whether the next experience is the beginning of a new episode.
    bool new_episode;

    // A compressor to encode and decode the stored frames, and a thread pool to parallelize the decompression.
    ZCompressor png;
    ThreadPool pool;

public:

    /**
     * Create a frame buffer.
     * @param capacity the number of experiences the buffer can store
     * @param frame_skip the number of times each action is repeated in the environment
     * @param n_steps the number of steps for which rewards are accumulated in multistep Q-learning
     * @param stack_size the number of stacked frame in each observation
     * @param screen_size: the size of the images used by the agent to learn
     * @param n_threads the number of threads to use for speeding up the decompression of tensors
     */
    FrameBuffer(int capacity, int frame_skip, int n_steps, int stack_size, int screen_size=84, int n_threads=1);

    /**
     * Add the frames of the next experience to the buffer.
     * @param experience_tuple the experience whose frames must be added to the buffer
     */
    void append(const ExperienceTuple &experience_tuple);

    /**
     * Retrieve the observations of the experience whose index is passed as parameters.
     * @param idx the index of the experience whose observations must be retrieved
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
};

#endif //FRAME_BUFFER_HPP
