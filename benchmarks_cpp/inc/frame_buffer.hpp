#ifndef FRAME_BUFFER_HPP
#define FRAME_BUFFER_HPP

#include <vector>
#include "circular_list.hpp"
#include "experience.hpp"
#include "frame.hpp"
#include "deque.hpp"


/**
 * Class storing references to the first frame of observations at time t and t + n_steps.
 */
class ObsReferences {

public:

    Frame frame_t;
    Frame frame_tn;

public:

    /**
     * Create an empty observation references.
     */
    ObsReferences();

    /**
     * Create a observation references.
     * @param frame_t the first frame of the observation at time t
     * @param frame_tn the first frame of the observation at time t + n_steps
     */
    ObsReferences(Frame frame_t, Frame frame_tn);
};


/**
 * A buffer allowing for storage and retrieval of experience observations.
 */
class FrameBuffer {

private:

    // Store the frame buffer's parameters.
    int frame_skip;
    int stack_size;
    int capacity;
    int n_steps;

    // A circular list storing all the buffer's frames.
    CircularList<Frame> frames;

    // A list storing the observation references of each experience.
    std::vector<ObsReferences> references;
    int current_ref;

    // A queue storing the frames of recent observations (for multistep Q-learning).
    Deque<Frame> past_obs_frames;

    // A boolean keeping track of whether the next experience is the beginning of a new episode.
    bool new_episode;

public:

    /**
     * Create a frame buffer.
     * @param capacity the number of experiences the buffer can store
     * @param frame_skip the number of times each action is repeated in the environment
     * @param n_steps the number of steps for which rewards are accumulated in multistep Q-learning
     * @param stack_size the number of stacked frame in each observation
     */
    FrameBuffer(int capacity, int frame_skip, int n_steps, int stack_size);

    /**
     * Add the frames of the next experience to the buffer.
     * @param experience_tuple the experience whose frames must be added to the buffer
     */
    void append(ExperienceTuple experience_tuple);

    /**
     * Retrieve the observations of the experience whose index is passed as parameters.
     * @param idx the index of the experience whose observations must be retrieved
     * @return the observations at time t and t + n_steps
     */
    std::tuple<torch::Tensor, torch::Tensor> operator[](int idx);

    /**
     * Retrieve the number of experiences stored in the buffer.
     * @return the number of experiences stored in the buffer
     */
    int length();

    /**
     * Empty the frame buffer.
     */
    void clear();

    /**
     * Add a frame to the buffer.
     * @param data the frame's data
     */
    void addFrame(torch::Tensor data);

    /**
     * Add an observation references to the buffer.
     * @param t_0 the index of the first frame in the queue of past observation frames
     * @param t_1 the index of the second frame in the queue of past observation frames
     */
    void addReference(int t_0, int t_1);

    /**
     * Retrieve an observation from the buffer.
     * @param idx the index of the first frame of the observation to retrieve
     * @return the observation
     */
    torch::Tensor getObservation(int idx);

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
    torch::Tensor encode(torch::Tensor frame);

    /**
     * Decode a frame to decompress it.
     * @param frame the encoded frame to decode
     * @return the decoded frame
     */
    torch::Tensor decode(torch::Tensor frame);
};

#endif //FRAME_BUFFER_HPP
