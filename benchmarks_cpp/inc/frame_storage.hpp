#ifndef FRAME_STORAGE_HPP
#define FRAME_STORAGE_HPP

#include <torch/extension.h>


/**
 * Class storing a list of frames inside a tensor.
 */
class FrameStorage {

public:

    // The initial and current number of frames that can be stored in the vector of frames.
    int initial_capacity;
    int capacity;

    // The number by which the capacity is increased when no space is left in the vector of frames.
    int capacity_incr;

    // The vector of frames.
    std::vector<torch::Tensor> frames;

    // The unique indices of the first and last frames stored in the vector of frames.
    // Note: unique in this context means that each frame has a different index.
    // Note: these indices may be larger than the tensor's capacity.
    int first_frame_index;
    int last_frame_index;

    // The indices at which the first and last frames are stored in the vector of frames.
    // Note: these indices are always smaller than the tensor's capacity.
    int first_frame;
    int last_frame;

public:

    /**
     * Create a frame storage.
     * @param capacity the initial number of frames the storage can contain
     * @param capacity_incr the number by which the capacity is increased when no space is left in the tensor
     */
    FrameStorage(int capacity, int capacity_incr=100000);

    /**
     * Add an frame to the storage.
     * @param frame the frame to add
     * @return the unique index of the frame that was added to the buffer
     */
    int append(torch::Tensor frame);

    /**
     * Resize the vector of frames, i.e., increasing its size by this->capacity_incr.
     */
    void resize_frames();

    /**
     * Retrieve the size of the vector of frames.
     */
    int size();

    /**
     * Remove the first frame from the storage.
     */
    void pop();

    /**
     * Retrieve the the unique index of first frame in the storage.
     * @return the unique index of the first frame in the storage
     */
    int top_index();

    /**
     * Clear the content of the frame storage.
     */
    void clear();

    /**
     * Retrieve a frame from the storage.
     * @param index the unique index of the frame to retrieve
     * @return the frame
     */
    torch::Tensor operator[](int index);
};

#endif //FRAME_STORAGE_HPP
