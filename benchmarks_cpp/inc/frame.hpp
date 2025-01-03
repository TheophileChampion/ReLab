#ifndef FRAME_HPP
#define FRAME_HPP

#include <torch/extension.h>


/**
 * Class storing a frame.
 */
class Frame {

public:

    int index;
    torch::Tensor data;

public:

    /**
     * Create an empty frame.
     */
    Frame();

    /**
     * Create a frame.
     * @param index the frame index
     * @param data the frame's data
     */
    Frame(int index, torch::Tensor data);

    /**
     * Copy of the attributes of the frame passed as parameters.
     * @param the frame whose attribute must be copied
     * @return a reference to the new frame
     */
    Frame &operator=(const Frame &other);
};

#endif //FRAME_HPP
