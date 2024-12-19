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
};

#endif //FRAME_HPP
