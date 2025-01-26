/**
 * @file frame_storage.hpp
 * @brief Declaration of the frame storage class.
 */

#ifndef FRAME_STORAGE_HPP
#define FRAME_STORAGE_HPP

#include <torch/extension.h>

namespace relab::agents::memory {

    /**
     * @brief Class storing a list of frames inside a vector of tensors.
     */
    class FrameStorage {

    public:

        /// @var initial_capacity
        /// The initial number of frames that can be stored in the vector.
        int initial_capacity;

        /// @var capacity
        /// The current number of frames that can be stored in the vector.
        int capacity;

        /// @var capacity_incr
        /// The increment size when expanding the storage's capacity.
        int capacity_incr;

        /// @var frames
        /// The vector storing all frames.
        std::vector<torch::Tensor> frames;

        /// @var first_frame_index
        /// The unique index of the first frame in storage (may exceed capacity).
        int first_frame_index;

        /// @var last_frame_index
        /// The unique index of the last frame in storage (may exceed capacity).
        int last_frame_index;

        /// @var first_frame
        /// The storage index of the first frame (always less than capacity).
        int first_frame;

        /// @var last_frame
        /// The storage index of the last frame (always less than capacity).
        int last_frame;

    public:

        /**
         * Create a frame storage.
         * @param capacity the initial number of frames the storage can contain
         * @param capacity_incr the number by which the capacity is increased when no space is left in the tensor
         */
        FrameStorage(int capacity, int capacity_incr=100000);

        /**
         * Add a frame to the storage.
         * @param frame the frame to add
         * @return the unique index of the frame that was added to the buffer
         */
        int append(const torch::Tensor &frame);

        /**
         * Resize the vector of frames, i.e., increasing its size by this->capacity_incr.
         */
        void resize_frames();

        /**
         * Retrieve the number of frames stored in the frame storage.
         * @return the number of frames stored in the frame storage
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

        /**
         * Load the frame storage from the checkpoint.
         * @param checkpoint a stream reading from the checkpoint file
         */
        void load(std::istream &checkpoint);

        /**
         * Save the frame storage in the checkpoint.
         * @param checkpoint a stream writing into the checkpoint file
         */
        void save(std::ostream &checkpoint);

        /**
         * Print the frame storage on the standard output.
         * @param verbose true if the full frame storage should be displayed, false otherwise
         * @param prefix the prefix to add an front of the optional information
         */
        void print(bool verbose=false, const std::string &prefix="");

        /**
         * Compare two frame storages.
         * @param lhs the frame storage on the left-hand-side of the equal sign
         * @param rhs the frame storage on the right-hand-side of the equal sign
         * @return true if the frame storages are identical, false otherwise
         */
        friend bool operator==(const FrameStorage &lhs, const FrameStorage &rhs);
    };
}

#endif //FRAME_STORAGE_HPP
