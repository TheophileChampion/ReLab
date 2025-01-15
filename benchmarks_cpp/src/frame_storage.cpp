#include "frame_storage.hpp"

FrameStorage::FrameStorage(int capacity, int capacity_incr)
    : initial_capacity(capacity), capacity(capacity), capacity_incr(capacity_incr),
      first_frame_index(0), last_frame_index(-1), first_frame(0), last_frame(-1) {

    // Allocate enough memory to store a number of frames equal to the storage capacity.
    this->frames.reserve(capacity);
}

int FrameStorage::append(const torch::Tensor &frame) {

    // Update the last frame indices.
    this->last_frame_index += 1;
    this->last_frame += 1;
    if (this->last_frame >= this->capacity) {
        this->last_frame %= this->capacity;
    }

    // Add the frame to the vector of frames.
    if (this->size() != this->capacity) {
        this->frames.push_back(frame);
    } else {
        // Resize the vector of frames if it is full.
        if (this->last_frame == this->first_frame) {
            this->resize_frames();
        }
        this->frames[this->last_frame] = frame;
    }
    return this->last_frame_index;
}

void FrameStorage::resize_frames() {

    // Resize the vector of frames.
    int capacity = this->capacity;
    this->capacity += this->capacity_incr;
    this->frames.resize(this->capacity);

    // Create space between the first and last frames.
    int n = capacity - this->first_frame;
    for (auto i = 0; i < n; i++) {
        this->frames[this->capacity - 1 - i] = this->frames[capacity - 1 - i];
    }

    // Update the first and last frame to reflect the new state of the vector of frames.
    this->first_frame = this->capacity - n;
    this->last_frame = this->first_frame - this->capacity_incr;
}

int FrameStorage::size() {
    return static_cast<int>(this->frames.size());
}

void FrameStorage::pop() {

    // Update the first frame indices.
    this->first_frame_index += 1;
    this->first_frame += 1;
    if (this->first_frame >= this->capacity) {
        this->first_frame %= this->capacity;
    }
}

int FrameStorage::top_index() {
    return this->first_frame_index;
}

void FrameStorage::clear() {

    // Reset the class attributes.
    this->capacity = this->initial_capacity;
    this->first_frame_index = 0;
    this->last_frame_index = -1;
    this->first_frame = 0;
    this->last_frame = -1;

    // Clear the vector of frames.
    this->frames.clear();
}

torch::Tensor FrameStorage::operator[](int index) {
    index -= this->first_frame_index;
    index = (index + this->first_frame) % this->capacity;
    return this->frames[index];
}
