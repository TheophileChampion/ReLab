#include "frame_storage.hpp"
#include "serialize.hpp"
#include "debug.hpp"

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
    if (static_cast<int>(this->frames.size()) != this->capacity) {
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
    return last_frame_index - first_frame_index;
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

void FrameStorage::load(std::istream &checkpoint) {

    // Load the frame buffer from the checkpoint.
    this->initial_capacity = serialize::load_value<int>(checkpoint);
    this->capacity = serialize::load_value<int>(checkpoint);
    this->capacity_incr = serialize::load_value<int>(checkpoint);
    this->frames = std::move(serialize::load_vector<torch::Tensor, float>(checkpoint));
    this->first_frame_index = serialize::load_value<int>(checkpoint);
    this->last_frame_index = serialize::load_value<int>(checkpoint);
    this->first_frame = serialize::load_value<int>(checkpoint);
    this->last_frame = serialize::load_value<int>(checkpoint);
}

void FrameStorage::save(std::ostream &checkpoint) {

    // Save the frame buffer in the checkpoint.
    serialize::save_value(this->initial_capacity, checkpoint);
    serialize::save_value(this->capacity, checkpoint);
    serialize::save_value(this->capacity_incr, checkpoint);
    serialize::save_vector<torch::Tensor, float>(this->frames, checkpoint);
    serialize::save_value(this->first_frame_index, checkpoint);
    serialize::save_value(this->last_frame_index, checkpoint);
    serialize::save_value(this->first_frame, checkpoint);
    serialize::save_value(this->last_frame, checkpoint);
}

void FrameStorage::print(bool verbose, const std::string &prefix) {

    // Display the most important information about the frame storage.
    std::cout << "FrameStorage[initial_capacity: " << this->initial_capacity << ", capacity: " << this->capacity
              << ", capacity_incr: " << this->capacity_incr << ", first_frame_index: " << this->first_frame_index
              << ", last_frame_index: " << this->last_frame_index << ", first_frame: " << this->first_frame
              << ", last_frame: " << this->last_frame << "]" << std::endl;

    // Display optional information about the frame storage.
    if (verbose == true) {
        std::cout << prefix << " #-> frames = ";
        debug::print_vector<torch::Tensor, float>(this->frames, this->first_frame, 2);
    }
}