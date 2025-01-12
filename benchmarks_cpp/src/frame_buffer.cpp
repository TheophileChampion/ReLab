#include "replay_buffer.hpp"
#include "frame_buffer.hpp"

using namespace torch::indexing;

FrameBuffer::FrameBuffer(int capacity, int frame_skip, int n_steps, int stack_size)
    : device(ReplayBuffer::getDevice()), frames(FrameStorage(capacity)), past_references(n_steps + 1) {

    // Store the frame buffer's parameters.
    this->frame_skip = frame_skip;
    this->stack_size = stack_size;
    this->capacity = capacity;
    this->n_steps = n_steps;

    // A list storing the observation references of each experience.
    std::vector<int> references_t(capacity);
    this->references_t = std::move(references_t);
    std::vector<int> references_tn(capacity);
    this->references_tn = std::move(references_tn);
    this->current_ref = 0;

    // Create the PNG compressor.
    this->png = Compressor::create();

    // A boolean keeping track of whether the next experience is the beginning of a new episode.
    this->new_episode = true;
}

void FrameBuffer::append(ExperienceTuple experience_tuple) {

    Experience experience = Experience(experience_tuple);

    // If the buffer is full, remove the oldest observation frames from the buffer.
    if (this->size() == this->capacity) {
        int first_frame_index = this->references_t[this->firstReference() % this->capacity];
        while (this->frames.top_index() <= first_frame_index) {
            this->frames.pop();
        }
    }

    // Add the frames of the observation at time t, if needed.
    if (this->new_episode == true) {
        auto obs = experience.obs.to(this->device);
        for (auto i = 0; i < this->stack_size; i++) {
            int reference = this->addFrame(this->encode(obs.index({i, Slice(), Slice()}).detach().clone()));
            if (i == 0) {
                this->past_references.push_back(reference);
            }
        }
    }

    // Add the frames of the observation at time t + 1.
    int n = std::min(this->frame_skip, this->stack_size);
    auto next_obs = experience.next_obs.to(this->device);
    for (auto i = n; i >= 1; i--) {
        int reference = this->addFrame(this->encode(next_obs.index({-i, Slice(), Slice()}).detach().clone()));
        if (i == 1) {
            this->past_references.push_back(reference + 1 - this->stack_size);
        }
    }

    // Update the observation references.
    if (experience.done == true) {

        // If the current episode has ended, keep track of all valid references.
        while (this->past_references.size() != 1) {
            this->addReference(0, -1);
            this->past_references.pop_front();
        }

        // Then, clear the queue of past observation frames.
        this->past_references.clear();

    } else if (static_cast<int>(this->past_references.size()) == this->n_steps + 1) {

        // If the current episode has not ended, but the queue of past observation frame is full,
        // then keep track of next valid reference (before it is discarded in the next call to append).
        this->addReference(0, this->n_steps);
    }

    // Keep track of whether the next experience is the beginning of a new episode.
    this->new_episode = experience.done;
}

std::tuple<torch::Tensor, torch::Tensor> FrameBuffer::operator[](int idx) {

    // Retrieve the index of first frame for the requested observations.
    idx = (this->firstReference() + idx) % this->capacity;
    int reference_t = this->references_t[idx];
    int reference_tn = this->references_tn[idx];

    // Retrieve the requested observations.
    return std::make_tuple(this->getObservation(reference_t), this->getObservation(reference_tn));
}

int FrameBuffer::size() {
    return std::min(this->current_ref, this->capacity);
}

void FrameBuffer::clear() {
    std::vector<int> references_t(this->capacity);
    this->references_t = std::move(references_t);
    std::vector<int> references_tn(this->capacity);
    this->references_tn = std::move(references_tn);
    this->frames.clear();
    this->past_references.clear();
    this->new_episode = true;
    this->current_ref = 0;
}

int FrameBuffer::addFrame(torch::Tensor frame) {
    return this->frames.append(frame);
}

void FrameBuffer::addReference(int t, int tn) {

    // Handle negative indices as indices from the end of the queue.
    if (tn < 0) {
        tn += this->past_references.size();
    }

    // Add the reference and increase the reference index.
    this->references_t[this->current_ref % this->capacity] = this->past_references[t];
    this->references_tn[this->current_ref % this->capacity] = this->past_references[tn];
    this->current_ref += 1;
}

torch::Tensor FrameBuffer::getObservation(int idx) {
    std::vector<torch::Tensor> frames;
    for (auto i = 0; i < this->stack_size; i++) {
        frames.push_back(this->decode(this->frames[idx + i]));
    }
    return torch::cat(frames);
}

int FrameBuffer::firstReference() {
    return (this->current_ref < this->capacity) ? 0 : this->current_ref;
}

torch::Tensor FrameBuffer::encode(torch::Tensor frame){
    return this->png->encode(frame.unsqueeze(0));
}

torch::Tensor FrameBuffer::decode(torch::Tensor frame) {
    return this->png->decode(frame);
}
