#include "frame_buffer.hpp"
#include "replay_buffer.hpp"
#include "timer.hpp"

using namespace torch::indexing;

FrameBuffer::FrameBuffer(
    int capacity, int frame_skip, int n_steps, int stack_size, int screen_size,
    CompressorType type, int n_threads
) : device(ReplayBuffer::getDevice()), frame_skip(frame_skip), stack_size(stack_size),
    capacity(capacity), n_steps(n_steps), screen_size(screen_size),
    frames(FrameStorage(capacity)), past_references(n_steps + 1), pool(n_threads) {

    // A list storing the observation references of each experience.
    std::vector<int> references_t(capacity);
    this->references_t = std::move(references_t);
    std::vector<int> references_tn(capacity);
    this->references_tn = std::move(references_tn);
    this->current_ref = 0;

    // Create the compressor used to compress and decompress the tensors.
    this->png = Compressor::create(screen_size, screen_size, type);

    // A boolean keeping track of whether the next experience is the beginning of a new episode.
    this->new_episode = true;
}

void FrameBuffer::append(const ExperienceTuple &experience_tuple) {

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
        for (auto i = 0; i < this->stack_size; i++) {
            int reference = this->addFrame(this->encode(experience.obs.index({i, Slice(), Slice()}).detach().clone()));
            if (i == 0) {
                this->past_references.push_back(reference);
            }
        }
    }

    // Add the frames of the observation at time t + 1.
    int n = std::min(this->frame_skip, this->stack_size);
    for (auto i = n; i >= 1; i--) {
        int reference = this->addFrame(this->encode(experience.next_obs.index({-i, Slice(), Slice()}).detach().clone()));
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

std::tuple<torch::Tensor, torch::Tensor> FrameBuffer::operator[](const torch::Tensor &indices) {

    int n_elements = indices.numel();
    torch::Tensor obs_batch = torch::zeros({n_elements, this->stack_size, this->screen_size, this->screen_size});
    torch::Tensor next_obs_batch = torch::zeros({n_elements, this->stack_size, this->screen_size, this->screen_size});

    // Retrieve the all the decoded observations.
    int frame_size = this->screen_size * this->screen_size;
    float *obs_batch_ptr = obs_batch.data_ptr<float>();
    float *next_obs_batch_ptr = next_obs_batch.data_ptr<float>();
    long *indices_ptr = indices.data_ptr<long>();
    for (auto i = 0; i < n_elements; i++) {

        // Retrieve the index of first frame for the requested observations.
        int idx = (*indices_ptr + this->firstReference()) % this->capacity;
        int reference_t = this->references_t[idx];
        int reference_tn = this->references_tn[idx];

        // Parallelize the decompression of the observations.
        this->pool.push([this, obs_batch_ptr, reference_t, frame_size] {
            float *ptr = obs_batch_ptr;
            for (auto j = 0; j < this->stack_size; j++) {
                this->png->decode(this->frames[reference_t + j], ptr);
                ptr += frame_size;
            }
        });
        this->pool.push([this, next_obs_batch_ptr, reference_tn, frame_size] {
            float *ptr = next_obs_batch_ptr;
            for (auto j = 0; j < this->stack_size; j++) {
                this->png->decode(this->frames[reference_tn + j], ptr);
                ptr += frame_size;
            }
        });
        obs_batch_ptr += frame_size * this->stack_size;
        next_obs_batch_ptr += frame_size * this->stack_size;

        // Move to the next experience index in the batch.
        ++indices_ptr;
    }
    this->pool.synchronize();

    // Returns the batch's observations.
    return std::make_tuple(obs_batch, next_obs_batch);
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

int FrameBuffer::addFrame(const torch::Tensor &frame) {
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

int FrameBuffer::firstReference() {
    return (this->current_ref < this->capacity) ? 0 : this->current_ref;
}

torch::Tensor FrameBuffer::encode(const torch::Tensor &frame) {
    return this->png->encode(frame);
}

torch::Tensor FrameBuffer::decode(const torch::Tensor &frame) {
    return this->png->decode(frame);
}
