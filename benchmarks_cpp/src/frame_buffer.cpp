#include "frame_buffer.hpp"

using namespace torch::indexing;

ObsReferences::ObsReferences() {}

ObsReferences::ObsReferences(Frame frame_t, Frame frame_tn) : frame_t(frame_t), frame_tn(frame_tn) {}

FrameBuffer::FrameBuffer(int capacity, int frame_skip, int n_steps, int stack_size)
    : frames(CircularList<Frame>(capacity)), past_obs_frames(n_steps + 1) {

    // Store the frame buffer's parameters.
    this->frame_skip = frame_skip;
    this->stack_size = stack_size;
    this->capacity = capacity;
    this->n_steps = n_steps;

    // A list storing the observation references of each experience.
    std::vector<ObsReferences> references(capacity);
    this->references = std::move(references);
    this->current_ref = 0;

    // A boolean keeping track of whether the next experience is the beginning of a new episode.
    this->new_episode = true;
}

void FrameBuffer::append(ExperienceTuple experience_tuple) {

    Experience experience = Experience(experience_tuple);

    // If the buffer is full, remove the oldest observation frames from the buffer.
    if (this->length() == this->capacity) {
        Frame first_frame = this->references[this->firstReference() % this->capacity].frame_t;
        while (this->frames[0].index != first_frame.index) {
            this->frames.pop();
        }
    }

    // Add the frames of the observation at time t, if needed.
    if (this->new_episode == true) {
        for (auto i = 0; i < this->stack_size; i++) {
            this->addFrame(this->encode(experience.obs.index({i, Slice(), Slice()}).detach().clone()));
        }
        this->past_obs_frames.push_back(this->frames[-this->stack_size]);
    }

    // Add the frames of the observation at time t + 1.
    int n = std::min(this->frame_skip, this->stack_size);
    for (auto i = n; i >= 1; i--) {
        this->addFrame(this->encode(experience.next_obs.index({-i, Slice(), Slice()}).detach().clone()));
    }
    this->past_obs_frames.push_back(this->frames[-this->stack_size]);

    // Update the observation references.
    if (experience.done == true) {

        // If the current episode has ended, keep track of all valid references.
        while (this->past_obs_frames.size() != 1) {
            this->addReference(0, -1);
            this->past_obs_frames.pop_front();
        }

        // Then, clear the queue of past observation frames.
        this->past_obs_frames.clear();

    } else if (static_cast<int>(this->past_obs_frames.size()) == this->n_steps + 1) {

        // If the current episode has not ended, but the queue of past observation frame is full,
        // then keep track of next valid reference (before it is discarded in the next call to append).
        this->addReference(0, this->n_steps);
    }

    // Keep track of whether the next experience is the beginning of a new episode.
    this->new_episode = experience.done;
}

std::tuple<torch::Tensor, torch::Tensor> FrameBuffer::operator[](int idx) {

    // Retrieve the index of first frame for the requested observations.
    ObsReferences reference = this->references[(this->firstReference() + idx) % this->capacity];
    int idx_t = reference.frame_t.index - this->frames[0].index;
    int idx_tn = reference.frame_tn.index - this->frames[0].index;

    // Retrieve the requested observations.
    return std::make_tuple(this->getObservation(idx_t), this->getObservation(idx_tn));
}

int FrameBuffer::length() {
    return std::min(this->current_ref, this->capacity);
}

void FrameBuffer::clear() {
    std::vector<ObsReferences> references(this->capacity);
    this->references = std::move(references);
    this->frames.clear();
    this->past_obs_frames.clear();
    this->new_episode = true;
    this->current_ref = 0;
}

void FrameBuffer::addFrame(torch::Tensor data) {
    this->frames.append(Frame(this->frames.getCurrentIndex(), data));
}

void FrameBuffer::addReference(int t_0, int t_1) {
    // Handle negative indices as indices from the end of the queue.
    if (t_1 < 0) {
        t_1 += this->past_obs_frames.size();
    }

    // Add the reference and increase the reference index.
    this->references[this->current_ref % this->capacity] =
        ObsReferences(this->past_obs_frames[t_0], this->past_obs_frames[t_1]);
    this->current_ref += 1;
}

torch::Tensor FrameBuffer::getObservation(int idx) {
    std::vector<torch::Tensor> frames;
    for (auto i = 0; i < this->stack_size; i++) {
        frames.push_back(this->decode(this->frames[idx + i].data));
    }
    return torch::cat(frames);
}

int FrameBuffer::firstReference() {
    return (this->current_ref < this->capacity) ? 0 : this->current_ref;
}

torch::Tensor FrameBuffer::encode(torch::Tensor frame){
    return frame.unsqueeze(0);
    // TODO return encode_png((frame * 255).type(torch.uint8).unsqueeze(dim=0));
}

torch::Tensor FrameBuffer::decode(torch::Tensor frame) {
    return frame;
    // TODO return decode_png(frame).type(torch.float32) / 255;
}
