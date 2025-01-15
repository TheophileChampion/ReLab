#include "experience.hpp"

Experience::Experience(torch::Tensor obs, int action, float reward, bool done, torch::Tensor next_obs) {
    this->obs = obs;
    this->action = action;
    this->reward = reward;
    this->done = done;
    this->next_obs = next_obs;
}

Experience::Experience(const ExperienceTuple &experience) {
    this->obs = std::get<0>(experience);
    this->action = std::get<1>(experience);
    this->reward = std::get<2>(experience);
    this->done = std::get<3>(experience);
    this->next_obs = std::get<4>(experience);
}
