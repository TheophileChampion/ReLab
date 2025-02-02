// Copyright 2025 Theophile Champion. No Rights Reserved.

#include <iostream>
#include <string>
#include "helpers/timer.hpp"

using namespace std::chrono;

namespace relab::helpers {

Timer::Timer(std::string name) {
    this->name = name;
}

void Timer::start() {
    this->start_time = high_resolution_clock::now();
}

void Timer::stop() {
    if (this->name != "") {
        std::cout << "[" << this->name << "]" << std::endl;
    }
    auto end_time = high_resolution_clock::now();
    std::cout << "Elapsed: " << duration_cast<microseconds>(end_time - this->start_time).count() / 1000.0 << std::endl;
}
}  // namespace relab::helpers
