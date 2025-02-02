// Copyright 2025 Theophile Champion. No Rights Reserved.
/**
 * @file timer.hpp
 * @brief Declaration of a timer class for timing code execution.
 */

#ifndef RELAB_CPP_INC_HELPERS_TIMER_HPP_
#define RELAB_CPP_INC_HELPERS_TIMER_HPP_

#include <chrono>
#include <string>

namespace relab::helpers {

/**
 * @brief A timer class allowing to time the execution of a piece of code.
 */
class Timer {
private:
  std::string name;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;

public:
  /**
   * Create a timer.
   * @param name the code whose runtime must be measured
   */
  Timer(std::string name = "");

  /**
   * Start the timer.
   */
  void start();

  /**
   * Stop the timer and display the runtime.
   */
  void stop();
};
} // namespace relab::helpers

#endif // RELAB_CPP_INC_HELPERS_TIMER_HPP_
