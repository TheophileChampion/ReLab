#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <string>

/**
 * A timer class allowing to time the execution of a piece of code.
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
    Timer(std::string name="");

    /**
     * Start the timer.
     */
    void start();

    /**
     * Stop the timer and display the runtime.
     */
    void stop();
};

#endif //TIMER_HPP
