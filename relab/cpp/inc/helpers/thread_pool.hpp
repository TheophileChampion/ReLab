// Copyright 2025 Theophile Champion. No Rights Reserved.
/**
 * @file thread_pool.hpp
 * @brief Declaration of a thread pool class for parallelizing computation.
 */

#ifndef RELAB_CPP_INC_HELPERS_THREAD_POOL_HPP_
#define RELAB_CPP_INC_HELPERS_THREAD_POOL_HPP_

#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace relab::helpers {

/**
 * @brief Class implementing a thread pool.
 */
class ThreadPool {
  private:
    // Vector to store worker threads.
    std::vector<std::thread> threads;

    // Queue of tasks.
    std::queue<std::function<void()>> tasks;

    // Mutexes to synchronize access to shared data.
    std::mutex queue_mutex;
    std::mutex counter_mutex;

    // Condition variable to signal changes in the state of the tasks queue.
    std::condition_variable cv;

    // Flag to indicate whether the thread pool should stop or not.
    bool stop = false;

    // Counters keeping track of the number of tasks submitted and executed.
    int tasks_pushed = 0;
    int tasks_finished = 0;

  public:
    /**
     * Creates a thread pool.
     * @param num_threads the number of thread threads in the pool
     */
    explicit ThreadPool(size_t num_threads);

    /**
     * Destroy the thread pool.
     */
    ~ThreadPool();

    /**
     * Push a task for execution by the thread pool.
     * @param task the task to execute
     */
    void push(const std::function<void()> &task);

    /**
     * Wait for all tasks to complete.
     */
    void synchronize();
};
}  // namespace relab::helpers

#endif  // RELAB_CPP_INC_HELPERS_THREAD_POOL_HPP_
