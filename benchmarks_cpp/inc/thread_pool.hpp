#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

using namespace std;

/**
 * Class implementing a thread pool.
 */
class ThreadPool {

private:

    // Vector to store worker threads.
    vector<thread> threads;

    // Queue of tasks.
    queue<function<void()>> tasks;

    // Mutexes to synchronize access to shared data.
    mutex queue_mutex;
    mutex counter_mutex;

    // Condition variable to signal changes in the state of the tasks queue.
    condition_variable cv;

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
    ThreadPool(size_t num_threads);

    /**
     * Destroy the thread pool.
     */
    ~ThreadPool();

    /**
     * Push a task for execution by the thread pool.
     * @param task the task to execute
     */
    void push(const function<void()> &task);

    /**
     * Wait for all tasks to complete.
     */
    void synchronize();
};

#endif //THREAD_POOL_HPP
