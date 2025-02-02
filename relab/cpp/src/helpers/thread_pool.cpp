// Copyright 2025 Theophile Champion. No Rights Reserved.

#include "helpers/thread_pool.hpp"

#include <utility>

using namespace std;

namespace relab::helpers {

ThreadPool::ThreadPool(size_t num_threads) {

  // Creating worker threads.
  for (size_t i = 0; i < num_threads; ++i) {
    this->threads.emplace_back([this] {
      function<void()> task;
      while (true) {
        {
          // Locking the queue so that data can be shared safely.
          unique_lock<mutex> lock(this->queue_mutex);

          // Waiting until there is a task to execute or the pool is stopped.
          this->cv.wait(lock, [this] { return !this->tasks.empty() || this->stop; });

          // Exit the thread in case the pool is stopped and there are no tasks.
          if (this->stop && this->tasks.empty()) {
            return;
          }

          // Get the next task from the queue.
          task = move(this->tasks.front());
          this->tasks.pop();
        }

        // Execute the task.
        task();

        // Keep track of the number of tasks executed.
        {
          unique_lock<mutex> lock(this->counter_mutex);
          ++this->tasks_finished;
        }
      }
    });
  }
}

ThreadPool::~ThreadPool() {

  // Lock the queue to update the stop flag safely.
  {
    unique_lock<mutex> lock(this->queue_mutex);
    this->stop = true;
  }

  // Notify all threads.
  this->cv.notify_all();

  // Joining all worker threads to ensure they have completed their tasks.
  for (auto &thread : this->threads) {
    thread.join();
  }
}

void ThreadPool::push(const function<void()> &task) {
  ++this->tasks_pushed;
  {
    std::unique_lock<std::mutex> lock(this->queue_mutex);
    this->tasks.emplace(move(task));
  }
  this->cv.notify_one();
}

void ThreadPool::synchronize() {
  std::unique_lock<std::mutex> lock(this->counter_mutex);
  bool stop = (this->tasks_pushed == this->tasks_finished);
  lock.unlock();
  while (!stop) {
    lock.lock();
    stop = (this->tasks_pushed == this->tasks_finished);
    lock.unlock();
  }
}
}  // namespace relab::helpers
