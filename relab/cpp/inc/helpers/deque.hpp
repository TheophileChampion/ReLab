// Copyright 2025 Theophile Champion. No Rights Reserved.
/**
 * @file deque.hpp
 * @brief Declaration of double-ended queue with a maximum length.
 */

#ifndef RELAB_CPP_INC_HELPERS_DEQUE_HPP_
#define RELAB_CPP_INC_HELPERS_DEQUE_HPP_

#include <cstdint>
#include <deque>
#include <fstream>

namespace relab::helpers {

/**
 * @brief A double-ended queue with a maximum length.
 */
template <class T> class Deque : public std::deque<T> {
private:
  int max_size;

public:
  /**
   * Create a double ended queue.
   * @param max_size the maximum length of the queue
   */
  explicit Deque(int max_size = -1);

  /**
   * Add an element at the end of the queue.
   * @param element the element to add
   */
  void push_back(T element);

  /**
   * Add an element at the front of the queue.
   * @param element the element to add
   */
  void push_front(T element);

  /**
   * Retrieve the element whose index is passed as parameters.
   * @param index the index
   * @return the element at the given index
   */
  T get(int index);

  /**
   * Load the double ended queue from the checkpoint.
   * @param checkpoint a stream reading from the checkpoint file
   */
  void load(std::istream &checkpoint);

  /**
   * Save the double ended queue in the checkpoint.
   * @param checkpoint a stream writing into the checkpoint file
   */
  void save(std::ostream &checkpoint);

  /**
   * Print the double ended queue on the standard output.
   */
  void print();

  /**
   * Compare two double ended queues.
   * @param lhs the double ended queue on the left-hand-side of the equal sign
   * @param rhs the double ended queue on the right-hand-side of the equal sign
   * @return true if the double ended queues are identical, false otherwise
   */
  template <class Type>
  friend bool operator==(const Deque<Type> &lhs, const Deque<Type> &rhs);
};

// Explicit instantiation of double ended queue.
template class Deque<int>;
template class Deque<float>;
template class Deque<bool>;
}  // namespace relab::helpers

#endif  // RELAB_CPP_INC_HELPERS_DEQUE_HPP_
