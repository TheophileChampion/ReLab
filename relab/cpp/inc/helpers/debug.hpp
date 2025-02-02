// Copyright 2025 Theophile Champion. No Rights Reserved.
/**
 * @file debug.hpp
 * @brief Helper functions to display variables in human-readable format.
 */

#ifndef RELAB_CPP_INC_HELPERS_DEBUG_HPP_
#define RELAB_CPP_INC_HELPERS_DEBUG_HPP_

#include <torch/extension.h>

#include <iostream>
#include <string>
#include <vector>

namespace relab::helpers {

/**
 * Print a tensor on the standard output.
 * @param tensor the tensor to display
 * @param max_n_elements the maximum number of tensor elements to display, by
 * default all elements are displayed
 */
template <class T> void print_tensor(const torch::Tensor &tensor, int max_n_elements = -1, bool new_line = true);

/**
 * Print a vector on the standard output.
 * @param vector the vector to display
 * @param max_n_elements the maximum number of vector elements to display, by
 * default all elements are displayed
 */
template <class T> void print_vector(const std::vector<T> &vector, int max_n_elements = -1);

/**
 * Print a vector of tensors on the standard output.
 * @param vector the vector of tensors
 * @param start the index corresponding to the first element in the vector
 * @param max_n_elements the maximum number of vector elements to display, by
 * default all elements are displayed
 */
template <class TensorType, class DataType>
void print_vector(const std::vector<TensorType> &vector, int start = 0, int max_n_elements = -1);

/**
 * Print a boolean on the standard output.
 * @param value the boolean to display
 */
void print_bool(bool value);

/**
 * Print an ellipse on the standard output.
 * @param max_n_elements the maximum number of vector elements to display
 * @param size the size of the container
 */
void print_ellipse(int max_n_elements, int size);

/**
 * Enumeration representing log levels..
 */
enum LogLevel { DEBUG = 0, INFO = 1, WARNING = 2, ERROR = 3, CRITICAL = 4, NO_LOGGING = 5 };

/**
 * @brief Class allowing the user to log messages of various levels.
 */
class Logger {
 private:
  // The current logging level and the logger's name.
  LogLevel level;
  std::string logger_name;

  /**
   * Convert a logging level into its corresponding string.
   * @params level the level whose string must be returned
   * @return the string corresponding to the level passed as parameter
   */
  std::string levelToString(LogLevel level);

 public:
  /**
   * Create a logger.
   * @params level the minimum level required for a message to be displayed
   * @params logger_name the logger's name
   */
  explicit Logger(LogLevel level = INFO, const std::string &logger_name = "root");

  /**
   * Log a debugging message.
   * @params message the message to be displayed
   */
  void debug(const std::string &message);

  /**
   * Log an information.
   * @params message the message to be displayed
   */
  void info(const std::string &message);

  /**
   * Log a warning for the user.
   * @params message the message to be displayed
   */
  void warning(const std::string &message);

  /**
   * Log a critical error.
   * @params message the message to be displayed
   */
  void critical(const std::string &message);

  /**
   * Log a message with a specified level.
   * @params message the message to be displayed
   */
  void log(LogLevel level, const std::string &message);
};

// Root logger.
static Logger logging = Logger();
}  // namespace relab::helpers

#endif  // RELAB_CPP_INC_HELPERS_DEBUG_HPP_
