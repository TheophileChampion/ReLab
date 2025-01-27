/**
 * @file debug.hpp
 * @brief Helper functions to display variables in human-readable format.
 */

#ifndef DEBUG_HPP
#define DEBUG_HPP

#include <torch/extension.h>
#include <vector>

namespace relab::helpers {

    /**
     * Print a tensor on the standard output.
     * @param tensor the tensor to display
     * @param max_n_elements the maximum number of tensor elements to display, by default all elements are displayed
     */
    template<class T>
    void print_tensor(const torch::Tensor &tensor, int max_n_elements=-1, bool new_line=true);

    /**
     * Print a vector on the standard output.
     * @param vector the vector to display
     * @param max_n_elements the maximum number of vector elements to display, by default all elements are displayed
     */
    template<class T>
    void print_vector(const std::vector<T> &vector, int max_n_elements=-1);

    /**
     * Print a vector of tensors on the standard output.
     * @param vector the vector of tensors
     * @param start the index corresponding to the first element in the vector
     * @param max_n_elements the maximum number of vector elements to display, by default all elements are displayed
     */
    template<class TensorType, class DataType>
    void print_vector(const std::vector<TensorType> &vector, int start=0, int max_n_elements=-1);

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

    #include <ctime>
    #include <fstream>
    #include <iostream>
    #include <sstream>

    // Enum to represent log levels.
    enum LogLevel {
        DEBUG = 0,
        INFO = 1,
        WARNING = 2,
        ERROR = 3,
        CRITICAL = 4
    };

    // Root logger.
    static Logger logging = Logger();

    class Logger {

    private:

        // TODO doxygen documentation
        // Stream where to log messages.
        std::ostream stream;
        LogLevel level;
        std::string logger_name;

        // Converts log level to a string for output.
        std::string levelToString(LogLevel level)
        {
            switch (level) {
            case DEBUG:
                return "DEBUG";
            case INFO:
                return "INFO";
            case WARNING:
                return "WARNING";
            case ERROR:
                return "ERROR";
            case CRITICAL:
                return "CRITICAL";
            default:
                return "UNKNOWN";
            }
        }

    public:

        // Constructor: Opens the log file in append mode
        Logger(const std::ostream &stream=std::cout, LogLevel level=LogLevel.INFO, const std::string &logger_name="root")
            : stream(stream), level(level), logger_name(logger_name)
        {
            // TODO move
        }

        void debug(const std::string &message) {
            this->log(LogLevel.DEBUG, message);
        }

        void info(const std::string &message) {
            this->log(LogLevel.INFO, message);
        }

        void warning(const std::string &message) {
            this->log(LogLevel.WARNING, message);
        }

        void critical(const std::string &message) {
            this->log(LogLevel.CRITICAL, message);
        }

        // Logs a message with a given log level
        void log(LogLevel level, const std::string &message)
        {
            // Create log entry.
            std::ostringstream logEntry;
            logEntry << this->levelToString(level) << ":" << this->logger_name << ":" << message;

            // Output to console.
            this->stream << logEntry.str() << endl;
        }
    };
}

#endif //DEBUG_HPP
