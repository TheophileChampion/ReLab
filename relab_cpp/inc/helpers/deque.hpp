#ifndef DEQUE_HPP
#define DEQUE_HPP

#include <deque>
#include <cstdint>
#include <fstream>

namespace relab::helpers {

    /**
     * A deque with a maximum length.
     */
    template<class T>
    class Deque : public std::deque<T> {

    private:

        int max_size;

    public:

        /**
         * Create a deque.
         * @param max_size the maximum length of the queue
         */
        Deque(int max_size=-1);

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
         * @return the element
         */
        T get(int index);

        /**
         * Remove all the elements of the deque.
         */
        void clear();

        /**
         * Remove an elements from the end of the deque.
         */
        void pop_back();

        /**
         * Remove an elements from the front of the deque.
         */
        void pop_front();

        /**
         * Return the size of the deque.
         * @return the size of the deque
         */
        std::uint64_t size();

        /*
         * Load the deque from the checkpoint.
         * @param checkpoint a stream reading from the checkpoint file
         */
        void load(std::istream &checkpoint);

        /*
         * Save the deque in the checkpoint.
         * @param checkpoint a stream writing into the checkpoint file
         */
        void save(std::ostream &checkpoint);

        /**
         * Print the deque on the standard output.
         */
        void print();
    };

    // Explicit instantiation of double ended queue.
    template class Deque<int>;
    template class Deque<float>;
    template class Deque<bool>;
}

#endif //DEQUE_HPP
