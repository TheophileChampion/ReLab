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

        /**
         * Compare two deques.
         * @param lhs the deque on the left-hand-side of the equal sign
         * @param rhs the deque on the right-hand-side of the equal sign
         * @return true if the deques are identical, false otherwise
         */
        template<class Type>
        friend bool operator==(const Deque<Type> &lhs, const Deque<Type> &rhs);
    };

    // Explicit instantiation of double ended queue.
    template class Deque<int>;
    template class Deque<float>;
    template class Deque<bool>;
}

#endif //DEQUE_HPP
