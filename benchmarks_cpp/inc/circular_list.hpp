#ifndef CIRCULAR_LIST_HPP
#define CIRCULAR_LIST_HPP

#include <vector>
#include "frame.hpp"


/**
 * Class storing a circular index similar to polar coordinate.
 */
class CircularIndex {

public:

    int angle;
    int radius;

public:

    /**
     * Create a circular index.
     * @param angle the angle at which the element is stored
     * @param radius the radius at which the element is stored
     */
    CircularIndex(int angle, int radius);
};


/**
 * A list that can contain an infinite number of elements where:
 *  - removal of the first element is O(N / C)
 *  - addition of an element at the end of the list is amortized O(1)
 *  - access and modification of elements are O(1)
 *
 * N = number of elements in the list
 * C = the estimated size of the list (how big we expect the list to be)
 */
template<class T>
class CircularList {

private:

    int expected_size;
    T **elements;
    int first_index;
    int current_index;

public:

    /**
     * Create a circular list.
     * @param expected_size the estimated size of the list (how big we expect the list to be)
     */
    CircularList(int expected_size);

    /**
     * Add an element at the end of the list.
     * @param element the element to add
     */
    void append(T element);

    /**
     * Remove the first element from the list.
     * @return the removed element
     */
    T pop();

    /**
     * Retrieve an element from the list.
     * @param index the index of the element to retrieve
     * @return the element
     */
    T &operator[](int index);

    /**
     * Compute the circular index corresponding to the provided index.
     * @param index the index whose circular index must be computed
     * @param first_index the index to consider as the first element in the circular list
     * @return the circular index
     */
    CircularIndex circularIndex(int index, int first_index=-1);

    /**
     * Retrieve the number of elements currently stored in the list.
     * @return the number of elements currently stored in the list
     */
    int length();

    /**
     * Clear the content of the circular list.
     */
    void clear();

    /**
     * Retrieve the index of the next element to add.
     * @return the index
     */
    int getCurrentIndex();
};

// Explicit instantiation of circular list.
template class CircularList<Frame>;

#endif //CIRCULAR_LIST_HPP
