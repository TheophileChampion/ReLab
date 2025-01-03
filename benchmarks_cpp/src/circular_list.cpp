#include "circular_list.hpp"

CircularIndex::CircularIndex(int angle, int radius) {
    this->angle = angle;
    this->radius = radius;
}

template<class T>
CircularList<T>::CircularList(int expected_size) {
    this->expected_size = expected_size;
    this->elements = new T *[expected_size]{};
    this->first_index = 0;
    this->current_index = 0;
}

template<class T>
void CircularList<T>::append(T element) {
    CircularIndex c_index = this->circularIndex(this->current_index, 0);
    int n = this->length() / this->expected_size + 1;
    std::cout << "list.append: [n: " << n << "]" << std::endl;
    auto elements = new T [n]{};
    for (auto i = 0; i < n - 1; i++) {
        elements[i] = this->elements[c_index.angle][i];
    }
    elements[n - 1] = element;
    delete this->elements[c_index.angle];
    this->elements[c_index.angle] = std::move(elements);
    this->current_index += 1;
}

template<class T>
T CircularList<T>::pop() {

    // Retrieve the element that must be removed from the list.
    CircularIndex c_index = this->circularIndex(this->first_index, 0);
    T element = this->elements[c_index.angle][0];

    // Remove the element from the list.
    int n = this->length() / this->expected_size + 1;
    auto elements = new T [n]{};
    for (auto i = 0; i < n; i++) {
        elements[i] = this->elements[c_index.angle][i + 1];
    }
    delete this->elements[c_index.angle];
    this->elements[c_index.angle] = elements;

    // Increase the first index, and return the removed element.
    this->first_index += 1;
    return element;
}

template<class T>
T &CircularList<T>::operator[](int index) {
    CircularIndex c_index = this->circularIndex(index);
    return this->elements[c_index.angle][c_index.radius];
}

template<class T>
CircularIndex CircularList<T>::circularIndex(int index, int first_index) {
    if (index < 0) {
        index += this->length();
    }
    if (first_index == -1) {
        first_index = this->first_index;
    }
    return CircularIndex((index + first_index) % this->expected_size, index / this->expected_size);
}

template<class T>
int CircularList<T>::length() {
    return this->current_index - this->first_index;
}

template<class T>
void CircularList<T>::clear() {
    for (auto i = 0; i < this->length(); i++) {
        CircularIndex index = this->circularIndex(i);
        delete this->elements[index.angle];
        this->elements[index.angle] = nullptr;
    }
    delete this->elements;
    this->elements = new T *[expected_size]{};
    this->current_index = 0;
    this->first_index = 0;
}

template<class T>
int CircularList<T>::getCurrentIndex() {
    return this->current_index;
}
