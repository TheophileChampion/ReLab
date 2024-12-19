#include "circular_list.hpp"

CircularIndex::CircularIndex(int angle, int radius) {
    this->angle = angle;
    this->radius = radius;
}

template<class T>
CircularList<T>::CircularList(int expected_size) {
    this->expected_size = expected_size;
    std::vector<std::vector<T>> elements(expected_size);
    this->elements = std::move(elements);
    this->first_index = 0;
    this->current_index = 0;
}

template<class T>
void CircularList<T>::append(T element) {
    CircularIndex c_index = this->circularIndex(this->current_index, 0);
    this->elements[c_index.angle].push_back(element);
    this->current_index += 1;
}

template<class T>
T CircularList<T>::pop() {
    CircularIndex c_index = this->circularIndex(this->first_index, 0);
    T element = this->elements[c_index.angle][0];
    this->elements[c_index.angle].erase(this->elements[c_index.angle].begin());
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
    std::vector<std::vector<T>> elements(this->expected_size);
    this->elements = std::move(elements);
    this->current_index = 0;
    this->first_index = 0;
}

template<class T>
int CircularList<T>::getCurrentIndex() {
    return this->current_index;
}
