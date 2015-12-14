#include <iostream>

// A vector class which implements begin(), end() and sum().

template<typename T>
class MyVector {
 public:
    explicit MyVector(size_t size);
    ~MyVector();

    T& operator[](size_t i);
    const T& operator[](size_t i) const;

    size_t size() const;

 private:
    T *_data;
    size_t _size;
};

// Method definitions:

template<typename T>
MyVector<T>::MyVector(size_t size) {
    _size = size;
    _data = new T[size];
}

template<typename T>
MyVector<T>::~MyVector() {
    delete[] _data;
}

template<typename T>
size_t MyVector<T>::size() const {
    return _size;
}

template<typename T>
T& MyVector<T>::operator[](size_t i) {
    if (i >= size())
        throw std::out_of_range("Vector::operator[]");

    return _data[i];
}

template<typename T>
const T& MyVector<T>::operator[](size_t i) const {
    if (i >= size())
        throw std::out_of_range("Vector::operator[]");

    return _data[i];
}

// To support iterating over vectors:

template<typename T>
const T *begin(const MyVector<T>& v) {
    return v.size() ? &v[0] : nullptr;
}

template<typename T>
const T *end(const MyVector<T>& v) {
    return begin(v) + v.size();
}

// Summation operator:

template<typename T>
T &sum(const MyVector<T> &vec, T *acc) {
    for (auto &v : vec)
        *acc += v;

    return *acc;
}


int main(int argc, char **argv) {
    std::cout << "Hello, world!" << std::endl;

    MyVector<int> v(5);

    v[0] = 0;
    v[1] = 1;
    v[2] = 2;
    v[3] = 3;
    v[4] = 4;

    std::cout << "Vector contents: ";
    for (auto i : v)
        std::cout << i << " ";
    std::cout << std::endl;

    int i = 0;
    std::cout << "Vector sum: " << sum(v, &i);

    return 0;
}
