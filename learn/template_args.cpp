#include <iostream>
#include <array>

// Single template argument:
template<typename array>
void print_array_1(array &a) {
    std::cout << "{ ";
    for (auto i : a)
        std::cout << i << " ";
    std::cout << "}";
}

// Explicit template arguments:
template<typename T, size_t n>
void print_array_2(std::array<T, n> &a) {
    std::cout << "{ ";
    for (auto i : a)
        std::cout << i << " ";
    std::cout << "}";
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    std::array<int, 3> a = {{1, 2, 3}};

    std::cout << "Array contains: ";
    print_array_1(a);
    std::cout << std::endl;

    std::cout << "Array contains: ";
    print_array_2(a);
    std::cout << std::endl;

    return 0;
}
