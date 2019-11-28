#include <iostream>

void print_arr1(int* arr, size_t n) {
  for (size_t i = 0; i < n; ++i) std::cout << arr[i] << ' ';
  std::cout << std::endl;
}

void print_arr2(int arr[5]) {
  for (size_t i = 0; i < 5; ++i) std::cout << arr[i] << ' ';
  std::cout << std::endl;
}

void double_arr(int arr[5]) {
  for (size_t i = 0; i < 5; ++i) arr[i] *= 2;
}

int main() {
  int arr1[] = {1, 2, 3, 4, 5};
  int arr2[5] = {1, 2, 3, 4, 5};
  int arr3[5] = {1, 2, 3};
  int arr4[5];
  int arr5[5] = {};

  print_arr1(arr1, sizeof(arr1) / sizeof(arr1[0]));
  print_arr2(arr2);
  print_arr2(arr3);
  print_arr2(arr4);
  print_arr2(arr5);

  double_arr(arr1);
  print_arr2(arr1);

  return 0;
}
