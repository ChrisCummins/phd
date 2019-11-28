#include "./tests.h"

#include <type_traits>
#include <ustl/type_traits>

////////////////////
// Helper Classes //
////////////////////

TEST(std_type_traits, helper_classes) {
  static_assert(std::true_type::value, "error");
  static_assert(!std::false_type::value, "error");
  const bool t = std::true_type();
  const bool f = std::false_type();
  ASSERT_TRUE(t);
  ASSERT_FALSE(f);

  static_assert(std::true_type(), "error");
  static_assert(!std::false_type(), "error");
}

TEST(ustl_type_traits, helper_classes) {
  static_assert(ustl::true_type::value, "error");
  static_assert(!ustl::false_type::value, "error");
  const bool t = ustl::true_type();
  const bool f = ustl::false_type();
  ASSERT_TRUE(t);
  ASSERT_FALSE(f);

  static_assert(ustl::true_type(), "error");
  static_assert(!ustl::false_type(), "error");
}

/////////////////////////////
// Primary type categories //
/////////////////////////////

TEST(std_type_traits, is_void) {
  static_assert(std::is_void<void>::value, "error");
  static_assert(!std::is_void<int>::value, "error");
  static_assert(std::is_void<const void>::value, "error");
  static_assert(std::is_void<const volatile void>::value, "error");

  // c++17 usage:
  // static_assert(std::is_void_v<void>, "error");
  // static_assert(!std::is_void_v<int>, "error");
}

TEST(ustl_type_traits, is_void) {
  static_assert(ustl::is_void<void>::value, "error");
  static_assert(!ustl::is_void<int>::value, "error");
  static_assert(ustl::is_void<const void>::value, "error");
  static_assert(ustl::is_void<const volatile void>::value, "error");

  // c++17 usage:
  static_assert(ustl::is_void_v<void>, "error");
  static_assert(!ustl::is_void_v<int>, "error");
}

TEST(std_type_traits, is_integral) {
  int x;

  static_assert(std::is_integral<int>::value, "error");
  static_assert(!std::is_integral<float>::value, "error");
  static_assert(std::is_integral<decltype(x)>::value, "error");
  static_assert(std::is_integral<const int>::value, "error");
  static_assert(std::is_integral<unsigned int>::value, "error");
  static_assert(std::is_integral<const unsigned long long>::value,
                "error");  // NOLINT

  // c++17 usage:
  // static_assert(std::is_integral_v<int>);
  // static_assert(!std::is_integral_v<float>);
}

TEST(ustl_type_traits, is_integral) {
  int x;

  static_assert(ustl::is_integral<int>::value, "error");
  static_assert(!ustl::is_integral<float>::value, "error");
  static_assert(ustl::is_integral<decltype(x)>::value, "error");
  static_assert(ustl::is_integral<const int>::value, "error");
  static_assert(ustl::is_integral<unsigned int>::value, "error");
  static_assert(ustl::is_integral<const unsigned long long>::value,
                "error");  // NOLINT

  // c++17 usage:
  static_assert(ustl::is_integral_v<int>, "error");
  static_assert(!ustl::is_integral_v<float>, "error");
}

TEST(std_type_traits, is_pointer) {
  int x;
  int *px;

  static_assert(!std::is_pointer<decltype(x)>::value, "error");
  static_assert(std::is_pointer<decltype(px)>::value, "error");
  static_assert(std::is_pointer<decltype(px) *>::value, "error");
  static_assert(!std::is_pointer<float>::value, "error");
  static_assert(std::is_pointer<const float *>::value, "error");

  // c++17 usage:
  // static_assert(std::is_pointer_v<int*>, "error");
  // static_assert(!std::is_pointer_v<int>, "error");
}

TEST(ustl_type_traits, is_pointer) {
  int x;
  int *px;

  static_assert(!ustl::is_pointer<decltype(x)>::value, "error");
  static_assert(ustl::is_pointer<decltype(px)>::value, "error");
  static_assert(ustl::is_pointer<decltype(px) *>::value, "error");
  static_assert(!ustl::is_pointer<float>::value, "error");
  static_assert(ustl::is_pointer<const float *>::value, "error");

  // c++17 usage:
  static_assert(ustl::is_pointer_v<int *>, "error");
  static_assert(!ustl::is_pointer_v<int>, "error");
}

////////////////////////
// Type relationships //
////////////////////////

// TEST(std_type_traits, is_same) {
//   int x;

//   static_assert(std::is_same<int, int>::value);
//   static_assert(!std::is_same<int, float>::value);
//   static_assert(std::is_same<int, decltype(x)>::value);
//   // cv qualifiers count:
//   static_assert(!std::is_same<const int, decltype(x)>::value);

//   // c++17 usage:
//   static_assert(std::is_same_v<int, int>);
//   static_assert(!std::is_same_v<int, float>);
// }

TEST(ustl_type_traits, is_same) {
  int x;

  static_assert(ustl::is_same<int, int>::value, "error");
  static_assert(!ustl::is_same<int, float>::value, "error");
  static_assert(ustl::is_same<int, decltype(x)>::value, "error");
  // cv qualifiers count:
  static_assert(!ustl::is_same<const int, decltype(x)>::value, "error");

  // c++17 usage:
  static_assert(ustl::is_same_v<int, int>, "error");
  static_assert(!ustl::is_same_v<int, float>, "error");
}

/////////////////////////////////
// Const-volatility specifiers //
/////////////////////////////////

TEST(std_type_traits, remove_const) {
  static_assert(std::is_same<int, std::remove_const<const int>::type>::value,
                "error");
  static_assert(std::is_same<int, std::remove_const<int>::type>::value,
                "error");
}

TEST(ustl_type_traits, remove_const) {
  static_assert(ustl::is_same<int, ustl::remove_const<const int>::type>::value,
                "error");
  static_assert(ustl::is_same<int, ustl::remove_const<int>::type>::value,
                "error");
}

TEST(std_type_traits, remove_volatile) {
  static_assert(
      std::is_same<int, std::remove_volatile<volatile int>::type>::value,
      "error");
  static_assert(std::is_same<int, std::remove_volatile<int>::type>::value,
                "error");
}

TEST(ustl_type_traits, remove_volatile) {
  static_assert(
      ustl::is_same<int, ustl::remove_volatile<volatile int>::type>::value,
      "error");
  static_assert(ustl::is_same<int, ustl::remove_volatile<int>::type>::value,
                "error");
}

TEST(std_type_traits, remove_cv) {
  static_assert(
      std::is_same<int, std::remove_cv<const volatile int>::type>::value,
      "error");
  static_assert(
      std::is_same<int, std::remove_volatile<volatile int>::type>::value,
      "error");
  static_assert(std::is_same<int, std::remove_cv<const int>::type>::value,
                "error");
  static_assert(std::is_same<int, std::remove_cv<int>::type>::value, "error");
}

TEST(ustl_type_traits, remove_cv) {
  static_assert(
      ustl::is_same<int, ustl::remove_cv<const volatile int>::type>::value,
      "error");
  static_assert(
      ustl::is_same<int, ustl::remove_volatile<volatile int>::type>::value,
      "error");
  static_assert(ustl::is_same<int, ustl::remove_cv<const int>::type>::value,
                "error");
  static_assert(ustl::is_same<int, ustl::remove_cv<int>::type>::value, "error");
}

////////////////
// References //
////////////////

TEST(std_type_traits, remove_reference) {
  static_assert(std::is_same<int, std::remove_reference<int>::type>::value,
                "error");
  static_assert(std::is_same<int, std::remove_reference<int &>::type>::value,
                "error");
  static_assert(std::is_same<int, std::remove_reference<int &&>::type>::value,
                "error");
  static_assert(
      std::is_same<const int, std::remove_reference_t<const int &&>>::value,
      "error");
}

TEST(ustl_type_traits, remove_reference) {
  static_assert(ustl::is_same<int, ustl::remove_reference<int>::type>::value,
                "error");
  static_assert(ustl::is_same<int, ustl::remove_reference<int &>::type>::value,
                "error");
  static_assert(ustl::is_same<int, ustl::remove_reference<int &&>::type>::value,
                "error");
  static_assert(
      ustl::is_same<const int, ustl::remove_reference_t<const int &&>>::value,
      "error");
}

TEST(std_type_traits, add_lvalue_reference) {
  static_assert(
      std::is_same<int &, std::add_lvalue_reference<int>::type>::value,
      "error");
  static_assert(
      std::is_same<int &, std::add_lvalue_reference<int &>::type>::value,
      "error");
  static_assert(
      std::is_same<int &, std::add_lvalue_reference<int &&>::type>::value,
      "error");
  static_assert(std::is_same<const int &,
                             std::add_lvalue_reference_t<const int &>>::value,
                "error");
}

TEST(ustl_type_traits, add_lvalue_reference) {
  static_assert(
      ustl::is_same<int &, ustl::add_lvalue_reference<int>::type>::value,
      "error");
  static_assert(
      ustl::is_same<int &, ustl::add_lvalue_reference<int &>::type>::value,
      "error");
  static_assert(
      ustl::is_same<int &, ustl::add_lvalue_reference<int &&>::type>::value,
      "error");
  static_assert(ustl::is_same<const int &,
                              ustl::add_lvalue_reference_t<const int &>>::value,
                "error");
}

TEST(std_type_traits, add_rvalue_reference) {
  static_assert(
      std::is_same<int &&, std::add_rvalue_reference<int>::type>::value,
      "error");
  static_assert(
      std::is_same<int &, std::add_rvalue_reference<int &>::type>::value,
      "error");
  static_assert(
      std::is_same<int &&, std::add_rvalue_reference<int &&>::type>::value,
      "error");
  static_assert(std::is_same<const int &&,
                             std::add_rvalue_reference<const int>::type>::value,
                "error");
  static_assert(std::is_same<const int &,
                             std::add_rvalue_reference_t<const int &>>::value,
                "error");
}

TEST(ustl_type_traits, add_rvalue_reference) {
  static_assert(
      ustl::is_same<int &&, ustl::add_rvalue_reference<int>::type>::value,
      "error");
  static_assert(
      ustl::is_same<int &, ustl::add_rvalue_reference<int &>::type>::value,
      "error");
  static_assert(
      ustl::is_same<int &&, ustl::add_rvalue_reference<int &&>::type>::value,
      "error");
  static_assert(
      ustl::is_same<const int &&,
                    ustl::add_rvalue_reference<const int>::type>::value,
      "error");
  static_assert(ustl::is_same<const int &,
                              ustl::add_rvalue_reference_t<const int &>>::value,
                "error");
}

//////////////
// Pointers //
//////////////

// TEST(std_type_traits, remove_pointer) {
//   static_assert(std::is_same_v<int, std::remove_pointer_t<int>>);
//   static_assert(std::is_same<int, std::remove_pointer<int*>::type>::value);
//   static_assert(std::is_same<int*, std::remove_pointer<int**>::type>::value);
//   static_assert(std::is_same<const int,
//                 std::remove_pointer<const int*>::type>::value);
//   static_assert(std::is_same<int,
//                 std::remove_pointer<int*const>::type>::value);
//   static_assert(std::is_same<int,
//                 std::remove_pointer<int*const volatile>::type>::value);
//   static_assert(std::is_same<int&,
//                 std::remove_pointer_t<int&>>::value);
// }

TEST(ustl_type_traits, remove_pointer) {
  static_assert(ustl::is_same_v<int, ustl::remove_pointer_t<int>>, "error");
  static_assert(ustl::is_same<int, ustl::remove_pointer<int *>::type>::value,
                "error");
  static_assert(ustl::is_same<int *, ustl::remove_pointer<int **>::type>::value,
                "error");
  static_assert(
      ustl::is_same<const int, ustl::remove_pointer<const int *>::type>::value,
      "error");
  static_assert(
      ustl::is_same<int, ustl::remove_pointer<int *const>::type>::value,
      "error");
  static_assert(
      ustl::is_same<int,
                    ustl::remove_pointer<int *const volatile>::type>::value,
      "error");
  static_assert(ustl::is_same<int &, ustl::remove_pointer_t<int &>>::value,
                "error");
}

////////////
// Arrays //
////////////

// TEST(std_type_traits, remove_extent) {
//   static_assert(std::is_same_v<int, std::remove_extent_t<int>>);
//   static_assert(std::is_same_v<int, std::remove_extent_t<int[]>>);
//   static_assert(std::is_same_v<int, std::remove_extent_t<int[10]>>);
//   static_assert(std::is_same_v<int[10][15],
//                 std::remove_extent<int[5][10][15]>::type>);
// }

TEST(ustl_type_traits, remove_extent) {
  static_assert(ustl::is_same_v<int, ustl::remove_extent_t<int>>, "error");
  static_assert(ustl::is_same_v<int, ustl::remove_extent_t<int[]>>, "error");
  static_assert(ustl::is_same_v<int, ustl::remove_extent_t<int[10]>>, "error");
  static_assert(
      ustl::is_same_v<int[10][15], ustl::remove_extent<int[5][10][15]>::type>,
      "error");
}
