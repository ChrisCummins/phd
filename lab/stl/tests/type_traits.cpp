#include "./tests.h"

#include <type_traits>
#include <ustl/type_traits>


////////////////////
// Helper Classes //
////////////////////

TEST(std_type_traits, helper_classes) {
  static_assert(std::true_type::value);
  static_assert(!std::false_type::value);
  const bool t = std::true_type();
  const bool f = std::false_type();
  ASSERT_TRUE(t);
  ASSERT_FALSE(f);

  static_assert(std::true_type());
  static_assert(!std::false_type());
}

TEST(ustl_type_traits, helper_classes) {
  static_assert(ustl::true_type::value);
  static_assert(!ustl::false_type::value);
  const bool t = ustl::true_type();
  const bool f = ustl::false_type();
  ASSERT_TRUE(t);
  ASSERT_FALSE(f);

  static_assert(ustl::true_type());
  static_assert(!ustl::false_type());
}


/////////////////////////////
// Primary type categories //
/////////////////////////////

TEST(std_type_traits, is_void) {
  static_assert(std::is_void<void>::value);
  static_assert(!std::is_void<int>::value);
  static_assert(std::is_void<const void>::value);
  static_assert(std::is_void<const volatile void>::value);

  // c++17 usage:
  static_assert(std::is_void_v<void>);
  static_assert(!std::is_void_v<int>);
}

TEST(ustl_type_traits, is_void) {
  static_assert(ustl::is_void<void>::value);
  static_assert(!ustl::is_void<int>::value);
  static_assert(ustl::is_void<const void>::value);
  static_assert(ustl::is_void<const volatile void>::value);

  // c++17 usage:
  static_assert(ustl::is_void_v<void>);
  static_assert(!ustl::is_void_v<int>);
}


TEST(std_type_traits, is_integral) {
  int x;

  static_assert(std::is_integral<int>::value);
  static_assert(!std::is_integral<float>::value);
  static_assert(std::is_integral<decltype(x)>::value);
  static_assert(std::is_integral<const int>::value);
  static_assert(std::is_integral<unsigned int>::value);
  static_assert(std::is_integral<const unsigned long long>::value);  // NOLINT

  // c++17 usage:
  static_assert(std::is_integral_v<int>);
  static_assert(!std::is_integral_v<float>);
}

TEST(ustl_type_traits, is_integral) {
  int x;

  static_assert(ustl::is_integral<int>::value);
  static_assert(!ustl::is_integral<float>::value);
  static_assert(ustl::is_integral<decltype(x)>::value);
  static_assert(ustl::is_integral<const int>::value);
  static_assert(ustl::is_integral<unsigned int>::value);
  static_assert(ustl::is_integral<const unsigned long long>::value);  // NOLINT

  // c++17 usage:
  static_assert(ustl::is_integral_v<int>);
  static_assert(!ustl::is_integral_v<float>);
}


TEST(std_type_traits, is_pointer) {
  int x;
  int *px;

  static_assert(!std::is_pointer<decltype(x)>::value);
  static_assert(std::is_pointer<decltype(px)>::value);
  static_assert(std::is_pointer<decltype(px)*>::value);
  static_assert(!std::is_pointer<float>::value);
  static_assert(std::is_pointer<const float*>::value);

  // c++17 usage:
  static_assert(std::is_pointer_v<int*>);
  static_assert(!std::is_pointer_v<int>);
}

TEST(ustl_type_traits, is_pointer) {
  int x;
  int *px;

  static_assert(!ustl::is_pointer<decltype(x)>::value);
  static_assert(ustl::is_pointer<decltype(px)>::value);
  static_assert(ustl::is_pointer<decltype(px)*>::value);
  static_assert(!ustl::is_pointer<float>::value);
  static_assert(ustl::is_pointer<const float*>::value);

  // c++17 usage:
  static_assert(ustl::is_pointer_v<int*>);
  static_assert(!ustl::is_pointer_v<int>);
}


////////////////////////
// Type relationships //
////////////////////////

TEST(std_type_traits, is_same) {
  int x;

  static_assert(std::is_same<int, int>::value);
  static_assert(!std::is_same<int, float>::value);
  static_assert(std::is_same<int, decltype(x)>::value);
  // cv qualifiers count:
  static_assert(!std::is_same<const int, decltype(x)>::value);

  // c++17 usage:
  static_assert(std::is_same_v<int, int>);
  static_assert(!std::is_same_v<int, float>);
}


TEST(ustl_type_traits, is_same) {
  int x;

  static_assert(ustl::is_same<int, int>::value);
  static_assert(!ustl::is_same<int, float>::value);
  static_assert(ustl::is_same<int, decltype(x)>::value);
  // cv qualifiers count:
  static_assert(!ustl::is_same<const int, decltype(x)>::value);

  // c++17 usage:
  static_assert(ustl::is_same_v<int, int>);
  static_assert(!ustl::is_same_v<int, float>);
}


/////////////////////////////////
// Const-volatility specifiers //
/////////////////////////////////


TEST(std_type_traits, remove_const) {
  static_assert(std::is_same<int, std::remove_const<const int>::type>::value);
  static_assert(std::is_same<int, std::remove_const<int>::type>::value);
}

TEST(ustl_type_traits, remove_const) {
  static_assert(ustl::is_same<int, ustl::remove_const<const int>::type>::value);
  static_assert(ustl::is_same<int, ustl::remove_const<int>::type>::value);
}


TEST(std_type_traits, remove_volatile) {
  static_assert(std::is_same<int,
                std::remove_volatile<volatile int>::type>::value);
  static_assert(std::is_same<int, std::remove_volatile<int>::type>::value);
}

TEST(ustl_type_traits, remove_volatile) {
  static_assert(ustl::is_same<int,
                ustl::remove_volatile<volatile int>::type>::value);
  static_assert(ustl::is_same<int, ustl::remove_volatile<int>::type>::value);
}


TEST(std_type_traits, remove_cv) {
  static_assert(std::is_same<int,
                std::remove_cv<const volatile int>::type>::value);
  static_assert(std::is_same<int,
                std::remove_volatile<volatile int>::type>::value);
  static_assert(std::is_same<int, std::remove_cv<const int>::type>::value);
  static_assert(std::is_same<int, std::remove_cv<int>::type>::value);
}

TEST(ustl_type_traits, remove_cv) {
  static_assert(ustl::is_same<int,
                ustl::remove_cv<const volatile int>::type>::value);
  static_assert(ustl::is_same<int,
                ustl::remove_volatile<volatile int>::type>::value);
  static_assert(ustl::is_same<int, ustl::remove_cv<const int>::type>::value);
  static_assert(ustl::is_same<int, ustl::remove_cv<int>::type>::value);
}
