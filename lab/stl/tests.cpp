/*
 * test.cpp - Unit tests.
 *
 * Each function with ustl has one or more unit tests associated with
 * it. For each ustl unit test, there is an identical test using the
 * std:: library to compare against. This ensures that the majority of
 * code is tested, and the behaviour of the code matches the STD
 * library.
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wundef"
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include <algorithm>
#include <ustl/algorithm>

#include <array>
#include <ustl/array>

#include <vector>
#include <ustl/vector>

#include <map>
#include <ustl/map>

#include <unordered_map>
#include <ustl/unordered_map>

//
// Helper functions & objects.
//

template<typename T>
class InverseComp {
 public:
  bool operator()(const T &a, const T &b) { return a > b; }
};

class Comparable {
 public:
  Comparable() : data(0) {}
  explicit Comparable(const int _data) : Comparable(_data, 0) {}
  Comparable(const int _data, const int _nc) : data(_data), nc(_nc) {}
  ~Comparable() {}

  int data;
  int nc;
  bool operator<(const Comparable &rhs) const { return data < rhs.data; }
  bool operator==(const Comparable &rhs) const { return data == rhs.data; }
};

static bool inverse_comp(const int &a, const int &b) { return a > b; }

// lambdas
auto is_even = [](const int &x) -> bool{ return !(x % 2); };
auto increment = [](int &x) -> void{ x++; };

template<typename VectorType, typename T>
bool vector_equal(const VectorType &v, std::initializer_list<T> il) {
  auto vit = v.begin();
  auto ilit = il.begin();

  while (vit != v.end())
    if (*vit++ != *ilit++)
      return false;
  return true;
}


/////////////////
// Algorithms: //
/////////////////

// all_of

TEST(std_algorithm, all_of) {
  std::vector<int> a{2, 4, 6, 8};
  std::vector<int> b{2, 4, 6, 7, 8};

  ASSERT_TRUE(std::all_of(a.begin(), a.end(), is_even));
  ASSERT_FALSE(std::all_of(b.begin(), b.end(), is_even));
}

TEST(ustl_algorithm, all_of) {
  ustl::vector<int> a{2, 4, 6, 8};
  ustl::vector<int> b{2, 4, 6, 7, 8};

  ASSERT_TRUE(ustl::all_of(a.begin(), a.end(), is_even));
  ASSERT_FALSE(ustl::all_of(b.begin(), b.end(), is_even));
}

// any_of

TEST(std_algorithm, any_of) {
  std::vector<int> a{2, 4, 6, 8};
  std::vector<int> b{1, 3, 5, 7};
  std::vector<int> c{1, 3, 5, 6, 7};

  ASSERT_TRUE(std::any_of(a.begin(), a.end(), is_even));
  ASSERT_FALSE(std::any_of(b.begin(), b.end(), is_even));
  ASSERT_TRUE(std::any_of(c.begin(), c.end(), is_even));
}

TEST(ustl_algorithm, any_of) {
  ustl::vector<int> a{2, 4, 6, 8};
  ustl::vector<int> b{1, 3, 5, 7};
  ustl::vector<int> c{1, 3, 5, 6, 7};

  ASSERT_TRUE(ustl::any_of(a.begin(), a.end(), is_even));
  ASSERT_FALSE(ustl::any_of(b.begin(), b.end(), is_even));
  ASSERT_TRUE(ustl::any_of(c.begin(), c.end(), is_even));
}

// none_of

TEST(std_algorithm, none_of) {
  std::vector<int> a{2, 4, 6, 8};
  std::vector<int> b{1, 3, 5, 7};
  std::vector<int> c{1, 3, 5, 6, 7};

  ASSERT_FALSE(std::none_of(a.begin(), a.end(), is_even));
  ASSERT_TRUE(std::none_of(b.begin(), b.end(), is_even));
  ASSERT_FALSE(std::none_of(c.begin(), c.end(), is_even));
}

TEST(ustl_algorithm, none_of) {
  ustl::vector<int> a{2, 4, 6, 8};
  ustl::vector<int> b{1, 3, 5, 7};
  ustl::vector<int> c{1, 3, 5, 6, 7};

  ASSERT_FALSE(ustl::none_of(a.begin(), a.end(), is_even));
  ASSERT_TRUE(ustl::none_of(b.begin(), b.end(), is_even));
  ASSERT_FALSE(ustl::none_of(c.begin(), c.end(), is_even));
}

// for_each

TEST(std_algorithm, for_each) {
  std::vector<int> in{1, 2, 3, 4, 5};
  std::vector<int> out{2, 3, 4, 5, 6};

  std::for_each(in.begin(), in.end(), increment);

  for (size_t i = 0; i < 5; i++)
    ASSERT_EQ(in[i], out[i]);
}

TEST(ustl_algorithm, for_each) {
  ustl::vector<int> in{1, 2, 3, 4, 5};
  ustl::vector<int> out{2, 3, 4, 5, 6};

  ustl::for_each(in.begin(), in.end(), increment);

  for (size_t i = 0; i < 5; i++)
    ASSERT_EQ(in[i], out[i]);
}

// find

TEST(std_algorithm, find) {
  std::vector<int> a{1, 2, 3, 4, 5};
  std::vector<int> b{2, 4, 6, 8, 10};

  ASSERT_EQ(3, *std::find(a.begin(), a.end(), 3));
  ASSERT_EQ(b.end(), std::find(b.begin(), b.end(), 3));
}

TEST(ustl_algorithm, find) {
  ustl::vector<int> a{1, 2, 3, 4, 5};
  ustl::vector<int> b{2, 4, 6, 8, 10};

  ASSERT_EQ(3, *ustl::find(a.begin(), a.end(), 3));
  ASSERT_EQ(b.end(), ustl::find(b.begin(), b.end(), 3));
}

// find_if()

TEST(std_algorithm, find_if) {
  std::vector<int> a{1, 3, 5, 7, 9};
  std::vector<int> b{2, 4, 6, 8, 10};

  ASSERT_EQ(a.end(), std::find_if(a.begin(), a.end(), is_even));
  ASSERT_EQ(b.begin(), std::find_if(b.begin(), b.end(), is_even));
}

TEST(ustl_algorithm, find_if) {
  ustl::vector<int> a{1, 3, 5, 7, 9};
  ustl::vector<int> b{2, 4, 6, 8, 10};

  ASSERT_EQ(a.end(), ustl::find_if(a.begin(), a.end(), is_even));
  ASSERT_EQ(b.begin(), ustl::find_if(b.begin(), b.end(), is_even));
}

// find_if_not()

TEST(std_algorithm, find_if_not) {
  std::vector<int> a{1, 3, 5, 7, 9};
  std::vector<int> b{2, 4, 6, 8, 10};

  ASSERT_EQ(a.begin(), std::find_if_not(a.begin(), a.end(), is_even));
  ASSERT_EQ(b.end(), std::find_if_not(b.begin(), b.end(), is_even));
}

TEST(ustl_algorithm, find_if_not) {
  ustl::vector<int> a{1, 3, 5, 7, 9};
  ustl::vector<int> b{2, 4, 6, 8, 10};

  ASSERT_EQ(a.begin(), ustl::find_if_not(a.begin(), a.end(), is_even));
  ASSERT_EQ(b.end(), ustl::find_if_not(b.begin(), b.end(), is_even));
}

// find_end()

TEST(std_algorithm, find_end) {
  std::vector<int> a{1, 2, 3, 4, 5, 6, 1, 2, 3, 9};
  std::vector<int> seq{1, 2, 3};
  std::vector<int> notseq{0, 1};

  ASSERT_EQ(a.begin() + 6, std::find_end(a.begin(), a.end(),
                                         seq.begin(), seq.end()));
  ASSERT_EQ(a.end(), std::find_end(a.begin(), a.end(),
                                   notseq.begin(), notseq.end()));
}

TEST(ustl_algorithm, find_end) {
  ustl::vector<int> a{1, 2, 3, 4, 5, 6, 1, 2, 3, 9};
  ustl::vector<int> seq{1, 2, 3};
  ustl::vector<int> notseq{0, 1};

  ASSERT_EQ(a.begin() + 6, ustl::find_end(a.begin(), a.end(),
                                          seq.begin(), seq.end()));
  ASSERT_EQ(a.end(), ustl::find_end(a.begin(), a.end(),
                                    notseq.begin(), notseq.end()));
}

// find_first_of()

TEST(std_algorithm, find_first_of) {
  std::vector<int> a{1, 2, 3, 4, 5, 6, 1, 2, 3, 9};
  std::vector<int> seq{-1, 2, 3};
  std::vector<int> notseq{0, -1};

  ASSERT_EQ(a.begin() + 1, std::find_first_of(a.begin(), a.end(),
                                              seq.begin(), seq.end()));
  ASSERT_EQ(a.end(), std::find_first_of(a.begin(), a.end(),
                                        notseq.begin(), notseq.end()));
}

TEST(ustl_algorithm, find_first_of) {
  ustl::vector<int> a{1, 2, 3, 4, 5, 6, 1, 2, 3, 9};
  ustl::vector<int> seq{-1, 2, 3};
  ustl::vector<int> notseq{0, -1};

  ASSERT_EQ(a.begin() + 1, ustl::find_first_of(a.begin(), a.end(),
                                               seq.begin(), seq.end()));
  ASSERT_EQ(a.end(), ustl::find_first_of(a.begin(), a.end(),
                                         notseq.begin(), notseq.end()));
}

// adjacent_find()

TEST(std_algorithm, adjacent_find) {
  std::vector<int> a{1, 1, 2, 3, 4, 5, 5};
  std::vector<int> b{0, 1, 2, 3, 3, 4, 5};

  ASSERT_EQ(a.begin(), std::adjacent_find(a.begin(), a.end()));
  ASSERT_EQ(b.begin() + 3, std::adjacent_find(b.begin(), b.end()));
}

TEST(ustl_algorithm, adjacent_find) {
  ustl::vector<int> a{1, 1, 2, 3, 4, 5, 5};
  ustl::vector<int> b{0, 1, 2, 3, 3, 4, 5};

  ASSERT_EQ(a.begin(), ustl::adjacent_find(a.begin(), a.end()));
  ASSERT_EQ(b.begin() + 3, ustl::adjacent_find(b.begin(), b.end()));
}

// count_if()

TEST(std_algorithm, count_if) {
  std::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> b{1, 3, 5, 7, 9};

  ASSERT_EQ(5, std::count_if(a.begin(), a.end(), is_even));
  ASSERT_EQ(0, std::count_if(b.begin(), b.end(), is_even));
}

TEST(ustl_algorithm, count_if) {
  ustl::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  ustl::vector<int> b{1, 3, 5, 7, 9};

  ASSERT_EQ(5, ustl::count_if(a.begin(), a.end(), is_even));
  ASSERT_EQ(0, ustl::count_if(b.begin(), b.end(), is_even));
}

// count()

TEST(std_algorithm, count) {
  std::vector<int> a{1, 2, 3, 1, 1};
  std::vector<int> b{1, 3, 5, 7, 9};

  ASSERT_EQ(3, std::count(a.begin(), a.end(), 1));
  ASSERT_EQ(0, std::count(b.begin(), b.end(), -10));
}

TEST(ustl_algorithm, count) {
  ustl::vector<int> a{1, 2, 3, 1, 1};
  ustl::vector<int> b{1, 3, 5, 7, 9};

  ASSERT_EQ(3, ustl::count(a.begin(), a.end(), 1));
  ASSERT_EQ(0, ustl::count(b.begin(), b.end(), -10));
}

// mismatch()

TEST(std_algorithm, mismatch) {
  std::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> b{1, 2, 3, 0, 0, 0, 0, 0};
  std::vector<int> c{0, 2, 3, 0, 0, 0, 0, 0};

  const auto res1 = std::mismatch(a.begin(), a.end(), b.begin());
  ASSERT_EQ(a.begin() + 3, res1.first);
  ASSERT_EQ(b.begin() + 3, res1.second);

  const auto res2 = std::mismatch(a.begin(), a.end(), c.begin());
  ASSERT_EQ(a.begin(), res2.first);
  ASSERT_EQ(c.begin(), res2.second);
}

TEST(ustl_algorithm, mismatch) {
  ustl::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8};
  ustl::vector<int> b{1, 2, 3, 0, 0, 0, 0, 0};
  ustl::vector<int> c{0, 2, 3, 0, 0, 0, 0, 0};

  const auto res1 = ustl::mismatch(a.begin(), a.end(), b.begin());
  ASSERT_EQ(a.begin() + 3, res1.first);
  ASSERT_EQ(b.begin() + 3, res1.second);

  const auto res2 = ustl::mismatch(a.begin(), a.end(), c.begin());
  ASSERT_EQ(a.begin(), res2.first);
  ASSERT_EQ(c.begin(), res2.second);
}

// equal()

TEST(std_algorithm, equal) {
  std::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> b{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> c{0, 2, 3, 0, 0, 0, 0, 0};

  ASSERT_TRUE(std::equal(a.begin(), a.end(), b.begin()));
  ASSERT_FALSE(std::equal(a.begin(), a.end(), c.begin()));
}

TEST(ustl_algorithm, equal) {
  ustl::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8};
  ustl::vector<int> b{1, 2, 3, 4, 5, 6, 7, 8};
  ustl::vector<int> c{0, 2, 3, 0, 0, 0, 0, 0};

  ASSERT_TRUE(ustl::equal(a.begin(), a.end(), b.begin()));
  ASSERT_FALSE(ustl::equal(a.begin(), a.end(), c.begin()));
}

// is_permutation()

TEST(std_algorithm, is_permutation) {
  std::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> b{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> c{8, 2, 3, 4, 5, 7, 6, 1};
  std::vector<int> d{0, 0, 0, 0, 1, 1, 2, 3};

  ASSERT_TRUE(std::equal(a.begin(), a.end(), b.begin()));
  ASSERT_FALSE(std::equal(a.begin(), a.end(), c.begin()));
  ASSERT_FALSE(std::equal(a.begin(), a.end(), d.begin()));
}

TEST(ustl_algorithm, is_permutation) {
  ustl::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8};
  ustl::vector<int> b{1, 2, 3, 4, 5, 6, 7, 8};
  ustl::vector<int> c{8, 2, 3, 4, 5, 7, 6, 1};
  ustl::vector<int> d{0, 0, 0, 0, 1, 1, 2, 3};

  ASSERT_TRUE(ustl::equal(a.begin(), a.end(), b.begin()));
  ASSERT_FALSE(ustl::equal(a.begin(), a.end(), c.begin()));
  ASSERT_FALSE(ustl::equal(a.begin(), a.end(), d.begin()));
}

// search()

TEST(std_algorithm, search) {
  std::vector<int> a{0, 1, 2, 3, 4, 5, 1, 2, 3, 9};
  std::vector<int> seq{1, 2, 3};
  std::vector<int> notseq{-10, -11};

  ASSERT_EQ(a.begin() + 1, std::search(a.begin(), a.end(),
                                       seq.begin(), seq.end()));
  ASSERT_EQ(a.end(), std::search(a.begin(), a.end(),
                                 notseq.begin(), notseq.end()));
}

TEST(ustl_algorithm, search) {
  ustl::vector<int> a{0, 1, 2, 3, 4, 5, 1, 2, 3, 9};
  ustl::vector<int> seq{1, 2, 3};
  ustl::vector<int> notseq{-10, -11};

  ASSERT_EQ(a.begin() + 1, ustl::search(a.begin(), a.end(),
                                        seq.begin(), seq.end()));
  ASSERT_EQ(a.end(), ustl::search(a.begin(), a.end(),
                                  notseq.begin(), notseq.end()));
}

// search_n()

TEST(std_algorithm, search_n) {
  std::vector<int> a{0, 0, 1, 1, 0, 1, 1, 0, 0, 0};

  ASSERT_EQ(a.begin() + 7, std::search_n(a.begin(), a.end(), 3, 0));
  ASSERT_EQ(a.end(), std::search_n(a.begin(), a.end(), 3, 1));

  // Check against behaviour of std library.
  ASSERT_EQ(std::search_n(a.begin(), a.end(), 3, 0),
            std::search_n(a.begin(), a.end(), 3, 0));
  ASSERT_EQ(std::search_n(a.begin(), a.end(), 3, 1),
            std::search_n(a.begin(), a.end(), 3, 1));
}

TEST(ustl_algorithm, search_n) {
  ustl::vector<int> a{0, 0, 1, 1, 0, 1, 1, 0, 0, 0};

  ASSERT_EQ(a.begin() + 7, ustl::search_n(a.begin(), a.end(), 3, 0));
  ASSERT_EQ(a.end(), ustl::search_n(a.begin(), a.end(), 3, 1));

  // Check against behaviour of std library.
  ASSERT_EQ(ustl::search_n(a.begin(), a.end(), 3, 0),
            std::search_n(a.begin(), a.end(), 3, 0));
  ASSERT_EQ(ustl::search_n(a.begin(), a.end(), 3, 1),
            std::search_n(a.begin(), a.end(), 3, 1));
}

// Merge

TEST(std_algorithm, merge) {
  std::array<int, 5> a{1, 3, 5, 7, 9};
  std::array<int, 5> b{2, 4, 6, 8, 10};
  std::array<int, 10> c;
  const std::array<int, 10> sorted{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::merge(a.begin(), a.end(), b.begin(), b.end(), c.begin());

  for (size_t i = 0; i < c.size(); i++)
    ASSERT_EQ(c[i], sorted[i]);
}

TEST(ustl_algorithm, merge) {
  ustl::array<int, 5> a{1, 3, 5, 7, 9};
  ustl::array<int, 5> b{2, 4, 6, 8, 10};
  ustl::array<int, 10> c;
  const ustl::array<int, 10> sorted{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  ustl::merge(a.begin(), a.end(), b.begin(), b.end(), c.begin());

  for (size_t i = 0; i < c.size(); i++)
    ASSERT_EQ(c[i], sorted[i]);
}

TEST(std_algorithm, merge_comp_lambda) {
  std::array<int, 5> a{9, 7, 5, 3, 1};
  std::array<int, 5> b{10, 8, 6, 4, 2};
  std::array<int, 10> c;
  const std::array<int, 10> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  std::merge(a.begin(), a.end(), b.begin(), b.end(), c.begin(),
             [](int x, int y) { return x > y; });

  for (size_t i = 0; i < c.size(); i++)
    ASSERT_EQ(c[i], sorted[i]);
}

TEST(ustl_algorithm, merge_comp_lambda) {
  ustl::array<int, 5> a{9, 7, 5, 3, 1};
  ustl::array<int, 5> b{10, 8, 6, 4, 2};
  ustl::array<int, 10> c;
  const ustl::array<int, 10> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  ustl::merge(a.begin(), a.end(), b.begin(), b.end(), c.begin(),
              [](int x, int y) { return x > y; });

  for (size_t i = 0; i < c.size(); i++)
    ASSERT_EQ(c[i], sorted[i]);
}

TEST(std_algorithm, merge_comp_funcobj) {
  std::array<int, 5> a{9, 7, 5, 3, 1};
  std::array<int, 5> b{10, 8, 6, 4, 2};
  std::array<int, 10> c;
  const std::array<int, 10> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  std::merge(a.begin(), a.end(), b.begin(), b.end(), c.begin(),
             InverseComp<int>());

  for (size_t i = 0; i < c.size(); i++)
    ASSERT_EQ(c[i], sorted[i]);
}

TEST(ustl_algorithm, merge_comp_funcobj) {
  ustl::array<int, 5> a{9, 7, 5, 3, 1};
  ustl::array<int, 5> b{10, 8, 6, 4, 2};
  ustl::array<int, 10> c;
  const ustl::array<int, 10> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  ustl::merge(a.begin(), a.end(), b.begin(), b.end(), c.begin(),
              InverseComp<int>());

  for (size_t i = 0; i < c.size(); i++)
    ASSERT_EQ(c[i], sorted[i]);
}

TEST(std_algorithm, merge_comp_func_ptr) {
  std::array<int, 5> a{9, 7, 5, 3, 1};
  std::array<int, 5> b{10, 8, 6, 4, 2};
  std::array<int, 10> c;
  const std::array<int, 10> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  std::merge(a.begin(), a.end(), b.begin(), b.end(), c.begin(),
             inverse_comp);

  for (size_t i = 0; i < c.size(); i++)
    ASSERT_EQ(c[i], sorted[i]);
}

TEST(ustl_algorithm, merge_comp_func_ptr) {
  ustl::array<int, 5> a{9, 7, 5, 3, 1};
  ustl::array<int, 5> b{10, 8, 6, 4, 2};
  ustl::array<int, 10> c;
  const ustl::array<int, 10> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  ustl::merge(a.begin(), a.end(), b.begin(), b.end(), c.begin(),
              inverse_comp);

  for (size_t i = 0; i < c.size(); i++)
    ASSERT_EQ(c[i], sorted[i]);
}

// Sort

TEST(std_algorithm, sort) {
  std::vector<int> a{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int> b{9, 8, 7, 6, 5, 4, 3, 2, 1};
  const std::vector<int> sorted{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());

  for (size_t i = 0; i < a.size(); i++)
    ASSERT_EQ(a[i], sorted[i]);

  for (size_t i = 0; i < b.size(); i++)
    ASSERT_EQ(b[i], sorted[i]);
}

TEST(ustl_algorithm, sort) {
  ustl::vector<int> a{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  ustl::vector<int> b{9, 8, 7, 6, 5, 4, 3, 2, 1};
  const ustl::vector<int> sorted{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  ustl::sort(a.begin(), a.end());
  ustl::sort(b.begin(), b.end());

  for (size_t i = 0; i < a.size(); i++)
    ASSERT_EQ(a[i], sorted[i]);

  for (size_t i = 0; i < b.size(); i++)
    ASSERT_EQ(b[i], sorted[i]);
}

TEST(std_algorithm, sort_comp_lambda) {
  std::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const std::vector<int> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  std::sort(a.begin(), a.end(), [](int x, int y) { return x > y; });

  for (size_t i = 0; i < a.size(); i++)
    ASSERT_EQ(a[i], sorted[i]);
}

TEST(ustl_algorithm, sort_comp_lambda) {
  ustl::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const ustl::vector<int> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  ustl::sort(a.begin(), a.end(), [](int x, int y) { return x > y; });

  for (size_t i = 0; i < a.size(); i++)
    ASSERT_EQ(a[i], sorted[i]);
}

TEST(std_algorithm, sort_comp_funcobj) {
  std::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const std::vector<int> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  std::sort(a.begin(), a.end(), InverseComp<int>());

  for (size_t i = 0; i < a.size(); i++)
    ASSERT_EQ(a[i], sorted[i]);
}

TEST(ustl_algorithm, sort_comp_funcobj) {
  ustl::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const ustl::vector<int> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  ustl::sort(a.begin(), a.end(), InverseComp<int>());

  for (size_t i = 0; i < a.size(); i++)
    ASSERT_EQ(a[i], sorted[i]);
}

TEST(std_algorithm, sort_comp_func_ptr) {
  std::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const std::vector<int> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  std::sort(a.begin(), a.end(), &inverse_comp);

  for (size_t i = 0; i < a.size(); i++)
    ASSERT_EQ(a[i], sorted[i]);
}

TEST(ustl_algorithm, sort_comp_func_ptr) {
  ustl::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const ustl::vector<int> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  ustl::sort(a.begin(), a.end(), &inverse_comp);

  for (size_t i = 0; i < a.size(); i++)
    ASSERT_EQ(a[i], sorted[i]);
}

// Stable sort

TEST(std_algorithm, stable_sort) {
  std::vector<Comparable> a{
    Comparable(0, 0),
        Comparable(2),
        Comparable(0, 1),
        Comparable(1)
        };

  std::stable_sort(a.begin(), a.end());

  ASSERT_EQ(a[0].data, 0);
  ASSERT_EQ(a[0].nc,   0);
  ASSERT_EQ(a[1].data, 0);
  ASSERT_EQ(a[1].nc,   1);
  ASSERT_EQ(a[2].data, 1);
  ASSERT_EQ(a[3].data, 2);
}

TEST(ustl_algorithm, stable_sort) {
  ustl::vector<Comparable> a{
    Comparable(0, 0),
        Comparable(2),
        Comparable(0, 1),
        Comparable(1)
        };

  ustl::stable_sort(a.begin(), a.end());

  ASSERT_EQ(a[0].data, 0);
  ASSERT_EQ(a[0].nc,   0);
  ASSERT_EQ(a[1].data, 0);
  ASSERT_EQ(a[1].nc,   1);
  ASSERT_EQ(a[2].data, 1);
  ASSERT_EQ(a[3].data, 2);
}

// is_sorted()

TEST(std_algorithm, is_sorted) {
  std::vector<int> a{1, 2, 3, 4, 5};
  std::vector<int> b{5, 4, 3, 1, 2};
  std::vector<int> c{1, 2, 3, 5, 4};
  std::vector<int> d{1, 1, 1, 1, 1};

  ASSERT_TRUE(std::is_sorted(a.begin(), a.end()));
  ASSERT_FALSE(std::is_sorted(b.begin(), b.end()));
  ASSERT_FALSE(std::is_sorted(c.begin(), c.end()));
  ASSERT_TRUE(std::is_sorted(d.begin(), d.end()));
}

TEST(ustl_algorithm, is_sorted) {
  ustl::vector<int> a{1, 2, 3, 4, 5};
  ustl::vector<int> b{5, 4, 3, 1, 2};
  ustl::vector<int> c{1, 2, 3, 5, 4};
  ustl::vector<int> d{1, 1, 1, 1, 1};

  ASSERT_TRUE(ustl::is_sorted(a.begin(), a.end()));
  ASSERT_FALSE(ustl::is_sorted(b.begin(), b.end()));
  ASSERT_FALSE(ustl::is_sorted(c.begin(), c.end()));
  ASSERT_TRUE(ustl::is_sorted(d.begin(), d.end()));
}

// min()

TEST(std_algorithm, min) {
  ASSERT_EQ(1, std::min(5, 1));
  ASSERT_EQ(1, std::min(1, 5));

  ASSERT_EQ(5, std::min(5, 1, inverse_comp));
  ASSERT_EQ(5, std::min(1, 5, inverse_comp));

  ASSERT_EQ(1, std::min({3, 4, 1, 10, 1000}));
  ASSERT_EQ(1000, std::min({3, 4, 1, 10, 1000}, inverse_comp));
}

TEST(ustl_algorithm, min) {
  ASSERT_EQ(1, ustl::min(5, 1));
  ASSERT_EQ(1, ustl::min(1, 5));

  ASSERT_EQ(5, ustl::min(5, 1, inverse_comp));
  ASSERT_EQ(5, ustl::min(1, 5, inverse_comp));

  ASSERT_EQ(1, ustl::min({3, 4, 1, 10, 1000}));
  ASSERT_EQ(1000, ustl::min({3, 4, 1, 10, 1000}, inverse_comp));
}

// max()

TEST(std_algorithm, max) {
  ASSERT_EQ(5, std::max(5, 1));
  ASSERT_EQ(5, std::max(1, 5));

  ASSERT_EQ(1, std::max(5, 1, inverse_comp));
  ASSERT_EQ(1, std::max(1, 5, inverse_comp));

  ASSERT_EQ(1000, std::max({3, 4, 1, 10, 1000}));
  ASSERT_EQ(1, std::max({3, 4, 1, 10, 1000}, inverse_comp));
}

TEST(ustl_algorithm, max) {
  ASSERT_EQ(5, ustl::max(5, 1));
  ASSERT_EQ(5, ustl::max(1, 5));

  ASSERT_EQ(1, ustl::max(5, 1, inverse_comp));
  ASSERT_EQ(1, ustl::max(1, 5, inverse_comp));

  ASSERT_EQ(1000, ustl::max({3, 4, 1, 10, 1000}));
  ASSERT_EQ(1, ustl::max({3, 4, 1, 10, 1000}, inverse_comp));
}

// minmax()

TEST(std_algorithm, minmax) {
  ASSERT_EQ(1, std::minmax(5, 1).first);
  ASSERT_EQ(5, std::minmax(5, 1).second);
  ASSERT_EQ(1, std::minmax(1, 5).first);
  ASSERT_EQ(5, std::minmax(1, 5).second);

  ASSERT_EQ(5, std::minmax(5, 1, inverse_comp).first);
  ASSERT_EQ(1, std::minmax(5, 1, inverse_comp).second);
  ASSERT_EQ(5, std::minmax(1, 5, inverse_comp).first);
  ASSERT_EQ(1, std::minmax(1, 5, inverse_comp).second);

  ASSERT_EQ(1, std::minmax({3, 4, 1, 10, 1000}).first);
  ASSERT_EQ(1000, std::minmax({3, 4, 1, 10, 1000}).second);
  ASSERT_EQ(1000, std::minmax({3, 4, 1, 10, 1000}, inverse_comp).first);
  ASSERT_EQ(1, std::minmax({3, 4, 1, 10, 1000}, inverse_comp).second);
}

TEST(ustl_algorithm, minmax) {
  ASSERT_EQ(1, ustl::minmax(5, 1).first);
  ASSERT_EQ(5, ustl::minmax(5, 1).second);
  ASSERT_EQ(1, ustl::minmax(1, 5).first);
  ASSERT_EQ(5, ustl::minmax(1, 5).second);

  ASSERT_EQ(5, ustl::minmax(5, 1, inverse_comp).first);
  ASSERT_EQ(1, ustl::minmax(5, 1, inverse_comp).second);
  ASSERT_EQ(5, ustl::minmax(1, 5, inverse_comp).first);
  ASSERT_EQ(1, ustl::minmax(1, 5, inverse_comp).second);

  ASSERT_EQ(1, ustl::minmax({3, 4, 1, 10, 1000}).first);
  ASSERT_EQ(1000, ustl::minmax({3, 4, 1, 10, 1000}).second);
  ASSERT_EQ(1000, ustl::minmax({3, 4, 1, 10, 1000}, inverse_comp).first);
  ASSERT_EQ(1, ustl::minmax({3, 4, 1, 10, 1000}, inverse_comp).second);
}

// Array tests

TEST(std_array, size) {
  std::array<int, 10> a;
  std::array<char, 1> b;
  std::array<int, 100> c;
  std::array<int, 0> d;

  ASSERT_EQ(10u, a.size());
  ASSERT_EQ(1u, b.size());
  ASSERT_EQ(100u, c.size());
  ASSERT_EQ(0u, d.size());
}

TEST(ustl_array, size) {
  ustl::array<int, 10> a;
  ustl::array<char, 1> b;
  ustl::array<int, 100> c;
  ustl::array<int, 0> d;

  ASSERT_EQ(10u, a.size());
  ASSERT_EQ(1u, b.size());
  ASSERT_EQ(100u, c.size());
  ASSERT_EQ(0u, d.size());
}

TEST(std_array, max_size) {
  std::array<int, 10> a;
  std::array<char, 1> b;
  std::array<int, 100> c;
  std::array<int, 0> d;

  ASSERT_EQ(10u, a.max_size());
  ASSERT_EQ(a.size(), a.max_size());
  ASSERT_EQ(1u, b.max_size());
  ASSERT_EQ(b.size(), b.max_size());
  ASSERT_EQ(100u, c.max_size());
  ASSERT_EQ(c.size(), c.max_size());
  ASSERT_EQ(0u, d.max_size());
  ASSERT_EQ(d.size(), d.max_size());
}

TEST(ustl_array, max_size) {
  ustl::array<int, 10> a;
  ustl::array<char, 1> b;
  ustl::array<int, 100> c;
  ustl::array<int, 0> d;

  ASSERT_EQ(10u, a.max_size());
  ASSERT_EQ(a.size(), a.max_size());
  ASSERT_EQ(1u, b.max_size());
  ASSERT_EQ(b.size(), b.max_size());
  ASSERT_EQ(100u, c.max_size());
  ASSERT_EQ(c.size(), c.max_size());
  ASSERT_EQ(0u, d.max_size());
  ASSERT_EQ(d.size(), d.max_size());
}

TEST(ustl_array, empty) {
  ustl::array<int, 10> a;
  ustl::array<char, 1> b;
  ustl::array<int, 100> c;
  ustl::array<int, 0> d;

  ASSERT_EQ(false, a.empty());
  ASSERT_EQ(false, b.empty());
  ASSERT_EQ(false, c.empty());
  ASSERT_EQ(true,  d.empty());
}

TEST(std_array, empty) {
  std::array<int, 10> a;
  std::array<char, 1> b;
  std::array<int, 100> c;
  std::array<int, 0> d;

  ASSERT_EQ(false, a.empty());
  ASSERT_EQ(false, b.empty());
  ASSERT_EQ(false, c.empty());
  ASSERT_EQ(true,  d.empty());
}

TEST(ustl_array, subscript) {
  ustl::array<int, 2> a;

  // Set values:
  a[0] = 1;
  a[1] = 2;

  // Test values:
  ASSERT_EQ(1, a[0]);
  ASSERT_EQ(2, a[1]);
}

TEST(std_array, subscript) {
  std::array<int, 2> a;

  // Set values:
  a[0] = 1;
  a[1] = 2;

  // Test values:
  ASSERT_EQ(1, a[0]);
  ASSERT_EQ(2, a[1]);
}

TEST(std_array, at) {
  std::array<int, 2> a;
  std::array<int, 0> b;

  // Set values:
  a[0] = 1;
  a[1] = 2;

  // Test values:
  ASSERT_EQ(1, a.at(0));
  ASSERT_EQ(2, a.at(1));

  // Out of range tests:
  try {
    a.at(2) = 3;
    FAIL();
  } catch (std::out_of_range) {}

  try {
    b.at(0) = 1;
    FAIL();
  } catch (std::out_of_range) {}
}

TEST(ustl_array, at) {
  ustl::array<int, 2> a;
  ustl::array<int, 0> b;

  // Set values:
  a[0] = 1;
  a[1] = 2;

  // Test values:
  ASSERT_EQ(1, a.at(0));
  ASSERT_EQ(2, a.at(1));

  // Out of range tests:
  try {
    a.at(2) = 3;
    FAIL();
  } catch (std::out_of_range) {}

  try {
    b.at(0) = 1;
    FAIL();
  } catch (std::out_of_range) {}
}

TEST(std_array, front) {
  std::array<int, 3> a;

  // Set values:
  a[0] = 1;
  a[1] = 2;
  a[2] = 3;

  // Test values:
  ASSERT_EQ(1, a.front());
}

TEST(ustl_array, front) {
  ustl::array<int, 3> a;

  // Set values:
  a[0] = 1;
  a[1] = 2;
  a[2] = 3;

  // Test values:
  ASSERT_EQ(1, a.front());
}

TEST(std_array, back) {
  std::array<int, 3> a;

  // Set values:
  a[0] = 1;
  a[1] = 2;
  a[2] = 3;

  // Test values:
  ASSERT_EQ(3, a.back());
}

TEST(ustl_array, back) {
  ustl::array<int, 3> a;

  // Set values:
  a[0] = 1;
  a[1] = 2;
  a[2] = 3;

  // Test values:
  ASSERT_EQ(3, a.back());
}

TEST(std_array, data) {
  std::array<int, 3> a;

  // Set values:
  a[0] = 1;
  a[1] = 2;
  a[2] = 3;

  // Get pointer:
  auto *d = a.data();

  // Test values:
  ASSERT_EQ(1, d[0]);
  ASSERT_EQ(2, d[1]);
  ASSERT_EQ(3, d[2]);
}

TEST(ustl_array, data) {
  ustl::array<int, 3> a;

  // Set values:
  a[0] = 1;
  a[1] = 2;
  a[2] = 3;

  // Get pointer:
  auto *d = a.data();

  // Test values:
  ASSERT_EQ(1, d[0]);
  ASSERT_EQ(2, d[1]);
  ASSERT_EQ(3, d[2]);
}


//////////////
// Vectors: //
//////////////


// vector constructors

TEST(std_vector, constructors) {
  std::vector<int> a(3);
  std::vector<double> b(3, 3.5f);
  std::vector<char> c = { 'a', 'b', 'c' };

  for (size_t i = 0; i < 3; i++)
    ASSERT_EQ(0, a[i]);

  for (size_t i = 0; i < 3; i++)
    ASSERT_EQ(3.5f, b[i]);

  ASSERT_EQ('a', c[0]);
  ASSERT_EQ('b', c[1]);
  ASSERT_EQ('c', c[2]);
}

TEST(ustl_vector, constructors) {
  ustl::vector<int> a(3);
  ustl::vector<double> b(3, 3.5f);
  ustl::vector<char> c = { 'a', 'b', 'c' };

  for (size_t i = 0; i < 3; i++)
    ASSERT_EQ(0, a[i]);

  for (size_t i = 0; i < 3; i++)
    ASSERT_EQ(3.5f, b[i]);

  ASSERT_EQ('a', c[0]);
  ASSERT_EQ('b', c[1]);
  ASSERT_EQ('c', c[2]);
}


// vector capacity tests

TEST(std_vector_capacity, size) {
  std::vector<int> a;
  std::vector<int> b(5);
  std::vector<double> c(6, 3.5f);
  std::vector<char> d = { 'a', 'b', 'c' };

  ASSERT_EQ(0u, a.size());
  ASSERT_EQ(5u, b.size());
  ASSERT_EQ(6u, c.size());
  ASSERT_EQ(3u, d.size());
}

TEST(ustl_vector_capacity, size) {
  ustl::vector<int> a;
  ustl::vector<int> b(5);
  ustl::vector<double> c(6, 3.5f);
  ustl::vector<char> d = { 'a', 'b', 'c' };

  ASSERT_EQ(0u, a.size());
  ASSERT_EQ(5u, b.size());
  ASSERT_EQ(6u, c.size());
  ASSERT_EQ(3u, d.size());
}

TEST(std_vector_capacity, max_size) {
  std::vector<int> a;
  std::vector<int> b(5);
  std::vector<double> c(6, 3.5f);
  std::vector<char> d = { 'a', 'b', 'c' };

  ASSERT_TRUE(a.size() < a.max_size());
  ASSERT_TRUE(b.size() < b.max_size());
  ASSERT_TRUE(c.size() < c.max_size());

  ASSERT_EQ(a.max_size(), b.max_size());
}

TEST(ustl_vector_capacity, max_size) {
  ustl::vector<int> a;
  ustl::vector<int> b(5);
  ustl::vector<double> c(6, 3.5f);
  ustl::vector<char> d = { 'a', 'b', 'c' };

  ASSERT_TRUE(a.size() < a.max_size());
  ASSERT_TRUE(b.size() < b.max_size());
  ASSERT_TRUE(c.size() < c.max_size());

  ASSERT_EQ(a.max_size(), b.max_size());
}

TEST(std_vector_capacity, resize) {
  std::vector<int> a{1, 2, 3, 4, 5};

  size_t orig_cap = a.capacity();

  // Resize down. This won't reduce the capacity.
  a.resize(3);
  ASSERT_EQ(3u, a.size());
  ASSERT_EQ(orig_cap, a.capacity());


  a.push_back(4);
  ASSERT_EQ(4u, a.size());

  // Resize up.
  a.resize(5);
  a[4] = 11;

  ASSERT_EQ(4, a[3]);
  ASSERT_EQ(11, a[4]);

  // Resize and fill.
  a.resize(10, -1);
  ASSERT_EQ(-1, a.back());
}

TEST(ustl_vector_capacity, resize) {
  ustl::vector<int> a{1, 2, 3, 4, 5};

  size_t orig_cap = a.capacity();

  // Resize down. This won't reduce the capacity.
  a.resize(3);
  ASSERT_EQ(3u, a.size());
  ASSERT_EQ(orig_cap, a.capacity());


  a.push_back(4);
  ASSERT_EQ(4u, a.size());

  // Resize up.
  a.resize(5);
  a[4] = 11;

  ASSERT_EQ(4, a[3]);
  ASSERT_EQ(11, a[4]);

  // Resize and fill.
  a.resize(10, -1);
  ASSERT_EQ(-1, a.back());
}

TEST(std_vector_capacity, capacity) {
  std::vector<int> a;
  std::vector<int> b(5);
  std::vector<double> c(6, 3.5f);
  std::vector<char> d = { 'a', 'b', 'c' };

  ASSERT_TRUE(a.capacity() >= a.size());
  ASSERT_TRUE(b.capacity() >= b.size());
  ASSERT_TRUE(c.capacity() >= c.size());
  ASSERT_TRUE(d.capacity() >= d.size());
}

TEST(ustl_vector_capacity, capacity) {
  ustl::vector<int> a;
  ustl::vector<int> b(5);
  ustl::vector<double> c(6, 3.5f);
  ustl::vector<char> d = { 'a', 'b', 'c' };

  ASSERT_TRUE(a.capacity() >= a.size());
  ASSERT_TRUE(b.capacity() >= b.size());
  ASSERT_TRUE(c.capacity() >= c.size());
  ASSERT_TRUE(d.capacity() >= d.size());
}

TEST(std_vector_capacity, empty) {
  std::vector<int> a;
  std::vector<int> b(0);
  std::vector<int> c{1, 2, 3};

  ASSERT_TRUE(a.empty());
  ASSERT_TRUE(b.empty());
  ASSERT_FALSE(c.empty());
}

TEST(ustl_vector_capacity, empty) {
  ustl::vector<int> a;
  ustl::vector<int> b(0);
  ustl::vector<int> c{1, 2, 3};

  ASSERT_TRUE(a.empty());
  ASSERT_TRUE(b.empty());
  ASSERT_FALSE(c.empty());
}

TEST(std_vector_capacity, reserve) {
  std::vector<int> a(100);
  const size_t original_capacity = a.capacity();

  a.reserve(3);
  ASSERT_EQ(a.capacity(), original_capacity);

  a.reserve(10000);
  ASSERT_TRUE(a.capacity() > original_capacity);
}

TEST(ustl_vector_capacity, reserve) {
  ustl::vector<int> a(100);
  const size_t original_capacity = a.capacity();

  a.reserve(3);
  ASSERT_EQ(a.capacity(), original_capacity);

  a.reserve(10000);
  ASSERT_TRUE(a.capacity() > original_capacity);
}

TEST(std_vector_capacity, shrink_to_fit) {
  std::vector<int> a(1000);
  const size_t original_capacity = a.capacity();

  a[2] = 10;

  a.shrink_to_fit();
  ASSERT_EQ(original_capacity, a.capacity());

  a.resize(3);
  ASSERT_EQ(10, a[2]);
  ASSERT_TRUE(a.capacity() > a.size());
  a.shrink_to_fit();

  // Check that value remains unchanged.
  ASSERT_EQ(10, a[2]);

  // Check that capacity has shrunk.
  ASSERT_TRUE(a.capacity() < original_capacity);
}

TEST(ustl_vector_capacity, shrink_to_fit) {
  ustl::vector<int> a(1000);
  const size_t original_capacity = a.capacity();

  a[2] = 10;

  a.shrink_to_fit();
  ASSERT_EQ(original_capacity, a.capacity());

  a.resize(3);
  ASSERT_EQ(10, a[2]);
  ASSERT_TRUE(a.capacity() > a.size());
  a.shrink_to_fit();

  // Check that value remains unchanged.
  ASSERT_EQ(10, a[2]);

  // Check that capacity has shrunk.
  ASSERT_TRUE(a.capacity() < original_capacity);
}


// vector_iterators:

TEST(std_vector_iterators, begin) {
  std::vector<int> a{1, 2, 3, 4, 5};

  auto it = a.begin();
  ASSERT_EQ(1, *it++);
  ASSERT_EQ(2, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(4, it[1]);

  const auto cit = a.begin();
  ASSERT_EQ(1, *cit);
}

TEST(ustl_vector_iterators, begin) {
  ustl::vector<int> a{1, 2, 3, 4, 5};

  auto it = a.begin();
  ASSERT_EQ(1, *it++);
  ASSERT_EQ(2, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(4, it[1]);

  const auto cit = a.begin();
  ASSERT_EQ(1, *cit);
}

TEST(std_vector_iterators, rbegin) {
  std::vector<int> a{1, 2, 3, 4, 5};

  auto it = a.rbegin();
  ASSERT_EQ(5, *it++);
  ASSERT_EQ(4, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(2, it[1]);
}

TEST(ustl_vector_iterators, rbegin) {
  ustl::vector<int> a{1, 2, 3, 4, 5};

  auto it = a.rbegin();
  ASSERT_EQ(5, *it++);
  ASSERT_EQ(4, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(2, it[1]);
}

TEST(std_vector_iterators, cbegin) {
  std::vector<int> a{1, 2, 3, 4, 5};

  auto it = a.cbegin();
  ASSERT_EQ(1, *it++);
  ASSERT_EQ(2, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(4, it[1]);
}

TEST(ustl_vector_iterators, cbegin) {
  ustl::vector<int> a{1, 2, 3, 4, 5};

  auto it = a.cbegin();
  ASSERT_EQ(1, *it++);
  ASSERT_EQ(2, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(4, it[1]);
}

TEST(std_vector_iterators, crbegin) {
  std::vector<int> a{1, 2, 3, 4, 5};

  auto it = a.crbegin();
  ASSERT_EQ(5, *it++);
  ASSERT_EQ(4, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(2, it[1]);
}

TEST(ustl_vector_iterators, crbegin) {
  ustl::vector<int> a{1, 2, 3, 4, 5};

  auto it = a.crbegin();
  ASSERT_EQ(5, *it++);
  ASSERT_EQ(4, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(2, it[1]);
}

TEST(std_vector_iterators, end) {
  std::vector<int> a{1, 2, 3, 4, 5};
  std::vector<int> b;

  auto first = a.begin(), last = a.end();

  while (first != last)
    b.push_back(*first++);

  for (size_t i = 0; i < 5; i++)
    ASSERT_EQ(b[i], a[i]);
}

TEST(ustl_vector_iterators, end) {
  ustl::vector<int> a{1, 2, 3, 4, 5};
  ustl::vector<int> b;

  auto first = a.begin(), last = a.end();

  while (first != last)
    b.push_back(*first++);

  for (size_t i = 0; i < 5; i++)
    ASSERT_EQ(b[i], a[i]);
}

TEST(std_vector_iterators, rend) {
  std::vector<int> a{1, 2, 3, 4, 5};
  std::vector<int> b{5, 4, 3, 2, 1};
  std::vector<int> rev;

  auto first = a.rbegin(), last = a.rend();

  while (first != last)
    rev.push_back(*first++);

  for (size_t i = 0; i < 5; i++)
    ASSERT_EQ(rev[i], b[i]);
}

TEST(ustl_vector_iterators, rend) {
  ustl::vector<int> a{1, 2, 3, 4, 5};
  ustl::vector<int> b{5, 4, 3, 2, 1};
  ustl::vector<int> rev;

  auto first = a.rbegin(), last = a.rend();

  while (first != last)
    rev.push_back(*first++);

  for (size_t i = 0; i < 5; i++)
    ASSERT_EQ(rev[i], b[i]);
}

TEST(std_vector_iterators, cend) {
  std::vector<int> a{1, 2, 3, 4, 5};
  std::vector<int> b;

  auto first = a.cbegin(), last = a.cend();

  while (first != last)
    b.push_back(*first++);

  for (size_t i = 0; i < 5; i++)
    ASSERT_EQ(b[i], a[i]);
}

TEST(ustl_vector_iterators, cend) {
  ustl::vector<int> a{1, 2, 3, 4, 5};
  ustl::vector<int> b;

  auto first = a.cbegin(), last = a.cend();

  while (first != last)
    b.push_back(*first++);

  for (size_t i = 0; i < 5; i++)
    ASSERT_EQ(b[i], a[i]);
}

TEST(std_vector_iterators, crend) {
  std::vector<int> a{1, 2, 3, 4, 5};
  std::vector<int> b{5, 4, 3, 2, 1};
  std::vector<int> rev;

  auto first = a.crbegin(), last = a.crend();

  while (first != last)
    rev.push_back(*first++);

  for (size_t i = 0; i < 5; i++)
    ASSERT_EQ(rev[i], b[i]);
}

TEST(ustl_vector_iterators, crend) {
  ustl::vector<int> a{1, 2, 3, 4, 5};
  ustl::vector<int> b{5, 4, 3, 2, 1};
  ustl::vector<int> rev;

  auto first = a.crbegin(), last = a.crend();

  while (first != last)
    rev.push_back(*first++);

  for (size_t i = 0; i < 5; i++)
    ASSERT_EQ(rev[i], b[i]);
}


// vector element access:

TEST(std_vector_access, front) {
  std::vector<int> a(3);
  std::vector<double> b(3, 3.5f);
  std::vector<char> c = { 'a', 'b', 'c' };

  ASSERT_EQ(0, a.front());
  ASSERT_EQ(3.5, b.front());
  ASSERT_EQ('a', c.front());
}

TEST(ustl_vector_access, front) {
  ustl::vector<int> a(3);
  ustl::vector<double> b(3, 3.5f);
  ustl::vector<char> c = { 'a', 'b', 'c' };

  ASSERT_EQ(0, a.front());
  ASSERT_EQ(3.5, b.front());
  ASSERT_EQ('a', c.front());
}

TEST(std_vector_access, back) {
  std::vector<int> a(3);
  std::vector<double> b(3, 3.5f);
  std::vector<char> c = { 'a', 'b', 'c' };

  ASSERT_EQ(0, a.back());
  ASSERT_EQ(3.5, b.back());
  ASSERT_EQ('c', c.back());
}

TEST(ustl_vector_access, back) {
  ustl::vector<int> a(3);
  ustl::vector<double> b(3, 3.5f);
  ustl::vector<char> c = { 'a', 'b', 'c' };

  ASSERT_EQ(0, a.back());
  ASSERT_EQ(3.5, b.back());
  ASSERT_EQ('c', c.back());
}

TEST(std_vector_access, at) {
  std::vector<int> a(3);
  std::vector<double> b(3, 3.5f);
  std::vector<char> c = { 'a', 'b', 'c' };

  for (size_t i = 0; i < 3; i++) {
    ASSERT_EQ(0, a.at(i));
    ASSERT_EQ(a[i], a.at(i));
  }
  try {
    a.at(3);
    FAIL();
  } catch (std::out_of_range &e) {}

  for (size_t i = 0; i < 3; i++) {
    ASSERT_EQ(3.5f, b.at(i));
    ASSERT_EQ(b[i], b.at(i));
  }
  try {
    b.at(3);
    FAIL();
  } catch (std::out_of_range &e) {}

  ASSERT_EQ('a', c.at(0));
  ASSERT_EQ('b', c.at(1));
  ASSERT_EQ('c', c.at(2));
  try {
    c.at(3);
    FAIL();
  } catch (std::out_of_range &e) {}
}

TEST(ustl_vector_access, at) {
  ustl::vector<int> a(3);
  ustl::vector<double> b(3, 3.5f);
  ustl::vector<char> c = { 'a', 'b', 'c' };

  for (size_t i = 0; i < 3; i++) {
    ASSERT_EQ(0, a.at(i));
    ASSERT_EQ(a[i], a.at(i));
  }
  try {
    a.at(3);
    FAIL();
  } catch (std::out_of_range &e) {}

  for (size_t i = 0; i < 3; i++) {
    ASSERT_EQ(3.5f, b.at(i));
    ASSERT_EQ(b[i], b.at(i));
  }
  try {
    b.at(3);
    FAIL();
  } catch (std::out_of_range &e) {}

  ASSERT_EQ('a', c.at(0));
  ASSERT_EQ('b', c.at(1));
  ASSERT_EQ('c', c.at(2));
  try {
    c.at(3);
    FAIL();
  } catch (std::out_of_range &e) {}
}


// vector modifiers:

TEST(std_vector_modifiers, assign) {
  std::vector<int> a;
  std::vector<int> b{1, 2, 3};

  a.assign(b.begin(), b.end());
  for (size_t i = 0; i < 3; i++)
    ASSERT_EQ(a[i], b[i]);

  a.assign(std::vector<int>::size_type(100), 3);
  for (size_t i = 0; i < 100; i++)
    ASSERT_EQ(a[i], 3);

  a.assign({0, 1, 2, 3});
  for (size_t i = 0; i < 4; i++)
    ASSERT_EQ(a[i], static_cast<int>(i));

  ASSERT_EQ(a.size(), std::vector<int>::size_type(4));
}

TEST(ustl_vector_modifiers, assign) {
  ustl::vector<int> a;
  ustl::vector<int> b{1, 2, 3};

  a.assign(b.begin(), b.end());
  for (size_t i = 0; i < 3; i++)
    ASSERT_EQ(a[i], b[i]);

  a.assign(ustl::vector<int>::size_type(100), 3);
  for (size_t i = 0; i < 100; i++)
    ASSERT_EQ(a[i], 3);

  a.assign({0, 1, 2, 3});
  for (size_t i = 0; i < 4; i++)
    ASSERT_EQ(a[i], static_cast<int>(i));

  ASSERT_EQ(a.size(), ustl::vector<int>::size_type(4));
}

TEST(std_vector_modifiers, push_back) {
  std::vector<int> a;

  a.push_back(1);
  a.push_back(2);

  ASSERT_EQ(a.size(), std::vector<int>::size_type(2));
  ASSERT_EQ(a[0], 1);
  ASSERT_EQ(a[1], 2);
}

TEST(ustl_vector_modifiers, push_back) {
  ustl::vector<int> a;

  a.push_back(1);
  a.push_back(2);

  ASSERT_EQ(a.size(), ustl::vector<int>::size_type(2));
  ASSERT_EQ(a[0], 1);
  ASSERT_EQ(a[1], 2);
}

TEST(std_vector_modifiers, pop_back) {
  std::vector<int> a{1, 2, 3};

  a.pop_back();
  a.pop_back();
  ASSERT_EQ(a.size(), std::vector<int>::size_type(1));
  ASSERT_EQ(a[0], 1);
}

TEST(ustl_vector_modifiers, pop_back) {
  ustl::vector<int> a{1, 2, 3};

  a.pop_back();
  a.pop_back();
  ASSERT_EQ(a.size(), ustl::vector<int>::size_type(1));
  ASSERT_EQ(a[0], 1);
}

TEST(std_vector_modifiers, insert) {
  std::vector<int> a{1, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> b{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> c{1, 2, 3, 0, 0, 0, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> d{-1, -2, -3, 1, 2, 3, 0, 0, 0, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> e{
    -1, -2, -3, 1, 2, 3, 0, 0, 0, 4, 5, 6, 7, 8, 9, -1, -2, 10};

  std::vector<int> ins{-1, -2, -3};

  a.insert(a.begin() + 1, 2);
  ASSERT_EQ(a.size(), std::vector<int>::size_type(10));
  for (size_t i = 0; i < a.size(); i++)
    ASSERT_EQ(a[i], b[i]);

  a.insert(a.begin() + 3, std::vector<int>::size_type(3), static_cast<int>(0));
  ASSERT_EQ(a.size(), std::vector<int>::size_type(13));
  for (size_t i = 0; i < a.size(); i++)
    ASSERT_EQ(a[i], c[i]);

  a.insert(a.begin(), ins.begin(), ins.end());
  ASSERT_EQ(a.size(), std::vector<int>::size_type(16));
  for (size_t i = 0; i < a.size(); i++)
    ASSERT_EQ(a[i], d[i]);

  a.insert(a.end() - 1, {-1, -2});
  ASSERT_EQ(a.size(), std::vector<int>::size_type(18));
  for (size_t i = 0; i < a.size(); i++)
    ASSERT_EQ(a[i], e[i]);
}

TEST(ustl_vector_modifiers, insert) {
  ustl::vector<int> a{1, 3, 4, 5, 6, 7, 8, 9, 10};
  ustl::vector<int> b{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  ustl::vector<int> c{1, 2, 3, 0, 0, 0, 4, 5, 6, 7, 8, 9, 10};
  ustl::vector<int> d{-1, -2, -3, 1, 2, 3, 0, 0, 0, 4, 5, 6, 7, 8, 9, 10};
  ustl::vector<int> e{
    -1, -2, -3, 1, 2, 3, 0, 0, 0, 4, 5, 6, 7, 8, 9, -1, -2, 10};

  ustl::vector<int> ins{-1, -2, -3};

  a.insert(a.begin() + 1, 2);
  ASSERT_EQ(a.size(), ustl::vector<int>::size_type(10));
  for (size_t i = 0; i < a.size(); i++)
    ASSERT_EQ(a[i], b[i]);

  a.insert(a.begin() + 3, ustl::vector<int>::size_type(3), static_cast<int>(0));
  ASSERT_EQ(a.size(), ustl::vector<int>::size_type(13));
  for (size_t i = 0; i < a.size(); i++)
    ASSERT_EQ(a[i], c[i]);

  a.insert(a.begin(), ins.begin(), ins.end());
  ASSERT_EQ(a.size(), ustl::vector<int>::size_type(16));
  for (size_t i = 0; i < a.size(); i++)
    ASSERT_EQ(a[i], d[i]);

  a.insert(a.end() - 1, {-1, -2});
  ASSERT_EQ(a.size(), ustl::vector<int>::size_type(18));
  for (size_t i = 0; i < a.size(); i++)
    ASSERT_EQ(a[i], e[i]);
}

TEST(std_vector_modifiers, swap) {
  std::vector<int> a{1, 2, 3};
  std::vector<int> b{4, 5};

  a.swap(b);

  ASSERT_EQ(std::vector<int>::size_type(2), a.size());
  ASSERT_EQ(std::vector<int>::size_type(3), b.size());
  ASSERT_TRUE(vector_equal(a, {4, 5}));
  ASSERT_TRUE(vector_equal(b, {1, 2, 3}));
}

TEST(ustl_vector_modifiers, swap) {
  ustl::vector<int> a{1, 2, 3};
  ustl::vector<int> b{4, 5};

  a.swap(b);

  ASSERT_EQ(ustl::vector<int>::size_type(2), a.size());
  ASSERT_EQ(ustl::vector<int>::size_type(3), b.size());
  vector_equal(a, {4, 5});
  vector_equal(b, {1, 2, 3});
}

TEST(std_vector_modifiers, swap_overload) {
  std::vector<int> a{1, 2, 3};
  std::vector<int> b{4, 5};

  std::swap(a, b);

  ASSERT_EQ(std::vector<int>::size_type(2), a.size());
  ASSERT_EQ(std::vector<int>::size_type(3), b.size());
  vector_equal(a, {4, 5});
  vector_equal(b, {1, 2, 3});
}

TEST(ustl_vector_modifiers, swap_overload) {
  ustl::vector<int> a{1, 2, 3};
  ustl::vector<int> b{4, 5};

  ustl::swap(a, b);

  ASSERT_EQ(ustl::vector<int>::size_type(2), a.size());
  ASSERT_EQ(ustl::vector<int>::size_type(3), b.size());
  vector_equal(a, {4, 5});
  vector_equal(b, {1, 2, 3});
}

// relational operators

TEST(std_vector, rational_ops) {
  std::vector<int> a{1, 2, 3};
  std::vector<int> b{4, 5, 6};
  std::vector<int> c{1, 2, 3};
  std::vector<int> d{1, 2, 3, 4};

  ASSERT_TRUE(a == c);
  ASSERT_FALSE(a == b);
  ASSERT_TRUE(a != b);
  ASSERT_FALSE(a != c);

  ASSERT_TRUE(a < b);
  ASSERT_FALSE(b < a);
  ASSERT_TRUE(a <= c);
  ASSERT_TRUE(a <= b);
  ASSERT_TRUE(b > a);
  ASSERT_TRUE(b >= a);

  ASSERT_TRUE(a < d);
  ASSERT_FALSE(d >= b);
  ASSERT_TRUE(d >= a);
  ASSERT_TRUE(d > a);
}

TEST(ustl_vector, relational_ops) {
  ustl::vector<int> a{1, 2, 3};
  ustl::vector<int> b{4, 5, 6};
  ustl::vector<int> c{1, 2, 3};
  ustl::vector<int> d{1, 2, 3, 4};

  ASSERT_TRUE(a == c);
  ASSERT_FALSE(a == b);
  ASSERT_TRUE(a != b);
  ASSERT_FALSE(a != c);

  ASSERT_TRUE(a < b);
  ASSERT_FALSE(b < a);
  ASSERT_TRUE(a <= c);
  ASSERT_TRUE(a <= b);
  ASSERT_TRUE(b > a);
  ASSERT_TRUE(b >= a);

  ASSERT_TRUE(a < d);
  ASSERT_FALSE(d >= b);
  ASSERT_TRUE(d >= a);
  ASSERT_TRUE(d > a);
}


int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
