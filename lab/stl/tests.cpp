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

// Algorithm tests

// all_of

TEST(algorithm, all_of) {
  ustl::vector<int> a{2, 4, 6, 8};
  ustl::vector<int> b{2, 4, 6, 7, 8};

  ASSERT_TRUE(ustl::all_of(a.begin(), a.end(), is_even));
  ASSERT_FALSE(ustl::all_of(b.begin(), b.end(), is_even));
}

// any_of

TEST(algorithm, any_of) {
  ustl::vector<int> a{2, 4, 6, 8};
  ustl::vector<int> b{1, 3, 5, 7};
  ustl::vector<int> c{1, 3, 5, 6, 7};

  ASSERT_TRUE(ustl::any_of(a.begin(), a.end(), is_even));
  ASSERT_FALSE(ustl::any_of(b.begin(), b.end(), is_even));
  ASSERT_TRUE(ustl::any_of(c.begin(), c.end(), is_even));
}

// none_of

TEST(algorithm, none_of) {
  ustl::vector<int> a{2, 4, 6, 8};
  ustl::vector<int> b{1, 3, 5, 7};
  ustl::vector<int> c{1, 3, 5, 6, 7};

  ASSERT_FALSE(ustl::none_of(a.begin(), a.end(), is_even));
  ASSERT_TRUE(ustl::none_of(b.begin(), b.end(), is_even));
  ASSERT_FALSE(ustl::none_of(c.begin(), c.end(), is_even));
}

// for_each

TEST(algorithm, for_each) {
  ustl::vector<int> in{1, 2, 3, 4, 5};
  ustl::vector<int> out{2, 3, 4, 5, 6};

  ustl::for_each(in.begin(), in.end(), increment);

  for (size_t i = 0; i < 5; i++)
    ASSERT_EQ(in[i], out[i]);
}

// find

TEST(algorithm, find) {
  ustl::vector<int> a{1, 2, 3, 4, 5};
  ustl::vector<int> b{2, 4, 6, 8, 10};

  ASSERT_EQ(3, *ustl::find(a.begin(), a.end(), 3));
  ASSERT_EQ(b.end(), ustl::find(b.begin(), b.end(), 3));
}

// find_if()

TEST(algorithm, find_if) {
  ustl::vector<int> a{1, 3, 5, 7, 9};
  ustl::vector<int> b{2, 4, 6, 8, 10};

  ASSERT_EQ(a.end(), ustl::find_if(a.begin(), a.end(), is_even));
  ASSERT_EQ(b.begin(), ustl::find_if(b.begin(), b.end(), is_even));
}

// find_if_not()

TEST(algorithm, find_if_not) {
  ustl::vector<int> a{1, 3, 5, 7, 9};
  ustl::vector<int> b{2, 4, 6, 8, 10};

  ASSERT_EQ(a.begin(), ustl::find_if_not(a.begin(), a.end(), is_even));
  ASSERT_EQ(b.end(), ustl::find_if_not(b.begin(), b.end(), is_even));
}

// find_end()

TEST(algorithm, find_end) {
  ustl::vector<int> a{1, 2, 3, 4, 5, 6, 1, 2, 3, 9};
  ustl::vector<int> seq{1, 2, 3};
  ustl::vector<int> notseq{0, 1};

  ASSERT_EQ(&a[6], ustl::find_end(a.begin(), a.end(),
                                  seq.begin(), seq.end()));
  ASSERT_EQ(a.end(), ustl::find_end(a.begin(), a.end(),
                                    notseq.begin(), notseq.end()));

  // Check against behaviour of std library.
  ASSERT_EQ(ustl::find_end(a.begin(), a.end(),
                           seq.begin(), seq.end()),
            std::find_end(a.begin(), a.end(),
                       seq.begin(), seq.end()));
  ASSERT_EQ(ustl::find_end(a.begin(), a.end(),
                           notseq.begin(), notseq.end()),
            std::find_end(a.begin(), a.end(),
                          notseq.begin(), notseq.end()));
}

// find_first_of()

TEST(algorithm, find_first_of) {
  ustl::vector<int> a{1, 2, 3, 4, 5, 6, 1, 2, 3, 9};
  ustl::vector<int> seq{-1, 2, 3};
  ustl::vector<int> notseq{0, -1};

  ASSERT_EQ(&a[1], ustl::find_first_of(a.begin(), a.end(),
                                       seq.begin(), seq.end()));
  ASSERT_EQ(a.end(), ustl::find_first_of(a.begin(), a.end(),
                                         notseq.begin(), notseq.end()));

  // Check against behaviour of std library.
  ASSERT_EQ(ustl::find_first_of(a.begin(), a.end(),
                                seq.begin(), seq.end()),
            std::find_first_of(a.begin(), a.end(),
                               seq.begin(), seq.end()));
  ASSERT_EQ(ustl::find_first_of(a.begin(), a.end(),
                                notseq.begin(), notseq.end()),
            std::find_first_of(a.begin(), a.end(),
                               notseq.begin(), notseq.end()));
}

// adjacent_find()

TEST(algorithm, adjacent_find) {
  ustl::vector<int> a{1, 1, 2, 3, 4, 5, 5};
  ustl::vector<int> b{0, 1, 2, 3, 3, 4, 5};

  ASSERT_EQ(a.begin(), ustl::adjacent_find(a.begin(), a.end()));
  ASSERT_EQ(&b[3], ustl::adjacent_find(b.begin(), b.end()));
}

// count_if()

TEST(algorithm, count_if) {
  ustl::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  ustl::vector<int> b{1, 3, 5, 7, 9};

  ASSERT_EQ(5, ustl::count_if(a.begin(), a.end(), is_even));
  ASSERT_EQ(0, ustl::count_if(b.begin(), b.end(), is_even));
}

// count()

TEST(algorithm, count) {
  ustl::vector<int> a{1, 2, 3, 1, 1};
  ustl::vector<int> b{1, 3, 5, 7, 9};

  ASSERT_EQ(3, ustl::count(a.begin(), a.end(), 1));
  ASSERT_EQ(0, ustl::count(b.begin(), b.end(), -10));
}

// mismatch()

TEST(algorithm, mismatch) {
  ustl::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8};
  ustl::vector<int> b{1, 2, 3, 0, 0, 0, 0, 0};
  ustl::vector<int> c{0, 2, 3, 0, 0, 0, 0, 0};

  ASSERT_EQ(&a[3], ustl::mismatch(a.begin(), a.end(), b.begin()).first);
  ASSERT_EQ(&b[3], ustl::mismatch(a.begin(), a.end(), b.begin()).second);

  ASSERT_EQ(a.begin(), ustl::mismatch(a.begin(), a.end(), c.begin()).first);
  ASSERT_EQ(c.begin(), ustl::mismatch(a.begin(), a.end(), c.begin()).second);
}

// equal()

TEST(algorithm, equal) {
  ustl::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8};
  ustl::vector<int> b{1, 2, 3, 4, 5, 6, 7, 8};
  ustl::vector<int> c{0, 2, 3, 0, 0, 0, 0, 0};

  ASSERT_TRUE(ustl::equal(a.begin(), a.end(), b.begin()));
  ASSERT_FALSE(ustl::equal(a.begin(), a.end(), c.begin()));
}

// is_permutation()

TEST(algorithm, is_permutation) {
  ustl::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8};
  ustl::vector<int> b{1, 2, 3, 4, 5, 6, 7, 8};
  ustl::vector<int> c{8, 2, 3, 4, 5, 7, 6, 1};
  ustl::vector<int> d{0, 0, 0, 0, 1, 1, 2, 3};

  ASSERT_TRUE(ustl::equal(a.begin(), a.end(), b.begin()));
  ASSERT_FALSE(ustl::equal(a.begin(), a.end(), c.begin()));
  ASSERT_FALSE(ustl::equal(a.begin(), a.end(), d.begin()));
}

// search()

TEST(algorithm, search) {
  ustl::vector<int> a{0, 1, 2, 3, 4, 5, 1, 2, 3, 9};
  ustl::vector<int> seq{1, 2, 3};
  ustl::vector<int> notseq{-10, -11};

  ASSERT_EQ(&a[1], ustl::search(a.begin(), a.end(),
                                seq.begin(), seq.end()));
  ASSERT_EQ(a.end(), ustl::search(a.begin(), a.end(),
                                  notseq.begin(), notseq.end()));

  // Check against behaviour of std library.
  ASSERT_EQ(ustl::search(a.begin(), a.end(),
                         seq.begin(), seq.end()),
            std::search(a.begin(), a.end(),
                        seq.begin(), seq.end()));
  ASSERT_EQ(ustl::search(a.begin(), a.end(),
                         notseq.begin(), notseq.end()),
            std::search(a.begin(), a.end(),
                        notseq.begin(), notseq.end()));
}

// search_n()

TEST(algorithm, search_n) {
  ustl::vector<int> a{0, 0, 1, 1, 0, 1, 1, 0, 0, 0};

  ASSERT_EQ(&a[7], ustl::search_n(a.begin(), a.end(), 3, 0));
  ASSERT_EQ(a.end(), ustl::search_n(a.begin(), a.end(), 3, 1));

  // Check against behaviour of std library.
  ASSERT_EQ(ustl::search_n(a.begin(), a.end(), 3, 0),
            std::search_n(a.begin(), a.end(), 3, 0));
  ASSERT_EQ(ustl::search_n(a.begin(), a.end(), 3, 1),
            std::search_n(a.begin(), a.end(), 3, 1));
}

// Merge

TEST(algorithm, merge) {
  ustl::array<int, 5> a{1, 3, 5, 7, 9};
  ustl::array<int, 5> b{2, 4, 6, 8, 10};
  ustl::array<int, 10> c;
  const ustl::array<int, 10> sorted{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  ustl::merge(a.begin(), a.end(), b.begin(), b.end(), c.begin());

  for (size_t i = 0; i < c.size(); i++)
    ASSERT_EQ(c[i], sorted[i]);
}

TEST(algorithm, merge_comp_lambda) {
  ustl::array<int, 5> a{9, 7, 5, 3, 1};
  ustl::array<int, 5> b{10, 8, 6, 4, 2};
  ustl::array<int, 10> c;
  const ustl::array<int, 10> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  ustl::merge(a.begin(), a.end(), b.begin(), b.end(), c.begin(),
              [](int x, int y) { return x > y; });

  for (size_t i = 0; i < c.size(); i++)
    ASSERT_EQ(c[i], sorted[i]);
}

TEST(algorithm, merge_comp_funcobj) {
  ustl::array<int, 5> a{9, 7, 5, 3, 1};
  ustl::array<int, 5> b{10, 8, 6, 4, 2};
  ustl::array<int, 10> c;
  const ustl::array<int, 10> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  ustl::merge(a.begin(), a.end(), b.begin(), b.end(), c.begin(),
              InverseComp<int>());

  for (size_t i = 0; i < c.size(); i++)
    ASSERT_EQ(c[i], sorted[i]);
}

TEST(algorithm, merge_comp_funcptr) {
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

TEST(algorithm, sort) {
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

TEST(algorithm, sort_comp_lambda) {
  ustl::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const ustl::vector<int> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  ustl::sort(a.begin(), a.end(), [](int x, int y) { return x > y; });

  for (size_t i = 0; i < a.size(); i++)
    ASSERT_EQ(a[i], sorted[i]);
}

TEST(algorithm, sort_comp_funcobj) {
  ustl::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const ustl::vector<int> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  ustl::sort(a.begin(), a.end(), InverseComp<int>());

  for (size_t i = 0; i < a.size(); i++)
    ASSERT_EQ(a[i], sorted[i]);
}

TEST(algorithm, sort_comp_funcptr) {
  ustl::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const ustl::vector<int> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  ustl::sort(a.begin(), a.end(), &inverse_comp);

  for (size_t i = 0; i < a.size(); i++)
    ASSERT_EQ(a[i], sorted[i]);
}

// Stable sort

TEST(algorithm, stable_sort) {
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

TEST(algorithm, is_sorted) {
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

TEST(algorithm, min) {
  ASSERT_EQ(1, ustl::min(5, 1));
  ASSERT_EQ(1, ustl::min(1, 5));

  ASSERT_EQ(5, ustl::min(5, 1, inverse_comp));
  ASSERT_EQ(5, ustl::min(1, 5, inverse_comp));

  ASSERT_EQ(1, ustl::min({3, 4, 1, 10, 1000}));
  ASSERT_EQ(1000, ustl::min({3, 4, 1, 10, 1000}, inverse_comp));
}

// max()

TEST(algorithm, max) {
  ASSERT_EQ(5, ustl::max(5, 1));
  ASSERT_EQ(5, ustl::max(1, 5));

  ASSERT_EQ(1, ustl::max(5, 1, inverse_comp));
  ASSERT_EQ(1, ustl::max(1, 5, inverse_comp));

  ASSERT_EQ(1000, ustl::max({3, 4, 1, 10, 1000}));
  ASSERT_EQ(1, ustl::max({3, 4, 1, 10, 1000}, inverse_comp));
}

// minmax()

TEST(algorithm, minmax) {
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

TEST(array, size) {
  ustl::array<int, 10> a;
  ustl::array<char, 1> b;
  ustl::array<int, 100> c;
  ustl::array<int, 0> d;

  ASSERT_EQ(10u, a.size());
  ASSERT_EQ(1u, b.size());
  ASSERT_EQ(100u, c.size());
  ASSERT_EQ(0u, d.size());
}

TEST(array, max_size) {
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

TEST(array, empty) {
  ustl::array<int, 10> a;
  ustl::array<char, 1> b;
  ustl::array<int, 100> c;
  ustl::array<int, 0> d;

  ASSERT_EQ(false, a.empty());
  ASSERT_EQ(false, b.empty());
  ASSERT_EQ(false, c.empty());
  ASSERT_EQ(true,  d.empty());
}

TEST(array, subscript) {
  ustl::array<int, 2> a;

  // Set values:
  a[0] = 1;
  a[1] = 2;

  // Test values:
  ASSERT_EQ(1, a[0]);
  ASSERT_EQ(2, a[1]);
}

TEST(array, at) {
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

TEST(array, front) {
  ustl::array<int, 3> a;

  // Set values:
  a[0] = 1;
  a[1] = 2;
  a[2] = 3;

  // Test values:
  ASSERT_EQ(1, a.front());
}

TEST(array, back) {
  ustl::array<int, 3> a;

  // Set values:
  a[0] = 1;
  a[1] = 2;
  a[2] = 3;

  // Test values:
  ASSERT_EQ(3, a.back());
}

TEST(array, data) {
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

TEST(vector, constructors) {
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

TEST(vector_capacity, size) {
  ustl::vector<int> a;
  ustl::vector<int> b(5);
  ustl::vector<double> c(6, 3.5f);
  ustl::vector<char> d = { 'a', 'b', 'c' };

  ASSERT_EQ(0u, a.size());
  ASSERT_EQ(5u, b.size());
  ASSERT_EQ(6u, c.size());
  ASSERT_EQ(3u, d.size());
}

TEST(vector_capacity, max_size) {
  ustl::vector<int> a;
  ustl::vector<int> b(5);
  ustl::vector<double> c(6, 3.5f);
  ustl::vector<char> d = { 'a', 'b', 'c' };

  ASSERT_TRUE(a.size() < a.max_size());
  ASSERT_TRUE(b.size() < b.max_size());
  ASSERT_TRUE(c.size() < c.max_size());

  ASSERT_EQ(a.max_size(), b.max_size());
}

TEST(vector_capacity, resize) {
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

TEST(vector_capacity, capacity) {
  ustl::vector<int> a;
  ustl::vector<int> b(5);
  ustl::vector<double> c(6, 3.5f);
  ustl::vector<char> d = { 'a', 'b', 'c' };

  ASSERT_TRUE(a.capacity() >= a.size());
  ASSERT_TRUE(b.capacity() >= b.size());
  ASSERT_TRUE(c.capacity() >= c.size());
  ASSERT_TRUE(d.capacity() >= d.size());
}

TEST(vector_capacity, empty) {
  ustl::vector<int> a;
  ustl::vector<int> b(0);
  ustl::vector<int> c{1, 2, 3};

  ASSERT_TRUE(a.empty());
  ASSERT_TRUE(b.empty());
  ASSERT_FALSE(c.empty());
}

TEST(vector_capacity, reserve) {
  ustl::vector<int> a(100);
  const size_t original_capacity = a.capacity();

  a.reserve(3);
  ASSERT_EQ(a.capacity(), original_capacity);

  a.reserve(10000);
  ASSERT_TRUE(a.capacity() > original_capacity);
}

TEST(vector_capacity, shrink_to_fit) {
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

TEST(vector_iterators, begin) {
  ustl::vector<int> a{1, 2, 3, 4, 5};

  auto it = a.begin();
  ASSERT_EQ(1, *it++);
  ASSERT_EQ(2, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(4, it[1]);

  const auto cit = a.begin();
  ASSERT_EQ(1, *cit);
}

TEST(vector_iterators, rbegin) {
  ustl::vector<int> a{1, 2, 3, 4, 5};

  auto it = a.rbegin();
  ASSERT_EQ(5, *it++);
  ASSERT_EQ(4, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(2, it[1]);
}

TEST(vector_iterators, cbegin) {
  ustl::vector<int> a{1, 2, 3, 4, 5};

  auto it = a.cbegin();
  ASSERT_EQ(1, *it++);
  ASSERT_EQ(2, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(4, it[1]);
}

TEST(vector_iterators, crbegin) {
  ustl::vector<int> a{1, 2, 3, 4, 5};

  auto it = a.crbegin();
  ASSERT_EQ(5, *it++);
  ASSERT_EQ(4, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(2, it[1]);
}

TEST(vector_iterators, end) {
  ustl::vector<int> a{1, 2, 3, 4, 5};
  ustl::vector<int> b;

  auto first = a.begin(), last = a.end();

  while (first != last)
    b.push_back(*first++);

  for (size_t i = 0; i < 5; i++)
    ASSERT_EQ(b[i], a[i]);
}

TEST(vector_iterators, rend) {
  ustl::vector<int> a{1, 2, 3, 4, 5};
  ustl::vector<int> b{5, 4, 3, 2, 1};
  ustl::vector<int> rev;

  auto first = a.rbegin(), last = a.rend();

  while (first != last)
    rev.push_back(*first++);

  for (size_t i = 0; i < 5; i++)
    ASSERT_EQ(rev[i], b[i]);
}

TEST(vector_iterators, cend) {
  ustl::vector<int> a{1, 2, 3, 4, 5};
  ustl::vector<int> b;

  auto first = a.cbegin(), last = a.cend();

  while (first != last)
    b.push_back(*first++);

  for (size_t i = 0; i < 5; i++)
    ASSERT_EQ(b[i], a[i]);
}

TEST(vector_iterators, crend) {
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

TEST(vector_access, front) {
  ustl::vector<int> a(3);
  ustl::vector<double> b(3, 3.5f);
  ustl::vector<char> c = { 'a', 'b', 'c' };

  ASSERT_EQ(0, a.front());
  ASSERT_EQ(3.5, b.front());
  ASSERT_EQ('a', c.front());
}

TEST(vector_access, back) {
  ustl::vector<int> a(3);
  ustl::vector<double> b(3, 3.5f);
  ustl::vector<char> c = { 'a', 'b', 'c' };

  ASSERT_EQ(0, a.back());
  ASSERT_EQ(3.5, b.back());
  ASSERT_EQ('c', c.back());
}

TEST(vector_access, at) {
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

TEST(vector_modifiers, assign) {
  ustl::vector<int> a;
  std::vector<int> b{1, 2, 3};

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

TEST(vector_modifiers, push_back) {
  ustl::vector<int> a;

  a.push_back(1);
  a.push_back(2);

  ASSERT_EQ(a.size(), ustl::vector<int>::size_type(2));
  ASSERT_EQ(a[0], 1);
  ASSERT_EQ(a[1], 2);
}

TEST(vector_modifiers, pop_back) {
  ustl::vector<int> a{1, 2, 3};

  a.pop_back();
  a.pop_back();
  ASSERT_EQ(a.size(), ustl::vector<int>::size_type(1));
  ASSERT_EQ(a[0], 1);
}

TEST(vector_modifiers, insert) {
  ustl::vector<int> a{1, 3, 4, 5, 6, 7, 8, 9, 10};
  ustl::vector<int> b{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  ustl::vector<int> c{1, 2, 3, 0, 0, 0, 4, 5, 6, 7, 8, 9, 10};
  ustl::vector<int> d{-1, -2, -3, 1, 2, 3, 0, 0, 0, 4, 5, 6, 7, 8, 9, 10};
  ustl::vector<int> e{
    -1, -2, -3, 1, 2, 3, 0, 0, 0, 4, 5, 6, 7, 8, 9, -1, -2, 10};

  ustl::vector<int> ins{-1, -2, -3};

  a.insert(&a[1], 2);
  ASSERT_EQ(a.size(), ustl::vector<int>::size_type(10));
  for (size_t i = 0; i < a.size(); i++)
    ASSERT_EQ(a[i], b[i]);

  a.insert(&a[3], ustl::vector<int>::size_type(3), static_cast<int>(0));
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


int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
