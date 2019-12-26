#include <algorithm>
#include <array>
#include <ustl/algorithm>
#include <ustl/array>
#include <ustl/vector>
#include <vector>

#include "./tests.h"

bool inverse_comp(const int &a, const int &b) { return a > b; }

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

  for (size_t i = 0; i < 5; i++) ASSERT_EQ(in[i], out[i]);
}

TEST(ustl_algorithm, for_each) {
  ustl::vector<int> in{1, 2, 3, 4, 5};
  ustl::vector<int> out{2, 3, 4, 5, 6};

  ustl::for_each(in.begin(), in.end(), increment);

  for (size_t i = 0; i < 5; i++) ASSERT_EQ(in[i], out[i]);
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

  ASSERT_EQ(a.begin() + 6,
            std::find_end(a.begin(), a.end(), seq.begin(), seq.end()));
  ASSERT_EQ(a.end(),
            std::find_end(a.begin(), a.end(), notseq.begin(), notseq.end()));
}

TEST(ustl_algorithm, find_end) {
  ustl::vector<int> a{1, 2, 3, 4, 5, 6, 1, 2, 3, 9};
  ustl::vector<int> seq{1, 2, 3};
  ustl::vector<int> notseq{0, 1};

  ASSERT_EQ(a.begin() + 6,
            ustl::find_end(a.begin(), a.end(), seq.begin(), seq.end()));
  ASSERT_EQ(a.end(),
            ustl::find_end(a.begin(), a.end(), notseq.begin(), notseq.end()));
}

// find_first_of()

TEST(std_algorithm, find_first_of) {
  std::vector<int> a{1, 2, 3, 4, 5, 6, 1, 2, 3, 9};
  std::vector<int> seq{-1, 2, 3};
  std::vector<int> notseq{0, -1};

  ASSERT_EQ(a.begin() + 1,
            std::find_first_of(a.begin(), a.end(), seq.begin(), seq.end()));
  ASSERT_EQ(a.end(), std::find_first_of(a.begin(), a.end(), notseq.begin(),
                                        notseq.end()));
}

TEST(ustl_algorithm, find_first_of) {
  ustl::vector<int> a{1, 2, 3, 4, 5, 6, 1, 2, 3, 9};
  ustl::vector<int> seq{-1, 2, 3};
  ustl::vector<int> notseq{0, -1};

  ASSERT_EQ(a.begin() + 1,
            ustl::find_first_of(a.begin(), a.end(), seq.begin(), seq.end()));
  ASSERT_EQ(a.end(), ustl::find_first_of(a.begin(), a.end(), notseq.begin(),
                                         notseq.end()));
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

  ASSERT_EQ(a.begin() + 1,
            std::search(a.begin(), a.end(), seq.begin(), seq.end()));
  ASSERT_EQ(a.end(),
            std::search(a.begin(), a.end(), notseq.begin(), notseq.end()));
}

TEST(ustl_algorithm, search) {
  ustl::vector<int> a{0, 1, 2, 3, 4, 5, 1, 2, 3, 9};
  ustl::vector<int> seq{1, 2, 3};
  ustl::vector<int> notseq{-10, -11};

  ASSERT_EQ(a.begin() + 1,
            ustl::search(a.begin(), a.end(), seq.begin(), seq.end()));
  ASSERT_EQ(a.end(),
            ustl::search(a.begin(), a.end(), notseq.begin(), notseq.end()));
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

  for (size_t i = 0; i < c.size(); i++) ASSERT_EQ(c[i], sorted[i]);
}

TEST(ustl_algorithm, merge) {
  ustl::array<int, 5> a{1, 3, 5, 7, 9};
  ustl::array<int, 5> b{2, 4, 6, 8, 10};
  ustl::array<int, 10> c;
  const ustl::array<int, 10> sorted{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  ustl::merge(a.begin(), a.end(), b.begin(), b.end(), c.begin());

  for (size_t i = 0; i < c.size(); i++) ASSERT_EQ(c[i], sorted[i]);
}

TEST(std_algorithm, merge_comp_lambda) {
  std::array<int, 5> a{9, 7, 5, 3, 1};
  std::array<int, 5> b{10, 8, 6, 4, 2};
  std::array<int, 10> c;
  const std::array<int, 10> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  std::merge(a.begin(), a.end(), b.begin(), b.end(), c.begin(),
             [](int x, int y) { return x > y; });

  for (size_t i = 0; i < c.size(); i++) ASSERT_EQ(c[i], sorted[i]);
}

TEST(ustl_algorithm, merge_comp_lambda) {
  ustl::array<int, 5> a{9, 7, 5, 3, 1};
  ustl::array<int, 5> b{10, 8, 6, 4, 2};
  ustl::array<int, 10> c;
  const ustl::array<int, 10> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  ustl::merge(a.begin(), a.end(), b.begin(), b.end(), c.begin(),
              [](int x, int y) { return x > y; });

  for (size_t i = 0; i < c.size(); i++) ASSERT_EQ(c[i], sorted[i]);
}

TEST(std_algorithm, merge_comp_funcobj) {
  std::array<int, 5> a{9, 7, 5, 3, 1};
  std::array<int, 5> b{10, 8, 6, 4, 2};
  std::array<int, 10> c;
  const std::array<int, 10> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  std::merge(a.begin(), a.end(), b.begin(), b.end(), c.begin(),
             InverseComp<int>());

  for (size_t i = 0; i < c.size(); i++) ASSERT_EQ(c[i], sorted[i]);
}

TEST(ustl_algorithm, merge_comp_funcobj) {
  ustl::array<int, 5> a{9, 7, 5, 3, 1};
  ustl::array<int, 5> b{10, 8, 6, 4, 2};
  ustl::array<int, 10> c;
  const ustl::array<int, 10> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  ustl::merge(a.begin(), a.end(), b.begin(), b.end(), c.begin(),
              InverseComp<int>());

  for (size_t i = 0; i < c.size(); i++) ASSERT_EQ(c[i], sorted[i]);
}

TEST(std_algorithm, merge_comp_func_ptr) {
  std::array<int, 5> a{9, 7, 5, 3, 1};
  std::array<int, 5> b{10, 8, 6, 4, 2};
  std::array<int, 10> c;
  const std::array<int, 10> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  std::merge(a.begin(), a.end(), b.begin(), b.end(), c.begin(), inverse_comp);

  for (size_t i = 0; i < c.size(); i++) ASSERT_EQ(c[i], sorted[i]);
}

TEST(ustl_algorithm, merge_comp_func_ptr) {
  ustl::array<int, 5> a{9, 7, 5, 3, 1};
  ustl::array<int, 5> b{10, 8, 6, 4, 2};
  ustl::array<int, 10> c;
  const ustl::array<int, 10> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  ustl::merge(a.begin(), a.end(), b.begin(), b.end(), c.begin(), inverse_comp);

  for (size_t i = 0; i < c.size(); i++) ASSERT_EQ(c[i], sorted[i]);
}

// Sort

TEST(std_algorithm, sort) {
  std::vector<int> a{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int> b{9, 8, 7, 6, 5, 4, 3, 2, 1};
  const std::vector<int> sorted{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());

  for (size_t i = 0; i < a.size(); i++) ASSERT_EQ(a[i], sorted[i]);

  for (size_t i = 0; i < b.size(); i++) ASSERT_EQ(b[i], sorted[i]);
}

TEST(ustl_algorithm, sort) {
  ustl::vector<int> a{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  ustl::vector<int> b{9, 8, 7, 6, 5, 4, 3, 2, 1};
  const ustl::vector<int> sorted{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  ustl::sort(a.begin(), a.end());
  ustl::sort(b.begin(), b.end());

  for (size_t i = 0; i < a.size(); i++) ASSERT_EQ(a[i], sorted[i]);

  for (size_t i = 0; i < b.size(); i++) ASSERT_EQ(b[i], sorted[i]);
}

TEST(std_algorithm, sort_comp_lambda) {
  std::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const std::vector<int> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  std::sort(a.begin(), a.end(), [](int x, int y) { return x > y; });

  for (size_t i = 0; i < a.size(); i++) ASSERT_EQ(a[i], sorted[i]);
}

TEST(ustl_algorithm, sort_comp_lambda) {
  ustl::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const ustl::vector<int> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  ustl::sort(a.begin(), a.end(), [](int x, int y) { return x > y; });

  for (size_t i = 0; i < a.size(); i++) ASSERT_EQ(a[i], sorted[i]);
}

TEST(std_algorithm, sort_comp_funcobj) {
  std::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const std::vector<int> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  std::sort(a.begin(), a.end(), InverseComp<int>());

  for (size_t i = 0; i < a.size(); i++) ASSERT_EQ(a[i], sorted[i]);
}

TEST(ustl_algorithm, sort_comp_funcobj) {
  ustl::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const ustl::vector<int> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  ustl::sort(a.begin(), a.end(), InverseComp<int>());

  for (size_t i = 0; i < a.size(); i++) ASSERT_EQ(a[i], sorted[i]);
}

TEST(std_algorithm, sort_comp_func_ptr) {
  std::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const std::vector<int> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  std::sort(a.begin(), a.end(), &inverse_comp);

  for (size_t i = 0; i < a.size(); i++) ASSERT_EQ(a[i], sorted[i]);
}

TEST(ustl_algorithm, sort_comp_func_ptr) {
  ustl::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const ustl::vector<int> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  ustl::sort(a.begin(), a.end(), &inverse_comp);

  for (size_t i = 0; i < a.size(); i++) ASSERT_EQ(a[i], sorted[i]);
}

// Stable sort

TEST(std_algorithm, stable_sort) {
  std::vector<Comparable<int>> a{Comparable<int>(0, 0), Comparable<int>(2),
                                 Comparable<int>(0, 1), Comparable<int>(1)};

  std::stable_sort(a.begin(), a.end());

  ASSERT_EQ(a[0].data, 0);
  ASSERT_EQ(a[0].nc, 0);
  ASSERT_EQ(a[1].data, 0);
  ASSERT_EQ(a[1].nc, 1);
  ASSERT_EQ(a[2].data, 1);
  ASSERT_EQ(a[3].data, 2);
}

TEST(ustl_algorithm, stable_sort) {
  ustl::vector<Comparable<int>> a{Comparable<int>(0, 0), Comparable<int>(2),
                                  Comparable<int>(0, 1), Comparable<int>(1)};

  ustl::stable_sort(a.begin(), a.end());

  ASSERT_EQ(a[0].data, 0);
  ASSERT_EQ(a[0].nc, 0);
  ASSERT_EQ(a[1].data, 0);
  ASSERT_EQ(a[1].nc, 1);
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

TEST_MAIN();
