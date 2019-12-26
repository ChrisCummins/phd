#include <array>
#include <iostream>
#include <utility>
#include <vector>

#include "./container.hpp"
#include "labm8/cpp/test.h"

namespace pat {

namespace detail {

namespace {  // anonymous

//
// Base skeleton class.
//
template <typename Kernel>
class skeleton {
 public:
  explicit skeleton(const Kernel& kernel) : _kernel(kernel) {}

  Kernel kernel() { return _kernel; }

 private:
  Kernel _kernel;
};

}  // anonymous namespace

//
// Map skeleton.
//
template <typename Kernel>
class map : public skeleton<Kernel> {
 public:
  using skeleton<Kernel>::template skeleton;

  template <typename Container>
  Container operator()(const Container& in) {
    const auto kernel = this->kernel();
    Container ret;

    auto out = ret.begin();
    for (auto it = in.begin(); it != in.end(); ++it) {
      *out++ = kernel(*it);
    }

    return ret;
  }
};

//
// Zip skeleton.
//
template <typename Kernel>
class zip : public skeleton<Kernel> {
 public:
  using skeleton<Kernel>::template skeleton;

  template <typename Container>
  Container operator()(const Container& a, const Container& b) {
    const auto kernel = this->kernel();
    Container ret;

    assert(a.size() == b.size());

    auto out = ret.begin();
    for (auto it = std::make_pair(a.begin(), b.begin()); it.first != a.end();
         ++it.first, ++it.second) {
      *out++ = kernel(*it.first, *it.second);
    }

    return ret;
  }
};

//
// Reduce skeleton.
//
template <typename Kernel>
class reduce : public skeleton<Kernel> {
 public:
  using skeleton<Kernel>::template skeleton;

  template <typename Container, typename Return>
  typename Container::value_type operator()(const Container& in,
                                            const Return& base) {
    const auto kernel = this->kernel();
    auto ret = base;

    for (auto it = in.begin(); it != in.end(); ++it) ret = kernel(*it, ret);

    return ret;
  }
};

}  // namespace detail

//
// Skeleton factories.
//

template <typename Kernel>
auto map(const Kernel& kernel) {
  return detail::map<Kernel>(kernel);  // NOLINT
}

template <typename Kernel>
auto zip(const Kernel& kernel) {
  return detail::zip<Kernel>(kernel);
}

template <typename Kernel>
auto reduce(const Kernel& kernel) {
  return detail::reduce<Kernel>(kernel);
}

}  // namespace pat

///////////
// Tests //
///////////
template <typename Container>
void init(Container& in) {
  size_t i = 0;

  for (auto it = in.begin(); it != in.end(); ++it)
    *it = static_cast<typename Container::value_type>(i++);
}

TEST(patterns, container_1d) {
  pat::container<int, 10> a{0};

  ASSERT_EQ(10u, a.size());
  ASSERT_EQ(10u, a.dimen_size());
  ASSERT_EQ(1u, a.stride());
  ASSERT_EQ(a.dimen_size(), a.volume());

  for (auto& val : a) ASSERT_EQ(0, val);

  a[5] = 10;
  ASSERT_EQ(10, a[5]);
}

TEST(patterns, container_1d_data) {
  std::array<int, 10> d;
  d.fill(15);
  d[5] = 10;

  pat::container<int, d.size()> a{std::move(d)};

  ASSERT_EQ(10u, a.size());
  ASSERT_EQ(10u, a.dimen_size());
  ASSERT_EQ(1u, a.stride());
  ASSERT_EQ(a.dimen_size(), a.volume());

  ASSERT_EQ(15, a[0]);
  ASSERT_EQ(10, a[5]);
}

TEST(patterns, container_2d) {
  pat::container<int, 10, 10> a{2};

  ASSERT_EQ(100u, a.size());
  ASSERT_EQ(10u, a.stride());

  for (auto& val : a) ASSERT_EQ(2, val);

  a[5][3] = 10;
  ASSERT_EQ(10, a[5][3]);
}

TEST(patterns, container_2d_data) {
  std::array<int, 10 * 10> d;
  d.fill(15);

  pat::container<int, 10, 10> a{std::move(d)};

  ASSERT_EQ(100u, a.size());
  ASSERT_EQ(10u, a.dimen_size());

  for (auto& val : a) ASSERT_EQ(15, val);
}

TEST(patterns, container_3d) {
  pat::container<int, 5, 4, 3> a{0xCEC};

  ASSERT_EQ(60u, a.size());

  ASSERT_EQ(5u, a.dimen_size());
  ASSERT_EQ(4u, a[0].dimen_size());
  ASSERT_EQ(4u, a[1].dimen_size());
  ASSERT_EQ(3u, a[0][0].dimen_size());
  ASSERT_EQ(3u, a[1][2].dimen_size());

  for (auto& val : a) ASSERT_EQ(0xCEC, val);

  for (size_t k = 0; k < a.dimen_size(); ++k)
    for (size_t j = 0; j < a[k].dimen_size(); ++j)
      for (size_t i = 0; i < a[k][j].dimen_size(); ++i)
        ASSERT_EQ(0xCEC, a[k][j][i]);

  a[3][3][2] = 10;
  ASSERT_EQ(10, a[3][3][2]);
}

TEST(patterns, container_4d) {
  pat::container<int, 5, 4, 3, 2> a(5);

  ASSERT_EQ(120u, a.size());

  for (auto& val : a) ASSERT_EQ(5, val);

  for (size_t l = 0; l < a.dimen_size(); ++l)
    for (size_t k = 0; k < a[l].dimen_size(); ++k)
      for (size_t j = 0; j < a[l][k].dimen_size(); ++j)
        for (size_t i = 0; i < a[l][k][j].dimen_size(); ++i)
          ASSERT_EQ(5, a[l][k][j][i]);

  a[1][8][5][3] = 10;
  ASSERT_EQ(10, a[1][8][5][3]);
}

TEST(patterns, map1) {
  auto doubler = pat::map([=](const int& a) { return 2 * a; });

  pat::container<int, 100> a;
  init(a);

  auto b = doubler(a);

  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) ASSERT_EQ(a[i] * 2, b[i]);
}

TEST(patterns, zip1) {
  auto f = pat::zip([=](const int& a, const int& b) { return a + b; });

  pat::container<int, 100> a, b;
  init(a);
  init(b);

  auto c = f(a, b);

  ASSERT_EQ(a.size(), c.size());
  for (size_t i = 0; i < a.size(); ++i) ASSERT_EQ(a[i] + b[i], c[i]);
}

TEST(patterns, reduce1) {
  auto f = pat::reduce([=](const int& a, const int& b) { return a + b; });

  pat::container<int, 100> a;
  init(a);

  auto b = f(a, 0);

  auto gs = 0;
  for (size_t i = 0; i < a.size(); ++i) gs += a[i];

  ASSERT_EQ(gs, b);
}

TEST_MAIN();