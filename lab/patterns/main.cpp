#include <array>
#include <iostream>
#include <utility>
#include <vector>

#include <phd/test>

namespace pat {

namespace detail {

namespace {  // anonymous

//
// Base skeleton class.
//
template<typename Kernel>
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
template<typename Kernel>
class map : public skeleton<Kernel> {
 public:
  using skeleton<Kernel>::template skeleton;

  template<typename Container>
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
template<typename Kernel>
class zip : public skeleton<Kernel> {
 public:
  using skeleton<Kernel>::template skeleton;

  template<typename Container>
  Container operator()(const Container& a, const Container& b) {
    const auto kernel = this->kernel();
    Container ret;

    assert(a.size() == b.size());

    auto out = ret.begin();
    for (auto it = std::make_pair(a.begin(), b.begin());
         it.first != a.end();
         ++it.first, ++it.second) {
      *out++ = kernel(*it.first, *it.second);
    }

    return ret;
  }
};


//
// Reduce skeleton.
//
template<typename Kernel>
class reduce : public skeleton<Kernel> {
 public:
  using skeleton<Kernel>::template skeleton;

  template<typename Container, typename Return>
  typename Container::value_type operator()(
      const Container& in,
      const Return& base) {
    const auto kernel = this->kernel();
    auto ret = base;

    for (auto it = in.begin(); it != in.end(); ++it)
      ret = kernel(*it, ret);

    return ret;
  }
};

}  // namespace detail


//
// Skeleton factories.
//

template<typename Kernel>
auto map(const Kernel& kernel) {
  return detail::map<Kernel>(kernel);
}


template<typename Kernel>
auto zip(const Kernel& kernel) {
  return detail::zip<Kernel>(kernel);
}


template<typename Kernel>
auto reduce(const Kernel& kernel) {
  return detail::reduce<Kernel>(kernel);
}


//
// Container class
//
template<typename T, size_t d1, size_t... dimensions>
class container {
 public:
  using value_type = typename container<T, dimensions...>::value_type;
  using size_type = typename container<T, dimensions...>::size_type;

  constexpr size_type stride() const { return d1; }

  constexpr size_type size() const { return _parent.size() * stride(); }

  auto& operator[](const size_t index) {
    return _parent;
  }

  auto begin() { return _parent.begin(); }
  auto end() { return _parent.end(); }

  auto begin() const { return _parent.begin(); }
  auto end() const { return _parent.end(); }

 private:
  container<T, dimensions...> _parent;
};


template<typename T, size_t dn>
class container<T, dn> {
 public:
  using value_type = T;
  using size_type = size_t;

  constexpr size_type stride() const { return 1u; }

  constexpr size_type size() const { return _data.size(); }

  value_type& operator[](const size_t index) {
    return _data[index];
  }

  const value_type& operator[](const size_t index) const {
    return _data[index];
  }

  auto begin() { return _data.begin(); }
  auto end() { return _data.end(); }

  auto begin() const { return _data.begin(); }
  auto end() const { return _data.end(); }

 private:
  std::array<T, dn> _data;
};

}  // namespace pat


///////////
// Tests //
///////////
template<typename Container>
void init(Container& in) {
  size_t i = 0;

  for (auto it = in.begin(); it != in.end(); ++it)
    *it = static_cast<typename Container::value_type>(i++);
}

TEST(patterns, container) {
  pat::container<int, 10> a;
  pat::container<int, 10, 10> b;
  pat::container<int, 20, 10, 10> c;

  ASSERT_EQ(10u, a.size());
  ASSERT_EQ(100u, b.size());
  ASSERT_EQ(2000u, c.size());

  ASSERT_EQ(1u, a.stride());
  ASSERT_EQ(10u, b.stride());
  ASSERT_EQ(20u, c.stride());
}

TEST(patterns, map1) {
  auto doubler = pat::map([=](const int& a) { return 2 * a; });

  pat::container<int, 100> a;
  init(a);

  auto b = doubler(a);

  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i)
    ASSERT_EQ(a[i] * 2, b[i]);
}

TEST(patterns, zip1) {
  auto f = pat::zip([=](const int& a, const int& b) { return a + b; });

  pat::container<int, 100> a, b;
  init(a); init(b);

  auto c = f(a, b);

  ASSERT_EQ(a.size(), c.size());
  for (size_t i = 0; i < a.size(); ++i)
    ASSERT_EQ(a[i] + b[i], c[i]);
}

TEST(patterns, reduce1) {
  auto f = pat::reduce([=](const int& a, const int& b) { return a + b; });

  pat::container<int, 100> a;
  init(a);

  auto b = f(a, 0);

  auto gs = 0;
  for (size_t i = 0; i < a.size(); ++i)
    gs += a[i];

  ASSERT_EQ(gs, b);
}

PHD_MAIN();
