#include <array>
#include <iostream>
#include <utility>
#include <vector>

#include <phd/test>

#if 0
# define DEBUG(x) std::cerr << x << "\n";
#else
# define DEBUG(x)
#endif

namespace pat {

namespace detail {

template<size_t v1, size_t... vs>
struct product {
  static constexpr size_t call() {
    return v1 * product<vs...>::call();
  }
};

template<size_t v1, size_t v2>
struct product<v1, v2> {
  static constexpr size_t call() {
    return v1 * v2;
  }
};

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
  return detail::map<Kernel>(kernel);  // NOLINT
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
template<size_t size_, typename T, size_t d1, size_t... dimensions>
class container_impl {
 private:
  using parent_type = container_impl<size_, T, dimensions...>;

 public:
  using value_type = typename parent_type::value_type;
  using size_type = typename parent_type::size_type;
  using storage_class = typename parent_type::storage_class;

  container_impl() {}
  explicit container_impl(const value_type& fill) : _parent(fill) {}

  static constexpr size_type size() { return size_; }
  static constexpr size_type dimen_size() { return d1; }
  static constexpr size_type dimen() { return sizeof...(dimensions) + 1u; }
  constexpr size_type stride() const { return _parent.volume(); }
  constexpr size_t volume() const { return _parent.volume() * dimen_size(); }

  auto operator[](const size_t index) {
    using view_type = array_view<parent_type, storage_class, dimen() - 1>;
    return view_type(index * stride(), _parent);
  }

  value_type* data() { return _parent.data(); }
  const value_type* data() const { return _parent.data(); }

  auto begin() { return _parent.begin(); }
  auto end() { return _parent.end(); }

  auto begin() const { return _parent.begin(); }
  auto end() const { return _parent.end(); }

  auto& parent() { return _parent; }

 private:
  //
  template<typename storage_class>
  class view_base {};

  template<typename, size_t>
  friend class array_view;

  /**
   * View where n >= 3 dimensions
   */
  template<typename parent_type_, typename storage_class, size_t ndim>
  class array_view : view_base<storage_class> {
   public:
    array_view(size_t offset, parent_type_& parent)
        : _offset(offset), _parent(parent) {
      DEBUG("array_view<" << ndim << ">(" << offset << ")");
    }

    auto operator[](const size_t index) {
      const auto flat_index = _offset + index * stride();
      assert(flat_index < size());
      DEBUG("_data<" << size() << ">::array_view<" << dimen()
            << ":" << stride() << ">[" << _offset << " + " << index << "]");
      return array_view<decltype(parent()), storage_class, ndim - 1>(
          flat_index, parent());
    }

    constexpr size_t size() const { return _parent.size(); }
    constexpr size_type dimen_size() const { return _parent.dimen_size(); }
    static constexpr size_type dimen() { return ndim; }
    constexpr size_t stride() const { return _parent.stride(); }
    auto& parent() { return _parent.parent(); }

   private:
    const size_t _offset;
    parent_type_& _parent;
  };

  /**
   * View for 2 dimensions
   */
  template<typename parent_type_, typename storage_class>
  class array_view<parent_type_, storage_class, 1> {
   public:
    array_view(size_t offset, parent_type_& parent)
        : _offset(offset), _parent(parent) {
      // DEBUG("array_view<1>(" << offset << ")");
    }

    value_type& operator[](const size_t index) {
      const auto flat_index = _offset + index * stride();
      assert(flat_index < size());
      // DEBUG("_data<" << size() << ">::array_view<" << dimen()
      //       << ":" << stride() << ">[" << _offset << " + " << index << "]");
      return _parent[flat_index];
    }

    constexpr size_t size() const { return _parent.size(); }
    constexpr size_type dimen_size() const { return _parent.dimen_size(); }
    static constexpr size_type dimen() { return 1u; }
    constexpr size_t stride() const { return _parent.stride(); }

   private:
    const size_t _offset;
    parent_type_& _parent;
  };

 private:
  parent_type _parent;
};


template<size_t size_, typename T, size_t dn>
class container_impl<size_, T, dn> {
 public:
  using value_type = T;
  using size_type = size_t;
  using storage_class = std::array<T, size_>;

  container_impl() {}
  explicit container_impl(const value_type& fill) { _data.fill(fill); }

  static constexpr size_type size() { return size_; }
  static constexpr size_type dimen_size() { return dn; }
  static constexpr size_type dimen() { return 1u; }
  constexpr size_type stride() const { return 1u; }
  constexpr size_t volume() const { return dimen_size(); }

  value_type& operator[](const size_t index) {
    DEBUG("container_impl<" << size() << ">[" << index << "]");
    return _data[index];
  }

  const value_type& operator[](const size_t index) const {
    DEBUG("container_impl<" << size() << ">[" << index << "]");
    return _data[index];
  }

  value_type* data() { return _data.data(); }
  const value_type* data() const { return _data.data(); }

  auto begin() { return _data.begin(); }
  auto end() { return _data.end(); }

  auto begin() const { return _data.begin(); }
  auto end() const { return _data.end(); }

 private:
  storage_class _data;
};

template<typename T, size_t d1, size_t... dimensions>
class container : public container_impl<
  detail::product<d1, dimensions...>::call(), T, d1, dimensions...> {
 public:
  using container_impl<detail::product<d1, dimensions...>::call(),
                       T, d1, dimensions...>::container_impl;
};

template<typename T, size_t size>
class container<T, size> : public container_impl<size, T, size> {
 public:
  using container_impl<size, T, size>::container_impl;
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

TEST(patterns, container_1d) {
  pat::container<int, 10> a{0};

  ASSERT_EQ(10u, a.size());
  ASSERT_EQ(10u, a.dimen_size());
  ASSERT_EQ(1u, a.stride());
  ASSERT_EQ(a.dimen_size(), a.volume());

  for (auto& val : a)
    ASSERT_EQ(0, val);

  a[5] = 10;
  ASSERT_EQ(10, a[5]);
}

TEST(patterns, container_2d) {
  pat::container<int, 10, 10> a{2};

  ASSERT_EQ(100u, a.size());
  ASSERT_EQ(10u, a.stride());

  for (auto& val : a)
    ASSERT_EQ(2, val);

  a[5][3] = 10;
  ASSERT_EQ(10, a[5][3]);
}

TEST(patterns, container_3d) {
  pat::container<int, 5, 4, 3> a{0xCEC};

  ASSERT_EQ(60u, a.size());

  ASSERT_EQ(5u, a.dimen_size());
  ASSERT_EQ(4u, a[0].dimen_size());
  ASSERT_EQ(4u, a[1].dimen_size());
  ASSERT_EQ(3u, a[0][0].dimen_size());
  ASSERT_EQ(3u, a[1][2].dimen_size());

  for (auto& val : a)
    ASSERT_EQ(0xCEC, val);

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

  for (auto& val : a)
    ASSERT_EQ(5, val);

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
