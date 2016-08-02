#pragma once

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

}  // namespace detail

/**
 * \class container_impl
 * \brief A statically sized, n-dimensional container.
 *
 * Implementation class. Use the container<> class.
 *
 * \tparam T The element type.
 * \tparam size_ The number of elements in the container.
 * \tparam storage_class_ The type used to store elements.
 * \tparam d1 The number of elements in the top dimension.
 * \tparam dimensions... The number of elements in subsequent dimensions.
 */
template<typename T, size_t size_, typename storage_class_,
         unsigned int d1, unsigned int... dimensions>
class container_impl {
 private:
  using parent_type = container_impl<T, size_, storage_class_, dimensions...>;

  template<typename storage_class>
  class view_base {};

  template<typename parent_type_, typename storage_class, size_t ndim>
  class array_view;

 public:
  using value_type = typename parent_type::value_type;
  using storage_class = storage_class_;
  using size_type = typename parent_type::size_type;
  using dimen_size_type = unsigned long;

  container_impl() {}
  explicit container_impl(const value_type& fill) : _parent(fill) {}
  explicit container_impl(storage_class&& data) : _parent(std::move(data)) {}

  static constexpr size_type size() { return size_; }
  static constexpr dimen_size_type dimen_size() { return d1; }
  static constexpr size_type dimen() { return sizeof...(dimensions) + 1u; }
  constexpr size_type stride() const { return _parent.volume(); }
  constexpr size_type volume() const { return _parent.volume() * dimen_size(); }

  auto operator[](const dimen_size_type index) {
    using view_type = array_view<parent_type, storage_class, dimen() - 1>;
    return view_type(index * stride(), _parent);
  }

  auto operator[](const dimen_size_type index) const {
    using view_type = array_view<parent_type, storage_class, dimen() - 1>;
    return view_type(index * stride(), _parent);
  }

  value_type* data() { return _parent.data(); }
  const value_type* data() const { return _parent.data(); }

  typename storage_class::iterator begin() { return _parent.begin(); }
  typename storage_class::iterator end() { return _parent.end(); }

  typename storage_class::const_iterator begin() const {
    return _parent.begin();
  }

  typename storage_class::const_iterator end() const {
    return _parent.end();
  }

  auto& parent() { return _parent; }

 private:
  /**
   * View where n >= 3 dimensions
   */
  template<typename parent_type_, typename storage_class, size_t ndim>
  class array_view : view_base<storage_class> {
   public:
    array_view(size_t offset, parent_type_& parent)
        : _offset(offset), _parent(parent) {}

    auto operator[](const dimen_size_type index) {
      using view_type = array_view<decltype(parent()), storage_class, ndim - 1>;
      const auto flat_index = _offset + index * stride();
      assert(flat_index < size());
      return view_type(flat_index, parent());
    }

    auto operator[](const dimen_size_type index) const {
      using view_type = array_view<decltype(parent()), storage_class, ndim - 1>;
      const auto flat_index = _offset + index * stride();
      assert(flat_index < size());
      return view_type(flat_index, parent());
    }

    constexpr size_type size() const { return _parent.size(); }
    constexpr dimen_size_type dimen_size() const { return _parent.dimen_size(); }
    static constexpr size_type dimen() { return ndim; }
    constexpr size_type stride() const { return _parent.stride(); }
    auto& parent() { return _parent.parent(); }

   private:
    const size_type _offset;
    parent_type_& _parent;
  };

  /**
   * View for 2 dimensions
   */
  template<typename parent_type_, typename storage_class>
  class array_view<parent_type_, storage_class, 1> {
   public:
    array_view(size_type offset, parent_type_& parent)
        : _offset(offset), _parent(parent) {}

    value_type& operator[](const dimen_size_type index) {
      const auto flat_index = _offset + index * stride();
      assert(flat_index < size());
      return _parent[flat_index];
    }

    const value_type& operator[](const dimen_size_type index) const {
      const auto flat_index = _offset + index * stride();
      assert(flat_index < size());
      return _parent[flat_index];
    }

    constexpr size_type size() const { return _parent.size(); }
    constexpr size_type dimen_size() const { return _parent.dimen_size(); }
    static constexpr size_type dimen() { return 1u; }
    constexpr size_type stride() const { return _parent.stride(); }

   private:
    const size_type _offset;
    parent_type_& _parent;
  };

 private:
  parent_type _parent;
};


/**
 * \class container_impl
 * \brief A statically sized, n-dimensional container.
 *
 * Implementation class for 1D container. Use the container<> class.
 *
 * \tparam T The element type.
 * \tparam size_ The number of elements in the container.
 * \tparam storage_class_ The type used to store elements.
 * \tparam dn The number of elements in the final dimension.
 */
template<typename T, size_t size_, typename storage_class_, unsigned int dn>
class container_impl<T, size_, storage_class_, dn> {
  static_assert(size_ > 0, "zero size container_impl");

 public:
  using value_type = T;
  using storage_class = storage_class_;
  using size_type = size_t;
  using dimen_size_type = unsigned long;

  container_impl() {}
  explicit container_impl(const value_type& fill) { _data.fill(fill); }
  explicit container_impl(storage_class&& data) : _data(std::move(data)) {}

  static constexpr size_type size() { return size_; }
  static constexpr dimen_size_type dimen_size() { return dn; }
  static constexpr size_type dimen() { return 1u; }
  constexpr size_type stride() const { return 1u; }
  constexpr size_type volume() const { return dimen_size(); }

  value_type& operator[](const dimen_size_type index) {
    assert(index < size());
    return _data[index];
  }

  const value_type& operator[](const dimen_size_type index) const {
    assert(index < size());
    return _data[index];
  }

  value_type* data() { return _data.data(); }
  const value_type* data() const { return _data.data(); }

  typename storage_class::iterator begin() { return _data.begin(); }
  typename storage_class::iterator end() { return _data.end(); }

  typename storage_class::const_iterator begin() const { return _data.begin(); }
  typename storage_class::const_iterator end() const { return _data.end(); }

 private:
  storage_class _data;
};


/**
 * \class container
 * \brief A statically sized, n-dimensional container.
 *
 * Container class.
 *
 * \tparam T The element type.
 * \tparam d1 The number of elements in the top dimension.
 * \tparam dimensions... The number of elements in subsequent dimensions.
 */
template<typename T, size_t d1, unsigned int... dimensions>
class container : public container_impl<
  T, detail::product<d1, dimensions...>::call(),
  std::array<T, detail::product<d1, dimensions...>::call()>,
  d1, dimensions...> {
 public:
  using container_impl<T, detail::product<d1, dimensions...>::call(),
                       std::array<T, detail::product<d1, dimensions...>::call()>,
                       d1, dimensions...>::container_impl;
};

/**
 * \class container
 * \brief A statically sized, n-dimensional container.
 *
 * 1D container class.
 *
 * \tparam T The element type.
 * \tparam size The number of elements in the container.
 */
template<typename T, unsigned int size>
class container<T, size> : public container_impl<
  T, size, std::array<T, size>, size> {
 public:
  using container_impl<T, size, std::array<T, size>, size>::container_impl;
};

}  // namespace pat
