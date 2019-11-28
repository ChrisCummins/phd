#ifndef HASH_MAP_H
#define HASH_MAP_H

#include <forward_list>
#include <ostream>
#include <utility>
#include <vector>

template <typename Key, typename T>
class hashmap {
 public:
  class iterator;
  using mapped_type = T;
  using key_type = Key;
  using value_type = std::pair<const key_type, mapped_type>;

 private:
  using bucket_type = std::forward_list<value_type>;
  using table_type = std::vector<bucket_type>;

 public:
  using local_iterator = typename bucket_type::iterator;

 private:
  using table_iterator = typename table_type::iterator;

 public:
  explicit hashmap(const size_t num_buckets = 8)
      : _table(num_buckets), _size(0) {}

  mapped_type& operator[](const key_type& key) {
    const size_t table_idx = _hasher(key) % _table.size();
    bucket_type& bucket = _table[table_idx];

    for (auto& pair : bucket) {
      if (pair.first == key) return pair.second;
    }

    bucket.emplace_front(key, mapped_type{});
    return bucket.front().second;
  }

  iterator begin() {
    auto it = _table.begin();
    while (it != _table.end() && (*it).empty()) it++;

    return iterator{_table, it, &(*it), (*it).begin()};
  }

  iterator end() { return iterator{_table, _table.end()}; }

 private:
  table_type _table;
  size_t _size;
  std::hash<key_type> _hasher;

 public:
  class iterator {
   public:
    iterator(table_type& table, table_iterator table_it, bucket_type* bucket,
             local_iterator bucket_it)
        : _table_it(table_it),
          _table(table),
          _bucket_it(bucket_it),
          _bucket(bucket) {}

    iterator(table_type& table, table_iterator table_it)
        : _table_it(table_it), _table(table) {}

    value_type& operator*() { return *_bucket_it; }

    iterator& operator++() {
      // No increment at the end of the table
      if (_table_it == _table.end()) return *this;

      _bucket_it++;

      // If we're at the end of a bucket, move to the next bucket in
      // the table.
      while (_bucket_it == _bucket->end()) {
        _table_it++;

        if (_table_it == _table.end()) {
          // We've reached the end of the table.
          _bucket = nullptr;
          _bucket_it = local_iterator{};
          break;
        } else {
          _bucket = &(*_table_it);
          _bucket_it = _bucket->begin();
        }
      }

      return *this;
    }

    iterator operator++(int n) {
      auto bucket_it = _bucket_it;
      auto bucket = _bucket;
      auto table_it = _table_it;

      ++*this;

      return iterator(_table, table_it, bucket, bucket_it);
    }

    friend bool operator==(const iterator& lhs, const iterator& rhs) {
      if (lhs._bucket && rhs._bucket) {
        return lhs._table_it == rhs._table_it && lhs._table == rhs._table &&
               lhs._bucket_it == rhs._bucket_it && lhs._bucket == rhs._bucket;
      } else {
        return lhs._table_it == rhs._table_it && lhs._table == rhs._table;
      }
    }

    friend bool operator!=(const iterator& lhs, const iterator& rhs) {
      return !(lhs == rhs);
    }

   private:
    table_iterator _table_it;
    table_type& _table;
    local_iterator _bucket_it;
    bucket_type* _bucket;
  };
};

template <typename Key, typename T>
std::ostream& operator<<(std::ostream& out, hashmap<Key, T>& map) {
  auto started = false;
  auto it = map.begin();
  while (it != map.end()) {
    if (started) out << "; ";
    started = true;
    out << '<' << (*it).first << ", " << (*it++).second << '>';
  }

  return out;
}

#endif  // HASH_MAP_H
