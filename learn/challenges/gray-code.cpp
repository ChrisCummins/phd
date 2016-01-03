// Implementation of Gray code

#include <cstdio>
#include <iostream>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
template<size_t nbits>
class Gray {
 public:
  Gray() {
    _nbits = nbits;
    _msig = nbits - 1;
    for (size_t i = 0; i < size(); i++)
      _bits[i] = false;
  }

  Gray &operator++() {
    int tail = _msig ? _msig - 1 : nbits - 1;

    if (_bits[_msig] && _bits[tail]) {
      _bits[tail] = false;
    } else {
      _msig = (_msig + 1) % static_cast<int>(_nbits);

      if (_bits[tail]) {
        _bits[tail] = false;
      } else {
        _bits[_msig] = true;
      }
    }

    return *this;
  }

  Gray &operator++(int n) { return ++*this; }

  const bool &operator[](const size_t index) {
    if (index >= _nbits)
      throw std::out_of_range("index >= size()");
    return _bits[index];
  }

  const bool &operator[](const size_t index) const {
    if (index >= _nbits)
      throw std::out_of_range("index >= size()");
    return _bits[index];
  }

  size_t size() const { return _nbits; }

  friend std::ostream &operator<<(std::ostream &o, const Gray &g) {
    for (int i = g.size() - 1; i >= 0; i--)
      o << (g[static_cast<size_t>(i)] ? '1' : '0');
    return o;
  }

 private:
  size_t _nbits;
  int _msig;
  bool _bits[nbits];
};
#pragma GCC diagnostic pop  // -Wpadded


int main(int argc, char **argv) {
  auto g = Gray<5>();

  for (auto i = 1; i <= 25; i++) {
    g++;
    printf("%02d   ", i);
    std::cout << g << std::endl;
  }

  return 0;
}
