// Implementation of Gray code

#include <bitset>
#include <cstdio>
#include <iostream>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
template <size_t nbits>
class Gray {
 public:
  Gray() : _msig(nbits - 1), _bits(0) {}

  Gray &operator++() {
    size_t tail = _msig ? _msig - 1 : nbits - 1;

    if (_bits[_msig] && _bits[tail]) {
      _bits[tail] = false;
    } else {
      _msig = (_msig + 1) % nbits;

      if (_bits[tail])
        _bits[tail] = false;
      else
        _bits[_msig] = true;
    }

    return *this;
  }

  Gray &operator++(int n) { return ++*this; }

  const bool &operator[](const size_t index) const { return _bits[index]; }

  size_t size() const { return nbits; }

  friend std::ostream &operator<<(std::ostream &o, const Gray &g) {
    return o << g._bits;
  }

 private:
  size_t _msig;
  std::bitset<nbits> _bits;
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
