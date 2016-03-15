# AC0 complexity

AC0 (formatted AC^0) is a complexity class in circuit complexity in
which all circuits have depth O(1) and polynomial size.

When used in the context of computational complexity, it can refer to
programs which could execute in finite time given an unlimited number
of processing elements. E.g.:

```
std::vector<int> v{1, 2, 3, 4, 5};
std::transform(v.begin(), v.end(), v.begin(),
               [](const auto& a){ return a * 2;});
```

Vectorisation could enable this to be computed in constant time.

[Further reading](https://en.wikipedia.org/wiki/AC0).
