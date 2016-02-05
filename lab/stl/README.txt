stl - A toy implementation of the C++14 Standard Template Library

These are my own implementations of the C++14 template library,
intended for educational rather than production use. Headers files are
included in 'include/ustl'. Unit tests in 'tests/', and benchmarks in
'benchmarks/'. A rough overview of my progress:

  include/ustl/algorithm        Fully implemented.
  include/ustl/array            Fully implemented.
  include/ustl/forward_list     Mostly implemented (75% functional).
  include/ustl/map              Work in progress.
  include/ustl/unordered_map    Work in progress.
  include/ustl/vector           Mostly implemented (95% functional).

The implementations are a drop in replace for the standard library,
using the 'ustl' namespace. Example program:

    #include <ustl/vector>
    #include <ustl/algorithm>

    int main(int argc, char **argv) {
      ustl::vector<int> vec{1, 5, 4, 3, 2};
      ustl::sort(vec.begin(), vec.end());

      ustl::forward_list<int> list(vec.begin(), vec.begin() + 3);
      ustl::for_each(list.begin(), list.end(),
                     [](int &x) { return 2 * x; });
      list.reverse();
      return *list.begin();
    }

The above program exits with return code 6.
