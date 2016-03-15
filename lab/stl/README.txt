stl - A toy implementation of the C++17 Standard Template Library

These are my own implementations of the C++14 template library,
intended for educational rather than production use. Headers files are
included in 'include/ustl'. Unit tests in 'tests/', and benchmarks in
'benchmarks/'. A rough overview of my progress:

  include/ustl/algorithm        Fully implemented.
  include/ustl/array            Fully implemented.
  include/ustl/forward_list     Mostly implemented (75% functional).
  include/ustl/list             Mostly implemented (75% functional).
  include/ustl/map              Work in progress.
  include/ustl/stack            Fully implemented.
  include/ustl/unordered_map    Partly implemented (50% functional).
  include/ustl/vector           Mostly implemented (95% functional).

The implementations are a drop in replace for the standard library,
using the 'ustl' namespace. Example program:

    #include <ustl/algorithm>
    #include <ustl/forward_list>
    #include <ustl/vector>

    int main(int argc, char **argv) {
      ustl::vector<int> vec{1, 5, 4, 3, 2};
      ustl::sort(vec.begin(), vec.end());

      ustl::forward_list<int> list(vec.begin(), vec.begin() + 3);
      ustl::forward_list<int>::iterator it = list.begin();
      while (it != list.end())
        *it++ *= 2;
      list.reverse();

      return list.front();
    }

The above program exits with return code 6.
