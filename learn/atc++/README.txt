Notes and source files from reading Bjarne Stroustrup's "A Tour of
C++", 2013.

Notes:

* The noexcept keyword specifies functions which do not throw
  exceptions, and can be parameterised by (constant)
  conditionals. E.g.

     double &operator[](const int index) noexcept(CONDITIONAL) { ... }

* static_assert(A, S) - Print message S if A is false.
