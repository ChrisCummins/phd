Notes and source files from reading Bjarne Stroustrup's "A Tour of
C++", 2013.

Notes:

* The noexcept keyword specifies functions which do not throw
  exceptions, and can be parameterised by (constant)
  conditionals. E.g.

     double &operator[](const int index) noexcept(CONDITIONAL) { ... }

* static_assert(A, S) - Print message S if A is false.

* Essential operations:

class X {
 public:
  X(arg1, arg2);              // "ordinary" constructor
  explicit X(arg);            // explicit constructor
  X();                        // default constructor
  ~X();                       // destructor
  X(const X &src);            // copy constructor
  X(X &&src);                 // move constructor
  X &operator=(const X &src); // copy assignment - clean up target then copy
  X &operator=(X &&src);      // move assignment - cleanup target then move
}
