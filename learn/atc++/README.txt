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

* If a class has a destructor, it probably needs user-defined or
  deleted copy and move operations.

* Use unique_prt<>() if you want to return a pointer to an object.

* An abstract class is one in which one or more methods are declared
  but not defined. Base classes can be either abstract or
  concrete. Abstract classes can't be instantiated.

* Use dynamic_cast to a pointer when failure to find the required
  class is valid (will return nullptr). Use dynamic_cast to a
  reference when failure to find class is considered a failure (will
  throw std::bad_cast).

* Keyword 'override' marks an intentionally overridden method.

* Return resource handlers (e.g. vector, shared_ptr, unique_ptr) by
  value, and rely on lightweight move operations for performance.
