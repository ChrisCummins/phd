# Print type name at compile time

To check the type of a variable at compile time, adapt the following
snippet:

```
int x;
static_assert(decltype(x)::__err);
```

The error message from trying to access the nonexistent `__err` member
gives away the type:

```
'decltype(x)' (aka 'int') is not a class, namespace, or enumeration
```

Note that `static_assert` without a message is C++17, and `decltype`
is C++11.
