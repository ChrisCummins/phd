# N-dimensional for loops

1 dimensional (boo, boring!):

```c++
for (int i = 0; i < 5; i++) {
  std::cout << "i = " << i << std::endl;
}
```

2 dimensional:

```c++
for (int j = 0, i = 0;
     j < 5;
     i++, j += i / 5, i %= 5) {
  std::cout <<  "j = " << j
            << " i = " << i
            << std::endl;
}
```

3 dimensional:

```c++
for (int k = 0, j = 0, i = 0;
     k < 5;
     i++, j += i / 5, k += j / 5, i %= 5, j %= 5) {
  std::cout <<  "k = " << k
            << " j = " << j
            << " i = " << i
            << std::endl;
}
```

4 dimensional:

```c++
for (int l = 0, k = 0, j = 0, i = 0;
     l < 5;
     i++, j += i / 5, k += j / 5, l += k / 5, i %= 5, j %= 5, k %= 5) {
  std::cout <<  "l = " << l
            << " k = " << k
            << " j = " << j
            << " i = " << i
            << std::endl;
}
```
