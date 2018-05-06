#!/usr/bin/env python3

if __name__ == "__main__":
  for j in range(1, 13):
    for i in range(1, 13):
      print("{:4d} ".format(j * i), end="")
    print()
