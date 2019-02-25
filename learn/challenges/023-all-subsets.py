#!/usr/bin/env python3.6


def all_subsets(arr):

  def all_subset_helper(msk, i, n):
    if i == n:
      print([arr[i] for i in range(len(msk)) if msk[i]])
      return

    for j in range(i, n):
      msk[i], msk[j] = True, False
      all_subset_helper(msk, i + 1, n)
      msk[i], msk[j] = False, True

  msk = [False] * len(arr)
  all_subset_helper(msk, 0, len(arr))


def all_permutations(a):

  def permutor(a, i, n):
    if i == n:
      print(''.join(a))
      return

    for j in range(i, n):
      a[i], a[j] = a[j], a[i]
      permutor(a, i + 1, n)
      a[i], a[j] = a[j], a[i]  # backtrack

  permutor(a, 0, len(a))


def main():
  a = list('abc')
  all_permutations(a)

  a = [1, 2, 3]
  all_subsets(a)


if __name__ == '__main__':
  main()
