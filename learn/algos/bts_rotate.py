#!/usr/bin/env python


def bts(arr, val, left, right):
  if right < left:
    return -1

  mid = (left + right) // 2
  if arr[mid] > val:
    right = mid - 1
  elif arr[mid] < val:
    left = mid + 1
  elif arr[mid] == val:
    return mid

  return bts(arr, val, left, right)


def find_pivot(arr, left, right):
  if right < left:
    return -1
  if right == left:
    return left

  mid = (left + right) // 2
  if mid < right and arr[mid] > arr[mid + 1]:
    return mid + 1
  if mid > left and arr[mid] < arr[mid - 1]:
    return mid
  if arr[left] >= arr[mid]:
    return find_pivot(arr, left, mid - 1)
  return find_pivot(arr, mid + 1, right)


def bts_rotated(arr, val):
  # find the pivot point
  i = find_pivot(arr, 0, len(arr) - 1)

  # search in the relevant part of the array
  if val >= arr[i]:
    return bts(arr, val, i, len(arr) - 1)
  else:
    return bts(arr, val, 0, i)


def main():
  tests = [
      ([5, 6, 7, 8, 9, 10, 1, 2, 3], 3, 8),
      ([5, 6, 7, 8, 9, 10, 1, 2, 3], 30, -1),
      ([30, 40, 50, 10, 20], 10, 3),
  ]

  for test in tests:
    arr, key, out = test
    actual_out = bts_rotated(arr, key)
    if actual_out != out:
      print(arr)
      print(key)
      print(out)
      print(actual_out)
      print()

  print("done.")


if __name__ == "__main__":
  main()
