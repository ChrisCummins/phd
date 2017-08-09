#!/usr/bin/env python3
import sys
import numpy as np

from progressbar import ProgressBar
from typing import Tuple

def brute_force(stock_prices):
    """
    Time = O(n**2)
    Space = O(1)
    """
    maxprofit = 0
    n = len(stock_prices)

    for i in range(n - 1):
        buy_price = stock_prices[i]
        for j in range(i + 1, n):
            sell_price = stock_prices[j]
            profit = sell_price - buy_price
            if profit > maxprofit:
                maxprofit = profit
    return maxprofit


def maxdiff(a: list) -> Tuple[int, int]:
    if len(a) <= 1:
        return 0, 0, 0

    maxdiff = -1
    maxright = a[-1]
    iright = -1
    ileft = 0

    for i in range(len(a) - 2, -1, -1):
        if a[i] > maxright:
            maxright = a[i]
            iright = i
        else:
            diff = maxright - a[i]
            if diff > maxdiff:
                maxdiff = diff
                ileft = i

    return maxdiff, ileft, iright


def maxdiff_stocks(stock_prices: list):
    """
    Time: O(n)
    Space: O(1)
    """
    md, il, ir = maxdiff(stock_prices)
    return md


def main():
    a = [10, 7, 5, 8, 11, 9]

    print(brute_force(a))
    print(maxdiff_stocks(a))

    assert brute_force([]) == maxdiff_stocks([])

    testcases = np.random.randint(low=1, high=1000, size=1000)

    for testcase in ProgressBar()(testcases):
        a = np.random.randint(low=1, high=100, size=testcase)
        bf = brute_force(a)
        md = maxdiff_stocks(a)
        if bf != md:
            print(f"stocks: {a}")
            print(f"brute force: {bf}")
            print(f"max-diff:    {md}")
            sys.exit(1)


if __name__ == "__main__":
    main()
