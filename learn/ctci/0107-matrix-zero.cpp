/*
 * Write a method to perform basic string compression using the counts
 * of repeated characters. For example, the string aabcccccaaa would
 * become a2b1c5a3. If the "compressed" string would not become
 * smaller than the original string, your method should return the
 * original string.
 */
#include "./ctci.h"

#include <set>

template<typename Element>
void matrixZero1(Element *const m, const size_t n) {
    if (n == 0 || m == nullptr)
        return;

    // Determine which rows and columns to zero.
    std::set<size_t> rows, cols;
    for (size_t i = 0; i < n * n; i++) {
        if (m[i] == 0) {
            auto x = i % n;
            auto y = i / n;

            rows.insert(y);
            cols.insert(x);
        }
    }

    for (auto row : rows)
        for (size_t i = row * n; i < (row + 1) * n; i++)
            m[i] = 0;

    for (auto col : cols)
        for (size_t i = col % n; i < n * n; i += n)
            m[i] = 0;
}

// Unit tests

TEST(Permutation, matrixZero1) {
    int m1[] = {
        1, 2, 3,
        4, 0, 6,
        7, 8, 9
    };
    const int m2[] = {
        1, 0, 3,
        0, 0, 0,
        7, 0, 9
    };

    matrixZero1(m1, 3);
    for (size_t i = 0; i < 9; i++)
        ASSERT_EQ(m2[i], m1[i]);

    float m3a[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    };
    const float m3[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    };

    matrixZero1(m3a, 3);
    for (size_t i = 0; i < 9; i++)
        ASSERT_EQ(m3a[i], m3[i]);

    int m4[] = {
         0,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };
    const int m5[] = {
         0,  0,  0,  0,
         0,  6,  7,  8,
         0, 10, 11, 12,
         0, 14, 15, 16
    };

    matrixZero1(m4, 4);
    for (size_t i = 0; i < 16; i++)
        ASSERT_EQ(m4[i], m5[i]);
}

// Benchmarks

static const size_t lengthMin = 8;
static const size_t lengthMax = 10 << 10;

void BM_baseline(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range_x());
    int *m = new int[n];

    while (state.KeepRunning()) {
        for (size_t i = 0; i < n; i++)
            m[i] = static_cast<int>(arc4random());

        benchmark::DoNotOptimize(*m);
    }

    delete[] m;
}
BENCHMARK(BM_baseline)->Range(lengthMin, lengthMax);

void BM_matrixZero1(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range_x());
    int *m = new int[n];

    while (state.KeepRunning()) {
        for (size_t i = 0; i < n; i++)
            m[i] = static_cast<int>(arc4random());

        matrixZero1(m, n);
        benchmark::DoNotOptimize(*m);
    }

    delete[] m;
}
BENCHMARK(BM_matrixZero1)->Range(lengthMin, lengthMax);

CTCI_MAIN();
