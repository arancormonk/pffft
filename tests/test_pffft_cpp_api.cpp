/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026 Mark Connor
 */

/*
  C++ wrapper contract test (C++98):
  - copy construction and assignment of pffft::Fft<> give independent
    instances (no shared setup or work memory -- these used to be
    freed twice) that compute the same results as the original;
  - convolveAccumulate() exists for every transform type and computes
    ab += a * b * scaling in the internal layout;
  - prepareLength() with an invalid length, or assigning an invalid
    instance, leaves an invalid instance (not the previous length);
  - AlignedVector allocation failure throws std::bad_alloc instead of
    handing std::vector a NULL or truncated block.
*/

#include "pffft/pffft.hpp"

#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <new>

namespace {

int num_failures = 0;

void fail(const char *type, int N, const char *what) {
  ++num_failures;
  std::printf("FAIL Fft<%s> N=%d: %s\n", type, N, what);
}

template<typename S> S urand() { return S(std::rand()) / S(RAND_MAX) * S(2) - S(1); }

template<typename T> struct Rand;
template<> struct Rand<float> { static float get() { return urand<float>(); } };
template<> struct Rand<double> { static double get() { return urand<double>(); } };
template<typename S> struct Rand< std::complex<S> > {
  static std::complex<S> get() { return std::complex<S>(urand<S>(), urand<S>()); }
};

template<typename T>
double max_abs_diff(const pffft::AlignedVector<T> &a, const pffft::AlignedVector<T> &b) {
  double d = 0;
  for (int i = 0; i < (int)a.size(); ++i) {
    const double v = std::abs(a[i] - b[i]);
    if (v > d) d = v;
  }
  return d;
}

template<typename T>
void test_type(const char *type, int N, int stackThresholdLen) {
  typedef pffft::Fft<T> Fft;
  typedef typename Fft::Scalar Scalar;
  // relative to the largest output magnitude: O(eps log2 N) for a correct result
  const double tol = 32 * std::log((double)N) / std::log(2.0) * (sizeof(Scalar) == sizeof(float) ? 1.2e-7 : 2.3e-16);

  Fft original(N, stackThresholdLen);
  if (!original.isValid()) { fail(type, N, "invalid"); return; }

  pffft::AlignedVector<T> x = original.valueVector(), y = original.valueVector();
  for (int i = 0; i < N; ++i) x[i] = Rand<T>::get();
  pffft::AlignedVector<typename Fft::Complex> X0 = original.spectrumVector(), X1 = original.spectrumVector();
  original.forward(x, X0);

  {
    Fft copy(original);  // copy construction
    if (!copy.isValid()) fail(type, N, "copy is invalid");
    copy.forward(x, X1);
    if (max_abs_diff(X0, X1) != 0) fail(type, N, "copy computes a different spectrum");
  }  // the copy's destructor must not release the original's memory

  {
    Fft assigned(Fft::minFFtsize(), stackThresholdLen);
    assigned = original;  // assignment, replacing a different length
    const Fft &alias = assigned;
    assigned = alias;     // self-assignment
    if (!assigned.isValid()) fail(type, N, "assigned instance is invalid");
    assigned.forward(x, X1);
    if (max_abs_diff(X0, X1) != 0) fail(type, N, "assigned instance computes a different spectrum");
  }

  original.forward(x, X1);
  if (max_abs_diff(X0, X1) != 0) fail(type, N, "original changed after its copies were destroyed");

  // convolveAccumulate: ab = D + scaling * A * B, back transform = N*d + scaling*N*(a (*) b)
  {
    pffft::AlignedVector<T> a = original.valueVector(), b = original.valueVector(), d = original.valueVector();
    pffft::AlignedVector<Scalar> A = original.internalLayoutVector(), B = original.internalLayoutVector();
    pffft::AlignedVector<Scalar> AB = original.internalLayoutVector(), P = original.internalLayoutVector();
    pffft::AlignedVector<T> got = original.valueVector(), want = original.valueVector();
    const Scalar scaling = Scalar(0.25);
    for (int i = 0; i < N; ++i) { a[i] = Rand<T>::get(); b[i] = Rand<T>::get(); d[i] = Rand<T>::get(); }
    original.forwardToInternalLayout(a, A);
    original.forwardToInternalLayout(b, B);
    original.forwardToInternalLayout(d, AB);
    original.convolveAccumulate(&A[0], &B[0], &AB[0], scaling);
    original.inverseFromInternalLayout(AB, got);
    // independent route: unscaled product, inverse, then combine in the signal domain
    original.convolve(&A[0], &B[0], &P[0]);
    original.inverseFromInternalLayout(P, want);
    for (int i = 0; i < N; ++i) want[i] = d[i] * Scalar(N) + want[i] * scaling;
    double peak = 0;
    for (int i = 0; i < N; ++i) if (std::abs(want[i]) > peak) peak = std::abs(want[i]);
    if (!(max_abs_diff(got, want) <= tol * peak)) fail(type, N, "convolveAccumulate result");
  }
}

template<typename T>
void test_invalid_lengths(const char *type) {
  typedef pffft::Fft<T> Fft;
  const int bad[] = { Fft::minFFtsize() / 2, Fft::minFFtsize() * 7 };
  for (int i = 0; i < 2; ++i) {
    if (bad[i] < 1 || Fft::isValidSize(bad[i])) continue;
    Fft f(64);
    if (f.prepareLength(bad[i])) fail(type, bad[i], "prepareLength accepted an invalid length");
    if (f.isValid() || f.getLength() != 0) fail(type, bad[i], "prepareLength kept the old length");
    Fft g(64), invalid(bad[i]);
    g = invalid;
    if (g.isValid() || g.getLength() != 0) fail(type, bad[i], "assigning an invalid instance kept the old length");
  }
  // a request no allocator can satisfy must throw, not return NULL
  // (more than max_size(): the int-truncated count used to allocate a few bytes)
  bool threw = false;
  pffft::PFAlloc<T> alloc;
  try {
    T *p = alloc.allocate(alloc.max_size() + 1);
    alloc.deallocate(p, 0);
  } catch (const std::bad_alloc &) {
    threw = true;
  }
  if (!threw) fail(type, 0, "PFAlloc::allocate(max_size() + 1) did not throw");
}

template<typename T>
void test_sizes(const char *type) {
  test_invalid_lengths<T>(type);
  const int sizes[] = { 32, 96, 480, 1024, 6144 };
  for (int i = 0; i < (int)(sizeof(sizes) / sizeof(sizes[0])); ++i) {
    test_type<T>(type, sizes[i], 4096);  // stack work below 4096, heap above
    test_type<T>(type, sizes[i], 0);     // always heap
  }
}

}  // namespace

int main() {
#ifdef PFFFT_ENABLE_FLOAT
  test_sizes<float>("float");
  test_sizes< std::complex<float> >("complex<float>");
#endif
#ifdef PFFFT_ENABLE_DOUBLE
  test_sizes<double>("double");
  test_sizes< std::complex<double> >("complex<double>");
#endif
  if (num_failures) {
    std::printf("%d FAILURES\n", num_failures);
    return 1;
  }
  std::printf("all checks passed\n");
  return 0;
}
