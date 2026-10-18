/* Standalone performance analysis driver; distributed under LICENSE.txt.
   Compile twice, defining PERF_DOUBLE for the double precision driver.
   Setup, validation, allocation and checksums are outside timed regions. */
#include "pffft/pffft.h"
#include "pffft/pffft_double.h"
#include "pffft/pffastconv.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef PERF_DOUBLE
typedef double scalar;
typedef PFFFTD_Setup setup_t;
#define PF(name) pffftd_##name
#define PRECISION "double"
#else
typedef float scalar;
typedef PFFFT_Setup setup_t;
#define PF(name) pffft_##name
#define PRECISION "float"
#endif

typedef std::chrono::steady_clock perf_clock;
static volatile double checksum_sink;

struct Buffer {
  scalar *data;
  explicit Buffer(int n) : data(static_cast<scalar *>(PF(aligned_malloc)(n * sizeof(scalar)))) {
    if (!data) throw std::bad_alloc();
    std::memset(data, 0, n * sizeof(scalar));
  }
  ~Buffer() { PF(aligned_free)(data); }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
};

static double elapsed(perf_clock::time_point start) {
  return std::chrono::duration<double>(perf_clock::now() - start).count();
}

template<class Function>
static void measure(const char *kind, const std::string &op, int n, int repeats,
                    double seconds, int output_count, scalar *output, Function run) {
  size_t count = 1;
  double calibration;
  do {
    perf_clock::time_point start = perf_clock::now();
    for (size_t i = 0; i < count; ++i) run();
    calibration = elapsed(start);
    if (calibration < 0.01) count *= 2;
  } while (calibration < 0.01);
  count = std::max<size_t>(1, static_cast<size_t>(count * seconds / calibration));
  for (int repeat = 0; repeat < repeats; ++repeat) {
    std::memset(output, 0, output_count * sizeof(scalar));
    perf_clock::time_point start = perf_clock::now();
    for (size_t i = 0; i < count; ++i) run();
    double duration = elapsed(start);
    double checksum = 0;
    for (int i = 0; i < output_count; ++i) {
      if (!std::isfinite(output[i])) throw std::runtime_error("nonfinite output");
      checksum += output[i];
    }
    checksum_sink = checksum;
    std::printf("%s,%s,%s,%s,%d,%d,%zu,%.9f,%.3f,%.9g\n", PRECISION,
                PF(simd_arch)(), kind, op.c_str(), n, repeat, count,
                duration, duration * 1e9 / count, checksum);
    std::fflush(stdout);
  }
}

static void fill(scalar *x, int n) {
  unsigned state = 1729;
  for (int i = 0; i < n; ++i) {
    state = state * 1664525u + 1013904223u;
    x[i] = static_cast<scalar>(static_cast<int>(state >> 16) - 32768) / 65536;
  }
}

static void check_equal(const scalar *a, const scalar *b, int n, double scale) {
  double tolerance = sizeof(scalar) == sizeof(float) ? 2e-5 : 2e-12;
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(a[i]) || std::fabs(a[i] * scale - b[i]) > tolerance) {
      std::fprintf(stderr, "validation failed at %d: %.17g vs %.17g\n",
                   i, a[i] * scale, static_cast<double>(b[i]));
      throw std::runtime_error("FFT validation failed");
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s real|complex OP N REPEATS SECONDS\n"
                 "OP: forward backward ordered-forward ordered-backward reorder "
                 "inplace-forward inplace-backward inplace-ordered-forward "
                 "inplace-ordered-backward zconv zconv-scale zconv-acc zp "
                 "pipeline fastconv fastconv-single\n", argv[0]);
    return 2;
  }
  try {
    const std::string kind(argv[1]), op(argv[2]);
    const int n = std::atoi(argv[3]), repeats = std::atoi(argv[4]);
    const double seconds = std::atof(argv[5]);
    const bool complex = kind == "complex";
    if ((kind != "real" && !complex) || n < 32 || n > (1 << 22) ||
        repeats < 1 || repeats > 100 || !(seconds > 0 && seconds <= 60))
      throw std::runtime_error("invalid arguments");
    const int count = complex ? 2 * n : n;
    Buffer input(count), spectrum(count), filter(count), product(count),
           output(count), work(count), ordered(count);
    fill(input.data, count);

#ifndef PERF_DOUBLE
    if (op == "fastconv" || op == "fastconv-single") {
      const int filter_len = n / 4 + 1;
      std::vector<float> coeffs(filter_len);
      fill(coeffs.data(), filter_len);
      for (int j = 0; j < filter_len; ++j) coeffs[j] /= filter_len;
      int block = n;
      int flags = complex ? PFFASTCONV_CPLX_INP_OUT : 0;
      if (op == "fastconv-single") flags |= PFFASTCONV_CPLX_SINGLE_FFT;
      PFFASTCONV_Setup *conv = pffastconv_new_setup(coeffs.data(), filter_len, &block, flags);
      if (!conv || block != n) throw std::runtime_error("unexpected convolution block size");
      int produced = pffastconv_apply(conv, input.data, n, output.data, 0);
      if (produced != n - filter_len + 1) throw std::runtime_error("incorrect output count");
      for (int j = 0; j < produced; ++j) {
        for (int part = 0; part < (complex ? 2 : 1); ++part) {
          double ref = 0;
          for (int k = 0; k < filter_len; ++k)
            ref += input.data[(complex ? 2 : 1) * (j + k) + part] *
                   static_cast<double>(coeffs[filter_len - 1 - k]);
          if (std::fabs(ref - output.data[(complex ? 2 : 1) * j + part]) > 2e-5)
            throw std::runtime_error("convolution validation failed");
        }
      }
      std::fprintf(stderr, "filter=%d block=%d produced=%d\n", filter_len, n, produced);
      measure(argv[1], op, n, repeats, seconds, produced * (complex ? 2 : 1), output.data,
              [&]() { pffastconv_apply(conv, input.data, n, output.data, 0); });
      pffastconv_destroy_setup(conv);
      return 0;
    }
#endif

    setup_t *setup = PF(new_setup)(n, complex ? PFFFT_COMPLEX : PFFFT_REAL);
    if (!setup) throw std::runtime_error("unsupported FFT size");
    PF(transform)(setup, input.data, spectrum.data, work.data, PFFFT_FORWARD);
    PF(transform)(setup, spectrum.data, output.data, work.data, PFFFT_BACKWARD);
    check_equal(output.data, input.data, count, 1.0 / n);
    PF(transform_ordered)(setup, input.data, ordered.data, work.data, PFFFT_FORWARD);
    PF(zreorder)(setup, spectrum.data, output.data, PFFFT_FORWARD);
    check_equal(output.data, ordered.data, count, 1.0);
    PF(transform_ordered)(setup, ordered.data, output.data, work.data, PFFFT_BACKWARD);
    check_equal(output.data, input.data, count, 1.0 / n);
    std::memcpy(output.data, input.data, count * sizeof(scalar));
    PF(transform)(setup, output.data, output.data, work.data, PFFFT_FORWARD);
    check_equal(output.data, spectrum.data, count, 1.0);
    PF(transform)(setup, output.data, output.data, work.data, PFFFT_BACKWARD);
    check_equal(output.data, input.data, count, 1.0 / n);
    std::memcpy(output.data, input.data, count * sizeof(scalar));
    PF(transform_ordered)(setup, output.data, output.data, work.data, PFFFT_FORWARD);
    check_equal(output.data, ordered.data, count, 1.0);
    PF(transform_ordered)(setup, output.data, output.data, work.data, PFFFT_BACKWARD);
    check_equal(output.data, input.data, count, 1.0 / n);
    for (int i = 0; i < count; ++i) filter.data[i] = spectrum.data[i] / n;
    if (op == "zp" && (complex || PF(zconvert_zp)(setup, spectrum.data, filter.data, 1) != 0))
      throw std::runtime_error("zero phase requires real FFT");

    /* Repeated unnormalized in-place transforms would overflow random input.
       Validate with random data above, then time zero input (including warmup)
       without adding a reset/copy to the measured transform. */
    if (op.find("inplace-") == 0) std::memset(output.data, 0, count * sizeof(scalar));

#define MEASURE(...) measure(argv[1], op, n, repeats, seconds, count, output.data, [&]() { __VA_ARGS__; })
    if (op == "inplace-forward")
      MEASURE(PF(transform)(setup, output.data, output.data, work.data, PFFFT_FORWARD));
    else if (op == "inplace-backward")
      MEASURE(PF(transform)(setup, output.data, output.data, work.data, PFFFT_BACKWARD));
    else if (op == "inplace-ordered-forward")
      MEASURE(PF(transform_ordered)(setup, output.data, output.data, work.data, PFFFT_FORWARD));
    else if (op == "inplace-ordered-backward")
      MEASURE(PF(transform_ordered)(setup, output.data, output.data, work.data, PFFFT_BACKWARD));
    else if (op == "forward")
      MEASURE(PF(transform)(setup, input.data, output.data, work.data, PFFFT_FORWARD));
    else if (op == "backward")
      MEASURE(PF(transform)(setup, spectrum.data, output.data, work.data, PFFFT_BACKWARD));
    else if (op == "ordered-forward")
      MEASURE(PF(transform_ordered)(setup, input.data, output.data, work.data, PFFFT_FORWARD));
    else if (op == "ordered-backward")
      MEASURE(PF(transform_ordered)(setup, ordered.data, output.data, work.data, PFFFT_BACKWARD));
    else if (op == "reorder")
      MEASURE(PF(zreorder)(setup, spectrum.data, output.data, PFFFT_FORWARD));
    else if (op == "zconv")
      MEASURE(PF(zconvolve)(setup, spectrum.data, filter.data, output.data));
    else if (op == "zconv-scale")
      MEASURE(PF(zconvolve_scale)(setup, spectrum.data, filter.data, output.data, scalar(0.5)));
    else if (op == "zconv-acc")
      MEASURE(PF(zconvolve_accumulate)(setup, spectrum.data, filter.data, output.data, scalar(0.001)));
    else if (op == "zp")
      MEASURE(PF(zconvolve_zp)(setup, spectrum.data, filter.data, output.data));
    else if (op == "pipeline")
      MEASURE(PF(transform)(setup, input.data, spectrum.data, work.data, PFFFT_FORWARD);
              PF(zconvolve)(setup, spectrum.data, filter.data, product.data);
              PF(transform)(setup, product.data, output.data, work.data, PFFFT_BACKWARD));
    else
      throw std::runtime_error("unknown operation");
#undef MEASURE
    PF(destroy_setup)(setup);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
