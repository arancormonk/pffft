/* Table-driven mixer regression tests. Distributed under LICENSE.txt. */
#include "pffft/pf_mixer.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static double worst_error;

static void check_value(complexf value, double re, double im, double tolerance) {
  const double error = std::max(std::fabs(value.i - re), std::fabs(value.q - im));
  if (!std::isfinite(error) || error > tolerance)
    std::fprintf(stderr, "table mixer error %.9g exceeds %.9g\n", error, tolerance);
  assert(std::isfinite(error) && error <= tolerance);
  worst_error = std::max(worst_error, error);
}

static void check_case(float rate, float phase, int n) {
  const complexf guard = { 42, -17 };
  shift_unroll_data_t table = shift_unroll_init(rate, std::max(n, 1));
  assert(table.dsin && table.dcos);
  std::vector<complexf> input(n+2, guard), output(n+2, guard);
  for (int i = 0; i < n; ++i) {
    input[i+1].i = float((i * 19) % 101 - 50) / 50;
    input[i+1].q = float((i * 13) % 103 - 51) / 51;
  }
  const std::vector<complexf> original = input;
  std::vector<complexf> inplace = input, aliased = input;
  const float next = shift_unroll_cc(input.data()+1, output.data()+1, n, &table, phase);
  const float next_in = shift_unroll_inp_c(inplace.data()+1, n, &table, phase);
  const float next_alias = shift_unroll_cc(aliased.data()+1, aliased.data()+1,
                                          n, &table, phase);
  /* All entry points must return the same phase bits. */
  assert(!std::memcmp(&next, &next_in, sizeof(next)));
  assert(!std::memcmp(&next, &next_alias, sizeof(next)));
  assert(std::isfinite(next) && std::fabs(next) <= 3.1415928f);
  assert(!std::memcmp(input.data(), original.data(), (n+2) * sizeof(complexf)));
  /* The existing table accumulates its phase in float. Allow its accumulated
     rounding error, while checking the rotation against independent double
     trigonometry. Small sizes also cover every vector tail and the first bin. */
  const double tolerance = 2e-6 + n * 3e-7;
  for (int i = 0; i < n; ++i) {
    const double angle = phase + i * static_cast<double>(table.phase_increment);
    const double c = std::cos(angle), s = std::sin(angle);
    const double re = c * input[i+1].i - s * input[i+1].q;
    const double im = s * input[i+1].i + c * input[i+1].q;
    check_value(output[i+1], re, im, tolerance);
    check_value(inplace[i+1], re, im, tolerance);
    check_value(aliased[i+1], re, im, tolerance);
    check_value(inplace[i+1], output[i+1].i, output[i+1].q, 5e-7);
    check_value(aliased[i+1], output[i+1].i, output[i+1].q, 5e-7);
  }
  const std::vector<complexf> *buffers[] = { &input, &output, &inplace, &aliased };
  for (size_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); ++i) {
    assert(!std::memcmp(&buffers[i]->front(), &guard, sizeof(guard)));
    assert(!std::memcmp(&buffers[i]->back(), &guard, sizeof(guard)));
  }
  shift_unroll_deinit(&table);
}

static void check_continuity() {
  const int n = 1024;
  const float start = .37f;
  shift_unroll_data_t table = shift_unroll_init(-.0009f, n);
  assert(table.dsin && table.dcos);
  std::vector<complexf> data(n);
  for (int i = 0; i < n; ++i) { data[i].i = .5f; data[i].q = .25f; }
  float phase = start;
  for (int offset = 0; offset < n;) {
    const int count = std::min(n-offset, offset % 17 + 1);
    phase = shift_unroll_inp_c(data.data()+offset, count, &table, phase);
    offset += count;
  }
  for (int i = 0; i < n; ++i) {
    const double a = start + i * static_cast<double>(table.phase_increment);
    check_value(data[i], .5 * std::cos(a) - .25 * std::sin(a),
                         .5 * std::sin(a) + .25 * std::cos(a), 2e-5);
  }
  /* Empty calls must preserve the phase bit for bit. */
  const float next_in = shift_unroll_inp_c(NULL, 0, &table, phase);
  const float next = shift_unroll_cc(NULL, NULL, 0, &table, phase);
  assert(!std::memcmp(&next_in, &phase, sizeof(phase)));
  assert(!std::memcmp(&next, &phase, sizeof(phase)));
  shift_unroll_deinit(&table);
}

int main() {
  const float rates[] = { 0, .0009f, -.0009f, .12345f, -.12345f, .499f, -.499f };
  const float phases[] = { 0, .37f, -2.1f };
  const int sizes[] = { 0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 127, 128,
                        129, 255, 256, 257, 1024, 4096 };
  int cases = 0;
  for (float rate : rates)
    for (float phase : phases)
      for (int n : sizes) {
        check_case(rate, phase, n);
        ++cases;
      }
  check_continuity();
  std::printf("%d table mixer cases and continuity passed; worst error %.9g\n",
              cases, worst_error);
  return 0;
}
