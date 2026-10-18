/* High resolution, bounded-working-set mixer benchmark. LICENSE.txt applies. */
#include "pffft/pf_mixer.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

typedef std::chrono::steady_clock mixer_clock;
static volatile double mixer_checksum;

template<class Initialize, class Apply>
static int measure(const char *op, int n, int repeats, double seconds, Initialize init, Apply apply) {
  const int blocks = 32;
  std::vector<complexf> data(n * blocks);
  for (int trial = 0; trial < repeats; ++trial) {
    for (size_t j = 0; j < data.size(); ++j) {
      data[j].i = 0.5f;
      data[j].q = 0.25f;
    }
    init();
    for (int j = 0; j < blocks; ++j) apply(data.data() + j*n);
    size_t iterations = 0;
    double duration;
    mixer_clock::time_point start = mixer_clock::now();
    do {
      for (int j = 0; j < blocks; ++j) apply(data.data() + j*n);
      iterations += blocks;
      duration = std::chrono::duration<double>(mixer_clock::now() - start).count();
    } while (duration < seconds);
    double checksum = 0;
    for (const complexf &x : data) {
      const double power = double(x.i) * x.i + double(x.q) * x.q;
      // Exclude nonfinite, underflowing or exploding repeated inputs.
      if (!std::isfinite(power) || power < 0.01 || power > 4) {
        std::fprintf(stderr, "invalid benchmark data: power=%g\n", power);
        return 1;
      }
      checksum += power;
    }
    mixer_checksum = checksum;
    std::printf("%s,%d,%d,%zu,%.9f,%.6f,%.9g\n", op, n, trial, iterations,
                duration, duration * 1e9 / (iterations * n), checksum);
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: %s recursive8|recursive4|table-inplace|table-out "
                        "N REPEATS SECONDS\n", argv[0]);
    return 2;
  }
  const int n = std::atoi(argv[2]), repeats = std::atoi(argv[3]);
  const double seconds = std::atof(argv[4]);
  if (n < 8 || n > 65536 || n % 8 || repeats < 1 || repeats > 100 ||
      !(seconds > 0 && seconds <= 1)) return 2;
  if (!std::strcmp(argv[1], "recursive8")) {
    shift_recursive_osc_t state;
    shift_recursive_osc_conf_t conf;
    return measure(argv[1], n, repeats, seconds,
                   [&]() { shift_recursive_osc_init(-0.0009f, 0, &conf, &state); },
                   [&](complexf *x) { shift_recursive_osc_inp_c(x, n, &conf, &state); });
  }
  if (!std::strcmp(argv[1], "recursive4") && have_sse_shift_mixer_impl()) {
    shift_recursive_osc_sse_t state;
    shift_recursive_osc_sse_conf_t conf;
    return measure(argv[1], n, repeats, seconds,
                   [&]() { shift_recursive_osc_sse_init(-0.0009f, 0, &conf, &state); },
                   [&](complexf *x) { shift_recursive_osc_sse_inp_c(x, n, &conf, &state); });
  }
  const bool inplace = !std::strcmp(argv[1], "table-inplace");
  if (inplace || !std::strcmp(argv[1], "table-out")) {
    shift_unroll_data_t table = shift_unroll_init(-0.0009f, n);
    if (!table.dsin || !table.dcos) return 1;
    const complexf sample = { 0.5f, 0.25f };
    std::vector<complexf> input(n, sample);
    float phase;
    const int result = measure(argv[1], n, repeats, seconds,
        [&]() { phase = 0.37f; },
        [&](complexf *x) {
          phase = inplace ? shift_unroll_inp_c(x, n, &table, phase)
                          : shift_unroll_cc(input.data(), x, n, &table, phase);
        });
    shift_unroll_deinit(&table);
    return result;
  }
  return 2;
}
