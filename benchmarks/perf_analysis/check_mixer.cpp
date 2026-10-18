/* Numerical and block-continuity checks for the experimental mixer.
   Distributed under LICENSE.txt. Compile against either PFDSP library. */
#include "pffft/pf_mixer.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char **argv) {
  const float rates[] = { -0.12345f, -0.0009f, 0, 0.001f, 0.125f, 0.249f };
  const int sizes[] = { 8, 32, 256, 1024, 8192 };
  const float pi = 3.14159265358979323846f;
  double worst = 0, worst_low_rate = 0;
  FILE *dump = argc == 2 ? std::fopen(argv[1], "wb") : NULL;
  if (argc > 2 || (argc == 2 && !dump)) return 2;
  for (float rate : rates) {
    for (int n : sizes) {
      std::vector<complexf> input(n), actual(n), chunks(n);
      for (int i = 0; i < n; ++i) {
        input[i].i = float((i * 19) % 101 - 50) / 50;
        input[i].q = float((i * 13) % 103 - 51) / 51;
      }
      actual = chunks = input;
      shift_recursive_osc_conf_t conf;
      shift_recursive_osc_t state, chunk_state;
      shift_recursive_osc_init(rate, 0.1f, &conf, &state);
      chunk_state = state;
      shift_recursive_osc_inp_c(actual.data(), n, &conf, &state);
      for (int i = 0; i < n; i += 32)
        shift_recursive_osc_inp_c(chunks.data() + i, std::min(32, n - i), &conf, &chunk_state);
      if (std::memcmp(actual.data(), chunks.data(), n * sizeof(complexf)) ||
          std::memcmp(&state, &chunk_state, sizeof(state))) {
        std::fprintf(stderr, "block continuity failed: rate=%g n=%d\n", rate, n);
        return 1;
      }
      for (int i = 0; i < n; ++i) {
        const double phase = 0.1f + i * static_cast<double>(rate * pi);
        const double c = std::cos(phase), s = std::sin(phase);
        const double re = c * input[i].i - s * input[i].q;
        const double im = s * input[i].i + c * input[i].q;
        const double error = std::max(std::fabs(re - actual[i].i), std::fabs(im - actual[i].q));
        /* The existing oscillator loses accuracy near an eight-sample
           phase step of pi. Report that domain separately; do not claim
           a timing prototype repairs the recurrence's conditioning. */
        if (!std::isfinite(error) || (std::fabs(rate) <= 0.01f && error > 1e-3)) {
          std::fprintf(stderr, "reference failed: rate=%g n=%d i=%d error=%g\n", rate, n, i, error);
          return 1;
        }
        worst = std::max(worst, error);
        if (std::fabs(rate) <= 0.01f) worst_low_rate = std::max(worst_low_rate, error);
      }
      if (dump && std::fwrite(actual.data(), sizeof(complexf), n, dump) != static_cast<size_t>(n))
        return 2;
    }
  }
  if (dump) std::fclose(dump);
  std::printf("30 continuity/finite-output cases passed; block continuity exact; "
              "worst absolute error %.9g; low-rate error %.9g (limit 1e-3)\n",
              worst, worst_low_rate);
  return 0;
}
