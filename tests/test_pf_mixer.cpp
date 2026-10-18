/* Recursive oscillator regression tests. Distributed under LICENSE.txt. */
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

static const double PI_REF = 3.14159265358979323846;
static const double ERROR_LIMIT = 1e-3;
static double worst_error;

static void check_sample(complexf actual, double re, double im, float rate, int i) {
  const double error = std::max(std::fabs(actual.i - re), std::fabs(actual.q - im));
  if (!std::isfinite(error) || error > ERROR_LIMIT)
    std::fprintf(stderr, "rate=%g sample=%d error=%g expected=(%g,%g) actual=(%g,%g)\n",
                 rate, i, error, re, im, actual.i, actual.q);
  assert(std::isfinite(error) && error <= ERROR_LIMIT);
  worst_error = std::max(worst_error, error);
}

static void check_case(float rate, float phase, int n) {
  /* Offset the buffers by one complex sample to exercise unaligned loads. */
  const complexf guard = { 42, -17 };
  std::vector<complexf> input(n+2, guard), mixed(n+2, guard), osc(n+2, guard);
  for (int i = 0; i < n; ++i) {
    input[i+1].i = float((i * 19) % 101 - 50) / 50;
    input[i+1].q = float((i * 13) % 103 - 51) / 51;
  }
  std::vector<complexf> inplace = input, chunks = input, sse = input;
  shift_recursive_osc_conf_t conf;
  shift_recursive_osc_t state, in_state, chunk_state, osc_state;
  shift_recursive_osc_init(rate, phase, &conf, &state);
  in_state = chunk_state = osc_state = state;
  shift_recursive_osc_cc(input.data()+1, mixed.data()+1, n, &conf, &state);
  shift_recursive_osc_inp_c(inplace.data()+1, n, &conf, &in_state);
  gen_recursive_osc_c(osc.data()+1, n, &conf, &osc_state);
  shift_recursive_osc_inp_c(chunks.data()+1, 0, &conf, &chunk_state);
  for (int i = 0; i < n;) {
    const int count = std::min(n - i, (i % 5 + 1) * PF_SHIFT_RECURSIVE_SIMD_SZ);
    shift_recursive_osc_inp_c(chunks.data()+1+i, count, &conf, &chunk_state);
    i += count;
  }
  assert(!std::memcmp(inplace.data(), chunks.data(), (n+2) * sizeof(complexf)));
  assert(!std::memcmp(&in_state, &chunk_state, sizeof(in_state)));
  assert(!std::memcmp(&in_state, &state, sizeof(state)));
  assert(!std::memcmp(&in_state, &osc_state, sizeof(state)));
  assert(!std::memcmp(inplace.data()+1, mixed.data()+1, n * sizeof(complexf)));

  if (have_sse_shift_mixer_impl()) {
    shift_recursive_osc_sse_conf_t sse_conf;
    shift_recursive_osc_sse_t sse_state, split_state;
    shift_recursive_osc_sse_init(rate, phase, &sse_conf, &sse_state);
    split_state = sse_state;
    shift_recursive_osc_sse_inp_c(sse.data()+1, n, &sse_conf, &sse_state);
    chunks = input;
    shift_recursive_osc_sse_inp_c(chunks.data()+1, 0, &sse_conf, &split_state);
    for (int i = 0; i < n; i += PF_SHIFT_RECURSIVE_SIMD_SSE_SZ)
      shift_recursive_osc_sse_inp_c(chunks.data()+1+i, PF_SHIFT_RECURSIVE_SIMD_SSE_SZ,
                                    &sse_conf, &split_state);
    assert(!std::memcmp(sse.data(), chunks.data(), (n+2) * sizeof(complexf)));
    assert(!std::memcmp(&sse_state, &split_state, sizeof(sse_state)));
  }

  for (int i = 0; i < n; ++i) {
    const double angle = phase + i * static_cast<double>(rate) * PI_REF;
    const double c = std::cos(angle), s = std::sin(angle);
    const double re = c * input[i+1].i - s * input[i+1].q;
    const double im = s * input[i+1].i + c * input[i+1].q;
    check_sample(mixed[i+1], re, im, rate, i);
    check_sample(osc[i+1], c, s, rate, i);
    if (have_sse_shift_mixer_impl()) check_sample(sse[i+1], re, im, rate, i);
  }
  const std::vector<complexf> *buffers[] = { &input, &mixed, &inplace, &chunks, &osc, &sse };
  for (size_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); ++i) {
    assert(!std::memcmp(&buffers[i]->front(), &guard, sizeof(guard)));
    assert(!std::memcmp(&buffers[i]->back(), &guard, sizeof(guard)));
  }
}

static void check_rate_changes() {
  const float rates[] = { -0.0009f, 0.125f, -0.249f, 0.25f, 1, -1, 0 };
  const int n = 1024;
  std::vector<complexf> output(n), sse(n);
  shift_recursive_osc_conf_t conf;
  shift_recursive_osc_t state;
  shift_recursive_osc_sse_conf_t sse_conf;
  shift_recursive_osc_sse_t sse_state;
  const float start = 0.1f;
  double phase = start;
  shift_recursive_osc_init(0, start, &conf, &state);
  if (have_sse_shift_mixer_impl())
    shift_recursive_osc_sse_init(0, start, &sse_conf, &sse_state);
  for (size_t j = 0; j < sizeof(rates) / sizeof(rates[0]); ++j) {
    const float rate = rates[j];
    const float u = state.u_cos[0], v = state.v_sin[0];
    shift_recursive_osc_update_rate(rate, &conf, &state);
    assert(state.u_cos[0] == u && state.v_sin[0] == v);
    gen_recursive_osc_c(output.data(), n, &conf, &state);
    if (have_sse_shift_mixer_impl()) {
      shift_recursive_osc_sse_update_rate(rate, &sse_conf, &sse_state);
      for (int i = 0; i < n; ++i) { sse[i].i = 1; sse[i].q = 0; }
      shift_recursive_osc_sse_inp_c(sse.data(), n, &sse_conf, &sse_state);
    }
    for (int i = 0; i < n; ++i) {
      const double angle = phase + i * static_cast<double>(rate) * PI_REF;
      check_sample(output[i], std::cos(angle), std::sin(angle), rate, i);
      if (have_sse_shift_mixer_impl())
        check_sample(sse[i], std::cos(angle), std::sin(angle), rate, i);
    }
    phase += n * static_cast<double>(rate) * PI_REF;
  }
}

int main() {
  /* Exact singularities, nearby rates, half-plane boundaries, negative rates,
     DC/Nyquist, and rates outside the principal interval. */
  const float rates[] = { 0, -0.0009f, 0.001f, -0.12345f, 0.0625f, -0.0625f,
    0.062499f, 0.062501f, 0.124999f, 0.125f, 0.125001f, -0.125f,
    -0.124999f, -0.125001f, 0.249f, 0.25f, -0.25f, 0.375f, 0.5f, 0.75f,
    0.999999f, 1, -1, 1.125f, 8.125f };
  const int sizes[] = { 0, 8, 32, 256, 8192, 65536 };
  const float phases[] = { 0, 0.1f };
  int cases = 0;
  for (size_t r = 0; r < sizeof(rates) / sizeof(rates[0]); ++r)
    for (size_t p = 0; p < sizeof(phases) / sizeof(phases[0]); ++p)
      for (size_t n = 0; n < sizeof(sizes) / sizeof(sizes[0]); ++n) {
        check_case(rates[r], phases[p], sizes[n]);
        ++cases;
      }
  check_rate_changes();
  std::printf("%d mixer cases and rate changes passed; worst absolute error %.9g\n",
              cases, worst_error);
  return 0;
}
