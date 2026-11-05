#pragma once

#include <inttypes.h>
#include <math.h>

#ifndef USE_RVV
#define USE_RVV 0
#endif
#if USE_RVV && defined(__riscv_vector)
#include <riscv_vector.h>
#define HAVE_RVV 1
#else
#define HAVE_RVV 0
#endif

#define HIDDEN 1536
#define QHEADS 12
#define KVHEADS 2
#define HDIM 128
#define KVDIM (KVHEADS * HDIM)
#define QK 32
#define Q4_BYTES 18
#define Q8_BYTES 34
#define ROPE_BASE 1000000.0f
#define RMS_EPS 1e-6f

typedef struct { uint16_t d; int8_t qs[QK]; } block_q8_0;
_Static_assert(sizeof(block_q8_0) == Q8_BYTES, "Q8_0 layout should be 34 bytes");

typedef struct {
    uint8_t *wq, *wk, *wv, *wo;
    float *norm, *bq, *bk, *bv;
} layer_t;

typedef struct {
    float rms, q8_input, q_proj, k_proj, v_proj, rope_cache, attention, q8_output, out_proj;
} phase_t;


