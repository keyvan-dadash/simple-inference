// One-token Qwen2.5-Coder-1.5B attention sublayer. No inference engine.
// Q4_0 weights x on-the-fly Q8_0 activations -> FP32 Q/K/V -> RoPE
// -> synthetic FP16 KV history + online GQA -> Q8_0 x Q4_0 output projection.
// Meant for isolated profiling and later fusion experiments, NOT text inference.
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>

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

static double seconds_now(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
        perror("clock_gettime"); exit(1);
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void die(const char *message) {
    fprintf(stderr, "error: %s\n", message); exit(1);
}

static uint32_t bits_of_float(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    return u;
}
static float float_of_bits(uint32_t u) {
    float x;
    memcpy(&x, &u, 4);
    return x;
}

// Portable IEEE-754 f16 conversions (no dependency on compiler __fp16).
static float f16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 31u;
    uint32_t mant = h & 1023u;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) return float_of_bits(sign);
        int e = -14;
        while ((mant & 1024u) == 0) { mant <<= 1; --e; }
        mant &= 1023u;
        bits = sign | ((uint32_t)(e + 127) << 23) | (mant << 13);
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    return float_of_bits(bits);
}

static uint16_t f32_to_f16(float x) {
    uint32_t b = bits_of_float(x);
    uint16_t sign = (uint16_t)((b >> 16) & 0x8000u);
    uint32_t mant = b & 0x7fffffu;
    int e = (int)((b >> 23) & 0xffu) - 127 + 15;
    if (((b >> 23) & 255u) == 255u)
        return (uint16_t)(sign | (mant ? 0x7e00u : 0x7c00u));
    if (e >= 31) return (uint16_t)(sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return sign;
        uint32_t m = mant | 0x800000u;
        int shift = 14 - e;
        uint32_t frac = m >> shift;
        uint32_t rem = m & ((1u << shift) - 1u);
        uint32_t tie = 1u << (shift - 1);
        if (rem > tie || (rem == tie && (frac & 1u))) ++frac;
        return (uint16_t)(sign | frac);
    }
    uint32_t rounded = mant + 0xfffu + ((mant >> 13) & 1u);
    if (rounded & 0x800000u) {
        ++e;
        rounded = 0;
        if (e >= 31) return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint16_t)e << 10) | (rounded >> 13));
}

static uint32_t rng_state = 0x5217ac8bu;
static uint32_t random_u32(void) {
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return rng_state = x;
}
static float random_float(void) {
    return ((float)(random_u32() & 0xffffu) * (1.0f / 32767.5f) - 1.0f);
}

static void quantize_q8(const float *src, block_q8_0 *dst, int count) {
    if (count % QK) die("Q8 quantization length must be multiple of 32");
    for (int b = 0; b < count / QK; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < QK; j++) {
            float v = fabsf(src[b * QK + j]);
            if (v > amax) amax = v;
        }
        float d = amax / 127.0f;
        dst[b].d = f32_to_f16(d);
        float inv = d == 0.0f ? 0.0f : 1.0f / d;
        for (int j = 0; j < QK; j++) {
            long q = lrintf(src[b * QK + j] * inv);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            dst[b].qs[j] = (int8_t)q;
        }
    }
}

static inline int32_t dot_32_q4q8(const uint8_t *w, const int8_t *a) {
#if HAVE_RVV
    // GGUF Q4_0: w[2+j] low nibble = element j, high = element j+16.
    // Compute 16 low + 16 high products with widening integer arithmetic.
    size_t vl = __riscv_vsetvl_e8m1(16);
    vuint8m1_t packed = __riscv_vle8_v_u8m1(w + 2, vl);
    vuint8m1_t low = __riscv_vsub_vx_u8m1(
        __riscv_vand_vx_u8m1(packed, 15, vl), 8, vl);
    vuint8m1_t high = __riscv_vsub_vx_u8m1(
        __riscv_vsrl_vx_u8m1(packed, 4, vl), 8, vl);
    vint8m1_t lo = __riscv_vreinterpret_v_u8m1_i8m1(low);
    vint8m1_t hi = __riscv_vreinterpret_v_u8m1_i8m1(high);
    vint8m1_t x0 = __riscv_vle8_v_i8m1(a, vl);
    vint8m1_t x1 = __riscv_vle8_v_i8m1(a + 16, vl);
    vint16m2_t products = __riscv_vwmul_vv_i16m2(lo, x0, vl);
    products = __riscv_vwmacc_vv_i16m2(products, hi, x1, vl);
    vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
    vint32m1_t sum = __riscv_vwredsum_vs_i16m2_i32m1(products, zero, vl);
    return __riscv_vmv_x_s_i32m1_i32(sum);
#else
    assert(0 == 1);
    int32_t result = 0;
    for (int j = 0; j < 16; ++j) {
        uint8_t q = w[2 + j];
        result += ((int32_t)(q & 15) - 8) * a[j];
        result += ((int32_t)(q >> 4) - 8) * a[j + 16];
    }
    return result;
#endif
}

static void matvec_q4q8(const uint8_t *weights, const block_q8_0 *input,
                        int inputs, int outputs, const float *bias, float *out) {
    int blocks = inputs / QK;
    for (int row = 0; row < outputs; ++row) {
        const uint8_t *wrow = weights + (size_t)row * blocks * Q4_BYTES;
        float sum = bias ? bias[row] : 0.0f;
        for (int b = 0; b < blocks; b++) {
            const uint8_t *wb = wrow + (size_t)b * Q4_BYTES;
            uint16_t hw = (uint16_t)wb[0] | ((uint16_t)wb[1] << 8);
            int32_t intsum = dot_32_q4q8(wb, input[b].qs);
            sum += (float)intsum * f16_to_f32(hw) * f16_to_f32(input[b].d);
        }
        out[row] = sum;
    }
}

static void rmsnorm(const float *input, const float *weight, float *out) {
    double square = 0.0;
    for (int i = 0; i < HIDDEN; i++) square += (double)input[i] * input[i];
    float inv = 1.0f / sqrtf((float)(square / HIDDEN) + RMS_EPS);
    for (int i = 0; i < HIDDEN; i++) out[i] = input[i] * inv * (weight ? weight[i] : 1.0f);
}

// Qwen rotate_half convention: pair d with d+HDIM/2.
static void rope(float *heads, int nheads, int position) {
    for (int h = 0; h < nheads; h++) {
        float *v = heads + h * HDIM;
        for (int i = 0; i < HDIM / 2; i++) {
            float theta = (float)position / powf(ROPE_BASE, (float)(2 * i) / HDIM);
            float c = cosf(theta), s = sinf(theta);
            float first = v[i], second = v[i + HDIM / 2];
            v[i] = first * c - second * s;
            v[i + HDIM / 2] = second * c + first * s;
        }
    }
}

// Standard per-head online softmax: no materialized S-by-S score buffer.
static void attention_one_head(const float *q, const uint16_t *key_cache,
                               const uint16_t *value_cache, int group,
                               int context, float *out) {
    float max_score = -INFINITY;
    float normalizer = 0.0f;
    memset(out, 0, HDIM * sizeof(*out));
    for (int t = 0; t < context; t++) {
        const uint16_t *k = key_cache + ((size_t)t * KVHEADS + group) * HDIM;
        const uint16_t *v = value_cache + ((size_t)t * KVHEADS + group) * HDIM;
        float score = 0.0f;
        for (int d = 0; d < HDIM; d++) score += q[d] * f16_to_f32(k[d]);
        score *= (1.0f / sqrtf((float)HDIM));
        float new_max = fmaxf(max_score, score);
        float alpha = expf(max_score - new_max);
        float beta = expf(score - new_max);
        normalizer = normalizer * alpha + beta;
        for (int d = 0; d < HDIM; d++)
            out[d] = out[d] * alpha + beta * f16_to_f32(v[d]);
        max_score = new_max;
    }
    float inv = 1.0f / normalizer;
    for (int d = 0; d < HDIM; d++) out[d] *= inv;
}

static void attention_all(const float *q, const uint16_t *kc,
                          const uint16_t *vc, int context, float *out) {
    for (int h = 0; h < QHEADS; ++h)
        attention_one_head(q + h * HDIM, kc, vc, h / (QHEADS / KVHEADS),
                           context, out + h * HDIM);
}

static void *load_file(const char *dir, const char *name, size_t expected,
                       int optional) {
    char path[1024];
    int n = snprintf(path, sizeof(path), "%s/%s", dir, name);
    if (n < 0 || (size_t)n >= sizeof(path)) die("Weights directory path too long");
    FILE *f = fopen(path, "rb");
    if (!f) {
        if (optional && errno == ENOENT) return NULL;
        perror(path); exit(1);
    }
    if (fseek(f, 0, SEEK_END) != 0 || ftell(f) != (long)expected ||
        fseek(f, 0, SEEK_SET) != 0) {
        fprintf(stderr, "Wrong size for %s (expected %zu bytes)\n", path, expected);
        exit(1);
    }
    void *buf = malloc(expected);
    if (!buf) die("Out of memory loading weights");
    if (fread(buf, 1, expected, f) != expected) die("Failed to read weight file");
    fclose(f);
    return buf;
}

static layer_t load_layer(const char *dir) {
    layer_t l = {0};
    l.wq = load_file(dir, "q.q4_0", (size_t)HIDDEN * HIDDEN / QK * Q4_BYTES, 0);
    l.wk = load_file(dir, "k.q4_0", (size_t)KVDIM * HIDDEN / QK * Q4_BYTES, 0);
    l.wv = load_file(dir, "v.q4_0", (size_t)KVDIM * HIDDEN / QK * Q4_BYTES, 0);
    l.wo = load_file(dir, "o.q4_0", (size_t)HIDDEN * HIDDEN / QK * Q4_BYTES, 0);
    l.norm = load_file(dir, "norm.f32", HIDDEN * sizeof(float), 1);
    l.bq = load_file(dir, "q_bias.f32", HIDDEN * sizeof(float), 1);
    l.bk = load_file(dir, "k_bias.f32", KVDIM * sizeof(float), 1);
    l.bv = load_file(dir, "v_bias.f32", KVDIM * sizeof(float), 1);
    return l;
}

static void free_layer(layer_t *l) {
    free(l->wq); free(l->wk); free(l->wv); free(l->wo);
    free(l->norm); free(l->bq); free(l->bk); free(l->bv);
}

static void fill_kv_history(uint16_t *kc, uint16_t *vc, int context) {
    // Prior context is synthetic and already RoPE-compatible in the sense that
    // it is precomputed cache data. No preceding transformer layers are run.
    for (size_t i = 0, n = (size_t)(context - 1) * KVDIM; i < n; ++i) {
        kc[i] = f32_to_f16(random_float() * 0.25f);
        vc[i] = f32_to_f16(random_float() * 0.25f);
    }
}

static phase_t run_layer(const layer_t *l, const float *x,
                         uint16_t *kc, uint16_t *vc, int context,
                         float *output) {
    float normed[HIDDEN], q[HIDDEN], k[KVDIM], v[KVDIM], attn[HIDDEN];
    block_q8_0 inq8[HIDDEN / QK], attq8[HIDDEN / QK];
    phase_t p = {0};
    double t = seconds_now(), next;

    rmsnorm(x, l->norm, normed);
    next = seconds_now(); p.rms = next - t; t = next;

    quantize_q8(normed, inq8, HIDDEN);
    next = seconds_now(); p.q8_input = next - t; t = next;

    matvec_q4q8(l->wq, inq8, HIDDEN, HIDDEN, l->bq, q);
    next = seconds_now(); p.q_proj = (float)(next - t); t = next;
    matvec_q4q8(l->wk, inq8, HIDDEN, KVDIM, l->bk, k);
    next = seconds_now(); p.k_proj = (float)(next - t); t = next;
    matvec_q4q8(l->wv, inq8, HIDDEN, KVDIM, l->bv, v);
    next = seconds_now(); p.v_proj = (float)(next - t); t = next;

    rope(q, QHEADS, context - 1);
    rope(k, KVHEADS, context - 1);
    size_t new_token = (size_t)(context - 1) * KVDIM;
    for (int i = 0; i < KVDIM; ++i) {
        kc[new_token + (size_t)i] = f32_to_f16(k[i]);
        vc[new_token + (size_t)i] = f32_to_f16(v[i]);
    }
    next = seconds_now(); p.rope_cache = next - t; t = next;

    attention_all(q, kc, vc, context, attn);
    next = seconds_now(); p.attention = next - t; t = next;

    quantize_q8(attn, attq8, HIDDEN);
    next = seconds_now(); p.q8_output = next - t; t = next;

    matvec_q4q8(l->wo, attq8, HIDDEN, HIDDEN, NULL, output);
    next = seconds_now(); p.out_proj = next - t;
    return p;
}

// Human-readable sizes are binary KiB/MiB. These are storage footprints,
// NOT the bytes transferred by hardware: data can be read many times.
static void format_bytes(size_t bytes, char *buffer, size_t n) {
    if (bytes >= 1024u * 1024u)
        snprintf(buffer, n, "%.2f MiB", (double)bytes / (1024.0 * 1024.0));
    else if (bytes >= 1024u)
        snprintf(buffer, n, "%.2f KiB", (double)bytes / 1024.0);
    else
        snprintf(buffer, n, "%zu B", bytes);
}

static void data_label(char *dest, size_t n, const char *label, size_t bytes) {
    char size[32];
    format_bytes(bytes, size, sizeof(size));
    snprintf(dest, n, "%s %s", label, size);
}

typedef struct {
    const char *operation;
    const char *result;
    char data[36];
    float seconds;
} report_row_t;

static void report(const phase_t *p, const layer_t *l, int context,
                   int repeat, const float *out) {
    const size_t q_bytes = (size_t)HIDDEN * HIDDEN / QK * Q4_BYTES;
    const size_t kv_proj_bytes = (size_t)KVDIM * HIDDEN / QK * Q4_BYTES;
    const size_t q8_bytes = (size_t)(HIDDEN / QK) * Q8_BYTES;
    const size_t kv_cache_bytes = (size_t)context * KVDIM * sizeof(uint16_t) * 2u;
    const size_t new_kv_bytes = (size_t)KVDIM * sizeof(uint16_t) * 2u;
    const size_t all_weights = q_bytes * 2u + kv_proj_bytes * 2u;
    const size_t optional_weights = (l->norm ? HIDDEN * sizeof(float) : 0u) +
        (l->bq ? HIDDEN * sizeof(float) : 0u) +
        (l->bk ? KVDIM * sizeof(float) : 0u) +
        (l->bv ? KVDIM * sizeof(float) : 0u);
    report_row_t rows[] = {
        {"RMSNorm",            "FP32 [1536]",     "", p->rms},
        {"Quantize input",     "Q8_0 [48 blocks]", "", p->q8_input},
        {"Q projection",       "FP32 [12 x 128]", "", p->q_proj},
        {"K projection",       "FP32 [2 x 128]",  "", p->k_proj},
        {"V projection",       "FP32 [2 x 128]",  "", p->v_proj},
        {"RoPE + KV append",   "KV cache update", "", p->rope_cache},
        {"GQA online attention","FP32 [12 x 128]","", p->attention},
        {"Quantize attention", "Q8_0 [48 blocks]", "", p->q8_output},
        {"Output projection",  "FP32 [1536]",     "", p->out_proj},
    };
    const size_t count = sizeof(rows) / sizeof(rows[0]);
    if (l->norm)
        data_label(rows[0].data, sizeof(rows[0].data), "Norm", HIDDEN * sizeof(float));
    else
        snprintf(rows[0].data, sizeof(rows[0].data), "Norm (absent)");
    data_label(rows[1].data, sizeof(rows[1].data), "Q8 buffer", q8_bytes);
    data_label(rows[2].data, sizeof(rows[2].data), "Wq Q4_0", q_bytes);
    data_label(rows[3].data, sizeof(rows[3].data), "Wk Q4_0", kv_proj_bytes);
    data_label(rows[4].data, sizeof(rows[4].data), "Wv Q4_0", kv_proj_bytes);
    data_label(rows[5].data, sizeof(rows[5].data), "New K+V", new_kv_bytes);
    data_label(rows[6].data, sizeof(rows[6].data), "KV cache", kv_cache_bytes);
    data_label(rows[7].data, sizeof(rows[7].data), "Q8 buffer", q8_bytes);
    data_label(rows[8].data, sizeof(rows[8].data), "Wo Q4_0", q_bytes);

    double total = 0.0;
    for (size_t i = 0; i < count; i++) total += rows[i].seconds;
    double checksum = 0.0;
    for (int i = 0; i < HIDDEN; ++i) {
        if (!isfinite(out[i])) die("Non-finite projection output");
        checksum += (double)out[i];
    }

    char wsize[32], extrasize[32], kvsize[32];
    format_bytes(all_weights, wsize, sizeof(wsize));
    format_bytes(optional_weights, extrasize, sizeof(extrasize));
    format_bytes(kv_cache_bytes, kvsize, sizeof(kvsize));

    printf("\nQwen2.5-Coder-1.5B | one attention sublayer (decode)\n");
    printf("Context: %d tokens  |  Repeats: %d  |  RVV dot: %s\n",
           context, repeat, HAVE_RVV ? "enabled" : "disabled (scalar fallback)");
    printf("Weights: Q4_0  |  Matvec activations: Q8_0  |  KV cache: FP16\n");
    printf("Timings exclude model loading, synthetic input generation and setup.\n\n");

    puts("---------------------------------------------------------------------------------------------------------------");
    printf(" %3s  %-22s  %-18s  %-22s %10s %10s %7s\n",
           "#", "Operation (in order)", "Output", "Weights / data", "ms/run", "ms total", "Share");
    puts("---------------------------------------------------------------------------------------------------------------");
    for (size_t i = 0; i < count; i++) {
        const double ms_total = (double)rows[i].seconds * 1000.0;
        const double ms_run = ms_total / (double)repeat;
        const double share = total > 0 ? 100.0 * rows[i].seconds / total : 0.0;
        printf(" %3zu  %-22s  %-18s  %-22s %10.3f %10.3f %6.1f%%\n",
               i + 1, rows[i].operation, rows[i].result, rows[i].data,
               ms_run, ms_total, share);
    }
    puts("---------------------------------------------------------------------------------------------------------------");
    printf("      %-22s  %-18s  %-22s %10.3f %10.3f %6.1f%%\n",
           "TOTAL (timed)", "", "", total * 1000.0 / repeat,
           total * 1000.0, 100.0);

    const double projection = (double)p->q_proj + p->k_proj + p->v_proj;
    const double fusion_window = (double)p->attention + p->q8_output + p->out_proj;
    printf("\nQKV projections: %.3f ms/run (%.1f%% of timed work)\n",
           projection * 1000.0 / repeat,
           total > 0 ? projection * 100.0 / total : 0.0);
    printf("Fusion target (attention + quantize + W_O): %.3f ms/run (%.1f%%)\n",
           fusion_window * 1000.0 / repeat,
           total > 0 ? fusion_window * 100.0 / total : 0.0);
    printf("Stored Q4_0 matrices: %s  |  norm/bias: %s  |  KV allocation: %s\n",
           wsize, extrasize, kvsize);
    printf("Output checksum: %.9g  |  First 4: [%.6g, %.6g, %.6g, %.6g]\n",
           checksum, out[0], out[1], out[2], out[3]);
    puts("Note: data sizes are footprints, not measured memory traffic; times are averages/sums over repeats.");
    puts("      Synthetic input/KV history; no full inference or fusion is performed.\n");
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "Usage: %s <weights_directory> [context=256] [repeat=1]\n", argv[0]);
        return 2;
    }
    char *end;
    long context = argc >= 3 ? strtol(argv[2], &end, 10) : 256;
    if (argc >= 3 && *end) die("Invalid context argument");
    long repeat = argc >= 4 ? strtol(argv[3], &end, 10) : 1;
    if (argc >= 4 && *end) die("Invalid repeat argument");
    if (context < 1 || context > 32768 || repeat < 1 || repeat > 100000)
        die("Context must be 1..32768 and repeat 1..100000");

    layer_t l = load_layer(argv[1]);
    float *input = malloc(HIDDEN * sizeof(float));
    float *output = malloc(HIDDEN * sizeof(float));
    uint16_t *kc = malloc((size_t)context * KVDIM * sizeof(uint16_t));
    uint16_t *vc = malloc((size_t)context * KVDIM * sizeof(uint16_t));
    if (!input || !output || !kc || !vc) die("Out of memory creating synthetic inputs");
    for (int i = 0; i < HIDDEN; i++) input[i] = random_float();
    fill_kv_history(kc, vc, (int)context);
    phase_t summed = {0};
    for (long r = 0; r < repeat; r++) {
        phase_t p = run_layer(&l, input, kc, vc, (int)context, output);
        summed.rms += p.rms;
        summed.q8_input += p.q8_input;
        summed.q_proj += p.q_proj;
        summed.k_proj += p.k_proj;
        summed.v_proj += p.v_proj;
        summed.rope_cache += p.rope_cache;
        summed.attention += p.attention;
        summed.q8_output += p.q8_output;
        summed.out_proj += p.out_proj;
    }
    report(&summed, &l, (int)context, (int)repeat, output);
    free(input); free(output); free(kc); free(vc);
    free_layer(&l);
    return 0;
}
