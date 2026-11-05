#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "defines.h"

// No block_q8_0, layer_t, or phase_t typedefs here.

static double seconds_now(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
        perror("clock_gettime");
        exit(1);
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

// Shared standard/online attention kernels: same Q and FP16 KV tensors.
#include "attention_kernels.h"

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
                         float *output, attention_method_t method,
                         float *scores, float *attn_copy,
                         block_q8_0 *q8_copy) {
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

    attention_compute(method, q, kc, vc, context, attn, scores);
    next = seconds_now(); p.attention = next - t; t = next;

    quantize_q8(attn, attq8, HIDDEN);
    next = seconds_now(); p.q8_output = next - t; t = next;

    matvec_q4q8(l->wo, attq8, HIDDEN, HIDDEN, NULL, output);
    next = seconds_now(); p.out_proj = next - t;
    // Optional correctness captures are outside every timed stage.
    if (attn_copy) memcpy(attn_copy, attn, sizeof(attn));
    if (q8_copy) memcpy(q8_copy, attq8, sizeof(attq8));
    return p;
}

#include "fused_layer.h"
#include "register_fusion.h"
#include "soa_staged.h"

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
                   int repeat, const float *out, attention_method_t method) {
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
        {method == ATTENTION_FLASH ? "GQA online (flash)" : "GQA score/softmax",
         "FP32 [12 x 128]", "", p->attention},
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
    printf("Attention: %s | Context: %d tokens | Repeats: %d | RVV dot: %s\n",
           attention_name(method), context, repeat,
           HAVE_RVV ? "enabled" : "disabled (scalar fallback)");
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
    puts("      Flash mode fuses score/softmax/value computation ONLY; no attention-to-W_O fusion.\n");
}


static void phase_add(phase_t *sum, phase_t p) {
    sum->rms       += p.rms;
    sum->q8_input  += p.q8_input;
    sum->q_proj    += p.q_proj;
    sum->k_proj    += p.k_proj;
    sum->v_proj    += p.v_proj;
    sum->rope_cache+= p.rope_cache;
    sum->attention += p.attention;
    sum->q8_output += p.q8_output;
    sum->out_proj  += p.out_proj;
}

static void phase_vector(const phase_t *p, double t[9]) {
    t[0]=p->rms;       t[1]=p->q8_input; t[2]=p->q_proj;
    t[3]=p->k_proj;    t[4]=p->v_proj;   t[5]=p->rope_cache;
    t[6]=p->attention; t[7]=p->q8_output;t[8]=p->out_proj;
}

static void report_compare(const phase_t *st, const phase_t *fl,
                           const layer_t *l, int context, int repeat,
                           const float *out_st, const float *out_fl,
                           const float *att_st, const float *att_fl,
                           const block_q8_0 *q8_st,
                           const block_q8_0 *q8_fl) {
    const char *ops[] = {
        "RMSNorm", "Quantize input", "Q projection", "K projection",
        "V projection", "RoPE + KV append", "GQA attention",
        "Quantize attention", "Output projection"
    };
    const char *shapes[] = {
        "FP32 [1536]", "Q8_0 [48 blocks]", "FP32 [12 x 128]",
        "FP32 [2 x 128]", "FP32 [2 x 128]", "KV cache update",
        "FP32 [12 x 128]", "Q8_0 [48 blocks]", "FP32 [1536]"
    };
    char datas[9][36];
    const size_t qsize = (size_t)HIDDEN * HIDDEN / QK * Q4_BYTES;
    const size_t kvsize = (size_t)KVDIM * HIDDEN / QK * Q4_BYTES;
    const size_t cache_size = (size_t)context * KVDIM * sizeof(uint16_t) * 2;
    const size_t q8size = (HIDDEN / QK) * Q8_BYTES;
    data_label(datas[0], sizeof(datas[0]), l->norm ? "Norm" : "Norm absent",
               l->norm ? HIDDEN * sizeof(float) : 0);
    data_label(datas[1], sizeof(datas[1]), "Q8 buffer", q8size);
    data_label(datas[2], sizeof(datas[2]), "Wq Q4_0", qsize);
    data_label(datas[3], sizeof(datas[3]), "Wk Q4_0", kvsize);
    data_label(datas[4], sizeof(datas[4]), "Wv Q4_0", kvsize);
    data_label(datas[5], sizeof(datas[5]), "New K+V", KVDIM * sizeof(uint16_t) * 2);
    data_label(datas[6], sizeof(datas[6]), "KV cache", cache_size);
    data_label(datas[7], sizeof(datas[7]), "Q8 buffer", q8size);
    data_label(datas[8], sizeof(datas[8]), "Wo Q4_0", qsize);

    double a[9], b[9], ta=0.0, tb=0.0;
    phase_vector(st,a); phase_vector(fl,b);
    puts("\nQwen2.5-Coder-1.5B | single attention sublayer | STANDARD vs FLASH");
    printf("Context: %d | Repeats: %d per mode | RVV quantized dot: %s\n",
           context, repeat, HAVE_RVV ? "yes" : "no (scalar fallback)");
    puts("Weights Q4_0 | Activations Q8_0 | KV FP16 | Same tensors and W_O kernel");
    puts("FLASH = existing streaming online-softmax; STANDARD = materialized QK scores.");
    puts("Timings exclude model loading, synthetic setup, captures and output comparison.\n");
    puts("----------------------------------------------------------------------------------------------------------------------");
    printf(" %3s  %-21s %-18s %-21s %12s %12s %10s\n",
           "#", "Operation", "Output", "Weights/data", "Std ms/run", "Flash ms/run", "Speedup");
    puts("----------------------------------------------------------------------------------------------------------------------");
    for (int i=0; i<9; ++i) {
        ta += a[i]; tb += b[i];
        const double ams=a[i]*1000.0/repeat, bms=b[i]*1000.0/repeat;
        printf(" %3d  %-21s %-18s %-21s %12.3f %12.3f %9.3fx\n",
               i+1,ops[i],shapes[i],datas[i],ams,bms,bms>0?ams/bms:0.0);
    }
    puts("----------------------------------------------------------------------------------------------------------------------");
    printf("      %-21s %-18s %-21s %12.3f %12.3f %9.3fx\n",
           "TOTAL (timed)","","",ta*1000.0/repeat,tb*1000.0/repeat,
           tb>0?ta/tb:0.0);
    double fa=(a[6]+a[7]+a[8])*1000.0/repeat;
    double fb=(b[6]+b[7]+b[8])*1000.0/repeat;
    printf("      %-21s %-18s %-21s %12.3f %12.3f %9.3fx\n",
           "Future fusion region","attention + Q8 + W_O", "",
           fa,fb,fb>0?fa/fb:0.0);

    double max_att_diff=0.0, max_att_rel=0.0;
    double max_out_diff=0.0, max_out_rel=0.0;
    double st_sum=0.0, fl_sum=0.0;
    for (int i=0; i<HIDDEN; ++i) {
        if (!isfinite(out_st[i]) || !isfinite(out_fl[i]) ||
            !isfinite(att_st[i]) || !isfinite(att_fl[i]))
            die("Non-finite output in attention comparison");
        double d=fabs((double)att_st[i]-att_fl[i]);
        double denom=fmax(fabs((double)att_st[i]),fabs((double)att_fl[i]));
        if(d>max_att_diff) max_att_diff=d;
        if(denom>1e-6 && d/denom>max_att_rel) max_att_rel=d/denom;
        d=fabs((double)out_st[i]-out_fl[i]);
        denom=fmax(fabs((double)out_st[i]),fabs((double)out_fl[i]));
        if(d>max_out_diff) max_out_diff=d;
        if(denom>1e-6 && d/denom>max_out_rel) max_out_rel=d/denom;
        st_sum += out_st[i]; fl_sum += out_fl[i];
    }
    int changed_q8=0;
    for (int i=0; i<HIDDEN/QK; ++i)
        changed_q8 += memcmp(&q8_st[i],&q8_fl[i],sizeof(block_q8_0))!=0;
    printf("\nCorrectness (same projected inputs; FP32 rounding may differ):\n");
    printf("  Attention O[1536]: max_abs=%.7g, max_rel=%.7g\n",max_att_diff,max_att_rel);
    printf("  Quantized attention: %d / %d Q8_0 blocks differ\n",changed_q8,HIDDEN/QK);
    printf("  Final W_O result:   max_abs=%.7g, max_rel=%.7g\n",max_out_diff,max_out_rel);
    printf("  Output checksums: standard=%.9g, flash=%.9g\n",st_sum,fl_sum);
    if (max_att_diff > 1e-3 || changed_q8 > 0)
        puts("  NOTE: The online and standard reductions have different rounding; check Q8 differences before interpreting projection speedup.");
    puts("\nThis is attention-kernel comparison, not attention-to-W_O projection fusion.");
    puts("Std/Flash timings reflect C implementations, not optimized RVV FlashAttention-V.");
    puts("Mode order alternates on successive repeats to reduce first-run bias.\n");
}

static void fused_stats_add(fused_stats_t *sum, fused_stats_t p) {
    phase_add(&sum->phase, p.phase);
    sum->tcm_stage += p.tcm_stage;
}

static double checksum_and_validate(const float *y) {
    double v=0.0;
    for (int i=0; i<HIDDEN; ++i) {
        if (!isfinite(y[i])) die("Non-finite fused output");
        v += (double)y[i];
    }
    return v;
}

static void report_fusion(const phase_t *base, const fused_stats_t *fused,
                          int context, int repeat, int tcm_mode,
                          const float *y_ref, const float *y_fused,
                          const float *att_ref, const float *att_fused,
                          const block_q8_0 *q8_ref,
                          const block_q8_0 *q8_fused) {
    const char *names[9] = { "RMSNorm", "Quantize X", "Wq", "Wk", "Wv",
                             "RoPE + KV append", "Online attention (GQA)",
                             "Quantize O", "W_o" };
    double a[9], b[9], base_total=0.0;
    phase_vector(base,a);
    phase_vector(&fused->phase,b);
    for (int i=0;i<9;i++) base_total += a[i];
    double fused_compute = fused_total(fused) - fused->tcm_stage;
    printf("\nQwen2.5-Coder-1.5B | BASELINE vs 2-HEAD COLUMN-SoA FUSED\n");
    printf("One token | context=%d | repeats=%d/mode | RVV=%s | TCM=%s\n",
           context,repeat,HAVE_RVV ? "yes" : "no",tcm_mode ? "enabled" : "off");
    printf("All weights remain 4-bit Q4_0; activations Q8_0; KV FP16.\n");
    printf("One token, two heads, 256 output neurons per RVV tile.\n");
    if (HAVE_RVV) {
        puts("NOTE: Fused 'Online attention' timing now INCLUDES Q projection, Q RoPE, attention, Q8 and W_O.");
        puts("There are NO stage timers inside that register-resident core.");
    }
    puts("--------------------------------------------------------------------------");
    printf(" %-24s %15s %15s %10s\n", "Operation", "Baseline ms", "Fused ms", "Speedup");
    puts("--------------------------------------------------------------------------");
    for (int i=0;i<9;i++) {
        if (i==4) continue; // K/V are interleaved and timed together.
        double av=(a[i]+(i==3?a[4]:0))*1000.0/repeat;
        double fv=(b[i]+(i==3?b[4]:0))*1000.0/repeat;
        const char *label = i==3 ? "Wk + Wv (interleaved)" : names[i];
        printf(" %-24s %15.3f %15.3f %9.3fx\n",
               label,av,fv,fv>0 ? av/fv : 0.0);
    }
    puts("--------------------------------------------------------------------------");
    printf(" %-24s %15.3f %15.3f %9.3fx\n", "Compute only",
           base_total*1000/repeat, fused_compute*1000/repeat,
           fused_compute>0 ? base_total/fused_compute:0.0);
    printf(" %-24s %15s %15.3f\n", "TCM staging (14 tiles)", "-",
           fused->tcm_stage*1000/repeat);
    printf(" %-24s %15.3f %15.3f %9.3fx\n", "TOTAL (inc. staging)",
           base_total*1000/repeat, fused_total(fused)*1000/repeat,
           fused_total(fused)>0?base_total/fused_total(fused):0.0);
    double aw=(a[6]+a[7]+a[8])*1000/repeat;
    double bw=(b[6]+b[7]+b[8])*1000/repeat;
    printf(" %-24s %15.3f %15.3f %9.3fx\n", "Attention + Q8 + W_o",
           aw,bw,bw>0?aw/bw:0.0);
    double max_att=0, max_y=0, max_rel=0;
    for (int i=0;i<HIDDEN;i++) {
        double da=fabs((double)att_ref[i]-att_fused[i]);
        double dy=fabs((double)y_ref[i]-y_fused[i]);
        double denominator=fmax(fabs((double)y_ref[i]),fabs((double)y_fused[i]));
        if (da>max_att) max_att=da;
        if (dy>max_y) max_y=dy;
        if (denominator>1e-6 && dy/denominator>max_rel) max_rel=dy/denominator;
    }
    int q8diff=0;
    for (int i=0;i<HIDDEN/QK;i++)
        q8diff += memcmp(&q8_ref[i], &q8_fused[i],sizeof(block_q8_0)) != 0;
    printf("\nCorrectness: attention max_abs=%.8g, Q8 blocks different=%d/48\n",
           max_att,q8diff);
    printf("             output max_abs=%.8g, output max_rel=%.8g\n",max_y,max_rel);
    printf("             checksum baseline=%.10g fused=%.10g\n",
           checksum_and_validate(y_ref),checksum_and_validate(y_fused));
    puts("NOTE: vector dot-reduction and float accumulation may round differently.");
    puts("TCM staging is counted separately; initial file loading and repacking are not.");
    puts("Fused uses paired attention, not a prefill tile or a full-model inference engine.\n");
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 5) {
        fprintf(stderr,
                "Usage: %s <weights_directory> [context=256] [repeat=1] [standard|flash|compare|fused|fused-compare|reference|soa-staged|soa-fused]\n",
                argv[0]);
        return 2;
    }
    char *end = NULL;
    long context = argc >= 3 ? strtol(argv[2], &end, 10) : 256;
    if (argc >= 3 && *end) die("Invalid context argument");
    long repeat = argc >= 4 ? strtol(argv[3], &end, 10) : 1;
    if (argc >= 4 && *end) die("Invalid repeat argument");
    if (context < 1 || context > 32768 || repeat < 1 || repeat > 100000)
        die("Context must be 1..32768 and repeat 1..100000");
    const char *mode = argc >= 5 ? argv[4] : "compare";
    int compare = strcmp(mode,"compare")==0;
    int reference_mode = strcmp(mode,"reference")==0;
    int staged_mode = strcmp(mode,"soa-staged")==0;
    int fused_mode = strcmp(mode,"fused")==0 ||
                     strcmp(mode,"fused-compare")==0 ||
                     strcmp(mode,"soa-fused")==0 || staged_mode;
    int fused_compare = strcmp(mode,"fused-compare")==0;
    int selected = strcmp(mode,"standard")==0 ? ATTENTION_STANDARD : ATTENTION_FLASH;
    if (!compare && !fused_mode && !reference_mode &&
        strcmp(mode,"standard") && strcmp(mode,"flash"))
        die("Attention mode must be standard, flash, compare, fused, fused-compare, reference, soa-staged or soa-fused");

    layer_t l = load_layer(argv[1]);
    float *input = malloc(HIDDEN * sizeof(float));
    uint16_t *kc = malloc((size_t)context * KVDIM * sizeof(uint16_t));
    uint16_t *vc = malloc((size_t)context * KVDIM * sizeof(uint16_t));
    float *scores = malloc((size_t)QHEADS * (size_t)context * sizeof(float));
    float *output[2] = {malloc(HIDDEN*sizeof(float)), NULL};
    float *attn[2] = {NULL,NULL};
    block_q8_0 *q8[2] = {NULL,NULL};
    if (compare) {
        output[1]=malloc(HIDDEN*sizeof(float));
        attn[0]=malloc(HIDDEN*sizeof(float));
        attn[1]=malloc(HIDDEN*sizeof(float));
        q8[0]=malloc((HIDDEN/QK)*sizeof(block_q8_0));
        q8[1]=malloc((HIDDEN/QK)*sizeof(block_q8_0));
    }
    if (!input || !kc || !vc || !scores || !output[0] ||
        (compare && (!output[1] || !attn[0] || !attn[1] || !q8[0] || !q8[1])))
        die("Out of memory creating synthetic inputs and comparison buffers");
    for (int i=0;i<HIDDEN;i++) input[i]=random_float();
    fill_kv_history(kc,vc,(int)context);

    phase_t sums[2]={0};
    if (fused_mode) {
        const char *tcmstr=getenv("FUSED_TCM");
        int tcm_enabled=tcmstr && atoi(tcmstr)!=0;
        const char *blockstr=getenv("TCM_BLOCK");
        int tcm_block=blockstr ? atoi(blockstr) : 0;
        fused_weights_t fw=fused_weights_load(argv[1], tcm_enabled, tcm_block);
        float *fy=malloc(HIDDEN*sizeof(float));
        float *a0=malloc(HIDDEN*sizeof(float));
        float *a1=malloc(HIDDEN*sizeof(float));
        block_q8_0 *b0=malloc((HIDDEN/QK)*sizeof(block_q8_0));
        block_q8_0 *b1=malloc((HIDDEN/QK)*sizeof(block_q8_0));
        if (!fy||!a0||!a1||!b0||!b1) die("Out of memory for fused comparison");
        fused_stats_t fs={0};
        if (fused_compare) {
            // Both use the same already-loaded tensors; alternate run order.
            for (int i=0;i<4;i++) {
                if (i==0 || i==3)
                    (void)run_layer(&l,input,kc,vc,(int)context,output[0],
                                    ATTENTION_FLASH,scores,NULL,NULL);
                else
                    (void)run_layer_fused(&l,&fw,input,kc,vc,(int)context,fy,NULL,NULL);
            }
            for (long r=0;r<repeat;r++) {
                for (int j=0;j<2;j++) {
                    int method = (r&1L) ? 1-j : j;
                    if (method==0) {
                        phase_t pp=run_layer(&l,input,kc,vc,(int)context,output[0],
                                             ATTENTION_FLASH,scores,NULL,NULL);
                        phase_add(&sums[0],pp);
                    } else {
                        fused_stats_t fp=run_layer_fused(&l,&fw,input,kc,vc,
                                               (int)context,fy,NULL,NULL);
                        fused_stats_add(&fs,fp);
                    }
                }
            }
            /* Capture only during an untimed replay. Capturing O/Q8 to RAM
             * in the measured fused path would defeat register residency. */
            (void)run_layer(&l,input,kc,vc,(int)context,output[0],
                            ATTENTION_FLASH,scores,a0,b0);
            (void)run_layer_fused(&l,&fw,input,kc,vc,(int)context,fy,a1,b1);
            report_fusion(&sums[0],&fs,(int)context,(int)repeat,tcm_enabled,
                          output[0],fy,a0,a1,b0,b1);
        } else {
            for (long r=0;r<repeat;r++) {
                fused_stats_t fp = staged_mode
                    ? run_layer_soa_staged(&l,&fw,input,kc,vc,
                                           (int)context,fy,NULL,NULL)
                    : run_layer_fused(&l,&fw,input,kc,vc,
                                      (int)context,fy,NULL,NULL);
                fused_stats_add(&fs,fp);
            }
            double ch=checksum_and_validate(fy);
            printf("%s one-token | context=%ld repeats=%ld RVV=%s TCM=%s\n",
                   staged_mode ? "SOA STAGED" : "REGISTER FUSED",
                   context,repeat,HAVE_RVV?"yes":"no", tcm_enabled?"yes":"no");
            printf("Compute %.3f ms/token, staging %.3f ms/token, all %.3f ms/token\n",
                   (fused_total(&fs)-fs.tcm_stage)*1e3/repeat,
                   fs.tcm_stage*1e3/repeat,fused_total(&fs)*1e3/repeat);
            printf("Output checksum %.10g | first four %.6g %.6g %.6g %.6g\n",
                   ch,fy[0],fy[1],fy[2],fy[3]);
        }
        free(fy);free(a0);free(a1);free(b0);free(b1);
        fused_weights_free(&fw);
    } else if (compare) {
        // Alternating-order warmup, excluded from all measurements.
        for (int i=0;i<4;i++) {
            int method = (i==0 || i==3) ? ATTENTION_STANDARD : ATTENTION_FLASH;
            (void)run_layer(&l,input,kc,vc,(int)context,output[method],
                            (attention_method_t)method,scores,NULL,NULL);
        }
        for (long r=0;r<repeat;r++) {
            for (int j=0;j<2;j++) {
                int method = ((r&1L)==0) ? j : 1-j;
                phase_t p=run_layer(&l,input,kc,vc,(int)context,output[method],
                                    (attention_method_t)method,scores,
                                    attn[method],q8[method]);
                phase_add(&sums[method],p);
            }
        }
        report_compare(&sums[0],&sums[1],&l,(int)context,(int)repeat,
                       output[0],output[1],attn[0],attn[1],q8[0],q8[1]);
    } else {
        for (long r=0;r<repeat;r++) {
            phase_t p=run_layer(&l,input,kc,vc,(int)context,output[0],
                                (attention_method_t)selected,scores,NULL,NULL);
            phase_add(&sums[0],p);
        }
        report(&sums[0],&l,(int)context,(int)repeat,output[0],
               (attention_method_t)selected);
    }
    free(input); free(kc); free(vc); free(scores);
    for (int i=0;i<2;i++) {free(output[i]);free(attn[i]);free(q8[i]);}
    free_layer(&l);
    return 0;
}
