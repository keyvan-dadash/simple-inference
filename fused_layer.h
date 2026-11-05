#pragma once
#include "defines.h"

/*
 * One-token, two-query-head, column-major Q4_0 x Q8_0 fused path.
 *
 * Expects to be included AFTER the following existing definitions from
 * qwen_one_layer.c: layer_t, phase_t, block_q8_0, die(), seconds_now(),
 * f16_to_f32(), f32_to_f16(), rmsnorm(), quantize_q8(), rope().
 *
 * Binary repack format is defined in repack_weights.py:
 *   one 240-KiB tile: 48 x [32 x 128-byte packed column + 256 x f32 scales].
 *   each packed byte contains two Q4_0 weights from different output rows:
 *      low nibble: output j, high nibble: output j + 128, for j=0..127.
 *   the scales are f32 conversions of original Q4_0 f16 scales, unchanged
 *   numerically. The integer multiplication is still Q4_0 x Q8_0.
 *
 * Wq: 6 tiles (2 query heads each)
 * KV: 2 tiles (128 K outputs followed by 128 V outputs)
 * Wo: 6 tiles (one per ATTENTION INPUT head pair), each containing
 *     6 subtiles of 40KiB (256 output rows x 256 input values).
 *
 * TCM is opt-in: FUSED_TCM=1, TCM_BLOCK=0. One 240KiB tile is staged
 * at a time. Only use with the local K3 /dev/tcm driver and exclusive access.
 */
#include <fcntl.h>
#include <sched.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#define FUSED_COL_BYTES 128u
#define FUSED_GROUP_BYTES 5120u
#define FUSED_TILE_BYTES (48u * FUSED_GROUP_BYTES)
#define FUSED_WO_SUBTILE (8u * FUSED_GROUP_BYTES)

_Static_assert(HIDDEN == 1536 && HDIM == 128 && QK == 32,
               "This isolated fused kernel is specialized for Qwen2.5-Coder-1.5B");
_Static_assert(FUSED_TILE_BYTES == 245760, "Repacker tile geometry mismatch");

typedef struct {
    void *base;
    size_t block_size;
    size_t block_num;
} fused_tcm_info_t;

typedef struct {
    uint32_t block_id;
    uint32_t reserved;
    uint64_t phys;
    uint64_t size;
    uint64_t cpu_affinity_mask;
} fused_tcm_block_t;

#define FUSED_TCM_INFO_GET _IOR('c', 7, int)
#define FUSED_TCM_BLOCK_INFO_GET _IOWR('c', 9, int)

typedef struct {
    uint8_t *q, *kv, *o;
    int tcm_fd;
    uint8_t *tcm_map;
    uint8_t *tcm_tile;
    size_t tcm_map_bytes;
    int tcm_block;
} fused_weights_t;

typedef struct {
    phase_t phase;
    double tcm_stage; /* measured independently; included in total time */
} fused_stats_t;

static void *fused_load_blob(const char *dir, const char *name, size_t expected) {
    char filename[1024];
    int n = snprintf(filename, sizeof(filename), "%s/%s", dir, name);
    if (n < 0 || (size_t)n >= sizeof(filename)) die("Fused weight path too long");
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Missing %s: run `make repack WEIGHTS=%s` first\n", filename, dir);
        die("Cannot open repacked column-SoA weights");
    }
    if (fseek(fp, 0, SEEK_END) || ftell(fp) != (long)expected || fseek(fp, 0, SEEK_SET)) {
        fprintf(stderr, "Unexpected size for %s: expected %zu bytes\n", filename, expected);
        die("Invalid repacked weights: regenerate with repack_weights.py");
    }
    uint8_t *buf = malloc(expected);
    if (!buf || fread(buf, 1, expected, fp) != expected) die("Cannot load fused weights");
    fclose(fp);
    return buf;
}

static void fused_tcm_open(fused_weights_t *fw, int block) {
    if (block < 0) die("TCM_BLOCK must be non-negative");
    int fd = open("/dev/tcm", O_RDWR | O_CLOEXEC);
    if (fd < 0) { perror("/dev/tcm"); die("FUSED_TCM requested but device is unavailable"); }
    fused_tcm_info_t info = {0};
    if (ioctl(fd, FUSED_TCM_INFO_GET, &info) != 0 || info.block_num == 0 ||
        info.block_size < FUSED_TILE_BYTES || (size_t)block >= info.block_num) {
        perror("TCM_INFO_GET"); close(fd);
        die("TCM driver geometry unsuitable for a 240KiB tile");
    }
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        perror("flock(/dev/tcm)"); close(fd);
        die("TCM is owned by another process; avoid concurrent use");
    }
    fused_tcm_block_t bi = {.block_id = (uint32_t)block};
    if (ioctl(fd, FUSED_TCM_BLOCK_INFO_GET, &bi) != 0 ||
        bi.size != info.block_size || bi.cpu_affinity_mask == 0) {
        perror("TCM_BLOCK_INFO_GET"); close(fd);
        die("TCM block query failed");
    }
    int cpu = sched_getcpu();
    if (cpu < 0 || cpu >= 64 || !(bi.cpu_affinity_mask & (UINT64_C(1) << cpu))) {
        fprintf(stderr,"TCM block %d affinity=0x%016" PRIx64 ", CPU=%d\n",
                block, bi.cpu_affinity_mask, cpu);
        close(fd); die("Selected TCM block is not local to this CPU");
    }
    if (info.block_size > SIZE_MAX / info.block_num) { close(fd); die("TCM geometry overflow"); }
    size_t bytes = info.block_size * info.block_num;
    void *map = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { perror("mmap(/dev/tcm)"); close(fd); die("Unable to map TCM"); }
    fw->tcm_fd = fd;
    fw->tcm_map = map;
    fw->tcm_map_bytes = bytes;
    fw->tcm_tile = fw->tcm_map + (size_t)block * info.block_size;
    fw->tcm_block = block;
    fprintf(stderr,"FUSED_TCM enabled: block=%d, CPU=%d, block_bytes=%zu, tile_bytes=%u\n",
            block, cpu, info.block_size, (unsigned)FUSED_TILE_BYTES);
}

static fused_weights_t fused_weights_load(const char *dir, int use_tcm, int block) {
    fused_weights_t fw = {0};
    fw.tcm_fd = -1;
    fw.q = fused_load_blob(dir, "q.colsoa", 6u * FUSED_TILE_BYTES);
    fw.kv = fused_load_blob(dir, "kv.colsoa", 2u * FUSED_TILE_BYTES);
    fw.o = fused_load_blob(dir, "o.colsoa", 6u * FUSED_TILE_BYTES);
    if (use_tcm) fused_tcm_open(&fw, block);
    return fw;
}

static void fused_weights_free(fused_weights_t *fw) {
    if (fw->tcm_map) munmap(fw->tcm_map, fw->tcm_map_bytes);
    if (fw->tcm_fd >= 0) close(fw->tcm_fd);
    free(fw->q); free(fw->kv); free(fw->o);
    memset(fw, 0, sizeof(*fw));
}

static const uint8_t *fused_stage_tile(const fused_weights_t *fw,
                                      const uint8_t *from, fused_stats_t *stats) {
    if (!fw->tcm_tile) return from;
    double start = seconds_now();
    memcpy(fw->tcm_tile, from, FUSED_TILE_BYTES);
    stats->tcm_stage += seconds_now() - start;
    return fw->tcm_tile;
}

/*
 * RVV lanes are OUTPUT neurons, not input features.
 * Each e8m1 128-byte vector contains 128 pairs of INT4 weights, representing
 * a 256-output column. Two e32m4 accumulators cover all 256 outputs.
 * Each Q8 activation is read as one scalar from ordinary memory and reused
 * across the 256 integer lanes. The FP32 scales are loaded only at original
 * Q4_0/Q8_0 block boundaries, not on every activation.
 *
 * bias_lo and bias_hi are two optional 128-element arrays, used when add=0.
 * When add=1, the existing 256 FP32 output accumulators are RVV-loaded.
 * The same routine runs Wq, interleaved Wk/Wv, and W_O partial updates.
 */
static void fused_column_matvec(const uint8_t *tile,
                                const block_q8_0 *x,
                                int nblocks, const float *bias_lo,
                                const float *bias_hi,
                                float *output, int add) {
#if HAVE_RVV
    const size_t vl = __riscv_vsetvl_e32m4(128);
    if (vl != 128 || __riscv_vsetvl_e8m1(128) != 128)
        die("Fused RVV kernel expects VLEN >= 1024 bits (A100)");

    vfloat32m4_t out0 = add ? __riscv_vle32_v_f32m4(output, vl) :
        (bias_lo ? __riscv_vle32_v_f32m4(bias_lo, vl) :
                   __riscv_vfmv_v_f_f32m4(0.0f, vl));
    vfloat32m4_t out1 = add ? __riscv_vle32_v_f32m4(output + 128, vl) :
        (bias_hi ? __riscv_vle32_v_f32m4(bias_hi, vl) :
                   __riscv_vfmv_v_f_f32m4(0.0f, vl));

    for (int b = 0; b < nblocks; ++b) {
        const uint8_t *group = tile + (size_t)b * FUSED_GROUP_BYTES;
        vint32m4_t int0 = __riscv_vmv_v_x_i32m4(0, vl);
        vint32m4_t int1 = __riscv_vmv_v_x_i32m4(0, vl);
        for (int k = 0; k < QK; ++k) {
            const int16_t activation = x[b].qs[k]; // scalar L1 path
            vuint8m1_t packed = __riscv_vle8_v_u8m1(
                group + (size_t)k * FUSED_COL_BYTES, vl);
            vuint8m1_t lowbits = __riscv_vsub_vx_u8m1(
                __riscv_vand_vx_u8m1(packed, 15, vl), 8, vl);
            vuint8m1_t highbits = __riscv_vsub_vx_u8m1(
                __riscv_vsrl_vx_u8m1(packed, 4, vl), 8, vl);
            vint16m2_t low = __riscv_vsext_vf2_i16m2(
                __riscv_vreinterpret_v_u8m1_i8m1(lowbits), vl);
            vint16m2_t high = __riscv_vsext_vf2_i16m2(
                __riscv_vreinterpret_v_u8m1_i8m1(highbits), vl);
            int0 = __riscv_vwmacc_vx_i32m4(int0, activation, low, vl);
            int1 = __riscv_vwmacc_vx_i32m4(int1, activation, high, vl);
        }
        const float *scales = (const float *)(const void *)(group + 4096);
        const float dx = f16_to_f32(x[b].d);
        vfloat32m4_t d0 = __riscv_vle32_v_f32m4(scales, vl);
        vfloat32m4_t d1 = __riscv_vle32_v_f32m4(scales + 128, vl);
        vfloat32m4_t c0 = __riscv_vfcvt_f_x_v_f32m4(int0, vl);
        vfloat32m4_t c1 = __riscv_vfcvt_f_x_v_f32m4(int1, vl);
        c0 = __riscv_vfmul_vv_f32m4(c0, d0, vl);
        c1 = __riscv_vfmul_vv_f32m4(c1, d1, vl);
        c0 = __riscv_vfmul_vf_f32m4(c0, dx, vl);
        c1 = __riscv_vfmul_vf_f32m4(c1, dx, vl);
        out0 = __riscv_vfadd_vv_f32m4(out0, c0, vl);
        out1 = __riscv_vfadd_vv_f32m4(out1, c1, vl);
    }
    __riscv_vse32_v_f32m4(output, out0, vl);
    __riscv_vse32_v_f32m4(output + 128, out1, vl);
#else
    if (!add) {
        for (int j = 0; j < 128; ++j) {
            output[j] = bias_lo ? bias_lo[j] : 0.0f;
            output[j + 128] = bias_hi ? bias_hi[j] : 0.0f;
        }
    }
    for (int b = 0; b < nblocks; ++b) {
        const uint8_t *group = tile + (size_t)b * FUSED_GROUP_BYTES;
        const float *scales = (const float *)(const void *)(group + 4096);
        int32_t sums[256] = {0};
        for (int k = 0; k < QK; ++k) {
            int xval = x[b].qs[k];
            const uint8_t *column = group + (size_t)k * FUSED_COL_BYTES;
            for (int j = 0; j < 128; ++j) {
                uint8_t w = column[j];
                sums[j] += ((int)(w & 15) - 8) * xval;
                sums[j + 128] += ((int)(w >> 4) - 8) * xval;
            }
        }
        float dx = f16_to_f32(x[b].d);
        for (int j = 0; j < 256; ++j)
            output[j] += ((float)sums[j] * scales[j]) * dx;
    }
#endif
}

/* Computes both query heads against their shared FP16 K/V cache.
 * Scores, max, normalizer and expf remain scalar; RVV carries 128-element
 * Q, K, V, and output accumulators. No score matrix is materialized.
 */
static void fused_attention_pair(const float *qpair,
                                 const uint16_t *kc, const uint16_t *vc,
                                 int context, int group, float *opair) {
    const float inv_dim = 1.0f / sqrtf((float)HDIM);
    const size_t kv_stride = (size_t)KVDIM;
#if HAVE_RVV
    const size_t vl = __riscv_vsetvl_e32m4(128);
    if (vl != 128) die("Fused attention expects A100 VLEN >=1024");
    vfloat32m4_t q0 = __riscv_vle32_v_f32m4(qpair, vl);
    vfloat32m4_t q1 = __riscv_vle32_v_f32m4(qpair + HDIM, vl);
    vfloat32m4_t u0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t u1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m1_t zerov = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    float keyf[HDIM], valf[HDIM];
    for (int t = 0; t < context; ++t) {
        const uint16_t *k = kc + (size_t)t * kv_stride + (size_t)group * HDIM;
        const uint16_t *v = vc + (size_t)t * kv_stride + (size_t)group * HDIM;
        for (int d = 0; d < HDIM; ++d) keyf[d] = f16_to_f32(k[d]);
        vfloat32m4_t key = __riscv_vle32_v_f32m4(keyf, vl);
        vfloat32m4_t prod0 = __riscv_vfmul_vv_f32m4(q0, key, vl);
        vfloat32m4_t prod1 = __riscv_vfmul_vv_f32m4(q1, key, vl);
        vfloat32m1_t r0 = __riscv_vfredusum_vs_f32m4_f32m1(prod0, zerov, vl);
        vfloat32m1_t r1 = __riscv_vfredusum_vs_f32m4_f32m1(prod1, zerov, vl);
        float s0 = __riscv_vfmv_f_s_f32m1_f32(r0) * inv_dim;
        float s1 = __riscv_vfmv_f_s_f32m1_f32(r1) * inv_dim;
        // Same one-exp-per-token update as the optimized reference.
        float a0, a1, b0, b1;
        if (t == 0) {
            m0 = s0; m1 = s1;
            l0 = l1 = 1.0f;
            a0 = a1 = 0.0f;
            b0 = b1 = 1.0f;
        } else {
            if (s0 > m0) {
                a0 = expf(m0 - s0); b0 = 1.0f;
                l0 = l0 * a0 + 1.0f; m0 = s0;
            } else {
                a0 = 1.0f; b0 = expf(s0 - m0);
                l0 += b0;
            }
            if (s1 > m1) {
                a1 = expf(m1 - s1); b1 = 1.0f;
                l1 = l1 * a1 + 1.0f; m1 = s1;
            } else {
                a1 = 1.0f; b1 = expf(s1 - m1);
                l1 += b1;
            }
        }
        for (int d = 0; d < HDIM; ++d) valf[d] = f16_to_f32(v[d]);
        vfloat32m4_t values = __riscv_vle32_v_f32m4(valf, vl);
        if (t == 0) {
            u0 = values; u1 = values;
        } else {
            if (a0 == 1.0f)
                u0 = __riscv_vfadd_vv_f32m4(u0,
                    __riscv_vfmul_vf_f32m4(values, b0, vl), vl);
            else
                u0 = __riscv_vfadd_vv_f32m4(
                    __riscv_vfmul_vf_f32m4(u0, a0, vl), values, vl);
            if (a1 == 1.0f)
                u1 = __riscv_vfadd_vv_f32m4(u1,
                    __riscv_vfmul_vf_f32m4(values, b1, vl), vl);
            else
                u1 = __riscv_vfadd_vv_f32m4(
                    __riscv_vfmul_vf_f32m4(u1, a1, vl), values, vl);
        }
    }
    u0 = __riscv_vfmul_vf_f32m4(u0, 1.0f/l0, vl);
    u1 = __riscv_vfmul_vf_f32m4(u1, 1.0f/l1, vl);
    __riscv_vse32_v_f32m4(opair, u0, vl);
    __riscv_vse32_v_f32m4(opair + HDIM, u1, vl);
#else
    float m[2] = {-INFINITY, -INFINITY};
    float l[2] = {0.0f, 0.0f};
    memset(opair, 0, 256 * sizeof(float));
    for (int t = 0; t < context; ++t) {
        const uint16_t *k = kc + (size_t)t * kv_stride + (size_t)group * HDIM;
        const uint16_t *v = vc + (size_t)t * kv_stride + (size_t)group * HDIM;
        for (int h = 0; h < 2; ++h) {
            const float *qh = qpair + h * HDIM;
            float *out = opair + h * HDIM;
            float score = 0.0f;
            for (int d = 0; d < HDIM; ++d) score += qh[d] * f16_to_f32(k[d]);
            score *= inv_dim;
            if (t == 0) {
                m[h] = score; l[h] = 1.0f;
                for (int d=0;d<HDIM;d++) out[d]=f16_to_f32(v[d]);
            } else if (score > m[h]) {
                float alpha = expf(m[h] - score);
                l[h] = l[h] * alpha + 1.0f;
                for (int d=0;d<HDIM;d++) out[d] =
                    out[d] * alpha + f16_to_f32(v[d]);
                m[h] = score;
            } else {
                float beta = expf(score - m[h]);
                l[h] += beta;
                for (int d=0;d<HDIM;d++) out[d] += beta * f16_to_f32(v[d]);
            }
        }
    }
    for (int h = 0; h < 2; ++h)
        for (int d = 0; d < HDIM; ++d) opair[h*HDIM+d] *= 1.0f/l[h];
#endif
}

/* The experimental fusion boundary: after pair attention, quantize exactly
 * four Q8_0 blocks per head, and IMMEDIATELY apply the 256 input columns
 * of W_O to every Y output tile. Do not construct O[1536] or Oq[48] here.
 */
static void attention_wo_fused(const fused_weights_t *fw, const float *qpair,
                               const uint16_t *key_cache,
                               const uint16_t *value_cache,
                               int context, int group, int pair,
                               float *output_y, fused_stats_t *stats,
                               float *attention_capture,
                               block_q8_0 *quantized_capture) {
    float opair[256];
    block_q8_0 oq[8];
    double t = seconds_now();
    fused_attention_pair(qpair, key_cache, value_cache, context, group, opair);
    stats->phase.attention += (float)(seconds_now() - t);
    if (attention_capture)
        memcpy(attention_capture + pair*256, opair, sizeof(opair));

    t = seconds_now();
    quantize_q8(opair, oq, 256);
    stats->phase.q8_output += (float)(seconds_now() - t);
    if (quantized_capture)
        memcpy(quantized_capture + pair*8, oq, sizeof(oq));

    const uint8_t *wo = fused_stage_tile(fw,
        fw->o + (size_t)pair * FUSED_TILE_BYTES, stats);
    t = seconds_now();
    for (int ytile = 0; ytile < 6; ++ytile) {
        fused_column_matvec(wo + (size_t)ytile * FUSED_WO_SUBTILE,
                            oq, 8, NULL, NULL, output_y + ytile*256, 1);
    }
    stats->phase.out_proj += (float)(seconds_now() - t);
}

/* One token, 2 query heads at a time. Each K/V group computed once.
 * The input X and Q8 activations are ordinary scalar-cacheable memory.
 * Partial Y is a 6KiB ordinary-memory vector buffer, RVV updated six times.
 * Weight tiles are contiguous column-major Q4_0 SoA, optionally TCM staged.
 */
static fused_stats_t run_layer_legacy_fused(const layer_t *l,
                                     const fused_weights_t *fw,
                                     const float *x,
                                     uint16_t *kc, uint16_t *vc,
                                     int context, float *output,
                                     float *attn_capture,
                                     block_q8_0 *q8_capture) {
    float normed[HIDDEN];
    block_q8_0 inq8[HIDDEN/QK];
    fused_stats_t stats = {0};
    double t = seconds_now();
    rmsnorm(x, l->norm, normed);
    stats.phase.rms = (float)(seconds_now() - t);
    t = seconds_now();
    quantize_q8(normed, inq8, HIDDEN);
    stats.phase.q8_input = (float)(seconds_now() - t);

    memset(output, 0, HIDDEN*sizeof(float));
    for (int group = 0; group < KVHEADS; ++group) {
        const uint8_t *kv_tile = fused_stage_tile(fw,
            fw->kv + (size_t)group * FUSED_TILE_BYTES, &stats);
        float kvpair[256];
        t = seconds_now();
        fused_column_matvec(kv_tile, inq8, HIDDEN/QK,
                           l->bk ? l->bk + group*HDIM : NULL,
                           l->bv ? l->bv + group*HDIM : NULL,
                           kvpair, 0);
        double kv_elapsed = seconds_now() - t;
        stats.phase.k_proj += (float)(kv_elapsed * 0.5);
        stats.phase.v_proj += (float)(kv_elapsed * 0.5);

        t = seconds_now();
        rope(kvpair, 1, context-1);
        size_t kv_offset = (size_t)(context-1)*KVDIM + (size_t)group*HDIM;
        for (int d = 0; d < HDIM; ++d) {
            kc[kv_offset + d] = f32_to_f16(kvpair[d]);
            vc[kv_offset + d] = f32_to_f16(kvpair[d+HDIM]);
        }
        stats.phase.rope_cache += (float)(seconds_now()-t);

        for (int p = 0; p < 3; ++p) {
            int pair = group*3 + p;
            const uint8_t *q_tile = fused_stage_tile(fw,
                fw->q + (size_t)pair * FUSED_TILE_BYTES, &stats);
            float qpair[256];
            t = seconds_now();
            const float *bq = l->bq ? l->bq + pair*256 : NULL;
            fused_column_matvec(q_tile, inq8, HIDDEN/QK,
                                bq, bq ? bq+128 : NULL, qpair, 0);
            stats.phase.q_proj += (float)(seconds_now()-t);
            t = seconds_now();
            rope(qpair, 2, context-1);
            stats.phase.rope_cache += (float)(seconds_now()-t);
            attention_wo_fused(fw, qpair, kc, vc, context, group, pair,
                               output, &stats, attn_capture, q8_capture);
        }
    }
    return stats;
}

static double fused_total(const fused_stats_t *p) {
    const phase_t *t = &p->phase;
    return (double)t->rms + t->q8_input + t->q_proj + t->k_proj + t->v_proj +
           t->rope_cache + t->attention + t->q8_output + t->out_proj +
           p->tcm_stage;
}
