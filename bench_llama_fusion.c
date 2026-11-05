/*
 * Compare the SAME llama.cpp-style row-major Q4_0 x Q8_0 RVV dot-product
 * path with and without pairwise attention -> Q8 -> W_O accumulation.
 *
 * Deliberately does not use repacked column-SoA weights or the alternative
 * paired-RVV attention kernel. This is a SCHEDULE/MATERIALIZATION control,
 * not a test of a different matmul implementation.
 *
 * Build with the repo Makefile: make build/bench_llama_fusion
 * Run: make llama-baseline CONTEXT=256 REPEAT=30 WEIGHTS=weights
 *
 * qwen_one_layer.c currently puts its reference functions in a single C
 * translation unit; rename its main so this benchmark can reuse them without
 * copy-pasting reference math.
 */
#define main qwen_original_cli_main_for_llama_baseline
#include "qwen_one_layer.c"
#undef main

#include <assert.h>

#define LLAMA_PAIR_SIZE (2 * HDIM)
#define LLAMA_PAIR_BLOCKS (LLAMA_PAIR_SIZE / QK)

_Static_assert(HIDDEN % LLAMA_PAIR_SIZE == 0, "Head pair must cover hidden");
_Static_assert(LLAMA_PAIR_BLOCKS == 8, "Unexpected pair geometry");

/* Directly mirrors attention_flash() for one head. No Q/K/V reordering,
 * no RVV pairwise reductions, no score buffer, and identical FP16 conversion
 * and softmax recurrence to the original row-major attention baseline.
 */
static void llama_reference_attention_one_head(
        const float *qhead, const uint16_t *kc, const uint16_t *vc,
        int context, int group, float *out) {
    const float scale = 1.0f / sqrtf((float)HDIM);
    const size_t kv_stride = (size_t)KVHEADS * HDIM;
    if (context <= 0) {
        memset(out, 0, HDIM * sizeof(float));
        return;
    }
    const uint16_t *k = kc + (size_t)group * HDIM;
    const uint16_t *v = vc + (size_t)group * HDIM;
    float score = 0.0f;
    for (int d = 0; d < HDIM; ++d) score += qhead[d] * f16_to_f32(k[d]);
    float max_score = score * scale;
    float normalizer = 1.0f;
    for (int d = 0; d < HDIM; ++d) out[d] = f16_to_f32(v[d]);
    for (int t = 1; t < context; ++t) {
        k += kv_stride;
        v += kv_stride;
        score = 0.0f;
        for (int d = 0; d < HDIM; ++d) score += qhead[d] * f16_to_f32(k[d]);
        score *= scale;
        if (score > max_score) {
            const float alpha = expf(max_score - score);
            normalizer = normalizer * alpha + 1.0f;
            for (int d = 0; d < HDIM; ++d)
                out[d] = out[d] * alpha + f16_to_f32(v[d]);
            max_score = score;
        } else {
            const float beta = expf(score - max_score);
            normalizer += beta;
            for (int d = 0; d < HDIM; ++d)
                out[d] += beta * f16_to_f32(v[d]);
        }
    }
    const float inv = 1.0f / normalizer;
    for (int d = 0; d < HDIM; ++d) out[d] *= inv;
}

/* W_O is stored in original row-major Q4_0 blocks. For one attention pair
 * (256 input features), traverse the 8 corresponding blocks of EACH of the
 * 1536 output neurons and accumulate into the partial Y buffer. Reuse the
 * EXACT SAME dot_32_q4q8() function as the original reference matvec.
 *
 * The total dot-product count remains 6 * 1536 * 8 = 73728, the same as
 * the original W_O. Only the order (and extra Y loads/stores) changes.
 */
static void llama_reference_wo_pair_update(
        const uint8_t *wo, const block_q8_0 pair_q8[LLAMA_PAIR_BLOCKS],
        int pair, float *y) {
    const int blocks_per_row = HIDDEN / QK;
    for (int row = 0; row < HIDDEN; ++row) {
        const uint8_t *wrow = wo +
            ((size_t)row * blocks_per_row + (size_t)pair * LLAMA_PAIR_BLOCKS)
                * Q4_BYTES;
        float sum = y[row];
        for (int b = 0; b < LLAMA_PAIR_BLOCKS; ++b) {
            const uint8_t *wb = wrow + (size_t)b * Q4_BYTES;
            uint16_t hw = (uint16_t)wb[0] | ((uint16_t)wb[1] << 8);
            const int32_t dot = dot_32_q4q8(wb, pair_q8[b].qs);
            sum += (float)dot * f16_to_f32(hw) * f16_to_f32(pair_q8[b].d);
        }
        y[row] = sum;
    }
}

/* Reference Q/K/V projections are intentionally unchanged. This isolates
 * the attention -> Q8_0 -> W_O scheduling change. Output starts at zero and
 * the fused path never constructs a 1536-element attention O buffer.
 */
static phase_t llama_reference_run_pair_schedule(
        const layer_t *l, const float *x, uint16_t *kc, uint16_t *vc,
        int context, float *output, float *attn_copy, block_q8_0 *q8_copy) {
    float normed[HIDDEN], q[HIDDEN], k[KVDIM], v[KVDIM];
    block_q8_0 inq8[HIDDEN / QK];
    float opair[LLAMA_PAIR_SIZE];
    block_q8_0 oq[LLAMA_PAIR_BLOCKS];
    phase_t p = {0};
    double t = seconds_now();
    rmsnorm(x, l->norm, normed);
    p.rms = (float)(seconds_now() - t);
    t = seconds_now();
    quantize_q8(normed, inq8, HIDDEN);
    p.q8_input = (float)(seconds_now() - t);
    t = seconds_now();
    matvec_q4q8(l->wq, inq8, HIDDEN, HIDDEN, l->bq, q);
    p.q_proj = (float)(seconds_now() - t);
    t = seconds_now();
    matvec_q4q8(l->wk, inq8, HIDDEN, KVDIM, l->bk, k);
    p.k_proj = (float)(seconds_now() - t);
    t = seconds_now();
    matvec_q4q8(l->wv, inq8, HIDDEN, KVDIM, l->bv, v);
    p.v_proj = (float)(seconds_now() - t);

    t = seconds_now();
    rope(q, QHEADS, context - 1);
    rope(k, KVHEADS, context - 1);
    const size_t current = (size_t)(context - 1) * KVDIM;
    for (int j = 0; j < KVDIM; ++j) {
        kc[current + (size_t)j] = f32_to_f16(k[j]);
        vc[current + (size_t)j] = f32_to_f16(v[j]);
    }
    p.rope_cache = (float)(seconds_now() - t);

    memset(output, 0, HIDDEN * sizeof(float));
    for (int pair = 0; pair < HIDDEN / LLAMA_PAIR_SIZE; ++pair) {
        const int h0 = pair * 2;
        const int h1 = h0 + 1;
        const int kv_group = h0 / (QHEADS / KVHEADS);
        t = seconds_now();
        llama_reference_attention_one_head(q + h0 * HDIM, kc, vc,
                                            context, kv_group, opair);
        llama_reference_attention_one_head(q + h1 * HDIM, kc, vc,
                                            context, kv_group, opair + HDIM);
        p.attention += (float)(seconds_now() - t);
        if (attn_copy) memcpy(attn_copy + pair * LLAMA_PAIR_SIZE,
                              opair, sizeof(opair));
        t = seconds_now();
        quantize_q8(opair, oq, LLAMA_PAIR_SIZE);
        p.q8_output += (float)(seconds_now() - t);
        if (q8_copy) memcpy(q8_copy + pair * LLAMA_PAIR_BLOCKS,
                            oq, sizeof(oq));
        t = seconds_now();
        llama_reference_wo_pair_update(l->wo, oq, pair, output);
        p.out_proj += (float)(seconds_now() - t);
    }
    return p;
}

static double llama_phase_total(phase_t p) {
    return (double)p.rms + p.q8_input + p.q_proj + p.k_proj + p.v_proj +
           p.rope_cache + p.attention + p.q8_output + p.out_proj;
}

static int llama_compare_double(const void *a, const void *b) {
    const double x = *(const double *)a;
    const double y = *(const double *)b;
    return (x > y) - (x < y);
}

static double llama_median(const double *samples, int n) {
    double *scratch = malloc((size_t)n * sizeof(*scratch));
    if (!scratch) die("Failed to allocate median scratch");
    memcpy(scratch, samples, (size_t)n * sizeof(*scratch));
    qsort(scratch, (size_t)n, sizeof(*scratch), llama_compare_double);
    double x = (n & 1) ? scratch[n / 2] :
        0.5 * (scratch[n / 2 - 1] + scratch[n / 2]);
    free(scratch);
    return x;
}

static void llama_compare_vec(const char *name, const float *a,
                              const float *b, int count) {
    double max_abs = 0.0, rms_diff = 0.0, max_rel = 0.0;
    int bad = 0, changed = 0, nonzero = 0;
    for (int j = 0; j < count; ++j) {
        if (!isfinite(a[j]) || !isfinite(b[j])) {
            fprintf(stderr, "%s invalid value at %d: %.8g / %.8g\n",
                    name, j, a[j], b[j]);
            exit(3);
        }
        const double diff = fabs((double)a[j] - b[j]);
        const double scale = fmax(fabs((double)a[j]), fabs((double)b[j]));
        if (diff > max_abs) max_abs = diff;
        if (scale > 1e-7 && diff / scale > max_rel) max_rel = diff / scale;
        rms_diff += diff * diff;
        bad += diff > 2e-5 + 2e-5 * scale;
        changed += diff != 0.0;
        nonzero += a[j] != 0.0f;
    }
    printf("%-12s max_abs=%-11.3g max_rel=%-11.3g rms_diff=%-11.3g "
           "different=%d/%d bad=%d nonzero=%d\n",
           name, max_abs, max_rel, sqrt(rms_diff / count), changed, count,
           bad, nonzero);
    if (bad || !nonzero) die("Reference/fused mismatch or all-zero output");
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "Usage: %s <weights> [context=256] [repeat=30]\n", argv[0]);
        return 2;
    }
    char *end = NULL;
    long context = argc > 2 ? strtol(argv[2], &end, 10) : 256;
    if (argc > 2 && *end) die("Invalid context");
    long repeat = argc > 3 ? strtol(argv[3], &end, 10) : 30;
    if (argc > 3 && *end) die("Invalid repeat");
    if (context < 1 || context > 32768 || repeat < 1 || repeat > 100000)
        die("Context must be 1..32768, repeat 1..100000");

    layer_t layer = load_layer(argv[1]);
    float *x = malloc(HIDDEN * sizeof(float));
    float *y[2], *attention[2];
    block_q8_0 *q8[2];
    uint16_t *kc[2], *vc[2];
    for (int m = 0; m < 2; ++m) {
        y[m] = malloc(HIDDEN * sizeof(float));
        attention[m] = malloc(HIDDEN * sizeof(float));
        q8[m] = malloc((HIDDEN / QK) * sizeof(block_q8_0));
        kc[m] = malloc((size_t)context * KVDIM * sizeof(uint16_t));
        vc[m] = malloc((size_t)context * KVDIM * sizeof(uint16_t));
        if (!y[m] || !attention[m] || !q8[m] || !kc[m] || !vc[m]) die("OOM");
    }
    float *scores = malloc((size_t)QHEADS * (size_t)context * sizeof(float));
    if (!x || !scores) die("OOM input/scores");
    for (int j = 0; j < HIDDEN; ++j) x[j] = random_float();
    fill_kv_history(kc[0], vc[0], (int)context);
    const size_t cache_bytes = (size_t)context * KVDIM * sizeof(uint16_t);
    memcpy(kc[1], kc[0], cache_bytes - KVDIM * sizeof(uint16_t));
    memcpy(vc[1], vc[0], cache_bytes - KVDIM * sizeof(uint16_t));

    printf("LLAMA-STYLE Q4_0 x Q8_0 RVV: STAGED vs PAIRED SCHEDULE\n");
    printf("context=%ld repeat=%ld RVV=%s | original row-major GGUF blocks\n",
           context, repeat, HAVE_RVV ? "yes" : "no");
    printf("No repacking, no TCM, same dot_32_q4q8 in BOTH paths.\n");

    for (int warm = 0; warm < 4; ++warm) {
        int m = (warm == 0 || warm == 3) ? 0 : 1;
        if (m == 0)
            (void)run_layer(&layer, x, kc[m], vc[m], (int)context, y[m],
                            ATTENTION_FLASH, scores, NULL, NULL);
        else
            (void)llama_reference_run_pair_schedule(
                  &layer, x, kc[m], vc[m], (int)context, y[m], NULL, NULL);
    }

    double *wall[2], meanwall[2] = {0}, meanphase[2] = {0};
    phase_t sumphase[2] = {0};
    for (int m = 0; m < 2; ++m) {
        wall[m] = malloc((size_t)repeat * sizeof(double));
        if (!wall[m]) die("OOM timing samples");
    }
    for (int r = 0; r < repeat; ++r) {
        for (int j = 0; j < 2; ++j) {
            const int m = (r & 1) ? 1 - j : j;
            const double begin = seconds_now();
            phase_t p;
            if (m == 0)
                p = run_layer(&layer, x, kc[m], vc[m], (int)context, y[m],
                              ATTENTION_FLASH, scores, NULL, NULL);
            else
                p = llama_reference_run_pair_schedule(
                         &layer, x, kc[m], vc[m], (int)context, y[m], NULL, NULL);
            const double elapsed = seconds_now() - begin;
            wall[m][r] = elapsed;
            meanwall[m] += elapsed;
            meanphase[m] += llama_phase_total(p);
            phase_add(&sumphase[m], p);
        }
    }
    puts("\nTIMING (complete call; ms)");
    printf("%-20s %12s %12s %12s %12s\n",
           "Path", "wall median", "wall mean", "stage mean", "gap mean");
    for (int m = 0; m < 2; ++m)
        printf("%-20s %12.4f %12.4f %12.4f %12.4f\n",
               m == 0 ? "row-major staged" : "row-major paired",
               llama_median(wall[m], (int)repeat) * 1e3,
               meanwall[m] / repeat * 1e3,
               meanphase[m] / repeat * 1e3,
               (meanwall[m] - meanphase[m]) / repeat * 1e3);
    const double base = llama_median(wall[0], (int)repeat);
    const double fused = llama_median(wall[1], (int)repeat);
    printf("schedule speedup (staged / paired): %.4fx\n", base / fused);
    puts("Uses exactly the same Q4_0 blocks, RVV dot product and scalar-head online softmax.");
    puts("The only algorithmic difference is pairwise attention consumption and partial-Y W_O updates.");
    puts("\nSTAGE TIMER MEANS (ms/call)");
    const char *names[] = {"RMSNorm", "Quantize X", "Wq", "Wk", "Wv",
                         "RoPE + cache", "Online attention", "Quantize O", "W_o"};
    double p0[9], p1[9];
    phase_vector(&sumphase[0], p0);
    phase_vector(&sumphase[1], p1);
    for (int i = 0; i < 9; ++i)
        printf("%-20s %12.4f %12.4f\n", names[i],
               p0[i] * 1e3 / repeat, p1[i] * 1e3 / repeat);

    (void)run_layer(&layer, x, kc[0], vc[0], (int)context, y[0],
                    ATTENTION_FLASH, scores, attention[0], q8[0]);
    (void)llama_reference_run_pair_schedule(&layer, x, kc[1], vc[1],
                                             (int)context, y[1],
                                             attention[1], q8[1]);
    puts("\nVALIDATION (outside all timed regions)");
    llama_compare_vec("Attention", attention[0], attention[1], HIDDEN);
    int different_q8 = 0;
    for (int i = 0; i < HIDDEN / QK; ++i)
        different_q8 += memcmp(&q8[0][i], &q8[1][i], sizeof(block_q8_0)) != 0;
    printf("Q8 blocks differing: %d/48\n", different_q8);
    if (different_q8) die("Quantization changed");
    llama_compare_vec("Output Y", y[0], y[1], HIDDEN);
    printf("PASS: staged and paired llama-style arithmetic match within tolerance.\n");

    free(x); free(scores);
    for (int m = 0; m < 2; ++m) {
        free(y[m]); free(attention[m]); free(q8[m]);
        free(kc[m]); free(vc[m]); free(wall[m]);
    }
    free_layer(&layer);
    return 0;
}
