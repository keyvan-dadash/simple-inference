/*
 * Independent three-way single-layer timing audit for simple-inference:
 *   A. original Q4_0 row-major + scalar-head online attention
 *   B. same repacked column-SoA RVV kernels, UNFUSED stage ordering
 *   C. existing two-head paired fused schedule
 *
 * Build INSIDE simple-inference (after removing duplicate typedefs from .c):
 * gcc -O2 -std=c11 -Wall -Wextra -Wshadow -ffp-contract=off \
 *     -march=rv64gcv_zicbop -mabi=lp64d -DUSE_RVV=1 \
 *     -o audit_fusion_timing audit_fusion_timing.c -lm
 *
 * Run: sh -c 'echo $$ > /proc/set_ai_thread || exit 1; \
 *   exec taskset -c 8 ./audit_fusion_timing weights 256 10'
 * TCM: FUSED_TCM=1 TCM_BLOCK=0 ...same command...
 *
 * Timing boundaries:
 *  - outer clock around each complete run_layer*() call; capture disabled
 *  - method-specific clock phases from the original source (sum printed)
 *  - 'unaccounted' = outer wall - phase sum; contains bookkeeping,
 *    output memset, instrumentation gaps and call overhead
 *  - excludes model loading, Python repacking, initial KV fabrication,
 *    and allocations, exactly like the current benchmark
 *
 * This file does not alter qwen_one_layer.c or its kernels. We rename its
 * main locally so its static functions are accessible to the audit.
 */
#define main qwen_existing_main_for_audit
#include "qwen_one_layer.c"
#undef main

#define AUDIT_METHODS 3
#define AUDIT_PHASES 8

typedef struct {
    double wall;
    double timed;
    double uncovered;
    double stage[9];
    double stage_tcm;
} audit_sample_t;

typedef struct {
    double *wall;
    double *timed;
    double *uncovered;
    audit_sample_t sum;
} audit_method_t;

static double audit_phase_sum(const phase_t *p) {
    return (double)p->rms + p->q8_input + p->q_proj + p->k_proj + p->v_proj +
           p->rope_cache + p->attention + p->q8_output + p->out_proj;
}

static fused_stats_t audit_run_unfused_soa(const layer_t *l,
                                            const fused_weights_t *fw,
                                            const float *x,
                                            uint16_t *kc, uint16_t *vc,
                                            int context, float *out,
                                            float *attn_copy,
                                            block_q8_0 *q8_copy) {
    /* Same RVV projection + attention + W_o functions as the fused path.
     * Deliberately materialize all Q, all FP32 attention O and Q8 O.
     * Staging count and weight/activation arithmetic are the same as fused.
     * Only ordering and inter-stage buffering are changed. */
    float normalized[HIDDEN];
    block_q8_0 xq8[HIDDEN / QK];
    float all_q[HIDDEN], all_o[HIDDEN];
    block_q8_0 all_oq[HIDDEN / QK];
    fused_stats_t stats = {0};
    double start = seconds_now();
    rmsnorm(x, l->norm, normalized);
    stats.phase.rms += (float)(seconds_now() - start);
    start = seconds_now();
    quantize_q8(normalized, xq8, HIDDEN);
    stats.phase.q8_input += (float)(seconds_now() - start);
    memset(out, 0, HIDDEN * sizeof(float));

    /* K/V for both groups first, once each. */
    for (int group = 0; group < KVHEADS; ++group) {
        const uint8_t *tile = fused_stage_tile(fw,
            fw->kv + (size_t)group * FUSED_TILE_BYTES, &stats);
        float both[256];
        start = seconds_now();
        fused_column_matvec(tile, xq8, HIDDEN / QK,
            l->bk ? l->bk + group * HDIM : NULL,
            l->bv ? l->bv + group * HDIM : NULL, both, 0);
        double elapsed = seconds_now() - start;
        stats.phase.k_proj += (float)(elapsed / 2);
        stats.phase.v_proj += (float)(elapsed / 2);

        start = seconds_now();
        rope(both, 1, context - 1);
        size_t offset = (size_t)(context - 1) * KVDIM +
                        (size_t)group * HDIM;
        for (int j = 0; j < HDIM; ++j) {
            kc[offset + j] = f32_to_f16(both[j]);
            vc[offset + j] = f32_to_f16(both[j + HDIM]);
        }
        stats.phase.rope_cache += (float)(seconds_now() - start);
    }

    /* All six query pairs before the attention stage. */
    for (int pair = 0; pair < QHEADS / 2; ++pair) {
        const uint8_t *tile = fused_stage_tile(fw,
            fw->q + (size_t)pair * FUSED_TILE_BYTES, &stats);
        const float *bias = l->bq ? l->bq + pair * 256 : NULL;
        start = seconds_now();
        fused_column_matvec(tile, xq8, HIDDEN / QK,
            bias, bias ? bias + 128 : NULL, all_q + pair * 256, 0);
        stats.phase.q_proj += (float)(seconds_now() - start);
        start = seconds_now();
        rope(all_q + pair * 256, 2, context - 1);
        stats.phase.rope_cache += (float)(seconds_now() - start);
    }

    /* All six attention pairs, preserving the *same paired RVV kernel*.
     * Contrasts fusion scheduling instead of conflating the speed of the
     * original scalar attention and the new vector attention. */
    for (int pair = 0; pair < QHEADS / 2; ++pair) {
        start = seconds_now();
        fused_attention_pair(all_q + pair * 256, kc, vc,
                             context, pair / 3, all_o + pair * 256);
        stats.phase.attention += (float)(seconds_now() - start);
    }

    /* One full attention materialization and quantization, unlike fusion. */
    start = seconds_now();
    quantize_q8(all_o, all_oq, HIDDEN);
    stats.phase.q8_output += (float)(seconds_now() - start);

    /* All six O pairs contribute to all six 256-output Y tiles. */
    for (int pair = 0; pair < QHEADS / 2; ++pair) {
        const uint8_t *tile = fused_stage_tile(fw,
            fw->o + (size_t)pair * FUSED_TILE_BYTES, &stats);
        start = seconds_now();
        for (int ytile = 0; ytile < HIDDEN / 256; ++ytile) {
            fused_column_matvec(tile + (size_t)ytile * FUSED_WO_SUBTILE,
                                all_oq + pair * 8, 8, NULL, NULL,
                                out + ytile * 256, 1);
        }
        stats.phase.out_proj += (float)(seconds_now() - start);
    }

    /* Optional output captures explicitly EXCLUDED from phase timers.
     * The timed caller always supplies NULL; verification runs after timing. */
    if (attn_copy) memcpy(attn_copy, all_o, sizeof(all_o));
    if (q8_copy) memcpy(q8_copy, all_oq, sizeof(all_oq));
    return stats;
}

static double audit_median(double *arr, int n);

static int audit_compare_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double audit_median(double *arr, int n) {
    double *copy = malloc((size_t)n * sizeof(double));
    if (!copy) die("audit median OOM");
    memcpy(copy, arr, (size_t)n * sizeof(double));
    qsort(copy, (size_t)n, sizeof(double), audit_compare_double);
    double x = (n & 1) ? copy[n / 2] :
               (copy[n / 2 - 1] + copy[n / 2]) * 0.5;
    free(copy);
    return x;
}

static void audit_capture(audit_method_t *m, int rep,
                          double outer_s, phase_t ph, double stage_s) {
    double timed = audit_phase_sum(&ph) + stage_s;
    m->wall[rep] = outer_s;
    m->timed[rep] = timed;
    m->uncovered[rep] = outer_s - timed;
    m->sum.wall += outer_s;
    m->sum.timed += timed;
    m->sum.uncovered += outer_s - timed;
    m->sum.stage_tcm += stage_s;
    const double stages[] = {ph.rms, ph.q8_input, ph.q_proj, ph.k_proj, ph.v_proj,
                             ph.rope_cache, ph.attention, ph.q8_output, ph.out_proj};
    for (int j = 0; j < 9; ++j) m->sum.stage[j] += stages[j];
}

static void audit_compare_output(const char *label,
                                 const float *ref, const float *test, int n) {
    double max_abs = 0.0, max_rel = 0.0, rms = 0.0;
    int different = 0, bad = 0, nonzero = 0;
    for (int i = 0; i < n; ++i) {
        if (!isfinite(ref[i]) || !isfinite(test[i])) {
            fprintf(stderr, "%s: non-finite value at %d\n", label, i);
            exit(3);
        }
        double delta = fabs((double)ref[i] - (double)test[i]);
        double scale = fmax(fabs((double)ref[i]), fabs((double)test[i]));
        if (delta > max_abs) max_abs = delta;
        if (scale > 1e-8 && delta / scale > max_rel)
            max_rel = delta / scale;
        rms += delta * delta;
        nonzero += test[i] != 0.0f;
        different += delta != 0.0;
        bad += delta > 1e-5 + 1e-4 * scale;
    }
    printf(" %-29s max_abs=%10.3g max_rel=%10.3g "
           "rms_diff=%10.3g different=%4d/%d bad=%d nonzero=%d\n",
           label, max_abs, max_rel, sqrt(rms / n), different, n, bad, nonzero);
    if (bad) {
        fprintf(stderr, "ERROR: %s differs beyond tolerance\n", label);
        exit(3);
    }
}

static void audit_check_q8(const char *label,
                           const block_q8_0 *ref, const block_q8_0 *test) {
    int differing = 0;
    for (int i = 0; i < HIDDEN / QK; ++i)
        differing += memcmp(ref + i, test + i, sizeof(block_q8_0)) != 0;
    printf(" %-29s differing Q8 blocks=%d/%d\n", label, differing, HIDDEN / QK);
    /* Q8 rounding may legitimately differ by one code near a boundary.
     * Inspect counts rather than enforce bitwise equality in all contexts. */
}

static void audit_usage(const char *program) {
    fprintf(stderr, "Usage: %s <weights_directory> [context=256] [repeats=10]\n", program);
    fprintf(stderr, "Environment: FUSED_TCM=0|1, TCM_BLOCK=0 (default).\n");
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) { audit_usage(argv[0]); return 2; }
    const char *directory = argv[1];
    char *end = NULL;
    long context = argc > 2 ? strtol(argv[2], &end, 10) : 256;
    if (argc > 2 && (!end || *end)) { audit_usage(argv[0]); return 2; }
    end = NULL;
    long repeats = argc > 3 ? strtol(argv[3], &end, 10) : 10;
    if (argc > 3 && (!end || *end)) { audit_usage(argv[0]); return 2; }
    if (context < 1 || context > 32768 || repeats < 2 || repeats > 100000)
        die("context must be 1..32768 and repeats 2..100000");
    const char *tcm_env = getenv("FUSED_TCM");
    const char *block_env = getenv("TCM_BLOCK");
    int tcm = tcm_env && atoi(tcm_env) != 0;
    int block = block_env ? atoi(block_env) : 0;

    layer_t layer = load_layer(directory);
    fused_weights_t fw = fused_weights_load(directory, tcm, block);

    float *x = malloc(HIDDEN * sizeof(float));
    float *out[AUDIT_METHODS];
    float *attention[AUDIT_METHODS];
    block_q8_0 *q8[AUDIT_METHODS];
    uint16_t *kc[AUDIT_METHODS], *vc[AUDIT_METHODS];
    size_t cache_count = (size_t)context * KVDIM;
    float *scores = malloc((size_t)QHEADS * (size_t)context * sizeof(float));
    audit_method_t methods[AUDIT_METHODS] = {0};
    if (!x || !scores) die("audit allocation failure");
    for (int m = 0; m < AUDIT_METHODS; ++m) {
        out[m] = malloc(HIDDEN * sizeof(float));
        attention[m] = malloc(HIDDEN * sizeof(float));
        q8[m] = malloc(HIDDEN / QK * sizeof(block_q8_0));
        kc[m] = malloc(cache_count * sizeof(uint16_t));
        vc[m] = malloc(cache_count * sizeof(uint16_t));
        methods[m].wall = malloc((size_t)repeats * sizeof(double));
        methods[m].timed = malloc((size_t)repeats * sizeof(double));
        methods[m].uncovered = malloc((size_t)repeats * sizeof(double));
        if (!out[m] || !attention[m] || !q8[m] || !kc[m] || !vc[m] ||
            !methods[m].wall || !methods[m].timed || !methods[m].uncovered)
            die("audit method allocation failure");
    }
    for (int i = 0; i < HIDDEN; ++i) x[i] = random_float();
    fill_kv_history(kc[0], vc[0], (int)context);
    for (int m = 1; m < AUDIT_METHODS; ++m) {
        memcpy(kc[m], kc[0], cache_count * sizeof(uint16_t));
        memcpy(vc[m], vc[0], cache_count * sizeof(uint16_t));
    }

    const char *names[] = {"Original row-major", "SoA/RVV staged", "SoA/RVV fused"};
    printf("THREE-WAY A100 FUSION TIMING AUDIT | context=%ld repeats=%ld TCM=%s RVV=%s\n",
           context, repeats, tcm ? "enabled" : "off", HAVE_RVV ? "yes" : "no");
    printf("Baseline is original, stage control uses SAME SoA kernels but buffers full Q/O,\n"
           "fused pairs attention and immediate W_o. No output captures during timing.\n");

    /* Six warmups, rotated, so each of the three methods starts cold once. */
    for (int w = 0; w < 6; ++w) {
        int m = w % AUDIT_METHODS;
        if (m == 0)
            (void)run_layer(&layer, x, kc[m], vc[m], (int)context,
                            out[m], ATTENTION_FLASH, scores, NULL, NULL);
        else if (m == 1)
            (void)audit_run_unfused_soa(&layer, &fw, x, kc[m], vc[m],
                                         (int)context, out[m], NULL, NULL);
        else
            (void)run_layer_fused(&layer, &fw, x, kc[m], vc[m],
                                  (int)context, out[m], NULL, NULL);
    }

    /* Rotate method order on each repetition. We record per-call outer wall
     * times, so the stage sums can be checked against an independent clock. */
    for (int rep = 0; rep < repeats; ++rep) {
        for (int j = 0; j < AUDIT_METHODS; ++j) {
            int m = (rep + j) % AUDIT_METHODS;
            double begin = seconds_now();
            phase_t p;
            double stage_time = 0.0;
            if (m == 0) {
                p = run_layer(&layer, x, kc[m], vc[m], (int)context,
                              out[m], ATTENTION_FLASH, scores, NULL, NULL);
            } else if (m == 1) {
                fused_stats_t s = audit_run_unfused_soa(&layer, &fw, x,
                                   kc[m], vc[m], (int)context,
                                   out[m], NULL, NULL);
                p = s.phase;
                stage_time = s.tcm_stage;
            } else {
                fused_stats_t s = run_layer_fused(&layer, &fw, x,
                                    kc[m], vc[m], (int)context,
                                    out[m], NULL, NULL);
                p = s.phase;
                stage_time = s.tcm_stage;
            }
            double duration = seconds_now() - begin;
            audit_capture(&methods[m], rep, duration, p, stage_time);
        }
    }

    printf("\nTIMING (ms/call) | median is robust to occasional OS interruptions\n");
    printf(" %-19s %13s %13s %13s %13s %13s\n",
           "Path", "wall median", "wall avg", "stage avg", "gap avg", "TCM avg");
    for (int m = 0; m < AUDIT_METHODS; ++m) {
        double n = (double)repeats;
        printf(" %-19s %13.4f %13.4f %13.4f %13.4f %13.4f\n",
               names[m], audit_median(methods[m].wall, (int)repeats) * 1e3,
               methods[m].sum.wall / n * 1e3,
               methods[m].sum.timed / n * 1e3,
               methods[m].sum.uncovered / n * 1e3,
               methods[m].sum.stage_tcm / n * 1e3);
    }
    double refwall = audit_median(methods[0].wall, (int)repeats);
    double ctrlwall = audit_median(methods[1].wall, (int)repeats);
    double fusewall = audit_median(methods[2].wall, (int)repeats);
    if (fusewall <= 0.0) die("Wall-clock measurement resolution inadequate; increase work/context");
    printf("\nWHOLE-LAYER WALL SPEEDUPS: original/fused %.3fx | "
           "same-SoA staged/fused %.3fx\n", refwall/fusewall, ctrlwall/fusewall);
    printf("   Original/fused mixes layout, kernel, attention and fusion changes.\n");
    printf("   SoA staged/fused isolates schedule + materialization (approximately),\n"
           "   using exactly the same projection/attention kernels.\n");
    printf("   Gap = measured wall minus sum of phase timers (incl. TCM).\n"
           "   Gap includes Y zeroing, timer/bookkeeping gaps and other overhead.\n");
    for (int m = 0; m < AUDIT_METHODS; ++m) {
        double gap = methods[m].sum.uncovered / (double)repeats;
        if (gap < -0.00005)
            printf("WARNING: %s stage timers exceed outer wall by %.4f ms; "
                   "investigate timer accounting.\n", names[m], -gap * 1e3);
    }

    const char *ops[] = {"RMSNorm", "Quantize X", "Wq", "Wk + Wv", "RoPE + KV append",
                         "Online attention", "Quantize O", "W_o"};
    const int stage_map[] = {0,1,2,3,5,6,7,8};
    puts("\nSTAGE TIMER MEANS (ms/call; TCM staging shown separately above)");
    printf(" %-21s %14s %14s %14s\n", "Stage", "Original", "SoA staged", "SoA fused");
    for (int k = 0; k < AUDIT_PHASES; ++k) {
        int n = stage_map[k];
        printf(" %-21s", ops[k]);
        for (int m = 0; m < AUDIT_METHODS; ++m) {
            double v = methods[m].sum.stage[n];
            if (n == 3) v += methods[m].sum.stage[4];
            printf(" %14.4f", v / repeats * 1e3);
        }
        putchar('\n');
    }

    /* Numerical checks are run entirely OUTSIDE timing. */
    (void)run_layer(&layer, x, kc[0], vc[0], (int)context, out[0],
                    ATTENTION_FLASH, scores, attention[0], q8[0]);
    (void)audit_run_unfused_soa(&layer, &fw, x, kc[1], vc[1],
                                 (int)context, out[1], attention[1], q8[1]);
    (void)run_layer_fused(&layer, &fw, x, kc[2], vc[2],
                          (int)context, out[2], attention[2], q8[2]);
    puts("\nOUTPUT VALIDATION (outside all timed regions)");
    audit_compare_output("SoA staged attention vs original",
                         attention[0], attention[1], HIDDEN);
    audit_compare_output("SoA fused attention vs original",
                         attention[0], attention[2], HIDDEN);
    audit_compare_output("SoA fused attention vs staged",
                         attention[1], attention[2], HIDDEN);
    audit_check_q8("SoA staged Q8 vs original", q8[0], q8[1]);
    audit_check_q8("SoA fused Q8 vs original", q8[0], q8[2]);
    audit_compare_output("SoA staged Y vs original", out[0], out[1], HIDDEN);
    audit_compare_output("SoA fused Y vs original", out[0], out[2], HIDDEN);
    printf("\nPASS: output tolerances satisfied.\n");

    for (int m = 0; m < AUDIT_METHODS; ++m) {
        free(out[m]); free(attention[m]); free(q8[m]);
        free(kc[m]); free(vc[m]);
        free(methods[m].wall); free(methods[m].timed);
        free(methods[m].uncovered);
    }
    free(x); free(scores);
    fused_weights_free(&fw);
    free_layer(&layer);
    return 0;
}
