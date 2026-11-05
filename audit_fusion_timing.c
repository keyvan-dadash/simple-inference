/*
 * Independent three-way single-layer timing audit for simple-inference:
 *   A. original Q4_0 row-major + scalar-head online attention
 *   B. fully staged Q/K/V -> attention -> quantize -> W_O, using the
 *      EXACT SAME rf_* RVV numerical kernels as the register-fused path
 *   C. register-resident Q -> attention -> Q8 -> W_O path
 *
 * Build INSIDE simple-inference (after removing duplicate typedefs from .c):
 * gcc -O2 -std=c11 -Wall -Wextra -Wshadow -ffp-contract=off \
 *     -march=rv64gcv_zvfh_zfh_zicbop -mabi=lp64d -DUSE_RVV=1 \
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

/* Matched staged baseline is soa_staged.h; qwen_one_layer.c includes it.
 * Both staged and fused reuse rf_project256/rf_attention2/rf_quant32/rf_wo_pair. */

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

static int audit_check_q8(const char *label,
                          const block_q8_0 *ref, const block_q8_0 *test) {
    int differing = 0;
    for (int i = 0; i < HIDDEN / QK; ++i)
        differing += memcmp(ref + i, test + i, sizeof(block_q8_0)) != 0;
    printf(" %-29s differing Q8 blocks=%d/%d\n", label, differing, HIDDEN / QK);
    /* Staged/fused use the SAME quantization code, so a difference here is
     * diagnostic. The original row-major reference may differ slightly. */
    return differing;
}

/* This is intentionally OUTSIDE any timed region. It checks a genuinely
 * different set of inputs, KV histories, and context lengths. One captured
 * replay of the same hot token is not a sufficient correctness test. */
static void audit_validate_cases(const layer_t *layer, const fused_weights_t *fw,
                                 float *x, uint16_t **kc, uint16_t **vc,
                                 float **out, float **attention,
                                 block_q8_0 **q8, float *scores,
                                 int max_context) {
    float previous_o[HIDDEN];
    float no_capture_y[HIDDEN];
    int distinct_cases = 0;
    const int cases = 5;
    const int lengths[5] = {1, 2, 17, 64, max_context};
    puts("\nINDEPENDENT VALIDATION (untimed, fresh input/KV per case)");
    for (int trial = 0; trial < cases; ++trial) {
        const int context = lengths[trial] < max_context ?
                            lengths[trial] : max_context;
        /* All three implementations see exactly the same synthetic token. */
        for (int i = 0; i < HIDDEN; ++i) {
            const float r = random_float();
            if (trial == 1) x[i] = (i & 1) ? -0.125f - fabsf(r) : 0.125f + fabsf(r);
            else if (trial == 3) x[i] = (i % 13 == 0) ? 0.8f : r * 0.1f;
            else x[i] = r;
        }
        const float kmul = trial == 0 ? 0.25f :
                           trial == 1 ? 0.0125f :
                           trial == 2 ? 1.0f :
                           trial == 3 ? 0.0f : 0.5f;
        const size_t prefix = (size_t)(context - 1) * KVDIM;
        for (size_t j = 0; j < prefix; ++j) {
            kc[0][j] = f32_to_f16(random_float() * kmul);
            vc[0][j] = f32_to_f16(random_float() * 0.25f);
        }
        /* All untouched elements are irrelevant beyond the current context. */
        for (int m = 1; m < AUDIT_METHODS; ++m) {
            memcpy(kc[m], kc[0], prefix * sizeof(uint16_t));
            memcpy(vc[m], vc[0], prefix * sizeof(uint16_t));
        }
        /* First test the EXACT capture-free path used in timed calls. */
        (void)run_layer_soa_staged(layer, fw, x, kc[1], vc[1],
                                    context, out[1], NULL, NULL);
        (void)run_layer_fused(layer, fw, x, kc[2], vc[2],
                              context, out[2], NULL, NULL);
        audit_compare_output("NO-CAPTURE staged vs fused Y", out[1], out[2], HIDDEN);
        memcpy(no_capture_y, out[2], sizeof(no_capture_y));
        /* Then enable captures to inspect intermediate O and quantized O. */
        (void)run_layer(layer, x, kc[0], vc[0], context, out[0],
                        ATTENTION_FLASH, scores, attention[0], q8[0]);
        (void)run_layer_soa_staged(layer, fw, x, kc[1], vc[1],
                                    context, out[1], attention[1], q8[1]);
        (void)run_layer_fused(layer, fw, x, kc[2], vc[2],
                              context, out[2], attention[2], q8[2]);
        audit_compare_output("fused capture vs no-capture Y",
                             no_capture_y, out[2], HIDDEN);
        printf("Case %d: context=%d KV-history-scale=%g\n", trial + 1,
               context, (double)kmul);
        audit_compare_output("matched vs fused attention", attention[1], attention[2], HIDDEN);
        if (audit_check_q8("matched vs fused Q8", q8[1], q8[2]))
            die("Matched and fused Q8 differ: cannot attribute timings to fusion alone");
        audit_compare_output("matched vs fused Y", out[1], out[2], HIDDEN);
        /* Both implementations must store the same new FP16 K/V cache data. */
        const size_t offset = (size_t)(context - 1) * KVDIM;
        if (memcmp(kc[1] + offset, kc[2] + offset,
                   KVDIM * sizeof(uint16_t)) ||
            memcmp(vc[1] + offset, vc[2] + offset,
                   KVDIM * sizeof(uint16_t)))
            die("Staged/fused newly generated FP16 K/V differ");
        /* The independent original path remains a numerical reference.
         * We allow ordinary FP32 differences from reductions and expf. */
        audit_compare_output("reference vs matched O", attention[0], attention[1], HIDDEN);
        audit_compare_output("reference vs fused Y", out[0], out[2], HIDDEN);
        if (trial && memcmp(previous_o, attention[2], sizeof(previous_o)))
            ++distinct_cases;
        memcpy(previous_o, attention[2], sizeof(previous_o));
    }
    if (distinct_cases == 0)
        die("All independent inputs gave identical attention: suspect dead path");
    printf("PASS: %d independent cases, %d distinct output changes, "
           "matched staged and fused agree.\n", cases, distinct_cases);
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

    const char *names[] = {"Original row-major", "Matched RVV staged", "Register RVV fused"};
    printf("THREE-WAY A100 FUSION TIMING AUDIT | context=%ld repeats=%ld TCM=%s RVV=%s\n",
           context, repeats, tcm ? "enabled" : "off", HAVE_RVV ? "yes" : "no");
    printf("MATCHED CONTROL: staged and fused use the same rf_* math, Zvfh KV loads,\n"
           "softmax approximation, Q8 quantization and W_O kernel.\n"
           "Staged writes full Q, O and Q8 before W_O; fused retains head-pair RVV values.\n"
           "No intermediate captures during timing; independent wall clock per call.\n");

    /* Six warmups, rotated, so each of the three methods starts cold once. */
    for (int w = 0; w < 6; ++w) {
        int m = w % AUDIT_METHODS;
        if (m == 0)
            (void)run_layer(&layer, x, kc[m], vc[m], (int)context,
                            out[m], ATTENTION_FLASH, scores, NULL, NULL);
        else if (m == 1)
            (void)run_layer_soa_staged(&layer, &fw, x, kc[m], vc[m],
                                         (int)context, out[m], NULL, NULL);
        else
            (void)run_layer_fused(&layer, &fw, x, kc[m], vc[m],
                                  (int)context, out[m], NULL, NULL);
    }

    /* Rotate method order on each repetition. We record per-call outer wall
     * times, so the stage sums can be checked against an independent clock. */
    for (int rep = 0; rep < repeats; ++rep) {
        /* Different token each repetition, but identical token for all three
         * methods in that repetition. Input generation is OUTSIDE timers. */
        if (rep != 0)
            for (int i = 0; i < HIDDEN; ++i) x[i] = random_float();
        for (int j = 0; j < AUDIT_METHODS; ++j) {
            int m = (rep + j) % AUDIT_METHODS;
            double begin = seconds_now();
            phase_t p;
            double stage_time = 0.0;
            if (m == 0) {
                p = run_layer(&layer, x, kc[m], vc[m], (int)context,
                              out[m], ATTENTION_FLASH, scores, NULL, NULL);
            } else if (m == 1) {
                fused_stats_t s = run_layer_soa_staged(&layer, &fw, x,
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
           "matched staged/fused %.3fx\n", refwall/fusewall, ctrlwall/fusewall);
    printf("   Original/fused mixes layout and kernel changes; NOT a fusion-only ratio.\n");
    printf("   MATCHED staged/fused uses identical rf_* computations and weight data;\n"
           "   it isolates materialization / register handoff plus compiler effects.\n");
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
    puts("Note: fused 'Online attention' includes Q + RoPE + attention + Q8 + W_O;");
    puts("      fused Wq / Q8 O / W_O zeros are TIMING GROUPING, not omitted arithmetic.");
    puts("      staged K/V projection is included in RoPE+cache timing; Wk+Wv cells are zero.");
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
    (void)run_layer_soa_staged(&layer, &fw, x, kc[1], vc[1],
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
    audit_check_q8("matched staged Q8 vs original", q8[0], q8[1]);
    audit_check_q8("register fused Q8 vs original", q8[0], q8[2]);
    if (audit_check_q8("matched staged Q8 vs fused", q8[1], q8[2]))
        die("Matched staged/fused Q8 mismatch in standard validation");
    audit_compare_output("SoA staged Y vs original", out[0], out[1], HIDDEN);
    audit_compare_output("SoA fused Y vs original", out[0], out[2], HIDDEN);
    audit_validate_cases(&layer, &fw, x, kc, vc, out, attention, q8,
                         scores, (int)context);
    printf("\nPASS: repeated measurements plus independent numerical validation.\n");

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
