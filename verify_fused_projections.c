/*
 * Independent validation of the column-SoA fused Qwen attention sublayer.
 * Place next to qwen_one_layer.c, fused_layer.h, defines.h, attention_kernels.h.
 * Build using the existing build flags, e.g.:
 *   gcc -O2 -std=c11 -Wall -Wextra -Wshadow -ffp-contract=off \
 *     -march=rv64gcv_zicbop -mabi=lp64d -DUSE_RVV=1 \
 *     -o verify_fused_projections verify_fused_projections.c -lm
 * Run on A100:
 *   sh -c 'echo $$ > /proc/set_ai_thread || exit 1; \
 *     exec taskset -c 8 ./verify_fused_projections weights'
 *
 * No original source edits; including the existing C translation unit
 * deliberately reuses its private static reference and fused kernels.
 * It validates Q,K,V and Wo separately, so softmax saturation cannot
 * hide a projection error; then tests attention on non-saturated random KV.
 */
#define main original_qwen_benchmark_main_for_verify
#include "qwen_one_layer.c"
#undef main

static int check_vector(const char *label, const float *expected,
                        const float *actual, size_t n) {
    double max_abs = 0.0, max_rel = 0.0, sum_a = 0.0, sum_b = 0.0;
    double energy = 0.0;
    size_t exact = 0, bad = 0, nonzero = 0;
    for (size_t j = 0; j < n; ++j) {
        float a = expected[j], b = actual[j];
        if (!isfinite(a) || !isfinite(b)) {
            printf("%s: invalid float at j=%zu, ref=%g fused=%g\n",label,j,a,b);
            return 1;
        }
        double diff = fabs((double)a - b);
        double magnitude = fmax(fabs((double)a), fabs((double)b));
        double rel = diff / fmax(magnitude, 1e-7);
        if (diff > max_abs) max_abs = diff;
        if (rel > max_rel) max_rel = rel;
        if (memcmp(&a, &b, sizeof(a)) == 0) exact++;
        if (a != 0.0f) nonzero++;
        if (diff > 2e-5 + 2e-5 * magnitude) {
            if (bad < 4) printf("  BAD %s[%zu]: ref=%.9g fused=%.9g diff=%.4g\n",
                                label,j,a,b,diff);
            bad++;
        }
        sum_a += (double)a;
        sum_b += (double)b;
        energy += (double)a*a;
    }
    printf("  %-12s max_abs=%-12.6g max_rel=%-12.6g exact=%4zu/%-4zu bad=%-4zu nonzero=%-4zu rms=%-10.6g checksums %.9g / %.9g\n",
           label, max_abs, max_rel, exact, n, bad, nonzero,
           sqrt(energy/(double)n), sum_a, sum_b);
    if (nonzero == 0 || !isfinite(energy)) {
        fprintf(stderr, "  FAIL: %s is all zeros or invalid; benchmark not meaningful\n",label);
        return 1;
    }
    return bad != 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <weights_directory>\n", argv[0]);
        return 2;
    }
    layer_t layer = load_layer(argv[1]);
    fused_weights_t fw = fused_weights_load(argv[1], 0, 0); // Never touch TCM.
    float input[HIDDEN], normed[HIDDEN];
    block_q8_0 xq[HIDDEN / QK], oq[HIDDEN / QK];
    float qref[HIDDEN], qfused[HIDDEN];
    float kref[KVDIM], vref[KVDIM], kfused[KVDIM], vfused[KVDIM];
    float o[HIDDEN], yref[HIDDEN], yfused[HIDDEN];
    float both[2 * HDIM];
    int failures = 0;
    printf("INDEPENDENT A100 CHECK: original row-major Q4_0 against column-SoA RVV\n");
    printf("HIDDEN=%d, QHEADS=%d, RVV=%s, TCM=disabled\n",
           HIDDEN, QHEADS, HAVE_RVV ? "yes" : "no");
    for (int trial=0; trial<3; ++trial) {
        rng_state = UINT32_C(0x8f53491a) + (uint32_t)trial*UINT32_C(0x1020304);
        for (int i=0;i<HIDDEN;i++) input[i]=random_float();
        rmsnorm(input, layer.norm, normed);
        quantize_q8(normed, xq, HIDDEN);
        matvec_q4q8(layer.wq, xq, HIDDEN, HIDDEN, layer.bq, qref);
        matvec_q4q8(layer.wk, xq, HIDDEN, KVDIM, layer.bk, kref);
        matvec_q4q8(layer.wv, xq, HIDDEN, KVDIM, layer.bv, vref);
        for (int pair=0;pair<6;pair++) {
            const float *bias = layer.bq ? layer.bq + pair*256 : NULL;
            fused_column_matvec(fw.q + (size_t)pair*FUSED_TILE_BYTES,
                                xq, HIDDEN/QK,
                                bias, bias ? bias + 128 : NULL,
                                qfused + pair*256, 0);
        }
        for (int group=0;group<2;group++) {
            fused_column_matvec(fw.kv + (size_t)group*FUSED_TILE_BYTES,
                                xq, HIDDEN/QK,
                                layer.bk ? layer.bk + group*HDIM : NULL,
                                layer.bv ? layer.bv + group*HDIM : NULL,
                                both, 0);
            memcpy(kfused + group*HDIM, both, HDIM*sizeof(float));
            memcpy(vfused + group*HDIM, both+HDIM, HDIM*sizeof(float));
        }
        printf("\nTRIAL %d / 3: raw-weight reference vs SoA RVV projections\n", trial+1);
        failures += check_vector("Wq -> Q", qref, qfused, HIDDEN);
        failures += check_vector("Wk -> K", kref, kfused, KVDIM);
        failures += check_vector("Wv -> V", vref, vfused, KVDIM);

        // Wo is tested with deliberately randomized *full* attention O, to
        // prevent an accidental 1-hot softmax from masking quantization bugs.
        for (int i=0;i<HIDDEN;i++) o[i]=random_float();
        quantize_q8(o, oq, HIDDEN);
        matvec_q4q8(layer.wo, oq, HIDDEN, HIDDEN, NULL, yref);
        memset(yfused, 0, sizeof(yfused));
        for (int pair=0; pair<6; pair++) {
            const uint8_t *wo = fw.o + (size_t)pair*FUSED_TILE_BYTES;
            for (int tile=0;tile<6;tile++) {
                fused_column_matvec(wo + (size_t)tile*FUSED_WO_SUBTILE,
                                    oq + 8*pair, 8, NULL, NULL,
                                    yfused + 256*tile, 1);
            }
        }
        failures += check_vector("Wo -> Y", yref, yfused, HIDDEN);
    }

    // A second independent attention test deliberately avoids dependence on
    // the model's Q and K magnitudes: randomized Q, FP16 K and FP16 V.
    enum { CTX = 64 };
    float qa[HIDDEN], att_ref[HIDDEN], att_fused[HIDDEN];
    uint16_t *kc = malloc(CTX*KVDIM*sizeof(uint16_t));
    uint16_t *vc = malloc(CTX*KVDIM*sizeof(uint16_t));
    if (!kc || !vc) die("verify: allocation failed");
    rng_state = UINT32_C(0x91be2145);
    for (int i=0;i<HIDDEN;i++) qa[i]=random_float()*1.5f;
    for (int i=0;i<CTX*KVDIM;i++) {
        kc[i]=f32_to_f16(random_float()*1.5f);
        vc[i]=f32_to_f16(random_float()*1.5f);
    }
    attention_flash(qa,kc,vc,CTX,att_ref);
    for (int pair=0; pair<6; pair++)
        fused_attention_pair(qa + pair*256, kc, vc, CTX, pair/3,
                             att_fused + pair*256);
    printf("\nBALANCED SOFTMAX CONTROL: 64 synthetic FP16 KV tokens, Q independent of Wq\n");
    failures += check_vector("Attention", att_ref, att_fused, HIDDEN);
    free(kc); free(vc);
    fused_weights_free(&fw);
    free_layer(&layer);
    printf("\n%s: %d failed stage-comparisons\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
