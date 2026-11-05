#pragma once
/*
 * Matched unfused control for register_fusion.h.
 *
 * This is intentionally a FULLY STAGED execution with identical numerical
 * microkernels and KV format to run_layer_fused(). The only Q/O/Q8 difference
 * is compulsory materialization between complete stages:
 *   (1) all K/V and all six Q pairs, including RoPE;
 *   (2) all six two-head attention outputs;
 *   (3) all 48 Q8 blocks;
 *   (4) all six W_O input pairs contributing to Y.
 *
 * Use rf_project256, rf_rope128, rf_attention2, rf_quant32 and rf_wo_pair
 * WITHOUT changing their computation, scale handling or exp approximation.
 * Do not use the old fused_column_matvec()/fused_attention_pair() here.
 * This control specifically tests the memory boundaries, not arithmetic.
 */
static fused_stats_t run_layer_soa_staged(const layer_t *l,
                                          const fused_weights_t *fw,
                                          const float *x,
                                          uint16_t *kc, uint16_t *vc,
                                          int context, float *out,
                                          float *attn_copy,
                                          block_q8_0 *q8_copy) {
#if HAVE_RVV
    if (fw->tcm_tile)
        die("Matched staged baseline needs FUSED_TCM=0, like register fusion");

    float normed[HIDDEN];
    block_q8_0 xq8[HIDDEN / QK];
    float all_q[HIDDEN], all_o[HIDDEN];
    block_q8_0 all_oq[HIDDEN / QK];
    float cv[64], sv[64];
    fused_stats_t stats = {0};
    double t = seconds_now();

    /* Match the fused entry point exactly before any Q/K/V vectors exist. */
    rmsnorm(x, l->norm, normed);
    stats.phase.rms += (float)(seconds_now() - t);
    t = seconds_now();
    quantize_q8(normed, xq8, HIDDEN);
    stats.phase.q8_input += (float)(seconds_now() - t);

    t = seconds_now();
    for (int d = 0; d < 64; ++d) {
        const float theta = (float)(context - 1) /
                            powf(ROPE_BASE, (float)(2 * d) / HDIM);
        cv[d] = cosf(theta);
        sv[d] = sinf(theta);
    }
    stats.phase.rope_cache += (float)(seconds_now() - t);

    memset(out, 0, HIDDEN * sizeof(float));

    /* STAGE 1A: finish both K/V projections and persistent FP16 cache writes.
     * Same rf_project256, rf_rope128 and rf_f16_store64 as the fused path. */
    t = seconds_now();
    for (int group = 0; group < KVHEADS; ++group) {
        const uint8_t *kv_tile = fw->kv + (size_t)group * FUSED_TILE_BYTES;
        vfloat32m2_t k0, k1, v0, v1;
        vfloat32m2_t c = __riscv_vle32_v_f32m2(cv, 64);
        vfloat32m2_t s = __riscv_vle32_v_f32m2(sv, 64);
        rf_project256(kv_tile, xq8, HIDDEN / QK,
                      l->bk ? l->bk + group * HDIM : NULL,
                      l->bv ? l->bv + group * HDIM : NULL,
                      &k0, &k1, &v0, &v1);
        rf_rope128(&k0, &k1, c, s);
        const size_t off = (size_t)(context - 1) * KVDIM +
                           (size_t)group * HDIM;
        rf_f16_store64(kc + off, k0);
        rf_f16_store64(kc + off + 64, k1);
        rf_f16_store64(vc + off, v0);
        rf_f16_store64(vc + off + 64, v1);
    }
    /* Same accounting convention as fused: includes K/V and cache commit. */
    stats.phase.rope_cache += (float)(seconds_now() - t);

    /* STAGE 1B: complete ALL Q projections before starting ANY attention.
     * This is the deliberate Q materialization boundary. */
    t = seconds_now();
    for (int pair = 0; pair < QHEADS / 2; ++pair) {
        const uint8_t *qtile = fw->q + (size_t)pair * FUSED_TILE_BYTES;
        const float *qb = l->bq ? l->bq + (size_t)pair * 256 : NULL;
        vfloat32m2_t q00, q01, q10, q11;
        vfloat32m2_t c = __riscv_vle32_v_f32m2(cv, 64);
        vfloat32m2_t s = __riscv_vle32_v_f32m2(sv, 64);
        rf_project256(qtile, xq8, HIDDEN / QK,
                      qb, qb ? qb + 128 : NULL,
                      &q00, &q01, &q10, &q11);
        rf_rope128(&q00, &q01, c, s);
        rf_rope128(&q10, &q11, c, s);
        float *dest = all_q + (size_t)pair * 256;
        __riscv_vse32_v_f32m2(dest, q00, 64);
        __riscv_vse32_v_f32m2(dest + 64, q01, 64);
        __riscv_vse32_v_f32m2(dest + 128, q10, 64);
        __riscv_vse32_v_f32m2(dest + 192, q11, 64);
    }
    stats.phase.q_proj += (float)(seconds_now() - t);
    /* Compiler memory fence: all Q values must be materialized here. */
    __asm__ __volatile__("" ::: "memory");

    /* STAGE 2: compute ALL attention using the exact rf_attention2 kernel.
     * This is the deliberate full FP32 O materialization boundary. */
    t = seconds_now();
    for (int pair = 0; pair < QHEADS / 2; ++pair) {
        const float *q = all_q + (size_t)pair * 256;
        float *dest = all_o + (size_t)pair * 256;
        vfloat32m2_t o00, o01, o10, o11;
        rf_attention2(__riscv_vle32_v_f32m2(q, 64),
                      __riscv_vle32_v_f32m2(q + 64, 64),
                      __riscv_vle32_v_f32m2(q + 128, 64),
                      __riscv_vle32_v_f32m2(q + 192, 64),
                      kc, vc, context, pair / 3,
                      &o00, &o01, &o10, &o11);
        __riscv_vse32_v_f32m2(dest, o00, 64);
        __riscv_vse32_v_f32m2(dest + 64, o01, 64);
        __riscv_vse32_v_f32m2(dest + 128, o10, 64);
        __riscv_vse32_v_f32m2(dest + 192, o11, 64);
    }
    stats.phase.attention += (float)(seconds_now() - t);
    __asm__ __volatile__("" ::: "memory");

    /* STAGE 3: quantize ALL O after attention is completely finished.
     * The exact same rf_quant32 kernel is used as in register fusion. */
    t = seconds_now();
    for (int b = 0; b < HIDDEN / QK; ++b) {
        const float *chunk = all_o + (size_t)(b / 2) * 64;
        vfloat32m2_t v = __riscv_vle32_v_f32m2(chunk, 64);
        uint16_t d;
        vint8mf2_t qs = rf_quant32(v, (b & 1) ? 32 : 0, &d);
        all_oq[b].d = d;
        __riscv_vse8_v_i8mf2(all_oq[b].qs, qs, 32);
    }
    stats.phase.q8_output += (float)(seconds_now() - t);
    __asm__ __volatile__("" ::: "memory");

    /* STAGE 4: only now begin W_O. Reload its eight Q8 blocks per input
     * pair, then call the SAME rf_wo_pair used by register fusion. */
    t = seconds_now();
    for (int pair = 0; pair < QHEADS / 2; ++pair) {
        const uint8_t *wo = fw->o + (size_t)pair * FUSED_TILE_BYTES;
        const block_q8_0 *q = all_oq + (size_t)pair * 8;
        rf_wo_pair(wo, out,
                   __riscv_vle8_v_i8mf2(q[0].qs, 32),
                   __riscv_vle8_v_i8mf2(q[1].qs, 32),
                   __riscv_vle8_v_i8mf2(q[2].qs, 32),
                   __riscv_vle8_v_i8mf2(q[3].qs, 32),
                   __riscv_vle8_v_i8mf2(q[4].qs, 32),
                   __riscv_vle8_v_i8mf2(q[5].qs, 32),
                   __riscv_vle8_v_i8mf2(q[6].qs, 32),
                   __riscv_vle8_v_i8mf2(q[7].qs, 32),
                   q[0].d, q[1].d, q[2].d, q[3].d,
                   q[4].d, q[5].d, q[6].d, q[7].d);
    }
    stats.phase.out_proj += (float)(seconds_now() - t);

    /* Unnecessary for inference; only called during untimed validation. */
    if (attn_copy) memcpy(attn_copy, all_o, sizeof(all_o));
    if (q8_copy) memcpy(q8_copy, all_oq, sizeof(all_oq));
    return stats;
#else
    (void)l; (void)fw; (void)x; (void)kc; (void)vc; (void)context;
    (void)out; (void)attn_copy; (void)q8_copy;
    die("Matched SoA staged baseline requires RVV Zvfh (VECTOR=1)");
    fused_stats_t stats = {0};
    return stats;
#endif
}
