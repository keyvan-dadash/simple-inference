#pragma once
/*
 * Official unfused column-SoA/RVV baseline for single-token Qwen.
 *
 * Include AFTER fused_layer.h and after the core C helpers; this header is
 * intentionally specialized to the current single translation unit design.
 *
 * SAME kernels and same repacked weights as run_layer_fused():
 *   - fused_column_matvec() for Q/K/V and W_O
 *   - fused_attention_pair() for all two-head attention pairs
 *   - same Q8_0 quantization, RoPE, and FP16 KV cache
 * DIFFERENCE: stage boundaries; all Q, O and quantized O are materialized
 * before the next stage. No paired attention-to-W_O fusion here.
 *
 * TCM staging (optional) matches the fused path: 2 KV + 6 Q + 6 W_O tiles.
 */
#include "defines.h"

static fused_stats_t run_layer_soa_staged(const layer_t *l,
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
