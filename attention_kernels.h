#ifndef QWEN_ATTENTION_KERNELS_H
#define QWEN_ATTENTION_KERNELS_H

#include "defines.h"

#include <math.h>
#include <inttypes.h>
#include <memory.h>

/*
 * Two attention algorithms with identical input and output interfaces.
 * Include this header after defining QHEADS, KVHEADS, HDIM, and f16_to_f32().
 *
 * STANDARD: materializes all [QHEADS, context] FP32 scores, normalizes
 *           scores in-place, and makes a separate pass through V.
 * FLASH:    single streaming online-softmax pass per query head; does not
 *           materialize attention scores/probabilities.
 *
 * These are CPU-C mathematical reference implementations, not optimized
 * multi-head RVV / FlashAttention-V code. Both use FP16 K/V and FP32 math.
 * Neither fuses QKV projections or the W_O output projection.
 */

typedef enum {
    ATTENTION_STANDARD = 0,
    ATTENTION_FLASH = 1
} attention_method_t;

static inline const char *attention_name(attention_method_t method) {
    return method == ATTENTION_FLASH ? "flash" : "standard";
}

static void attention_standard(const float *q, const uint16_t *key_cache,
                               const uint16_t *value_cache, int context,
                               float *out, float *scores) {
    const float inv_head_dim = 1.0f / sqrtf((float)HDIM);
    // Stage 1: explicitly materialize all Q*K^T scores to memory.
    for (int h = 0; h < QHEADS; ++h) {
        const int group = h / (QHEADS / KVHEADS);
        const float *qh = q + (size_t)h * HDIM;
        float *row = scores + (size_t)h * context;
        for (int t = 0; t < context; ++t) {
            const uint16_t *k = key_cache +
                ((size_t)t * KVHEADS + group) * HDIM;
            float dot = 0.0f;
            for (int d = 0; d < HDIM; ++d)
                dot += qh[d] * f16_to_f32(k[d]);
            row[t] = dot * inv_head_dim;
        }
    }
    // Stage 2: softmax of the materialized scores, in-place.
    for (int h = 0; h < QHEADS; ++h) {
        float *row = scores + (size_t)h * context;
        float m = -INFINITY;
        for (int t = 0; t < context; ++t) m = fmaxf(m, row[t]);
        float denom = 0.0f;
        for (int t = 0; t < context; ++t) {
            row[t] = expf(row[t] - m);
            denom += row[t];
        }
        const float inv = 1.0f / denom;
        for (int t = 0; t < context; ++t) row[t] *= inv;
    }
    // Stage 3: weighted V sum, distinct from scores/softmax stage.
    for (int h = 0; h < QHEADS; ++h) {
        const int group = h / (QHEADS / KVHEADS);
        const float *row = scores + (size_t)h * context;
        float *oh = out + (size_t)h * HDIM;
        memset(oh, 0, (size_t)HDIM * sizeof(float));
        for (int t = 0; t < context; ++t) {
            const uint16_t *v = value_cache +
                ((size_t)t * KVHEADS + group) * HDIM;
            const float p = row[t];
            for (int d = 0; d < HDIM; ++d)
                oh[d] += p * f16_to_f32(v[d]);
        }
    }
}

static void attention_flash(const float *q,
                            const uint16_t *key_cache,
                            const uint16_t *value_cache,
                            int context,
                            float *out)
{
    const float scale = 1.0f / sqrtf((float)HDIM);
    const size_t kv_stride = (size_t)KVHEADS * HDIM;

    for (int h = 0; h < QHEADS; ++h) {

        const int group = h / (QHEADS / KVHEADS);

        const float *qh = q + (size_t)h * HDIM;
        float *oh = out + (size_t)h * HDIM;

        if (context <= 0) {
            memset(oh, 0, (size_t)HDIM * sizeof(float));
            continue;
        }

        const uint16_t *k =
            key_cache + (size_t)group * HDIM;

        const uint16_t *v =
            value_cache + (size_t)group * HDIM;

        // First token: initialize online softmax.
        float score = 0.0f;

        for (int d = 0; d < HDIM; ++d)
            score += qh[d] * f16_to_f32(k[d]);

        float max_score = score * scale;
        float normalizer = 1.0f;

        for (int d = 0; d < HDIM; ++d)
            oh[d] = f16_to_f32(v[d]);

        // Process remaining cached tokens.
        for (int t = 1; t < context; ++t) {

            k += kv_stride;
            v += kv_stride;

            score = 0.0f;

            for (int d = 0; d < HDIM; ++d)
                score += qh[d] * f16_to_f32(k[d]);

            score *= scale;

            if (score > max_score) {

                // New maximum: rescale previous accumulation.
                const float alpha =
                    expf(max_score - score);

                normalizer = normalizer * alpha + 1.0f;

                for (int d = 0; d < HDIM; ++d)
                    oh[d] = oh[d] * alpha + f16_to_f32(v[d]);

                max_score = score;

            } else {

                // Maximum unchanged: no rescaling needed.
                const float beta =
                    expf(score - max_score);

                normalizer += beta;

                for (int d = 0; d < HDIM; ++d)
                    oh[d] += beta * f16_to_f32(v[d]);
            }
        }

        // Final normalization.
        const float inv = 1.0f / normalizer;

        for (int d = 0; d < HDIM; ++d)
            oh[d] *= inv;
    }
}

static void attention_compute(attention_method_t method, const float *q,
                              const uint16_t *kc, const uint16_t *vc,
                              int context, float *out, float *score_workspace) {
    if (method == ATTENTION_FLASH)
        attention_flash(q, kc, vc, context, out);
    else
        attention_standard(q, kc, vc, context, out, score_workspace);
}

#endif // QWEN_ATTENTION_KERNELS_H
