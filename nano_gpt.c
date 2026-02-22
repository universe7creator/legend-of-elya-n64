/*
 * nano_gpt.c - Sophia Elya AI: World's First N64 LLM
 *
 * nano-GPT inference engine for N64 (MIPS R4300i + RSP)
 * Model: 4 layers, 128 embedding dim, 4 heads, vocab=256, ctx=64
 * Weights: Q8 quantized (int8 per weight), scales in float16 per 32-block
 * Activations: Q8.7 fixed-point (int16_t, 1.0 = 128)
 *
 * RSP acceleration strategy:
 *   - DMA weight tiles into DMEM (4KB) in 128-byte chunks
 *   - data_cache_hit_writeback_invalidate prefetch hints
 *   - Process dot products in blocks of 8 (RSP vector width)
 *
 * Memory budget (8MB RDRAM):
 *   - Weights in ROM (DMA'd on demand): ~784KB
 *   - KV cache (SGAIKVCache): 4*64*128*2 * 2 = 131KB
 *   - Activations (x, logits, scratch): ~3KB
 *   - wbuf: 1MB
 */

#include "nano_gpt.h"
#include <string.h>
#include <stdlib.h>
#include <malloc.h>
#include <libdragon.h>

/* -----------------------------------------------------------------------
 * Fixed-point utilities
 * Q8.7: int16_t where 128 = 1.0
 * ----------------------------------------------------------------------- */

#define FP_ONE        128       /* 1.0 in Q8.7 */
#define FP_SCALE      128
#define FP_HALF       64        /* 0.5 in Q8.7 */

/* Multiply two Q8.7 values, return Q8.7 */
static inline int16_t fp_mul(int16_t a, int16_t b)
{
    return (int16_t)(((int32_t)a * (int32_t)b) >> 7);
}

/* Saturating add for Q8.7 */
static inline int16_t fp_add_sat(int32_t acc)
{
    if (acc > 32767) return 32767;
    if (acc < -32768) return -32768;
    return (int16_t)acc;
}

/* -----------------------------------------------------------------------
 * float16 decode (weights are stored as IEEE 754 half-precision scales)
 * Returns Q8.7 fixed-point (multiply this by raw Q4 nibble)
 * ----------------------------------------------------------------------- */
static int16_t f16_to_fp_scale(uint16_t f16)
{
    /* Python/numpy stores float16 as little-endian (x86 native).
     * N64 is big-endian: byte-swap before decoding. */
    f16 = (uint16_t)((f16 >> 8) | (f16 << 8));
    /* IEEE 754 half: s(1) | exp(5) | frac(10) */
    uint32_t sign     = (f16 >> 15) & 1;
    uint32_t exp      = (f16 >> 10) & 0x1F;
    uint32_t frac     = f16 & 0x3FF;
    float val;

    if (exp == 0) {
        /* subnormal */
        val = (frac / 1024.0f) * (1.0f / 16384.0f);
    } else if (exp == 31) {
        /* inf/nan -> clamp */
        val = 65504.0f;
    } else {
        float mantissa = 1.0f + frac / 1024.0f;
        int e = (int)exp - 15;
        if (e >= 0) {
            val = mantissa * (float)(1u << (unsigned)e);
        } else {
            val = mantissa / (float)(1u << (unsigned)(-e));
        }
    }
    if (sign) val = -val;

    /* Convert to Q8.7: clamp to int16 range */
    int32_t fixed = (int32_t)(val * FP_SCALE);
    if (fixed > 32767) fixed = 32767;
    if (fixed < -32768) fixed = -32768;
    return (int16_t)fixed;
}

/* -----------------------------------------------------------------------
 * Q8 dequantize helper
 * packed: pointer to int8 array (1 weight per byte, signed)
 * scales: float16 scale per 32-weight block
 * idx:    weight index (0-based)
 * Returns dequantized value in Q8.7
 * ----------------------------------------------------------------------- */
static inline int16_t q8_dequant(const int8_t *packed, const uint16_t *scales, int idx)
{
    int8_t  w     = packed[idx];              /* signed: -128..+127 */
    int     block = idx / SGAI_Q_BLOCK;
    int16_t scale = f16_to_fp_scale(scales[block]);

    /* w * scale: w in [-128,127], scale in Q8.7.
     * Product is Q16.14 (int32); >>7 to get Q8.7. */
    return (int16_t)(((int32_t)w * (int32_t)scale) >> 7);
}

/* -----------------------------------------------------------------------
 * RSP-accelerated matrix multiply (Q8 weights x Q8.7 input)
 *
 * Computes: output[out_dim] = W[out_dim x in_dim] * input[in_dim]
 * W is Q8 packed (out_dim * in_dim bytes, signed int8), scales are float16.
 * output and input are Q8.7 fixed-point int16_t.
 *
 * RSP DMA strategy:
 *   We tile the weight matrix into 128-byte chunks (matching the RSP
 *   DMA granularity and DCACHE line size on R4300). For each output
 *   row we:
 *     1. Issue data_cache_hit_writeback_invalidate on the weight tile
 *        to get a clean DMA-ready line.
 *     2. Accumulate the dot product in int32 to avoid overflow.
 *     3. Scale back to Q8.7 and store.
 *
 * In a full RSP microcode implementation, steps 1-3 would be offloaded
 * to the RSP via DMA + vector MAC instructions (vmudh/vmadh on 8-lane
 * int16 vectors). Here we provide the CPU fallback with DMA hints so
 * the hardware prefetcher can pipeline weight loads.
 * ----------------------------------------------------------------------- */
void sgai_rsp_matmul_q8(const int8_t *weights, const uint16_t *scales,
                          const int16_t *input,   int16_t *output,
                          int in_dim, int out_dim)
{
    /* Weight tile size: 128 bytes = 128 Q8 weights = 1 output row
     * for in_dim=128. Prefetch 2 rows ahead. */
    const int TILE_BYTES = 128;

    for (int o = 0; o < out_dim; o++) {
        /* Prefetch next weight row into D-cache */
        int next_row = o + 2;
        if (next_row < out_dim) {
            const int8_t *next_ptr = weights + (next_row * in_dim);
            data_cache_hit_writeback_invalidate((void *)next_ptr, TILE_BYTES);
        }

        /* Dot product: row o of W (in_dim Q8 weights) dot input (Q8.7) */
        int32_t acc = 0;

        /* Process in blocks of 8 (RSP vector width) */
        const int8_t   *row_w = weights + (o * in_dim);
        const uint16_t *row_s = scales  + (o * in_dim / SGAI_Q_BLOCK);

        for (int i = 0; i < in_dim; i += 8) {
            /* Unroll 8 lanes - mirrors RSP vmudh 8-element vector op */
            int lim = (i + 8 < in_dim) ? i + 8 : in_dim;
            for (int j = i; j < lim; j++) {
                int16_t w_dq = q8_dequant(row_w, row_s, j);
                acc += (int32_t)w_dq * (int32_t)input[j];
            }
        }

        /* Shift back: both operands were Q8.7 so product is Q16.14, >>7 to Q8.7 */
        acc >>= 7;
        output[o] = fp_add_sat(acc);
    }
}

/* -----------------------------------------------------------------------
 * Softmax in-place (Q8.7 input, Q8.7 output scaled to sum=128)
 * Uses integer approximation: find max, subtract, exponentiate with
 * e^x ≈ 1 + x for small x (sufficient for attention weights at this scale)
 * ----------------------------------------------------------------------- */
void sgai_softmax_inplace(int16_t *vec, int len)
{
    /* Find max for numerical stability */
    int16_t max_val = vec[0];
    for (int i = 1; i < len; i++) {
        if (vec[i] > max_val) max_val = vec[i];
    }

    /* Compute exp(x - max) approximation and sum
     * For Q8.7: e^x ≈ 1 + x + x^2/2 (2nd-order Taylor, x in [-4,0]) */
    int32_t sum = 0;
    int32_t exp_vals[SGAI_N_EMBED]; /* enough for attention (max in_dim) */
    int lim = (len < SGAI_N_EMBED) ? len : SGAI_N_EMBED;

    for (int i = 0; i < lim; i++) {
        int32_t x = (int32_t)(vec[i] - max_val);  /* x <= 0, Q8.7 */
        /* Bit-shift exponential: e^x = 2^(x * log2(e))
         * In Q8.7: e_fp = 128 >> shift,  shift = (-x * 185) >> 14
         * (185/16384 ≈ log2(e)/128 = 1.4427/128)
         * This is monotone-decreasing unlike the broken Taylor at large |x|.
         * At x=0: shift=0, e=128 (=1.0). At x=-512 (=-4.0): shift=5, e=4. */
        int32_t e;
        if (x <= -1792) {          /* e^(-14) * 128 ≈ 0 — clamp to 1 */
            e = 1;
        } else {
            int32_t shift = ((int32_t)(-x) * 185) >> 14;
            e = (int32_t)FP_ONE >> shift;
            if (e < 1) e = 1;
        }
        exp_vals[i] = e;
        sum += e;
    }

    /* Normalize: output[i] = exp_vals[i] * FP_ONE / sum */
    if (sum == 0) sum = 1;
    for (int i = 0; i < lim; i++) {
        vec[i] = (int16_t)((exp_vals[i] * FP_ONE) / sum);
    }
}

/* ReLU for Q8.7 */
int16_t sgai_relu(int16_t x)
{
    return (x > 0) ? x : 0;
}

/* -----------------------------------------------------------------------
 * Layer normalization (simplified RMS norm)
 * Normalizes vec in-place. No learned scale/bias (nano-GPT omission).
 * ----------------------------------------------------------------------- */
static void rms_norm(int16_t *vec, int len)
{
    int64_t sum_sq = 0;
    for (int i = 0; i < len; i++) {
        sum_sq += (int64_t)vec[i] * vec[i];
    }
    /* RMS = sqrt(sum_sq / len), in Q8.7 */
    int32_t mean_sq = (int32_t)(sum_sq / len);
    /* Integer sqrt via Newton's method */
    if (mean_sq <= 0) return;
    int32_t rms = 128; /* initial guess for sqrt(mean_sq) */
    for (int iter = 0; iter < 8; iter++) {
        rms = (rms + mean_sq / rms) >> 1;
    }
    if (rms == 0) rms = 1;

    /* Normalize: vec[i] = vec[i] * FP_ONE / rms */
    for (int i = 0; i < len; i++) {
        vec[i] = (int16_t)(((int32_t)vec[i] * FP_ONE) / rms);
    }
}

/* -----------------------------------------------------------------------
 * Embedding lookup (Q8 int8 table)
 * Writes SGAI_N_EMBED Q8.7 values into out[]
 * Embedding table immediately follows SGAIHeader in ROM.
 * ----------------------------------------------------------------------- */
static void embed_lookup(const SGAIHeader *hdr, uint8_t token, int16_t *out)
{
    /* Embedding table: vocab * n_embed bytes (int8), right after header */
    const int8_t *emb_table = (const int8_t *)(hdr + 1);
    int offset = (int)token * SGAI_N_EMBED;

    if (hdr == NULL) {
        /* Null weights: deterministic hash-based embedding */
        for (int i = 0; i < SGAI_N_EMBED; i++) {
            uint32_t h = (uint32_t)token * 2654435761u + (uint32_t)i * 40503u;
            out[i] = (int16_t)((int8_t)(h >> 16));
        }
        return;
    }

    for (int i = 0; i < SGAI_N_EMBED; i++) {
        /* Q8 embedding: int8 weight, scale = 1/8 baked in during export
         * Python exports emb as int8 with max_abs clamped to ~1.0 range
         * Convert to Q8.7: direct cast — Python exports em2≈0.9922, so int8≈Q8.7 already */
        out[i] = (int16_t)emb_table[offset + i];
    }
}

/* -----------------------------------------------------------------------
 * Attention layer forward pass
 * Single multi-head attention block:
 *   1. Project to Q, K, V
 *   2. Store K, V in KV cache
 *   3. Compute attention scores (Q dot all cached K)
 *   4. Softmax + weighted sum of V
 *   5. Project output (Wo)
 *   6. Residual add
 *   7. FFN: x = x + ff2(relu(ff1(x)))
 * ----------------------------------------------------------------------- */
static void attention_layer(const SGAILayer *layer, SGAIKVCache *kv,
                             int layer_idx, int pos,
                             int16_t *x)
{
    static int16_t q[SGAI_N_EMBED];
    static int16_t k_cur[SGAI_N_EMBED];
    static int16_t v_cur[SGAI_N_EMBED];
    static int16_t attn_out[SGAI_N_EMBED];
    static int16_t ff_buf[SGAI_N_EMBED * 4];  /* FFN hidden (512) */
    static int16_t attn_scores[SGAI_CTX];
    static int16_t residual[SGAI_N_EMBED];

    /* Save residual for skip connection */
    memcpy(residual, x, SGAI_N_EMBED * sizeof(int16_t));

    /* Layer norm input */
    rms_norm(x, SGAI_N_EMBED);

    /* Project Q, K, V */
    sgai_rsp_matmul_q8(layer->wq, layer->sq, x, q,     SGAI_N_EMBED, SGAI_N_EMBED);
    sgai_rsp_matmul_q8(layer->wk, layer->sk, x, k_cur, SGAI_N_EMBED, SGAI_N_EMBED);
    sgai_rsp_matmul_q8(layer->wv, layer->sv, x, v_cur, SGAI_N_EMBED, SGAI_N_EMBED);

    /* Store K, V in cache at current position */
    if (pos < SGAI_CTX) {
        memcpy(kv->k[layer_idx][pos], k_cur, SGAI_N_EMBED * sizeof(int16_t));
        memcpy(kv->v[layer_idx][pos], v_cur, SGAI_N_EMBED * sizeof(int16_t));
    }

    /* Compute attention scores: Q dot K[0..pos] for each head */
    /* Multi-head: process head by head */
    memset(attn_out, 0, SGAI_N_EMBED * sizeof(int16_t));

    int n_ctx = (pos + 1 < SGAI_CTX) ? pos + 1 : SGAI_CTX;

    for (int h = 0; h < SGAI_N_HEADS; h++) {
        const int16_t *q_head = q + h * SGAI_HEAD_DIM;

        /* Attention scores for this head over all positions */
        for (int t = 0; t < n_ctx; t++) {
            const int16_t *k_head = kv->k[layer_idx][t] + h * SGAI_HEAD_DIM;
            int32_t score = 0;
            for (int d = 0; d < SGAI_HEAD_DIM; d++) {
                score += (int32_t)q_head[d] * (int32_t)k_head[d];
            }
            /* Scale by 1/sqrt(head_dim) = 1/sqrt(32) ≈ 0.177
             * In Q8.7: multiply by 23 (≈ 0.177 * 128) then >> 7 */
            score = (score * 23) >> 14;  /* >> 7 for Q8.7^2, >> 7 for scale */
            attn_scores[t] = fp_add_sat(score);
        }

        /* Causal mask: positions beyond pos already excluded by n_ctx */
        /* Softmax over attn_scores[0..n_ctx-1] */
        sgai_softmax_inplace(attn_scores, n_ctx);

        /* Weighted sum of V */
        int head_out_base = h * SGAI_HEAD_DIM;
        for (int d = 0; d < SGAI_HEAD_DIM; d++) {
            int32_t acc = 0;
            for (int t = 0; t < n_ctx; t++) {
                const int16_t *v_head = kv->v[layer_idx][t] + h * SGAI_HEAD_DIM;
                acc += (int32_t)attn_scores[t] * (int32_t)v_head[d];
            }
            attn_out[head_out_base + d] = fp_add_sat(acc >> 7);
        }
    }

    /* Output projection Wo */
    static int16_t proj_out[SGAI_N_EMBED];
    sgai_rsp_matmul_q8(layer->wo, layer->so, attn_out, proj_out,
                        SGAI_N_EMBED, SGAI_N_EMBED);

    /* Residual add: x = residual + proj_out */
    for (int i = 0; i < SGAI_N_EMBED; i++) {
        x[i] = fp_add_sat((int32_t)residual[i] + (int32_t)proj_out[i]);
    }

    /* ---- FFN block ---- */
    memcpy(residual, x, SGAI_N_EMBED * sizeof(int16_t));

    /* Layer norm before FFN */
    rms_norm(x, SGAI_N_EMBED);

    /* ff1: 128 -> 512 */
    sgai_rsp_matmul_q8(layer->wff1, layer->sff1, x, ff_buf,
                        SGAI_N_EMBED, SGAI_N_EMBED * 4);

    /* ReLU */
    for (int i = 0; i < SGAI_N_EMBED * 4; i++) {
        ff_buf[i] = sgai_relu(ff_buf[i]);
    }

    /* ff2: 512 -> 128 */
    static int16_t ff_out[SGAI_N_EMBED];
    sgai_rsp_matmul_q8(layer->wff2, layer->sff2, ff_buf, ff_out,
                        SGAI_N_EMBED * 4, SGAI_N_EMBED);

    /* Residual add */
    for (int i = 0; i < SGAI_N_EMBED; i++) {
        x[i] = fp_add_sat((int32_t)residual[i] + (int32_t)ff_out[i]);
    }
}

/* -----------------------------------------------------------------------
 * Logit projection: x[128] -> logits[256]
 * Uses a simple linear unembedding (tied embedding weights for efficiency).
 * For the null-weight demo, uses a byte-frequency prior biased toward
 * printable ASCII to produce semi-plausible character outputs.
 * ----------------------------------------------------------------------- */
static void project_to_logits(const SGAIHeader *hdr, const int16_t *x,
                               int16_t *logits)
{
    if (hdr == NULL) {
        /* Demo mode: logits biased toward printable ASCII (32-126) */
        for (int v = 0; v < SGAI_VOCAB; v++) {
            /* Base: prefer printable characters */
            int32_t base = (v >= 32 && v <= 126) ? FP_ONE : -(FP_ONE * 4);
            /* Mix in a projection of x[v % SGAI_N_EMBED] for variation */
            int32_t proj = x[v % SGAI_N_EMBED];
            logits[v] = fp_add_sat(base + (proj >> 2));
        }
        return;
    }

    /* Tied unembedding: use embedding table rows as projection vectors (Q8) */
    const int8_t *emb_table = (const int8_t *)(hdr + 1);
    for (int v = 0; v < SGAI_VOCAB; v++) {
        int32_t acc = 0;
        int offset = v * SGAI_N_EMBED;
        for (int i = 0; i < SGAI_N_EMBED; i++) {
            /* e_val in Q8.7 (same scale as embed_lookup) */
            int16_t e_val = (int16_t)emb_table[offset + i];
            acc += (int32_t)e_val * (int32_t)x[i];
        }
        logits[v] = fp_add_sat(acc >> 7);
    }
}

/* -----------------------------------------------------------------------
 * Temperature sampling
 * temperature_q8: temperature in Q8 (256 = 1.0). Lower = more greedy.
 * Returns sampled token index.
 * ----------------------------------------------------------------------- */
/* Hard post-softmax exclusion: recent tokens are zeroed AFTER softmax.
 * This guarantees no repetition regardless of logit magnitude — a token
 * with logit +20 still gets probability 0 once we hard-zero it.
 * hist[0]=most recent, hist[7]=oldest. Window of 8 prevents cycles
 * up to length 8 (covers "I am Sop" and similar short stuck loops). */
static uint8_t sample_logits(const int16_t *logits, uint32_t temperature_q8,
                               const uint8_t *hist, int n_hist)
{
    static int16_t probs[SGAI_VOCAB];
    memcpy(probs, logits, SGAI_VOCAB * sizeof(int16_t));

    if (temperature_q8 == 0) {
        /* Greedy: argmax over printable ASCII 32-126, excluding recent tokens */
        int best = -1;
        for (int i = 32; i <= 126; i++) {
            int skip = 0;
            for (int h = 0; h < n_hist && h < 8; h++)
                if (hist[h] == (uint8_t)i) { skip = 1; break; }
            if (skip) continue;
            if (best < 0 || probs[i] > probs[best]) best = i;
        }
        return (best >= 0) ? (uint8_t)best : 'a';
    }

    /* Apply temperature: T = temp_q8/256.
     * temp_q8=350 -> T~1.37 (softer/more diverse for overfitted model) */
    for (int i = 0; i < SGAI_VOCAB; i++) {
        int32_t scaled = ((int32_t)probs[i] * 256) / (int32_t)temperature_q8;
        probs[i] = fp_add_sat(scaled);
    }

    /* Softmax over full vocab */
    sgai_softmax_inplace(probs, SGAI_VOCAB);

    /* Step 1: zero non-printable tokens (outside ASCII 32-126) */
    for (int i = 0; i < SGAI_VOCAB; i++) {
        if (i < 32 || i > 126) probs[i] = 0;
    }

    /* Step 2: hard-zero recent tokens — guaranteed exclusion.
     * Zeroing AFTER softmax means even a +20 logit token gets prob=0.
     * Also zero any token that appears 2+ times in the window (frequency cap)
     * to break m→a→m→a style 2-token cycles with matching logit pairs. */
    for (int h = 0; h < n_hist && h < 16; h++) {
        uint8_t t = hist[h];
        if (t >= 32 && t <= 126) probs[t] = 0;
    }
    /* Frequency cap: count occurrences, zero any token seen >= 2 times */
    {
        uint8_t freq[256];
        memset(freq, 0, sizeof(freq));
        for (int h = 0; h < n_hist && h < 16; h++) freq[hist[h]]++;
        for (int i = 32; i <= 126; i++)
            if (freq[i] >= 2) probs[i] = 0;
    }

    /* Multinomial sample using MIPS CP0 Count register as RNG seed */
    uint32_t rng;
    asm volatile("mfc0 %0, $9" : "=r"(rng));
    rng ^= rng >> 16;
    rng *= 0x45d9f3b;
    rng ^= rng >> 16;

    /* Cumulative sum sampling over printable range */
    int32_t r = (int32_t)(rng & 0x7FFF);  /* [0, 32767] */
    int32_t csum = 0;
    int32_t total = 0;
    for (int i = 32; i <= 126; i++) total += probs[i];
    if (total == 0) {
        /* All candidates exhausted — scan printable range for first char not
         * in recent history. With 94 printable chars and a 16-token window
         * this almost always succeeds quickly. */
        for (int i = 32; i <= 126; i++) {
            int in_hist = 0;
            for (int h = 0; h < n_hist && h < 16; h++)
                if (hist[h] == (uint8_t)i) { in_hist = 1; break; }
            if (!in_hist) return (uint8_t)i;
        }
        return ' ';   /* absolute last resort */
    }

    for (int i = 32; i <= 126; i++) {
        csum += (int32_t)probs[i] * 32767 / total;
        if (r < csum) return (uint8_t)i;
    }
    return (uint8_t)(32 + (rng & 31));
}

/* -----------------------------------------------------------------------
 * Public API
 * ----------------------------------------------------------------------- */

void sgai_init(SGAIState *state, const void *rom_weights)
{
    memset(state, 0, sizeof(SGAIState));

    if (rom_weights != NULL) {
        const SGAIHeader *hdr = (const SGAIHeader *)rom_weights;
        /* Magic is written LE by Python; N64 reads 4 bytes as BE uint32.
         * Python: struct.pack('<I', 0x53454149) -> bytes [0x49,0x41,0x45,0x53]
         * N64 reads as BE: 0x49414553  (byte-swapped from SGAI_MAGIC)       */
        uint32_t m = hdr->magic;
        uint32_t m_host = ((m & 0xFF000000u) >> 24) | ((m & 0x00FF0000u) >> 8)
                        | ((m & 0x0000FF00u) <<  8) | ((m & 0x000000FFu) << 24);
        if (m_host == SGAI_MAGIC) {
            state->weights = hdr;
            state->is_loaded = 1;
        }
    }

    /* Allocate KV cache in RDRAM (8-byte aligned for DMA) */
    state->kv = (SGAIKVCache *)memalign(8, sizeof(SGAIKVCache));
    if (state->kv) {
        memset(state->kv, 0, sizeof(SGAIKVCache));
        state->kv->pos = 0;
    }

    state->seq_len = 0;
}

void sgai_reset(SGAIState *state)
{
    if (state->kv) {
        memset(state->kv, 0, sizeof(SGAIKVCache));
        state->kv->pos = 0;
    }
    state->seq_len = 0;
    memset(state->x, 0, sizeof(state->x));
    memset(state->logits, 0, sizeof(state->logits));
    memset(state->penalty_hist, 0, sizeof(state->penalty_hist));
    state->penalty_n = 0;
}

/*
 * Run one forward pass for a single input token.
 * Updates KV cache, returns next predicted token.
 */
uint8_t sgai_next_token(SGAIState *state, uint8_t input_token,
                          uint32_t temperature_q8)
{
    if (!state->kv) return 0;

    int pos = state->kv->pos;

    /* 1. Embedding lookup */
    embed_lookup(state->weights, input_token, state->x);

    /* 2. Run transformer layers */
    if (state->is_loaded && state->weights != NULL) {
        /* Weights layout after header:
         * - Embedding table: SGAI_VOCAB * SGAI_N_EMBED bytes (Q8 int8)
         * - n_layers SGAILayer structs */
        const uint8_t *after_hdr = (const uint8_t *)(state->weights + 1);
        size_t emb_table_bytes = SGAI_VOCAB * SGAI_N_EMBED;  /* Q8: 1 byte per weight */
        const SGAILayer *layers = (const SGAILayer *)(after_hdr + emb_table_bytes);

        for (int l = 0; l < SGAI_N_LAYERS; l++) {
            attention_layer(&layers[l], state->kv, l, pos, state->x);
        }
    } else {
        /* Demo mode: run with null weights (produces character-frequency biased output) */
        SGAILayer dummy;
        memset(&dummy, 0, sizeof(dummy));
        for (int l = 0; l < SGAI_N_LAYERS; l++) {
            attention_layer(&dummy, state->kv, l, pos, state->x);
        }
    }

    /* 3. Final layer norm */
    rms_norm(state->x, SGAI_N_EMBED);

    /* 4. Project to logits */
    project_to_logits(state->weights, state->x, state->logits);

    /* 5. Sample with window-based repetition penalty */
    uint8_t next_tok = sample_logits(state->logits, temperature_q8,
                                      state->penalty_hist, (int)state->penalty_n);

    /* Update penalty history only during generation (temp > 0).
     * Shift array right, insert new token at [0]. */
    if (temperature_q8 > 0) {
        int new_n = ((int)state->penalty_n < 16) ? (int)state->penalty_n + 1 : 16;
        for (int i = new_n - 1; i > 0; i--)
            state->penalty_hist[i] = state->penalty_hist[i - 1];
        state->penalty_hist[0] = next_tok;
        state->penalty_n = (uint8_t)new_n;
    }

    /* 6. Advance position in KV cache */
    if (state->kv->pos < SGAI_CTX - 1) {
        state->kv->pos++;
    } else {
        /* Context full: shift KV cache left (sliding window) */
        for (int l = 0; l < SGAI_N_LAYERS; l++) {
            for (int t = 0; t < SGAI_CTX - 1; t++) {
                memcpy(state->kv->k[l][t], state->kv->k[l][t + 1],
                       SGAI_N_EMBED * sizeof(int16_t));
                memcpy(state->kv->v[l][t], state->kv->v[l][t + 1],
                       SGAI_N_EMBED * sizeof(int16_t));
            }
        }
    }

    /* Store token in sequence */
    if (state->seq_len < SGAI_CTX) {
        state->tokens[state->seq_len++] = input_token;
    }

    return next_tok;
}

/*
 * Generate up to max_tokens tokens from a prompt.
 * Output written to caller-provided buffer (null-terminated).
 */
void sgai_generate(SGAIState *state, const uint8_t *prompt, int prompt_len,
                   uint8_t *output, int max_tokens, uint32_t temperature_q8)
{
    sgai_reset(state);

    /* Process prompt tokens (teacher-forcing: feed each prompt token,
     * discard output until last) */
    uint8_t tok = 0;
    for (int i = 0; i < prompt_len; i++) {
        tok = sgai_next_token(state, prompt[i], temperature_q8);
    }

    /* Generate output tokens */
    int out_idx = 0;
    while (out_idx < max_tokens - 1) {
        tok = sgai_next_token(state, tok, temperature_q8);
        if (tok == 0) break;  /* null terminator */
        output[out_idx++] = tok;
    }
    output[out_idx] = 0;
}
