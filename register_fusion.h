#pragma once
/*
 * RVV1.0/VLEN=1024 head-pair register-fusion experiment.
 * Only the normed input Q8 buffer, persistent FP16 KV cache, final/partial Y,
 * precomputed RoPE coefficients, and Q4 weight tiles are memory-resident.
 * The 256 Q projections, RoPE, online attention, 256 O values, Q8 quantization,
 * and W_O activation handoff are held in RVV values, not C arrays.
 *
 * This file is included after fused_layer.h and core qwen_one_layer.c helpers.
 * A100 Zvfh: native FP16 vector loads/stores and register-only widening /
 * narrowing, without temporary FP32 scratch arrays.
 *
 * REQUIREMENT: -ffp-contract=off; no calls or timing inside a live head pair.
 * A100 vector register file cannot hold full Y[1536], so partial 256-element
 * output tiles are written to Y after each pair. Those stores are necessary.
 */
#if HAVE_RVV
#if !defined(__riscv_zvfh)
#error "Register fusion needs Zvfh; build with -march=rv64gcv_zvfh_zfh_zicbop"
#endif

#define RF_INLINE static inline __attribute__((always_inline))

/* Scalar Q8 scale conversions are ALWAYS inlined, to avoid library calls that
 * could force spilling the live RVV accumulators. */
RF_INLINE float rf_scale_from_f16(uint16_t h) {
    uint32_t sign=((uint32_t)h & 0x8000u)<<16;
    uint32_t e=(h>>10)&31u, mant=h&1023u;
    if (e==0) {
        if (mant==0) { union {uint32_t u;float f;} z={.u=sign};return z.f; }
        float v=(float)mant*0x1p-24f;
        return sign ? -v : v;
    }
    uint32_t bits=sign | ((e==31 ? 255u : e+112u)<<23) | (mant<<13);
    union {uint32_t u;float f;} result={.u=bits};
    return result.f;
}

RF_INLINE uint16_t rf_scale_to_f16(float x) {
    union {float f;uint32_t u;} cvt={.f=x};
    uint32_t b=cvt.u;
    uint16_t sign=(uint16_t)((b>>16)&0x8000u);
    uint32_t mant=b&0x7fffffu;
    int e=(int)((b>>23)&255u)-127+15;
    if (((b>>23)&255u)==255u)
        return (uint16_t)(sign | (mant ? 0x7e00u : 0x7c00u));
    if (e>=31) return (uint16_t)(sign|0x7c00u);
    if (e<=0) {
        if (e<-10) return sign;
        uint32_t m=mant|0x800000u;
        int shift=14-e;
        uint32_t frac=m>>shift;
        uint32_t rem=m&((1u<<shift)-1u);
        uint32_t tie=1u<<(shift-1);
        if (rem>tie || (rem==tie && (frac&1u))) ++frac;
        return (uint16_t)(sign|frac);
    }
    uint32_t rounded=mant+0xfffu+((mant>>13)&1u);
    if (rounded&0x800000u) {++e;rounded=0;if(e>=31)return (uint16_t)(sign|0x7c00u);}
    return (uint16_t)(sign|((uint16_t)e<<10)|(rounded>>13));
}

/* Direct cache FP16 vector load and widening within the register file. */
RF_INLINE vfloat32m2_t rf_f16_load64(const uint16_t *src) {
    const size_t vl=64;
    vuint16m1_t bits=__riscv_vle16_v_u16m1(src,vl);
    vfloat16m1_t half=__riscv_vreinterpret_v_u16m1_f16m1(bits);
    return __riscv_vfwcvt_f_f_v_f32m2(half,vl);
}

/* Convert the new FP32 key/value outputs to the *persistent* FP16 KV cache.
 * No intermediate FP32 or FP16 arrays: narrowing is in RVV registers. */
RF_INLINE void rf_f16_store64(uint16_t *dst, vfloat32m2_t f) {
    const size_t vl=64;
    vfloat16m1_t half=__riscv_vfncvt_f_f_w_f16m1(f,vl);
    __riscv_vse16_v_u16m1(dst,__riscv_vreinterpret_v_f16m1_u16m1(half),vl);
}

/* Output-neuron-vectorized Q4_0 x Q8_0: four 64-lane FP32 register values
 * represent the 256 outputs. No FP32 output array is materialized. */
RF_INLINE void rf_project256(const uint8_t *tile, const block_q8_0 *x,
                              int nblocks, const float *blo, const float *bhi,
                              vfloat32m2_t *o0, vfloat32m2_t *o1,
                              vfloat32m2_t *o2, vfloat32m2_t *o3) {
    const size_t vl = 64;
    vfloat32m2_t a0 = blo ? __riscv_vle32_v_f32m2(blo,vl) : __riscv_vfmv_v_f_f32m2(0,vl);
    vfloat32m2_t a1 = blo ? __riscv_vle32_v_f32m2(blo+64,vl) : __riscv_vfmv_v_f_f32m2(0,vl);
    vfloat32m2_t a2 = bhi ? __riscv_vle32_v_f32m2(bhi,vl) : __riscv_vfmv_v_f_f32m2(0,vl);
    vfloat32m2_t a3 = bhi ? __riscv_vle32_v_f32m2(bhi+64,vl) : __riscv_vfmv_v_f_f32m2(0,vl);
    for (int b=0; b<nblocks; ++b) {
        const uint8_t *grp = tile + (size_t)b*FUSED_GROUP_BYTES;
        vint32m2_t i0=__riscv_vmv_v_x_i32m2(0,vl), i1=__riscv_vmv_v_x_i32m2(0,vl);
        vint32m2_t i2=__riscv_vmv_v_x_i32m2(0,vl), i3=__riscv_vmv_v_x_i32m2(0,vl);
        for (int k=0;k<32;++k) {
            const int16_t xk=x[b].qs[k];
            const uint8_t *col=grp + (size_t)k*128;
            vuint8mf2_t w0=__riscv_vle8_v_u8mf2(col,vl);
            vuint8mf2_t w1=__riscv_vle8_v_u8mf2(col+64,vl);
            vuint8mf2_t lo0=__riscv_vsub_vx_u8mf2(__riscv_vand_vx_u8mf2(w0,15,vl),8,vl);
            vuint8mf2_t lo1=__riscv_vsub_vx_u8mf2(__riscv_vand_vx_u8mf2(w1,15,vl),8,vl);
            vuint8mf2_t hi0=__riscv_vsub_vx_u8mf2(__riscv_vsrl_vx_u8mf2(w0,4,vl),8,vl);
            vuint8mf2_t hi1=__riscv_vsub_vx_u8mf2(__riscv_vsrl_vx_u8mf2(w1,4,vl),8,vl);
            i0=__riscv_vwmacc_vx_i32m2(i0,xk,__riscv_vsext_vf2_i16m1(__riscv_vreinterpret_v_u8mf2_i8mf2(lo0),vl),vl);
            i1=__riscv_vwmacc_vx_i32m2(i1,xk,__riscv_vsext_vf2_i16m1(__riscv_vreinterpret_v_u8mf2_i8mf2(lo1),vl),vl);
            i2=__riscv_vwmacc_vx_i32m2(i2,xk,__riscv_vsext_vf2_i16m1(__riscv_vreinterpret_v_u8mf2_i8mf2(hi0),vl),vl);
            i3=__riscv_vwmacc_vx_i32m2(i3,xk,__riscv_vsext_vf2_i16m1(__riscv_vreinterpret_v_u8mf2_i8mf2(hi1),vl),vl);
        }
        const float *d=(const float *)(const void *)(grp+4096);
        const float dx=rf_scale_from_f16(x[b].d);
        vfloat32m2_t c0=__riscv_vfcvt_f_x_v_f32m2(i0,vl);
        vfloat32m2_t c1=__riscv_vfcvt_f_x_v_f32m2(i1,vl);
        vfloat32m2_t c2=__riscv_vfcvt_f_x_v_f32m2(i2,vl);
        vfloat32m2_t c3=__riscv_vfcvt_f_x_v_f32m2(i3,vl);
        a0=__riscv_vfadd_vv_f32m2(a0,__riscv_vfmul_vf_f32m2(__riscv_vfmul_vv_f32m2(c0,__riscv_vle32_v_f32m2(d,vl),vl),dx,vl),vl);
        a1=__riscv_vfadd_vv_f32m2(a1,__riscv_vfmul_vf_f32m2(__riscv_vfmul_vv_f32m2(c1,__riscv_vle32_v_f32m2(d+64,vl),vl),dx,vl),vl);
        a2=__riscv_vfadd_vv_f32m2(a2,__riscv_vfmul_vf_f32m2(__riscv_vfmul_vv_f32m2(c2,__riscv_vle32_v_f32m2(d+128,vl),vl),dx,vl),vl);
        a3=__riscv_vfadd_vv_f32m2(a3,__riscv_vfmul_vf_f32m2(__riscv_vfmul_vv_f32m2(c3,__riscv_vle32_v_f32m2(d+192,vl),vl),dx,vl),vl);
    }
    *o0=a0; *o1=a1; *o2=a2; *o3=a3;
}

/* Qwen's rotate_half; requires precomputed trig tables because powf/cosf/sinf
 * calls while Q vectors are live would force caller-saved vector spills. */
RF_INLINE void rf_rope128(vfloat32m2_t *low, vfloat32m2_t *high,
                           vfloat32m2_t cs, vfloat32m2_t sn) {
    const size_t vl=64;
    vfloat32m2_t old=*low;
    *low=__riscv_vfsub_vv_f32m2(__riscv_vfmul_vv_f32m2(old,cs,vl),
                                  __riscv_vfmul_vv_f32m2(*high,sn,vl),vl);
    *high=__riscv_vfadd_vv_f32m2(__riscv_vfmul_vv_f32m2(*high,cs,vl),
                                   __riscv_vfmul_vv_f32m2(old,sn,vl),vl);
}

/* Inline polynomial exp for nonpositive x. No libm call can clobber live Q/O
 * vector registers. Degree-7 Taylor on [-ln2/2, +ln2/2]; absolute relative
 * approximation error about 1e-6 with f32 rounding (verify on A100). */
RF_INLINE float rf_exp_neg(float x) {
    if (x < -85.0f) return 0.0f;
    if (x >= 0.0f) return 1.0f;
    float z=x*1.4426950408889634f;
    int n=(int)(z<0.0f ? z-0.5f : z+0.5f);
    float r=x-(float)n*0.6931471805599453f;
    float p=1.0f/5040.0f;
    p=p*r+1.0f/720.0f;
    p=p*r+1.0f/120.0f;
    p=p*r+1.0f/24.0f;
    p=p*r+1.0f/6.0f;
    p=p*r+0.5f;
    p=p*r+1.0f;
    p=p*r+1.0f;
    uint32_t ebits=(uint32_t)(n+127)<<23;
    union {uint32_t u;float f;} pow2={.u=ebits};
    return p*pow2.f;
}

RF_INLINE float rf_sum64(vfloat32m2_t x) {
    vfloat32m1_t z=__riscv_vfmv_v_f_f32m1(0.0f,1);
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(x,z,64));
}

/* Streaming attention for two heads, both queries and all 256 weighted value
 * accumulators remain RVV values. FP16 cache is loaded as bits; f16->f32
 * conversion is vector-only. No keyf/valf/opair stack arrays. */
RF_INLINE void rf_attention2(vfloat32m2_t q00,vfloat32m2_t q01,
                              vfloat32m2_t q10,vfloat32m2_t q11,
                              const uint16_t *kc,const uint16_t *vc,
                              int context,int group,
                              vfloat32m2_t *o00,vfloat32m2_t *o01,
                              vfloat32m2_t *o10,vfloat32m2_t *o11) {
    const size_t vl=64;
    const float inv_dim=0.08838834764831845f; /* 1 / sqrt(128) */
    float m0=-INFINITY,m1=-INFINITY,l0=0,l1=0;
    vfloat32m2_t u00=__riscv_vfmv_v_f_f32m2(0.0f,vl),u01=u00,u10=u00,u11=u00;
    for(int t=0;t<context;++t) {
        const uint16_t *k=kc+(size_t)t*KVDIM+(size_t)group*HDIM;
        const uint16_t *v=vc+(size_t)t*KVDIM+(size_t)group*HDIM;
        vfloat32m2_t k0=rf_f16_load64(k),k1=rf_f16_load64(k+64);
        const float s0=(rf_sum64(__riscv_vfmul_vv_f32m2(q00,k0,vl))+
                        rf_sum64(__riscv_vfmul_vv_f32m2(q01,k1,vl)))*inv_dim;
        const float s1=(rf_sum64(__riscv_vfmul_vv_f32m2(q10,k0,vl))+
                        rf_sum64(__riscv_vfmul_vv_f32m2(q11,k1,vl)))*inv_dim;
        float a0,b0,a1,b1;
        if(t==0){m0=s0;m1=s1;l0=1.0f;l1=1.0f;a0=a1=0;b0=b1=1;}
        else {
            if(s0>m0){a0=rf_exp_neg(m0-s0);b0=1;l0=l0*a0+1;m0=s0;}
            else {a0=1;b0=rf_exp_neg(s0-m0);l0+=b0;}
            if(s1>m1){a1=rf_exp_neg(m1-s1);b1=1;l1=l1*a1+1;m1=s1;}
            else {a1=1;b1=rf_exp_neg(s1-m1);l1+=b1;}
        }
        vfloat32m2_t v0=rf_f16_load64(v),v1=rf_f16_load64(v+64);
        if(t==0){u00=u10=v0;u01=u11=v1;}
        else {
            if(a0==1.0f) {
                u00=__riscv_vfadd_vv_f32m2(u00,__riscv_vfmul_vf_f32m2(v0,b0,vl),vl);
                u01=__riscv_vfadd_vv_f32m2(u01,__riscv_vfmul_vf_f32m2(v1,b0,vl),vl);
            } else {
                u00=__riscv_vfadd_vv_f32m2(__riscv_vfmul_vf_f32m2(u00,a0,vl),v0,vl);
                u01=__riscv_vfadd_vv_f32m2(__riscv_vfmul_vf_f32m2(u01,a0,vl),v1,vl);
            }
            if(a1==1.0f) {
                u10=__riscv_vfadd_vv_f32m2(u10,__riscv_vfmul_vf_f32m2(v0,b1,vl),vl);
                u11=__riscv_vfadd_vv_f32m2(u11,__riscv_vfmul_vf_f32m2(v1,b1,vl),vl);
            } else {
                u10=__riscv_vfadd_vv_f32m2(__riscv_vfmul_vf_f32m2(u10,a1,vl),v0,vl);
                u11=__riscv_vfadd_vv_f32m2(__riscv_vfmul_vf_f32m2(u11,a1,vl),v1,vl);
            }
        }
    }
    *o00=__riscv_vfmul_vf_f32m2(u00,1.0f/l0,vl);
    *o01=__riscv_vfmul_vf_f32m2(u01,1.0f/l0,vl);
    *o10=__riscv_vfmul_vf_f32m2(u10,1.0f/l1,vl);
    *o11=__riscv_vfmul_vf_f32m2(u11,1.0f/l1,vl);
}

/* Quantize exactly 32 FP32 elements. Source vector may contain 64 elements;
 * offset=32 selects its upper half via vslidedown (no memory operations).
 * Uses RNE float->int conversion, then clamp [-127,127]. */
RF_INLINE vint8mf2_t rf_quant32(vfloat32m2_t src,int offset,uint16_t *scale_bits) {
    const size_t vl=32;
    vfloat32m2_t v= offset ? __riscv_vslidedown_vx_f32m2(src,32,vl) : src;
    vfloat32m2_t absv=__riscv_vfabs_v_f32m2(v,vl);
    vfloat32m1_t seed=__riscv_vfmv_v_f_f32m1(0.0f,1);
    float amax=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m2_f32m1(absv,seed,vl));
    float d=amax/127.0f;
    *scale_bits=rf_scale_to_f16(d); /* Q8 scale stored as exact FP16 bits */
    const float inv=d==0.0f ? 0.0f : 1.0f/d;
    vint32m2_t q=__riscv_vfcvt_x_f_v_i32m2(__riscv_vfmul_vf_f32m2(v,inv,vl),vl);
    q=__riscv_vmax_vx_i32m2(q,-127,vl);
    q=__riscv_vmin_vx_i32m2(q,127,vl);
    return __riscv_vncvt_x_x_w_i8mf2(__riscv_vncvt_x_x_w_i16m1(q,vl),vl);
}

/* Read one signed Q8 scalar from a 32-element register value. */
RF_INLINE int16_t rf_q8_lane(vint8mf2_t q,int idx) {
    return __riscv_vmv_x_s_i8mf2_i8(__riscv_vslidedown_vx_i8mf2(q,(size_t)idx,32));
}

/* One 128-neuron half of a W_O block: takes and returns the output
 * accumulator through the RISC-V vector calling convention, never through
 * a temporary FP32 output array.  Reduces vector register pressure. */
static __attribute__((noinline, riscv_vector_cc))
vfloat32m4_t rf_wo_block_half(const uint8_t *group,vint8mf2_t q,
                              uint16_t scale_bits,vfloat32m4_t sum,int high_half) {
    const size_t vl=128;
    vint32m4_t isum=__riscv_vmv_v_x_i32m4(0,vl);
    for(int k=0;k<32;++k) {
        const int16_t xk=rf_q8_lane(q,k);
        vuint8m1_t packed=__riscv_vle8_v_u8m1(group+(size_t)k*128,vl);
        vuint8m1_t bits=high_half ?
            __riscv_vsrl_vx_u8m1(packed,4,vl) :
            __riscv_vand_vx_u8m1(packed,15,vl);
        bits=__riscv_vsub_vx_u8m1(bits,8,vl);
        isum=__riscv_vwmacc_vx_i32m4(isum,xk,
            __riscv_vsext_vf2_i16m2(__riscv_vreinterpret_v_u8m1_i8m1(bits),vl),vl);
    }
    const float *scales=(const float *)(const void *)(group+4096);
    const float dx=rf_scale_from_f16(scale_bits);
    vfloat32m4_t a=__riscv_vfcvt_f_x_v_f32m4(isum,vl);
    a=__riscv_vfmul_vf_f32m4(__riscv_vfmul_vv_f32m4(a,
        __riscv_vle32_v_f32m4(scales+(high_half?128:0),vl),vl),dx,vl);
    return __riscv_vfadd_vv_f32m4(sum,a,vl);
}

/* Six Y tiles; no oq[] and no O[] memory. Pass compact Q8 blocks through
 * RVV argument registers to each 128-output subkernel. */
static __attribute__((noinline, riscv_vector_cc))
void rf_wo_pair(const uint8_t *wo,float *y,
                           vint8mf2_t q0,vint8mf2_t q1,vint8mf2_t q2,vint8mf2_t q3,
                           vint8mf2_t q4,vint8mf2_t q5,vint8mf2_t q6,vint8mf2_t q7,
                           uint16_t d0,uint16_t d1,uint16_t d2,uint16_t d3,
                           uint16_t d4,uint16_t d5,uint16_t d6,uint16_t d7) {
    for(int yt=0;yt<6;++yt) {
        const uint8_t *sub=wo+(size_t)yt*FUSED_WO_SUBTILE;
        vfloat32m4_t y0=__riscv_vle32_v_f32m4(y+(size_t)yt*256,128);
        y0=rf_wo_block_half(sub+0*FUSED_GROUP_BYTES,q0,d0,y0,0);
        y0=rf_wo_block_half(sub+1*FUSED_GROUP_BYTES,q1,d1,y0,0);
        y0=rf_wo_block_half(sub+2*FUSED_GROUP_BYTES,q2,d2,y0,0);
        y0=rf_wo_block_half(sub+3*FUSED_GROUP_BYTES,q3,d3,y0,0);
        y0=rf_wo_block_half(sub+4*FUSED_GROUP_BYTES,q4,d4,y0,0);
        y0=rf_wo_block_half(sub+5*FUSED_GROUP_BYTES,q5,d5,y0,0);
        y0=rf_wo_block_half(sub+6*FUSED_GROUP_BYTES,q6,d6,y0,0);
        y0=rf_wo_block_half(sub+7*FUSED_GROUP_BYTES,q7,d7,y0,0);
        __riscv_vse32_v_f32m4(y+(size_t)yt*256,y0,128);
        vfloat32m4_t y1=__riscv_vle32_v_f32m4(y+(size_t)yt*256+128,128);
        y1=rf_wo_block_half(sub+0*FUSED_GROUP_BYTES,q0,d0,y1,1);
        y1=rf_wo_block_half(sub+1*FUSED_GROUP_BYTES,q1,d1,y1,1);
        y1=rf_wo_block_half(sub+2*FUSED_GROUP_BYTES,q2,d2,y1,1);
        y1=rf_wo_block_half(sub+3*FUSED_GROUP_BYTES,q3,d3,y1,1);
        y1=rf_wo_block_half(sub+4*FUSED_GROUP_BYTES,q4,d4,y1,1);
        y1=rf_wo_block_half(sub+5*FUSED_GROUP_BYTES,q5,d5,y1,1);
        y1=rf_wo_block_half(sub+6*FUSED_GROUP_BYTES,q6,d6,y1,1);
        y1=rf_wo_block_half(sub+7*FUSED_GROUP_BYTES,q7,d7,y1,1);
        __riscv_vse32_v_f32m4(y+(size_t)yt*256+128,y1,128);
    }
}

/* One inlined fused region: projection -> RoPE -> attention -> quantization
 * -> all partial-W_O contributions. No timers or calls in vector lifetime.
 * Q and O never have an addressable FP32 scratch array. */
static __attribute__((noinline, riscv_vector_cc)) void rf_headpair_core(const uint8_t *qtile,const uint8_t *wo,
                                const block_q8_0 *xq8,const float *q_bias,
                                const uint16_t *kc,const uint16_t *vc,
                                int context,int group,
                                const float *cos_table,const float *sin_table,
                                float *y, float *attn_capture,
                                block_q8_0 *q8_capture) {
    vfloat32m2_t q00,q01,q10,q11;
    rf_project256(qtile,xq8,HIDDEN/QK,q_bias,q_bias?q_bias+128:NULL,&q00,&q01,&q10,&q11);
    vfloat32m2_t cosv=__riscv_vle32_v_f32m2(cos_table,64);
    vfloat32m2_t sinv=__riscv_vle32_v_f32m2(sin_table,64);
    rf_rope128(&q00,&q01,cosv,sinv);
    rf_rope128(&q10,&q11,cosv,sinv);
    vfloat32m2_t u00,u01,u10,u11;
    rf_attention2(q00,q01,q10,q11,kc,vc,context,group,&u00,&u01,&u10,&u11);
    /* Debug-only output capture: only called in an untimed verification replay. */
    if (attn_capture) {
        __riscv_vse32_v_f32m2(attn_capture+0,u00,64);
        __riscv_vse32_v_f32m2(attn_capture+64,u01,64);
        __riscv_vse32_v_f32m2(attn_capture+128,u10,64);
        __riscv_vse32_v_f32m2(attn_capture+192,u11,64);
    }
    uint16_t d0,d1,d2,d3,d4,d5,d6,d7;
    vint8mf2_t q0=rf_quant32(u00, 0,&d0),q1=rf_quant32(u00,32,&d1);
    vint8mf2_t q2=rf_quant32(u01, 0,&d2),q3=rf_quant32(u01,32,&d3);
    vint8mf2_t q4=rf_quant32(u10, 0,&d4),q5=rf_quant32(u10,32,&d5);
    vint8mf2_t q6=rf_quant32(u11, 0,&d6),q7=rf_quant32(u11,32,&d7);
    if (q8_capture) {
        q8_capture[0].d=d0; __riscv_vse8_v_i8mf2(q8_capture[0].qs,q0,32);
        q8_capture[1].d=d1; __riscv_vse8_v_i8mf2(q8_capture[1].qs,q1,32);
        q8_capture[2].d=d2; __riscv_vse8_v_i8mf2(q8_capture[2].qs,q2,32);
        q8_capture[3].d=d3; __riscv_vse8_v_i8mf2(q8_capture[3].qs,q3,32);
        q8_capture[4].d=d4; __riscv_vse8_v_i8mf2(q8_capture[4].qs,q4,32);
        q8_capture[5].d=d5; __riscv_vse8_v_i8mf2(q8_capture[5].qs,q5,32);
        q8_capture[6].d=d6; __riscv_vse8_v_i8mf2(q8_capture[6].qs,q6,32);
        q8_capture[7].d=d7; __riscv_vse8_v_i8mf2(q8_capture[7].qs,q7,32);
    }
    rf_wo_pair(wo,y,q0,q1,q2,q3,q4,q5,q6,q7,d0,d1,d2,d3,d4,d5,d6,d7);
}

#endif /* HAVE_RVV */

/* Drop-in entry point; on x86 test hosts legacy fallback is for source-level
 * integration tests only. Register residency is exclusively the RVV path. */
static fused_stats_t run_layer_fused(const layer_t *l,
                                              const fused_weights_t *fw,
                                              const float *x,
                                              uint16_t *kc,uint16_t *vc,
                                              int context,float *output,
                                              float *attn_capture,
                                              block_q8_0 *q8_capture) {
#if HAVE_RVV
    if (fw->tcm_tile)
        die("Register fusion requires FUSED_TCM=0 (TCM memcpy would clobber live vectors)");
    float normed[HIDDEN];
    block_q8_0 inq8[HIDDEN/QK];
    fused_stats_t stats={0};
    double t=seconds_now();
    rmsnorm(x,l->norm,normed);
    stats.phase.rms=(float)(seconds_now()-t);
    t=seconds_now();
    quantize_q8(normed,inq8,HIDDEN);
    stats.phase.q8_input=(float)(seconds_now()-t);
    /* Trig coefficient preparation precedes all register-resident vectors. */
    float cv[64],sv[64];
    t=seconds_now();
    for(int d=0;d<64;++d){
        const float theta=(float)(context-1)/powf(ROPE_BASE,(float)(2*d)/HDIM);
        cv[d]=cosf(theta);sv[d]=sinf(theta);
    }
    stats.phase.rope_cache+=(float)(seconds_now()-t);
    memset(output,0,HIDDEN*sizeof(float));
    for(int group=0;group<KVHEADS;++group){
        const uint8_t *kvtile=fw->kv+(size_t)group*FUSED_TILE_BYTES;
        vfloat32m2_t k0,k1,v0,v1;
        /* One continuous K/V projection + RoPE + FP16 cache commit.
         * NO time query between K/V projection and cache store. */
        t=seconds_now();
        vfloat32m2_t cosv=__riscv_vle32_v_f32m2(cv,64);
        vfloat32m2_t sinv=__riscv_vle32_v_f32m2(sv,64);
        rf_project256(kvtile,inq8,HIDDEN/QK,
                      l->bk?l->bk+group*HDIM:NULL,
                      l->bv?l->bv+group*HDIM:NULL,
                      &k0,&k1,&v0,&v1);
        rf_rope128(&k0,&k1,cosv,sinv);
        const size_t off=(size_t)(context-1)*KVDIM+(size_t)group*HDIM;
        rf_f16_store64(kc+off,k0);
        rf_f16_store64(kc+off+64,k1);
        rf_f16_store64(vc+off,v0);
        rf_f16_store64(vc+off+64,v1);
        /* Accounting convention: this timing includes K/V projection, K
         * RoPE, and persistent cache commit, all without a timer inside. */
        stats.phase.rope_cache+=(float)(seconds_now()-t);
        for(int p=0;p<3;++p){
            int pair=group*3+p;
            const uint8_t *qtile=fw->q+(size_t)pair*FUSED_TILE_BYTES;
            const uint8_t *wotile=fw->o+(size_t)pair*FUSED_TILE_BYTES;
            const float *qb=l->bq?l->bq+(size_t)pair*256:NULL;
            t=seconds_now();
            rf_headpair_core(qtile,wotile,inq8,qb,kc,vc,context,group,cv,sv,
                             output,attn_capture?attn_capture+(size_t)pair*256:NULL,
                             q8_capture?q8_capture+(size_t)pair*8:NULL);
            /* Single timing boundary: timer calls inside would force spills.
             * This phase INCLUDES Q, RoPE, attention, Q8 and W_O. */
            stats.phase.attention+=(float)(seconds_now()-t);
        }
    }
    return stats;
#else
    return run_layer_legacy_fused(l,fw,x,kc,vc,context,output,attn_capture,q8_capture);
#endif
}
