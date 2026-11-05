#pragma once
/*
 * Three 30-KiB independent TCM banks: Q, interleaved K/V, and W_O.
 * Each bank holds SIX contiguous 5,120-byte SoA quantization groups:
 * 256 output neurons x 192 input values (Q4 and their FP32 scales).
 * The rest of each 240-KiB projection tile MUST be processed too.
 *
 * STREAM: copy every segment into its bank before use, including 3.28 MiB
 *         of on-token weight copies; nothing survives the next token.
 * PIN:    keep first 30 KiB of one Q pair, one K/V group and one W_O pair
 *         stationary across invocations. Other groups come from DDR.
 * DDR:    exact same segmented kernels with no TCM traffic.
 *
 * KV CACHE IS ALWAYS FP16 DDR/L2; NEVER STAGED TO TCM.
 * This is a standalone experiment; the existing fused kernel is unchanged.
 */
#if HAVE_RVV
#define RT_BLOCKS 6
#define RT_BYTES (RT_BLOCKS * FUSED_GROUP_BYTES)
#define RT_CAPACITY (3u * RT_BYTES)
_Static_assert(RT_BYTES == 30720u, "unexpected SoA microtile geometry");
_Static_assert(RT_CAPACITY == 92160u, "unexpected three-bank geometry");

enum { RT_DDR=0, RT_STREAM=1, RT_PIN=2, RT_AIDMA=3, RT_AIDMA_PIN=4 };
enum { RT_KIND_Q=0, RT_KIND_KV=1, RT_KIND_WO=2 };
typedef struct {
    int mode, pinned_pair;
    uint8_t *q_bank, *kv_bank, *wo_bank;
#if defined(RT_ENABLE_AIDMA)
    struct rt_ai_pipeline *ai;
#endif
} rt_workspace_t;

#if defined(RT_ENABLE_AIDMA)
/* The implementation is in ai_dma_pipeline.h, included by the opt-in bench. */
static const uint8_t *rt_ai_take(rt_workspace_t *rt,int kind,int index,
                                 int yt,int start,const uint8_t *src,int n);
#endif

/* Inline scalar-only volatile stores keep weights in the scratchpad without
 * a C call that could clobber Q/O registers. Inspect GCC output on A100. */
RF_INLINE void rt_copy_tcm(uint8_t *dest,const uint8_t *src,size_t n) {
    typedef uint64_t rt_alias_u64 __attribute__((may_alias));
    volatile rt_alias_u64 *d=(volatile rt_alias_u64 *)(void *)dest;
    const rt_alias_u64 *s=(const rt_alias_u64 *)(const void *)src;
    for(size_t j=0;j<n/8;++j) d[j]=s[j];
}

static void rt_init(rt_workspace_t *rt,const fused_weights_t *fw,
                    int mode,int pair) {
    if(mode<RT_DDR || pair<0 || pair>=QHEADS/2)
        die("Invalid TCM 3-bank configuration");
#if defined(RT_ENABLE_AIDMA)
    if(mode>RT_AIDMA_PIN) die("Invalid TCM 3-bank mode");
    rt->ai=NULL;
#else
    if(mode>RT_PIN) die("AI-DMA mode requires the dedicated benchmark");
#endif
    if(!fw->tcm_tile) die("TCM 3-bank benchmark requires mapped TCM");
    /* The existing driver already requires >= 240KiB, enough for 90KiB. */
    rt->mode=mode;rt->pinned_pair=pair;
    rt->q_bank=fw->tcm_tile;
    rt->kv_bank=fw->tcm_tile+RT_BYTES;
    rt->wo_bank=fw->tcm_tile+2u*RT_BYTES;
    if(mode==RT_PIN) {
        /* One-time preload; not measured in per-token wall-clock latency. */
        rt_copy_tcm(rt->q_bank,fw->q+(size_t)pair*FUSED_TILE_BYTES,RT_BYTES);
        rt_copy_tcm(rt->kv_bank,fw->kv+(size_t)(pair/3)*FUSED_TILE_BYTES,RT_BYTES);
        rt_copy_tcm(rt->wo_bank,fw->o+(size_t)pair*FUSED_TILE_BYTES,RT_BYTES);
    }
}

RF_INLINE const uint8_t *rt_take(rt_workspace_t *rt,int kind,int index,
                                  int yt,int start,const uint8_t *src,int n) {
    if(rt->mode==RT_DDR) return src;
#if defined(RT_ENABLE_AIDMA)
    if(rt->mode==RT_AIDMA || rt->mode==RT_AIDMA_PIN)
        return rt_ai_take(rt,kind,index,yt,start,src,n);
#endif
    uint8_t *bank = kind==RT_KIND_Q ? rt->q_bank :
                    kind==RT_KIND_KV ? rt->kv_bank : rt->wo_bank;
    if(rt->mode==RT_PIN) {
        int resident= start==0 &&
            (kind==RT_KIND_Q ? index==rt->pinned_pair :
             kind==RT_KIND_KV ? index==rt->pinned_pair/3 :
             index==rt->pinned_pair && yt==0);
        return resident ? (const uint8_t *)bank : src;
    }
    /* STREAM mode: all re-staging is charged in outer wall clock. */
    rt_copy_tcm(bank,src,(size_t)n*FUSED_GROUP_BYTES);
    return bank;
}

/* Same FP32 math as rf_project256, with the initial accumulators supplied by
 * the caller instead of reset at each TCM segment. */
RF_INLINE void rt_project_add(const uint8_t *tile, const block_q8_0 *x,
                              int nblocks,
                              vfloat32m2_t *o0, vfloat32m2_t *o1,
                              vfloat32m2_t *o2, vfloat32m2_t *o3) {
    const size_t vl = 64;
    vfloat32m2_t a0=*o0,a1=*o1,a2=*o2,a3=*o3;
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

/* Iterate over 8 contiguous 30-KiB chunks for a 1,536-input projection.
 * Never round or store the accumulating 256 outputs at a chunk boundary. */
RF_INLINE void rt_project256(rt_workspace_t *rt,int kind,int index,
                              const uint8_t *tile,const block_q8_0 *x,int nblocks,
                              const float *blo,const float *bhi,
                              vfloat32m2_t *o0,vfloat32m2_t *o1,
                              vfloat32m2_t *o2,vfloat32m2_t *o3) {
    const size_t vl=64;
    vfloat32m2_t a0=blo?__riscv_vle32_v_f32m2(blo,vl):__riscv_vfmv_v_f_f32m2(0,vl);
    vfloat32m2_t a1=blo?__riscv_vle32_v_f32m2(blo+64,vl):__riscv_vfmv_v_f_f32m2(0,vl);
    vfloat32m2_t a2=bhi?__riscv_vle32_v_f32m2(bhi,vl):__riscv_vfmv_v_f_f32m2(0,vl);
    vfloat32m2_t a3=bhi?__riscv_vle32_v_f32m2(bhi+64,vl):__riscv_vfmv_v_f_f32m2(0,vl);
    for(int b=0;b<nblocks;b+=RT_BLOCKS) {
        int n=nblocks-b < RT_BLOCKS ? nblocks-b : RT_BLOCKS;
        const uint8_t *ptr=rt_take(rt,kind,index,-1,b,
                                    tile+(size_t)b*FUSED_GROUP_BYTES,n);
        rt_project_add(ptr,x+b,n,&a0,&a1,&a2,&a3);
    }
    *o0=a0;*o1=a1;*o2=a2;*o3=a3;
}

/* Same 8 original rf_wo_block_half microkernels for each of the 6 Y tiles.
 * Copy at most six adjacent weight groups into W_O's 30-KiB bank at once.
 * Keep all eight Q8 vectors live through this function; no Q8 array.
 */
static __attribute__((noinline, riscv_vector_cc))
void rt_wo_pair(rt_workspace_t *rt,int pair,const uint8_t *wo,float *y,
                   vint8mf2_t q0,vint8mf2_t q1,vint8mf2_t q2,vint8mf2_t q3,
                   vint8mf2_t q4,vint8mf2_t q5,vint8mf2_t q6,vint8mf2_t q7,
                   uint16_t d0,uint16_t d1,uint16_t d2,uint16_t d3,
                   uint16_t d4,uint16_t d5,uint16_t d6,uint16_t d7) {
    for(int yt=0;yt<6;++yt) {
        const uint8_t *sub=wo+(size_t)yt*FUSED_WO_SUBTILE;
        vfloat32m4_t y0=__riscv_vle32_v_f32m4(y+(size_t)yt*256,128);
        vfloat32m4_t y1=__riscv_vle32_v_f32m4(y+(size_t)yt*256+128,128);
        const uint8_t *p=rt_take(rt,RT_KIND_WO,pair,yt,0,sub,6);
        y0=rf_wo_block_half(p+0*FUSED_GROUP_BYTES,q0,d0,y0,0);
        y1=rf_wo_block_half(p+0*FUSED_GROUP_BYTES,q0,d0,y1,1);
        y0=rf_wo_block_half(p+1*FUSED_GROUP_BYTES,q1,d1,y0,0);
        y1=rf_wo_block_half(p+1*FUSED_GROUP_BYTES,q1,d1,y1,1);
        y0=rf_wo_block_half(p+2*FUSED_GROUP_BYTES,q2,d2,y0,0);
        y1=rf_wo_block_half(p+2*FUSED_GROUP_BYTES,q2,d2,y1,1);
        y0=rf_wo_block_half(p+3*FUSED_GROUP_BYTES,q3,d3,y0,0);
        y1=rf_wo_block_half(p+3*FUSED_GROUP_BYTES,q3,d3,y1,1);
        y0=rf_wo_block_half(p+4*FUSED_GROUP_BYTES,q4,d4,y0,0);
        y1=rf_wo_block_half(p+4*FUSED_GROUP_BYTES,q4,d4,y1,1);
        y0=rf_wo_block_half(p+5*FUSED_GROUP_BYTES,q5,d5,y0,0);
        y1=rf_wo_block_half(p+5*FUSED_GROUP_BYTES,q5,d5,y1,1);
        p=rt_take(rt,RT_KIND_WO,pair,yt,6,sub+6*FUSED_GROUP_BYTES,2);
        y0=rf_wo_block_half(p+0*FUSED_GROUP_BYTES,q6,d6,y0,0);
        y1=rf_wo_block_half(p+0*FUSED_GROUP_BYTES,q6,d6,y1,1);
        y0=rf_wo_block_half(p+1*FUSED_GROUP_BYTES,q7,d7,y0,0);
        y1=rf_wo_block_half(p+1*FUSED_GROUP_BYTES,q7,d7,y1,1);
        __riscv_vse32_v_f32m4(y+(size_t)yt*256,y0,128);
        __riscv_vse32_v_f32m4(y+(size_t)yt*256+128,y1,128);
    }
}

/* Same fused head-pair operations as rf_headpair_core; only weight addresses
 * are selected through the three banks. KV history is always normal memory. */
static __attribute__((noinline, riscv_vector_cc)) void rt_headpair_core(rt_workspace_t *rt,int pair,const uint8_t *qtile,const uint8_t *wo,
                                const block_q8_0 *xq8,const float *q_bias,
                                const uint16_t *kc,const uint16_t *vc,
                                int context,int group,
                                const float *cos_table,const float *sin_table,
                                float *y, float *attn_capture,
                                block_q8_0 *q8_capture) {
    vfloat32m2_t q00,q01,q10,q11;
    rt_project256(rt,RT_KIND_Q,pair,qtile,xq8,HIDDEN/QK,q_bias,q_bias?q_bias+128:NULL,&q00,&q01,&q10,&q11);
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
    rt_wo_pair(rt,pair,wo,y,q0,q1,q2,q3,q4,q5,q6,q7,d0,d1,d2,d3,d4,d5,d6,d7);
}


/* Full one-token path. All TCM copies are inside the outer benchmark clock;
 * no seconds_now calls while register-resident data is live. */
static fused_stats_t rt_run_layer(const layer_t *l,const fused_weights_t *fw,
                                  rt_workspace_t *rt,const float *x,
                                  uint16_t *kc,uint16_t *vc,int context,float *output,
                                  float *attn_capture,block_q8_0 *q8_capture) {
    float normed[HIDDEN];
    block_q8_0 inq8[HIDDEN/QK];
    fused_stats_t stats={0};
    double t=seconds_now();
    rmsnorm(x,l->norm,normed);
    stats.phase.rms=(float)(seconds_now()-t);
    t=seconds_now();
    quantize_q8(normed,inq8,HIDDEN);
    stats.phase.q8_input=(float)(seconds_now()-t);
    float cv[64],sv[64];
    t=seconds_now();
    for(int d=0;d<64;++d){
        const float theta=(float)(context-1)/powf(ROPE_BASE,(float)(2*d)/HDIM);
        cv[d]=cosf(theta);sv[d]=sinf(theta);
    }
    stats.phase.rope_cache+=(float)(seconds_now()-t);
    memset(output,0,HIDDEN*sizeof(float));
    for(int group=0;group<KVHEADS;++group) {
        const uint8_t *tile=fw->kv+(size_t)group*FUSED_TILE_BYTES;
        vfloat32m2_t k0,k1,v0,v1;
        t=seconds_now();
        vfloat32m2_t cosv=__riscv_vle32_v_f32m2(cv,64);
        vfloat32m2_t sinv=__riscv_vle32_v_f32m2(sv,64);
        rt_project256(rt,RT_KIND_KV,group,tile,inq8,HIDDEN/QK,
                      l->bk?l->bk+(size_t)group*HDIM:NULL,
                      l->bv?l->bv+(size_t)group*HDIM:NULL,&k0,&k1,&v0,&v1);
        rf_rope128(&k0,&k1,cosv,sinv);
        size_t off=(size_t)(context-1)*KVDIM+(size_t)group*HDIM;
        rf_f16_store64(kc+off,k0);
        rf_f16_store64(kc+off+64,k1);
        rf_f16_store64(vc+off,v0);
        rf_f16_store64(vc+off+64,v1);
        stats.phase.rope_cache+=(float)(seconds_now()-t);
        for(int p=0;p<3;++p) {
            int pair=group*3+p;
            t=seconds_now();
            rt_headpair_core(rt,pair,
                 fw->q+(size_t)pair*FUSED_TILE_BYTES,
                 fw->o+(size_t)pair*FUSED_TILE_BYTES,
                 inq8,l->bq?l->bq+(size_t)pair*256:NULL,kc,vc,context,group,
                 cv,sv,output,
                 attn_capture?attn_capture+(size_t)pair*256:NULL,
                 q8_capture?q8_capture+(size_t)pair*8:NULL);
            stats.phase.attention+=(float)(seconds_now()-t);
        }
    }
    return stats;
}
#endif /* HAVE_RVV */
