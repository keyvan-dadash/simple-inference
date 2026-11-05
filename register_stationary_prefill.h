#pragma once
/*
 * One-pair-at-a-time, register-resident CAUSAL PREFILL.
 * All input token activations are already available (one layer, not decode).
 *
 * TCM block 0 on CPU8 is 384 KiB. A complete two-head Q or KV projection
 * occupies 240 KiB; a complete two-head W_O occupies ANOTHER 240 KiB.
 * 480 KiB cannot fit in one 384 KiB TCM block. This truthful single-block
 * experiment uses 240 KiB resident Q, and 120 KiB resident W_O, while the
 * remaining 120 KiB of W_O is read directly from DDR/L2. No temporary Q,
 * attention O, or attention Q8 array exists. The only projected intermediate
 * explicitly stored is the MANDATORY FP16 K/V cache; Y is accumulated in DDR.
 *
 * K/V is computed first for every prompt token, using a 240-KiB resident
 * K/V tile, before replacing that region with Q. Each query head pair then
 * computes token0: Q -> RoPE -> online attention -> Q8 -> W_O -> partial Y,
 * token1: ... through token(N-1), before its weights are replaced.
 *
 * Coherent source mirroring occurs ONCE in ra_open() outside any benchmark.
 * Per batch: 88 hardware DMA requests, transferring 2640 KiB into TCM.
 * The previous 30KiB weight-stationary prefill and decode are untouched.
 */
#if HAVE_RVV && defined(RT_ENABLE_AIDMA)
#define RP_WO_TCM_SUBTILES 3
#define RP_RESIDENT_Q_BYTES FUSED_TILE_BYTES
#define RP_RESIDENT_WO_BYTES (RP_WO_TCM_SUBTILES * FUSED_WO_SUBTILE)
#define RP_RESIDENT_BYTES (RP_RESIDENT_Q_BYTES + RP_RESIDENT_WO_BYTES)
_Static_assert(RP_RESIDENT_Q_BYTES == 245760u, "Q pair size mismatch");
_Static_assert(RP_RESIDENT_WO_BYTES == 122880u, "W_O subtiles mismatch");
_Static_assert(RP_RESIDENT_BYTES == 368640u, "TCM footprint mismatch");
_Static_assert(RP_RESIDENT_BYTES < 384u * 1024u, "TCM capacity exceeded");

/* Exactly the original eight rf_wo_block_half accumulations, in the same
 * order. They consume Q8 vector arguments, with no FP32 O scratch arrays. */
#define RP_WO_ACCUM_8(sum, sub, half) do { \
    (sum)=rf_wo_block_half((sub)+0u*FUSED_GROUP_BYTES,q0,d0,(sum),(half)); \
    (sum)=rf_wo_block_half((sub)+1u*FUSED_GROUP_BYTES,q1,d1,(sum),(half)); \
    (sum)=rf_wo_block_half((sub)+2u*FUSED_GROUP_BYTES,q2,d2,(sum),(half)); \
    (sum)=rf_wo_block_half((sub)+3u*FUSED_GROUP_BYTES,q3,d3,(sum),(half)); \
    (sum)=rf_wo_block_half((sub)+4u*FUSED_GROUP_BYTES,q4,d4,(sum),(half)); \
    (sum)=rf_wo_block_half((sub)+5u*FUSED_GROUP_BYTES,q5,d5,(sum),(half)); \
    (sum)=rf_wo_block_half((sub)+6u*FUSED_GROUP_BYTES,q6,d6,(sum),(half)); \
    (sum)=rf_wo_block_half((sub)+7u*FUSED_GROUP_BYTES,q7,d7,(sum),(half)); \
} while(0)

static __attribute__((noinline, riscv_vector_cc)) void
rp_wo_pair_hybrid(const uint8_t *wo_tcm_prefix, const uint8_t *wo_ddr,
                   float *y,
                   vint8mf2_t q0,vint8mf2_t q1,vint8mf2_t q2,vint8mf2_t q3,
                   vint8mf2_t q4,vint8mf2_t q5,vint8mf2_t q6,vint8mf2_t q7,
                   uint16_t d0,uint16_t d1,uint16_t d2,uint16_t d3,
                   uint16_t d4,uint16_t d5,uint16_t d6,uint16_t d7) {
    for(int yt=0;yt<6;++yt) {
        const uint8_t *sub = yt<RP_WO_TCM_SUBTILES
            ? wo_tcm_prefix+(size_t)yt*FUSED_WO_SUBTILE
            : wo_ddr+(size_t)yt*FUSED_WO_SUBTILE;
        float *yrow=y+(size_t)yt*256u;
        vfloat32m4_t y0=__riscv_vle32_v_f32m4(yrow,128);
        RP_WO_ACCUM_8(y0,sub,0);
        __riscv_vse32_v_f32m4(yrow,y0,128);
        vfloat32m4_t y1=__riscv_vle32_v_f32m4(yrow+128,128);
        RP_WO_ACCUM_8(y1,sub,1);
        __riscv_vse32_v_f32m4(yrow+128,y1,128);
    }
}
#undef RP_WO_ACCUM_8

/* A single register-fused region per token/head-pair. The Q and attention
 * vectors never have addressable memory buffers, and no timing/DMA ioctl
 * appears in their lifetime. This is the exact existing rf_* math. */
static __attribute__((noinline, riscv_vector_cc)) void
rp_headpair_core(const uint8_t *q_tcm,
                 const uint8_t *wo_tcm_prefix,const uint8_t *wo_ddr,
                 const block_q8_0 *xq8,const float *qbias,
                 const uint16_t *kc,const uint16_t *vc,
                 int context,int group,
                 const float *cos_table,const float *sin_table,
                 float *y) {
    vfloat32m2_t q00,q01,q10,q11;
    rf_project256(q_tcm,xq8,HIDDEN/QK,qbias,qbias?qbias+128:NULL,
                  &q00,&q01,&q10,&q11);
    vfloat32m2_t cv=__riscv_vle32_v_f32m2(cos_table,64);
    vfloat32m2_t sv=__riscv_vle32_v_f32m2(sin_table,64);
    rf_rope128(&q00,&q01,cv,sv);
    rf_rope128(&q10,&q11,cv,sv);
    vfloat32m2_t u00,u01,u10,u11;
    rf_attention2(q00,q01,q10,q11,kc,vc,context,group,
                  &u00,&u01,&u10,&u11);
    uint16_t d0,d1,d2,d3,d4,d5,d6,d7;
    vint8mf2_t q0=rf_quant32(u00,0,&d0),q1=rf_quant32(u00,32,&d1);
    vint8mf2_t q2=rf_quant32(u01,0,&d2),q3=rf_quant32(u01,32,&d3);
    vint8mf2_t q4=rf_quant32(u10,0,&d4),q5=rf_quant32(u10,32,&d5);
    vint8mf2_t q6=rf_quant32(u11,0,&d6),q7=rf_quant32(u11,32,&d7);
    rp_wo_pair_hybrid(wo_tcm_prefix,wo_ddr,y,
                      q0,q1,q2,q3,q4,q5,q6,q7,
                      d0,d1,d2,d3,d4,d5,d6,d7);
}

typedef struct {
    block_q8_0 *xin;    /* input activations ONLY: N x 48 blocks */
    float *cv,*sv;      /* N x 64 RoPE values */
    int capacity;
} rp_workspace_t;

static void rp_workspace_open(rp_workspace_t *p,int ntok) {
    memset(p,0,sizeof(*p));
    p->capacity=ntok;
    p->xin=malloc((size_t)ntok*(HIDDEN/QK)*sizeof(block_q8_0));
    p->cv=malloc((size_t)ntok*64u*sizeof(float));
    p->sv=malloc((size_t)ntok*64u*sizeof(float));
    if(!p->xin||!p->cv||!p->sv)die("Cannot allocate prefill input workspace");
}
static void rp_workspace_close(rp_workspace_t *p) {
    free(p->xin);free(p->cv);free(p->sv);
    memset(p,0,sizeof(*p));
}

/* Synchronous staging before each 100-token residency window. Source is a
 * single coherent 240-KiB mmap per family, populated only once at startup.
 * This uses the dedicated AI-DMA hardware ring, NEVER rt_copy_tcm(). */
static void rp_stage(rt_ai_pipeline_t *a,const fused_weights_t *fw,
                     int kind,int index,uint8_t *tcm,size_t size) {
    const uint8_t *src=(kind==RT_KIND_KV?fw->kv:
                        kind==RT_KIND_Q?fw->q:fw->o)
                        +(size_t)index*FUSED_TILE_BYTES;
    if(!size || size>FUSED_TILE_BYTES || size%RA_BANK_BYTES)
        die("Noncontiguous / unaligned register-prefill weight tile");
    for(size_t off=0;off<size;off+=RA_BANK_BYTES) {
        ra_submit(a,kind,src+off,index,tcm+off,RA_BANK_BYTES);
        ra_wait(a,kind);
    }
    if(a->verify && memcmp(tcm,src,size)!=0)
        die("AI-DMA resident weights do not match DDR source");
}

/* prefix cache [0..prefix-1] already contains FP16 keys/values.
 * Every new cache entry is produced once, before any pair attention.
 * The causal attention bound remains prefix+t+1 for token t. */
static void rp_run_batch(const layer_t *l,const fused_weights_t *fw,
                         rt_ai_pipeline_t *a,rp_workspace_t *p,
                         const float *inputs,int ntok,int prefix,
                         uint16_t *kc,uint16_t *vc,float *y,int use_tcm) {
    if(!a || a->pin || ntok<1 || ntok>p->capacity ||
       prefix<0 || prefix>32768-ntok || (use_tcm!=0 && use_tcm!=1))
        die("Invalid register-stationary prefill arguments");
    if(use_tcm) {
        fused_tcm_info_t ti={0};
        if(!fw->tcm_tile || ioctl(fw->tcm_fd,FUSED_TCM_INFO_GET,&ti)!=0 ||
           ti.block_size<RP_RESIDENT_BYTES)
            die("Local TCM has insufficient space for 240KiB Q + 120KiB W_O");
    }
    uint8_t *const resident=fw->tcm_tile;
    uint8_t *const wo_resident=use_tcm ? fw->tcm_tile+RP_RESIDENT_Q_BYTES : NULL;
    float normed[HIDDEN];
    for(int t=0;t<ntok;++t) {
        rmsnorm(inputs+(size_t)t*HIDDEN,l->norm,normed);
        quantize_q8(normed,p->xin+(size_t)t*(HIDDEN/QK),HIDDEN);
        for(int d=0;d<64;++d) {
            const float theta=(float)(prefix+t)/powf(ROPE_BASE,(float)(2*d)/HDIM);
            p->cv[(size_t)t*64u+d]=cosf(theta);
            p->sv[(size_t)t*64u+d]=sinf(theta);
        }
    }
    memset(y,0,(size_t)ntok*HIDDEN*sizeof(float));

    /* Stage full K/V group once, and process every token with no temporary
     * FP32 K/V buffer. Only the permanent FP16 cache is updated. */
    for(int group=0;group<KVHEADS;++group) {
        if(use_tcm)rp_stage(a,fw,RT_KIND_KV,group,resident,FUSED_TILE_BYTES);
        const uint8_t *kvweight=use_tcm ? resident : fw->kv+(size_t)group*FUSED_TILE_BYTES;
        for(int t=0;t<ntok;++t) {
            const size_t vl=64;
            vfloat32m2_t k0,k1,v0,v1;
            const block_q8_0 *x=p->xin+(size_t)t*(HIDDEN/QK);
            const float *bk=l->bk?l->bk+(size_t)group*HDIM:NULL;
            const float *bv=l->bv?l->bv+(size_t)group*HDIM:NULL;
            rf_project256(kvweight,x,HIDDEN/QK,bk,bv,&k0,&k1,&v0,&v1);
            vfloat32m2_t cv=__riscv_vle32_v_f32m2(p->cv+(size_t)t*64u,vl);
            vfloat32m2_t sv=__riscv_vle32_v_f32m2(p->sv+(size_t)t*64u,vl);
            rf_rope128(&k0,&k1,cv,sv);
            size_t at=((size_t)prefix+(size_t)t)*KVDIM+(size_t)group*HDIM;
            rf_f16_store64(kc+at,k0);
            rf_f16_store64(kc+at+64u,k1);
            rf_f16_store64(vc+at,v0);
            rf_f16_store64(vc+at+64u,v1);
        }
    }

    /* One head pair's weight window: load complete Q + half of W_O into
     * TCM ONCE, then run ALL tokens through the *full fused region*.
     * Nothing writes out Q, O, or attention Q8, only partial Y. */
    for(int pair=0;pair<QHEADS/2;++pair) {
        if(use_tcm) {
            rp_stage(a,fw,RT_KIND_Q,pair,resident,FUSED_TILE_BYTES);
            rp_stage(a,fw,RT_KIND_WO,pair,wo_resident,RP_RESIDENT_WO_BYTES);
        }
        const uint8_t *wo_ddr=fw->o+(size_t)pair*FUSED_TILE_BYTES;
        const uint8_t *qweight=use_tcm?resident:fw->q+(size_t)pair*FUSED_TILE_BYTES;
        const uint8_t *wo_prefix=use_tcm?wo_resident:wo_ddr;
        const float *qb=l->bq?l->bq+(size_t)pair*256u:NULL;
        for(int t=0;t<ntok;++t) {
            rp_headpair_core(qweight,wo_prefix,wo_ddr,
                             p->xin+(size_t)t*(HIDDEN/QK),qb,
                             kc,vc,prefix+t+1,pair/3,
                             p->cv+(size_t)t*64u,p->sv+(size_t)t*64u,
                             y+(size_t)t*HIDDEN);
        }
    }
}
#endif /* HAVE_RVV && RT_ENABLE_AIDMA */
