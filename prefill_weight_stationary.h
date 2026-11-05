#pragma once
/*
 * Weight-stationary, CAUSAL single-layer PREFILL for the K3 A100.
 * Input x[t] is the already-available activation of token t at THIS layer.
 * Unlike decode, all batch inputs are known before this function begins.
 *
 * Every 30-KiB weight chunk is transferred ONCE per batch, then applied to
 * ALL input tokens before the two-slot TCM lane advances to the next chunk.
 * Next chunk's AI-DMA runs while the current chunk is computing. The three
 * Q/KV/W_O lanes have two 30-KiB slots each = 180 KiB TCM total, arranged
 * as alternating 90-KiB aggregate buffers. No memcpy() weights per token.
 *
 * The order is KV projection (all tokens) -> Q projection + masked causal
 * attention (all tokens, for each head pair) -> WO projection (all tokens).
 * Full-batch FP32 projection accumulators and Q8 attention activations are
 * stored in DDR; cross-token weight reuse CANNOT preserve live RVV registers
 * across all tile switches. KV history remains FP16 in DDR/L2.
 *
 * Include after qwen_one_layer.c, tcm_three_banks.h and ai_dma_pipeline.h;
 * requires RT_ENABLE_AIDMA and HAVE_RVV. All old decode code is untouched.
 */
#if HAVE_RVV && defined(RT_ENABLE_AIDMA)

#define PF_PROJ_GROUPS 48
#define PF_WO_GROUPS 8
#define PF_TCM_REQUIRED RA_STREAM_BYTES
_Static_assert(PF_TCM_REQUIRED==184320u, "Prefill TCM capacity mismatch");

typedef struct {
    int kind, index, yt, start, n;
} pf_tile_t;

static pf_tile_t pf_first(int kind) {
    pf_tile_t z={kind,0,0,0,RT_BLOCKS};
    return z;
}

/* Return the next tile of the same projection family, or n=0 at its end.
 * W_O uses 6+2 groups per 40-KiB output subtile (six Y subtiles per pair).
 */
static pf_tile_t pf_next(pf_tile_t x) {
    if(x.kind==RT_KIND_WO) {
        if(x.start==0) {x.start=6; x.n=2; return x;}
        x.start=0; x.n=6;
        if(++x.yt==6) {x.yt=0; ++x.index;}
        if(x.index>=QHEADS/2) x.n=0;
        return x;
    }
    x.start+=RT_BLOCKS;
    if(x.start==PF_PROJ_GROUPS) {
        x.start=0;
        ++x.index;
    }
    if(x.index>=(x.kind==RT_KIND_KV?KVHEADS:QHEADS/2))x.n=0;
    return x;
}

static const uint8_t *pf_source(const fused_weights_t *fw,pf_tile_t x) {
    const uint8_t *weights=x.kind==RT_KIND_Q?fw->q:
                           x.kind==RT_KIND_KV?fw->kv:fw->o;
    size_t group=(size_t)(x.kind==RT_KIND_WO?x.yt*PF_WO_GROUPS+x.start:x.start);
    return weights+(size_t)x.index*FUSED_TILE_BYTES+group*FUSED_GROUP_BYTES;
}

/* The *only* weight-transfer operation used during prefill. The /dev/udma
 * memcpy ioctl and rt_copy_tcm() are never invoked by this prefill path. */
static void pf_submit(rt_ai_pipeline_t *a,const fused_weights_t *fw,
                      pf_tile_t x,int slot) {
    if(x.n<=0 || x.n>RT_BLOCKS || slot<0 || slot>1 ||
       x.kind<RT_KIND_Q || x.kind>RT_KIND_WO)
        die("Invalid prefill DMA request");
    const uint8_t *src=pf_source(fw,x);
    ra_submit(a,x.kind,src,x.index,a->lane[x.kind].slot[slot],
              (size_t)x.n*FUSED_GROUP_BYTES);
}

/* Wait for the CURRENT DMA chunk, issue NEXT into the *opposite* buffer,
 * then let the caller compute on CURRENT for every token in the batch.
 * No transfer destination overlaps current's bytes. */
static const uint8_t *pf_take(rt_ai_pipeline_t *a,const fused_weights_t *fw,
                              pf_tile_t current,pf_tile_t next,int slot) {
    ra_wait(a,current.kind);
    const uint8_t *p=a->lane[current.kind].slot[slot];
    if(a->verify && memcmp(p,pf_source(fw,current),
                           (size_t)current.n*FUSED_GROUP_BYTES)!=0)
        die("AI-DMA prefill TCM weight tile failed byte-exact verification");
    if(next.n)pf_submit(a,fw,next,slot^1);
    return p;
}

/* Accumulated projected group for one token, stored in DDR between DMA tiles.
 * The math is exactly the existing rt_project_add() (4 x 64 FP32 lanes).
 */
static void pf_project_chunk(const uint8_t *weights,
                             const block_q8_0 *qx,int n,float *acc) {
    const size_t vl=64;
    vfloat32m2_t a0=__riscv_vle32_v_f32m2(acc,vl);
    vfloat32m2_t a1=__riscv_vle32_v_f32m2(acc+64,vl);
    vfloat32m2_t a2=__riscv_vle32_v_f32m2(acc+128,vl);
    vfloat32m2_t a3=__riscv_vle32_v_f32m2(acc+192,vl);
    rt_project_add(weights,qx,n,&a0,&a1,&a2,&a3);
    __riscv_vse32_v_f32m2(acc,a0,vl);
    __riscv_vse32_v_f32m2(acc+64,a1,vl);
    __riscv_vse32_v_f32m2(acc+128,a2,vl);
    __riscv_vse32_v_f32m2(acc+192,a3,vl);
}

/* Supply the same Q/K/V bias locations as the register-fusion decode path. */
static void pf_bias(float *acc,int ntok,const float *blo,const float *bhi) {
    for(int t=0;t<ntok;++t) {
        float *dst=acc+(size_t)t*256u;
        if(blo)memcpy(dst,blo,128u*sizeof(float));
        else memset(dst,0,128u*sizeof(float));
        if(bhi)memcpy(dst+128,bhi,128u*sizeof(float));
        else memset(dst+128,0,128u*sizeof(float));
    }
}

static void pf_store_q8(block_q8_0 *dst,vint8mf2_t v,uint16_t scale) {
    dst->d=scale;
    __riscv_vse8_v_i8mf2(dst->qs,v,32);
}

/* Compute W_O on the current weight chunk for ALL batch tokens. Each input
 * block uses the same rf_wo_block_half() kernels as the decode implementation.
 * Repeated Y stores are necessary because a batch has far more accumulators
 * than can remain in the vector register file at the same time.
 */
static void pf_wo_chunk(const uint8_t *weights,int n,int pair,int yt,
                        const block_q8_0 *oq,float *output,int ntok) {
    const size_t vl=128;
    for(int t=0;t<ntok;++t) {
        const block_q8_0 *q=oq+(size_t)t*(HIDDEN/QK)+(size_t)pair*8u;
        float *y=output+(size_t)t*HIDDEN+(size_t)yt*256u;
        vfloat32m4_t y0=__riscv_vle32_v_f32m4(y,vl);
        vfloat32m4_t y1=__riscv_vle32_v_f32m4(y+128,vl);
        for(int b=0;b<n;++b) {
            const block_q8_0 *qb=q+b;
            vint8mf2_t v=__riscv_vle8_v_i8mf2(qb->qs,32);
            const uint8_t *w=weights+(size_t)b*FUSED_GROUP_BYTES;
            y0=rf_wo_block_half(w,v,qb->d,y0,0);
            y1=rf_wo_block_half(w,v,qb->d,y1,1);
        }
        __riscv_vse32_v_f32m4(y,y0,vl);
        __riscv_vse32_v_f32m4(y+128,y1,vl);
    }
}

typedef struct {
    block_q8_0 *xin;      /* ntok x 48 input Q8 blocks */
    block_q8_0 *oq;       /* ntok x 48 attention Q8 blocks */
    float *acc;           /* ntok x 256 FP32 projection accumulators */
    float *cv,*sv;        /* ntok x 64 RoPE values */
    int capacity;
} pf_workspace_t;

static void pf_workspace_open(pf_workspace_t *p,int ntok) {
    memset(p,0,sizeof(*p));
    p->capacity=ntok;
    p->xin=malloc((size_t)ntok*(HIDDEN/QK)*sizeof(block_q8_0));
    p->oq=malloc((size_t)ntok*(HIDDEN/QK)*sizeof(block_q8_0));
    p->acc=malloc((size_t)ntok*256u*sizeof(float));
    p->cv=malloc((size_t)ntok*64u*sizeof(float));
    p->sv=malloc((size_t)ntok*64u*sizeof(float));
    if(!p->xin||!p->oq||!p->acc||!p->cv||!p->sv)
        die("Out of memory allocating prefill activation workspace");
}
static void pf_workspace_close(pf_workspace_t *p) {
    free(p->xin);free(p->oq);free(p->acc);free(p->cv);free(p->sv);
    memset(p,0,sizeof(*p));
}

/* x[ntok][HIDDEN] is an ALREADY AVAILABLE input sequence for THIS layer.
 * prefix keys/values at cache offsets [0,prefix) are already populated.
 * Appends all ntok K/V pairs at offsets [prefix,prefix+ntok).
 * Computes causal attention at token i with context=prefix+i+1: no future
 * token's keys can be used, even though all the new KVs are precomputed.
 * Returns one Y[HIDDEN] for each new token, in chronological order.
 *
 * PRECONDITION: ra_open(a,fw,RT_AIDMA,0) has succeeded and owns TCM.
 * Run in one AI-pinned CPU8 thread; no other users of this TCM/AI-DMA pair.
 */
static void pf_run_batch(const layer_t *l,const fused_weights_t *fw,
                         rt_ai_pipeline_t *a,pf_workspace_t *p,
                         const float *x,int ntok,int prefix,
                         uint16_t *kc,uint16_t *vc,float *output) {
    if(ntok<1 || ntok>p->capacity || prefix<0 || prefix>32768-ntok ||
       !fw->tcm_tile || a->pin)
        die("Invalid prefill configuration / TCM pipeline");
    const size_t vl=64;
    float normed[HIDDEN];
    for(int t=0;t<ntok;++t) {
        rmsnorm(x+(size_t)t*HIDDEN,l->norm,normed);
        quantize_q8(normed,p->xin+(size_t)t*(HIDDEN/QK),HIDDEN);
        for(int d=0;d<64;++d) {
            const float theta=(float)(prefix+t)/powf(ROPE_BASE,(float)(2*d)/HDIM);
            p->cv[(size_t)t*64u+d]=cosf(theta);
            p->sv[(size_t)t*64u+d]=sinf(theta);
        }
    }
    memset(output,0,(size_t)ntok*HIDDEN*sizeof(float));

    /* Phase 1: K/V weights stationary; all tokens traverse one tile before
     * its TCM slot is overwritten. A 30-KiB DMA runs during every sweep.
     * Both GQA groups' K/V are completed before any attention is evaluated.
     */
    pf_tile_t cur=pf_first(RT_KIND_KV);
    int slot=0;
    pf_submit(a,fw,cur,slot);
    for(int group=0;group<KVHEADS;++group) {
        const float *bk=l->bk?l->bk+(size_t)group*HDIM:NULL;
        const float *bv=l->bv?l->bv+(size_t)group*HDIM:NULL;
        pf_bias(p->acc,ntok,bk,bv);
        for(int start=0;start<PF_PROJ_GROUPS;start+=RT_BLOCKS) {
            if(cur.index!=group || cur.start!=start || cur.n!=RT_BLOCKS)
                die("AI-DMA KV prefetch sequence error");
            pf_tile_t next=pf_next(cur);
            const uint8_t *weights=pf_take(a,fw,cur,next,slot);
            for(int t=0;t<ntok;++t)
                pf_project_chunk(weights,
                    p->xin+(size_t)t*(HIDDEN/QK)+(size_t)start,
                    RT_BLOCKS,p->acc+(size_t)t*256u);
            cur=next;slot^=1;
        }
        for(int t=0;t<ntok;++t) {
            float *f=p->acc+(size_t)t*256u;
            vfloat32m2_t k0=__riscv_vle32_v_f32m2(f,vl);
            vfloat32m2_t k1=__riscv_vle32_v_f32m2(f+64,vl);
            vfloat32m2_t c=__riscv_vle32_v_f32m2(p->cv+(size_t)t*64u,vl);
            vfloat32m2_t s=__riscv_vle32_v_f32m2(p->sv+(size_t)t*64u,vl);
            rf_rope128(&k0,&k1,c,s);
            size_t pos=((size_t)prefix+(size_t)t)*KVDIM+(size_t)group*HDIM;
            rf_f16_store64(kc+pos,k0);
            rf_f16_store64(kc+pos+64,k1);
            rf_f16_store64(vc+pos,__riscv_vle32_v_f32m2(f+128,vl));
            rf_f16_store64(vc+pos+64,__riscv_vle32_v_f32m2(f+192,vl));
        }
    }
    if(cur.n!=0)die("AI-DMA KV did not cover all groups");

    /* Phase 2: Q weights stationary; for each pair all tokens finish Q,
     * then compute CAUSAL attention against the completed FP16 KV cache.
     * Compact 48-block-per-token attention Q8 results are retained for W_O.
     */
    cur=pf_first(RT_KIND_Q);slot=0;
    pf_submit(a,fw,cur,slot);
    for(int pair=0;pair<QHEADS/2;++pair) {
        const float *bias=l->bq?l->bq+(size_t)pair*256u:NULL;
        pf_bias(p->acc,ntok,bias,bias?bias+128:NULL);
        for(int start=0;start<PF_PROJ_GROUPS;start+=RT_BLOCKS) {
            if(cur.index!=pair || cur.start!=start || cur.n!=RT_BLOCKS)
                die("AI-DMA Q prefetch sequence error");
            pf_tile_t next=pf_next(cur);
            const uint8_t *weights=pf_take(a,fw,cur,next,slot);
            for(int t=0;t<ntok;++t)
                pf_project_chunk(weights,
                    p->xin+(size_t)t*(HIDDEN/QK)+(size_t)start,
                    RT_BLOCKS,p->acc+(size_t)t*256u);
            cur=next;slot^=1;
        }
        for(int t=0;t<ntok;++t) {
            const float *f=p->acc+(size_t)t*256u;
            vfloat32m2_t q00=__riscv_vle32_v_f32m2(f,vl);
            vfloat32m2_t q01=__riscv_vle32_v_f32m2(f+64,vl);
            vfloat32m2_t q10=__riscv_vle32_v_f32m2(f+128,vl);
            vfloat32m2_t q11=__riscv_vle32_v_f32m2(f+192,vl);
            vfloat32m2_t c=__riscv_vle32_v_f32m2(p->cv+(size_t)t*64u,vl);
            vfloat32m2_t s=__riscv_vle32_v_f32m2(p->sv+(size_t)t*64u,vl);
            rf_rope128(&q00,&q01,c,s);
            rf_rope128(&q10,&q11,c,s);
            vfloat32m2_t u00,u01,u10,u11;
            rf_attention2(q00,q01,q10,q11,kc,vc,prefix+t+1,pair/3,
                          &u00,&u01,&u10,&u11);
            block_q8_0 *q=p->oq+(size_t)t*(HIDDEN/QK)+(size_t)pair*8u;
            uint16_t scale;
            vint8mf2_t v;
            v=rf_quant32(u00,0,&scale);pf_store_q8(q+0,v,scale);
            v=rf_quant32(u00,32,&scale);pf_store_q8(q+1,v,scale);
            v=rf_quant32(u01,0,&scale);pf_store_q8(q+2,v,scale);
            v=rf_quant32(u01,32,&scale);pf_store_q8(q+3,v,scale);
            v=rf_quant32(u10,0,&scale);pf_store_q8(q+4,v,scale);
            v=rf_quant32(u10,32,&scale);pf_store_q8(q+5,v,scale);
            v=rf_quant32(u11,0,&scale);pf_store_q8(q+6,v,scale);
            v=rf_quant32(u11,32,&scale);pf_store_q8(q+7,v,scale);
        }
    }
    if(cur.n!=0)die("AI-DMA Q did not cover all groups");

    /* Phase 3: W_O weights stationary; each 30/10-KiB chunk contributes
     * to all tokens before the next chunk replaces its TCM buffer.
     * Keep the exact original q0..q7 ordering within each 40-KiB subtile.
     */
    cur=pf_first(RT_KIND_WO);slot=0;
    pf_submit(a,fw,cur,slot);
    for(int pair=0;pair<QHEADS/2;++pair) {
        for(int yt=0;yt<6;++yt) {
            for(int start=0;start<PF_WO_GROUPS;start+=RT_BLOCKS) {
                int n=(start==0)?6:2;
                if(cur.index!=pair || cur.yt!=yt || cur.start!=start || cur.n!=n)
                    die("AI-DMA W_O prefetch sequence error");
                pf_tile_t next=pf_next(cur);
                const uint8_t *weights=pf_take(a,fw,cur,next,slot);
                /* q[0..5], then q[6..7] -- exact FP32 sum order. */
                pf_wo_chunk(weights,n,pair,yt,
                    p->oq+(size_t)start,output,ntok);
                cur=next;slot^=1;
            }
        }
    }
    if(cur.n!=0)die("AI-DMA W_O did not cover all groups");
}
#endif /* HAVE_RVV && RT_ENABLE_AIDMA */
