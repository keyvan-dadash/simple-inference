/*
 * Opt-in K3 A100 weight-STATIONARY PREFILL comparison. No repository decode
 * entry points are modified. Requires genuine /dev/aidma_list + MSI AI-DMA.
 *
 * Benchmark semantics: N independent *already available* layer inputs for
 * N sequential prompt tokens, causal attention (position prefix+t) with K/V
 * history in FP16 DDR. The reference calls the original fused DDR kernel for
 * these exact inputs in chronological order, maintaining the same KV history.
 *
 * WARNING: published K3 AI-DMA driver may leak coherent descriptors per
 * request (~136 requests per prefill batch). Run only a few batches.
 */
#define main qwen_main_for_ai_dma_prefill
#include "qwen_one_layer.c"
#undef main
#define RT_ENABLE_AIDMA 1
#include "tcm_three_banks.h"
#include "ai_dma_pipeline.h"
#include "prefill_weight_stationary.h"

#if HAVE_RVV

static int pf_compare_f32(const float *ref,const float *got,size_t n) {
    double max_abs=0.,max_rel=0.;
    size_t wrong=0,first=0;
    for(size_t i=0;i<n;++i) {
        double x=ref[i],y=got[i],delta=fabs(x-y),scale=fmax(fabs(x),fabs(y));
        if(!isfinite(x)||!isfinite(y)||delta>1e-5+1e-4*scale) {
            if(wrong==0)first=i;
            ++wrong;
        }
        if(delta>max_abs)max_abs=delta;
        if(scale>1e-8 && delta/scale>max_rel)max_rel=delta/scale;
    }
    printf("Y: max_abs=%.7g max_rel=%.7g mismatches=%zu/%zu\n",
           max_abs,max_rel,wrong,n);
    if(wrong)fprintf(stderr,"First mismatch Y[%zu] reference=%g prefill=%g\n",
                     first,ref[first],got[first]);
    return wrong==0;
}

static void pf_seed_history(uint16_t *k,uint16_t *v,int prefix,int total) {
    memset(k,0,(size_t)total*KVDIM*sizeof(uint16_t));
    memset(v,0,(size_t)total*KVDIM*sizeof(uint16_t));
    if(prefix>0)fill_kv_history(k,v,prefix+1); /* helper fills context-1 slots */
}

int main(int argc,char **argv) {
    if(argc<2||argc>5) {
        fprintf(stderr,"Usage: %s weights [tokens=100] [prefix=0] [repeat=1]\n",argv[0]);
        return 2;
    }
    const char *dir=argv[1];
    int ntok=argc>2?atoi(argv[2]):100;
    int prefix=argc>3?atoi(argv[3]):0;
    int repeats=argc>4?atoi(argv[4]):1;
    if(ntok<1||ntok>256||prefix<0||prefix>32768-ntok||repeats<1||repeats>3)
        die("Prefill tokens must be 1..256, prefix >=0, repeats 1..3");
    int block=getenv("TCM_BLOCK")?atoi(getenv("TCM_BLOCK")):0;
    int verify_tiles=getenv("PREFILL_VERIFY_TILES")?atoi(getenv("PREFILL_VERIFY_TILES")):0;
    if(verify_tiles!=0 && verify_tiles!=1)die("PREFILL_VERIFY_TILES must be 0 or 1");
    layer_t l=load_layer(dir);
    fused_weights_t fw=fused_weights_load(dir,1,block);
    fused_weights_t ddr=fw;
    ddr.tcm_tile=NULL;ddr.tcm_map=NULL;ddr.tcm_fd=-1;
    rt_ai_pipeline_t dma={0};
    double startup=seconds_now();
    ra_open(&dma,&fw,RT_AIDMA,0); /* NO pinned weight; entire batch reuses tiles */
    startup=seconds_now()-startup;
    dma.verify=verify_tiles;
    pf_workspace_t work;
    pf_workspace_open(&work,ntok);

    size_t nx=(size_t)ntok*HIDDEN;
    size_t nk=(size_t)(prefix+ntok)*KVDIM;
    float *input=malloc(nx*sizeof(float));
    float *yref=malloc(nx*sizeof(float));
    float *ypf=malloc(nx*sizeof(float));
    uint16_t *kc0=malloc(nk*sizeof(uint16_t)),*vc0=malloc(nk*sizeof(uint16_t));
    uint16_t *kc1=malloc(nk*sizeof(uint16_t)),*vc1=malloc(nk*sizeof(uint16_t));
    if(!input||!yref||!ypf||!kc0||!vc0||!kc1||!vc1)
        die("Out of memory allocating batch/prefix buffers");
    for(size_t i=0;i<nx;++i)input[i]=random_float();
    puts("K3 AI-DMA WEIGHT-STATIONARY CAUSAL PREFILL");
    printf("tokens=%d prefix=%d repeats=%d | TCM block=%d | tile verification=%s\n",
           ntok,prefix,repeats,block,verify_tiles?"ON (timed)":"OFF (validate outputs)");
    puts("Double buffers: 2 x (30 KiB Q + 30 KiB KV + 30 KiB W_O) = 180 KiB TCM.");
    puts("Each DMA weight chunk is processed for ALL prompt tokens before the next chunk is used.");
    puts("K/V all positions first; Q + causal masked attention second; W_O last.");
    puts("FP32 intermediates and attention Q8 live in DDR; FP16 K/V stays in DDR/L2.");
    printf("Startup coherent-source mirroring: %.3f ms (excluded).\n",startup*1e3);
    puts("WARNING: K3 vendor AI-DMA may leak coherent descriptor pages per request.");

    double bsum=0,psum=0;
    for(int rep=0;rep<repeats;++rep) {
        pf_seed_history(kc0,vc0,prefix,prefix+ntok);
        memcpy(kc1,kc0,nk*sizeof(uint16_t));
        memcpy(vc1,vc0,nk*sizeof(uint16_t));
        double bt0=0,pt0=0;
        size_t before=dma.requests,bytes_before=dma.bytes;
        for(int order=0;order<2;++order) {
            int method=(rep&1)?1-order:order;
            double t0=seconds_now();
            if(method==0) {
                for(int t=0;t<ntok;++t)
                    (void)run_layer_fused(&l,&ddr,input+(size_t)t*HIDDEN,
                        kc0,vc0,prefix+t+1,yref+(size_t)t*HIDDEN,NULL,NULL);
                bt0=seconds_now()-t0;
            } else {
                pf_run_batch(&l,&fw,&dma,&work,input,ntok,prefix,kc1,vc1,ypf);
                pt0=seconds_now()-t0;
            }
        }
        size_t requests=dma.requests-before;
        size_t bytes=dma.bytes-bytes_before;
        printf("\nBatch %d: sequential DDR %.3f ms (%.3f ms/token), "
               "weight-stationary AI-DMA %.3f ms (%.3f ms/token)\n",
               rep+1,bt0*1e3,bt0*1e3/ntok,pt0*1e3,pt0*1e3/ntok);
        printf("AI-DMA: %zu requests, %.2f KiB total (%.3f KiB/token amortized)\n",
               requests,(double)bytes/1024.,(double)bytes/(1024.*ntok));
        if(requests!=136 || bytes!=3360u*1024u)
            die("Unexpected prefill DMA request count/bytes; tiles missing or duplicated");
        int pass=pf_compare_f32(yref,ypf,nx);
        if(memcmp(kc0,kc1,nk*sizeof(uint16_t))!=0 ||
           memcmp(vc0,vc1,nk*sizeof(uint16_t))!=0) {
            fputs("FAIL: FP16 K/V cache differs from sequential decode baseline\n",stderr);
            pass=0;
        } else puts("FP16 K/V: exact match for every prefix and batch position");
        if(!pass)die("Prefill result differs from original fused sequential DDR");
        bsum+=bt0;psum+=pt0;
    }
    printf("\nPASS: causal prefill output and K/V agree across all %d tokens.\n",ntok);
    printf("Average: sequential DDR %.3f ms vs prefill AI-DMA %.3f ms; speedup %.3fx\n",
           bsum*1e3/repeats,psum*1e3/repeats,bsum/psum);
    printf("AI-DMA total: %zu requests; no CPU weight-copy fallback.\n",dma.requests);
    free(input);free(yref);free(ypf);
    free(kc0);free(vc0);free(kc1);free(vc1);
    pf_workspace_close(&work);ra_close(&dma);fused_weights_free(&fw);free_layer(&l);
    return 0;
}
#else
int main(void){fputs("AI-DMA prefill requires RV64GCV/Zvfh\n",stderr);return 2;}
#endif
