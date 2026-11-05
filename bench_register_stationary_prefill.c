/* K3 A100 fused-register, pair-stationary causal prefill.
 * Three-way control:
 * (1) original sequential fused DDR,
 * (2) pair-stationary register fusion with DDR weights,
 * (3) identical pair-stationary fusion with partial TCM weights via AI-DMA.
 * The same rf_* kernels, Q8 input, causal masking, and output order are used
 * for the matched (2)/(3) pair. No decode or older benchmark is modified.
 */
#define main qwen_main_for_register_stationary_prefill
#include "qwen_one_layer.c"
#undef main
#define RT_ENABLE_AIDMA 1
#include "tcm_three_banks.h"
#include "ai_dma_pipeline.h"
#include "register_stationary_prefill.h"

#if HAVE_RVV
static void rp_seed_history(uint16_t *k,uint16_t *v,int prefix,int total) {
    memset(k,0,(size_t)total*KVDIM*sizeof(uint16_t));
    memset(v,0,(size_t)total*KVDIM*sizeof(uint16_t));
    if(prefix>0)fill_kv_history(k,v,prefix+1);
}

static int rp_check_y(const float *ref,const float *got,size_t n,
                       const char *name) {
    size_t bad=0,first=0;
    double max_abs=0.0,max_rel=0.0;
    for(size_t i=0;i<n;++i) {
        double r=ref[i],g=got[i],diff=fabs(r-g),scale=fmax(fabs(r),fabs(g));
        if(diff>max_abs)max_abs=diff;
        if(scale>1e-8 && diff/scale>max_rel)max_rel=diff/scale;
        if(!isfinite(r)||!isfinite(g)||diff>1e-5+1e-4*scale) {
            if(!bad)first=i;
            ++bad;
        }
    }
    printf("%s Y: max_abs=%.8g max_rel=%.8g mismatches=%zu/%zu\n",
           name,max_abs,max_rel,bad,n);
    if(bad)printf("First mismatch token=%zu output=%zu: baseline=%.9g new=%.9g\n",
                  first/HIDDEN,first%HIDDEN,ref[first],got[first]);
    return bad==0;
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
        die("tokens 1..256; valid causal prefix; repeats 1..3 required");
    int block=getenv("TCM_BLOCK")?atoi(getenv("TCM_BLOCK")):0;
    int verify=getenv("PREFILL_VERIFY_TILES")?atoi(getenv("PREFILL_VERIFY_TILES")):0;
    if(verify!=0 && verify!=1)die("PREFILL_VERIFY_TILES must be 0 or 1");
    layer_t l=load_layer(dir);
    fused_weights_t fw=fused_weights_load(dir,1,block);
    fused_weights_t baseline_weights=fw;
    baseline_weights.tcm_fd=-1;
    baseline_weights.tcm_map=NULL;
    baseline_weights.tcm_tile=NULL;
    rt_ai_pipeline_t a={0};
    double startup_begin=seconds_now();
    ra_open(&a,&fw,RT_AIDMA,0);
    double startup=(seconds_now()-startup_begin)*1e3;
    a.verify=verify;
    rp_workspace_t work;
    rp_workspace_open(&work,ntok);

    size_t nx=(size_t)ntok*HIDDEN;
    size_t nk=(size_t)(prefix+ntok)*KVDIM;
    float *input=malloc(nx*sizeof(float));
    float *outputs[3]={0};
    uint16_t *keys[3]={0},*values[3]={0};
    for(int m=0;m<3;++m) {
        outputs[m]=malloc(nx*sizeof(float));
        keys[m]=malloc(nk*sizeof(uint16_t));
        values[m]=malloc(nk*sizeof(uint16_t));
        if(!outputs[m] || !keys[m] || !values[m])
            die("Unable to allocate benchmark output / KV cache");
    }
    if(!input)die("Unable to allocate benchmark inputs");
    for(size_t i=0;i<nx;++i)input[i]=random_float();
    puts("K3 REGISTER-RESIDENT / HEAD-PAIR-STATIONARY CAUSAL PREFILL");
    printf("tokens=%d prefix=%d repeats=%d | TCM block=%d | verify weights=%s\n",
           ntok,prefix,repeats,block,verify?"yes":"no");
    puts("TCM: 240KiB complete Q/KV tile + 120KiB W_O prefix = 360KiB.");
    puts("Remaining 120KiB W_O per pair: normal DDR/L2 (480KiB Q+W_O exceeds 384KiB).");
    puts("Pair loop: resident weights -> token0 Q/RoPE/attention/Q8/W_O/Y -> token1 -> ... -> next pair.");
    puts("No Q, attention O, or attention Q8 materialization; input Q8, RoPE, persistent KV and partial Y are stored.");
    puts("Matched DDR control executes the SAME register-fused pair-major kernel without DMA/TCM.");
    printf("Startup coherent-source mirroring: %.3f ms (excluded).\n",startup);
    puts("WARNING: vendor AI-DMA may leak driver descriptors; limit repetitions.");

    double sums[3]={0};
    const char *name[3]={"Sequential fused DDR","Pair-stationary fused DDR","Pair-stationary fused AI-DMA/TCM"};
    for(int rep=0;rep<repeats;++rep) {
        rp_seed_history(keys[0],values[0],prefix,prefix+ntok);
        for(int m=1;m<3;++m) {
            memcpy(keys[m],keys[0],nk*sizeof(uint16_t));
            memcpy(values[m],values[0],nk*sizeof(uint16_t));
        }
        double wall[3]={0};
        size_t req0=a.requests,bytes0=a.bytes;
        for(int order=0;order<3;++order) {
            int method=(order+rep)%3;
            double started=seconds_now();
            if(method==0) {
                for(int t=0;t<ntok;++t)
                    (void)run_layer_fused(&l,&baseline_weights,
                          input+(size_t)t*HIDDEN,keys[0],values[0],prefix+t+1,
                          outputs[0]+(size_t)t*HIDDEN,NULL,NULL);
            } else {
                rp_run_batch(&l,&fw,&a,&work,input,ntok,prefix,
                             keys[method],values[method],outputs[method],method==2);
            }
            wall[method]=seconds_now()-started;
        }
        size_t req=a.requests-req0,bytes=a.bytes-bytes0;
        printf("\nBatch %d:\n",rep+1);
        for(int m=0;m<3;++m)
            printf("  %-32s %9.3f ms (%7.4f ms/token)\n",
                   name[m],wall[m]*1e3,wall[m]*1e3/ntok);
        printf("AI-DMA: %zu requests, %.2f KiB total, %.3f KiB/token amortized\n",
               req,(double)bytes/1024.,(double)bytes/(1024.*ntok));
        if(req!=88 || bytes!=2640u*1024u)
            die("Incorrect resident DMA tile schedule (expected 88 requests / 2640 KiB)");
        for(int m=1;m<3;++m) {
            int pass=rp_check_y(outputs[0],outputs[m],nx,name[m]);
            if(memcmp(keys[0],keys[m],nk*sizeof(uint16_t))!=0 ||
               memcmp(values[0],values[m],nk*sizeof(uint16_t))!=0) {
                printf("FAIL: FP16 KV differs for %s\n",name[m]);
                pass=0;
            } else printf("%s FP16 KV: bit-for-bit exact\n",name[m]);
            if(!pass)die("Register-stationary prefill failed independent baseline validation");
        }
        for(int m=0;m<3;++m)sums[m]+=wall[m];
    }
    printf("\nPASS: %d-token causal register-resident prefill, all outputs and KV match.\n",ntok);
    printf("Means: sequential DDR %.3f ms; matched pair DDR %.3f ms; AI-DMA TCM %.3f ms\n",
           sums[0]*1e3/repeats,sums[1]*1e3/repeats,sums[2]*1e3/repeats);
    printf("Speedups: sequential/TCM %.3fx; matched-DDR/TCM %.3fx\n",
           sums[0]/sums[2],sums[1]/sums[2]);
    printf("Total AI-DMA requests: %zu (coherent buffer allocation excluded).\n",a.requests);
    free(input);
    for(int m=0;m<3;++m) {free(outputs[m]);free(keys[m]);free(values[m]);}
    rp_workspace_close(&work);ra_close(&a);fused_weights_free(&fw);free_layer(&l);
    return 0;
}
#else
int main(void) {
    fputs("Register-stationary prefill requires RV64GCV + Zvfh\n",stderr);
    return 2;
}
#endif
