/* Opt-in genuine K3 AI-DMA weight-prefetch experiment.
 * Existing register-fused arithmetic and KV cache are unchanged.
 * No per-token userspace payload copies; source mirrors are built ONCE.
 * WARNING: published K3 AI-DMA driver may leak kernel DMA descriptors;
 * keep repetitions small, avoid running concurrently with vendor EP.
 */
#define main qwen_main_for_ai_dma_fusion
#include "qwen_one_layer.c"
#undef main
#define RT_ENABLE_AIDMA 1
#include "tcm_three_banks.h"
#include "ai_dma_pipeline.h"

#if HAVE_RVV

static int rt_cmp_double(const void *a,const void *b) {
    double x=*(const double *)a,y=*(const double *)b;
    return (x>y)-(x<y);
}
static double rt_median(const double *x,int n) {
    double *tmp=malloc((size_t)n*sizeof(double));
    if(!tmp) die("Out of memory calculating median");
    memcpy(tmp,x,(size_t)n*sizeof(double));
    qsort(tmp,(size_t)n,sizeof(double),rt_cmp_double);
    double m=(n&1)?tmp[n/2]:(tmp[n/2-1]+tmp[n/2])*0.5;
    free(tmp);return m;
}
static void rt_check_f32(const char *label,const float *a,const float *b,int n) {
    double maxabs=0,maxrel=0;
    int mismatches=0;
    for(int j=0;j<n;++j){
        if(!isfinite(a[j])||!isfinite(b[j])) die("Non-finite TCM comparison result");
        double d=fabs((double)a[j]-b[j]);
        double s=fmax(fabs((double)a[j]),fabs((double)b[j]));
        if(d>maxabs)maxabs=d;
        if(s>1e-8&&d/s>maxrel)maxrel=d/s;
        if(d>1e-5+1e-4*s)++mismatches;
    }
    printf("  %-25s max_abs=%10.3g max_rel=%10.3g bad=%d/%d\n",
            label,maxabs,maxrel,mismatches,n);
    if(mismatches) die("Tiled and fused results differ");
}
static void rt_check_q8(const block_q8_0 *a,const block_q8_0 *b) {
    int diffs=0;
    for(int j=0;j<HIDDEN/QK;++j) diffs+=memcmp(a+j,b+j,sizeof(*a))!=0;
    printf("  Q8 blocks different: %d/%d\n",diffs,HIDDEN/QK);
    if(diffs) die("Q8 blocks differ between tiled and fused methods");
}
static void rt_check_kv(const uint16_t *a,const uint16_t *b,int context) {
    size_t off=(size_t)(context-1)*KVDIM;
    if(memcmp(a+off,b+off,KVDIM*sizeof(uint16_t)))
        die("New FP16 K/V differs between methods");
}

int main(int argc,char **argv) {
    if(argc<2||argc>5) {
        fprintf(stderr,"Usage: %s weights [context=256] [repeat=5] [mode=1 (DMA stream)|2 (DMA stream+pin)]\n",argv[0]);
        return 2;
    }
    const char *dir=argv[1];
    int context=argc>2?atoi(argv[2]):256;
    int repeats=argc>3?atoi(argv[3]):5;
    int dma_mode=argc>4?atoi(argv[4]):2;
    int mode=dma_mode==2?RT_AIDMA_PIN:RT_AIDMA;
    int pin_pair=getenv("TCM_PIN_PAIR")?atoi(getenv("TCM_PIN_PAIR")):0;
    int block=getenv("TCM_BLOCK")?atoi(getenv("TCM_BLOCK")):0;
    if(context<1||context>32768||repeats<3||repeats>8||
       (dma_mode!=1&&dma_mode!=2)||pin_pair<0||pin_pair>=6)
        die("Invalid context, repeats (3..8), DMA mode (1|2) or pinned pair");

    layer_t l=load_layer(dir);
    fused_weights_t fw=fused_weights_load(dir,1,block); /* exclusive lock */
    /* Existing fused path requires a DDR-only weight descriptor. Share the
     * loaded weight arrays; the original fw owns and releases them. */
    fused_weights_t ddr_fw=fw;
    ddr_fw.tcm_tile=NULL;
    ddr_fw.tcm_map=NULL;
    ddr_fw.tcm_fd=-1;
    rt_workspace_t tcm={0}, ddr={0};
    rt_init(&ddr,&fw,RT_DDR,pin_pair);
    rt_init(&tcm,&fw,mode,pin_pair);
    rt_ai_pipeline_t dma={0};
    double preload_begin=seconds_now();
    ra_open(&dma,&fw,mode,pin_pair);
    double preload_ms=(seconds_now()-preload_begin)*1e3;
    tcm.ai=&dma;

    float *x=malloc(HIDDEN*sizeof(float));
    float *y[3];
    uint16_t *kc[3],*vc[3];
    double *times[3];
    size_t cache_len=(size_t)context*KVDIM;
    for(int m=0;m<3;++m){
        y[m]=malloc(HIDDEN*sizeof(float));
        kc[m]=malloc(cache_len*sizeof(uint16_t));
        vc[m]=malloc(cache_len*sizeof(uint16_t));
        times[m]=malloc((size_t)repeats*sizeof(double));
        if(!y[m]||!kc[m]||!vc[m]||!times[m])die("TCM benchmark allocation failure");
    }
    if(!x)die("TCM benchmark allocation failure");
    memset(kc[0],0,cache_len*sizeof(uint16_t));
    memset(vc[0],0,cache_len*sizeof(uint16_t));
    fill_kv_history(kc[0],vc[0],context);
    for(int m=1;m<3;++m){
        memcpy(kc[m],kc[0],cache_len*sizeof(uint16_t));
        memcpy(vc[m],vc[0],cache_len*sizeof(uint16_t));
    }
    const char *names[3]={"Original fused DDR", "Tiled weights DDR", "Tiled + AI-DMA TCM"};
    const char *mode_name=dma_mode==1?"AI-DMA STREAM":"AI-DMA STREAM + cross-token PIN";
    printf("K3 AI-DMA REGISTER-FUSION EXPERIMENT | context=%d repeat=%d | %s\n",context,repeats,mode_name);
    puts("TCM: two 30KiB streaming slots for each Q/KV/W_O family (180KiB total).");
    if(dma.pin)printf("Cross-token resident: 30KiB Q pair=%d, KV group=%d, W_O pair=%d (+90KiB).\n",
                      pin_pair,pin_pair/3,pin_pair);
    printf("Startup DMA source mirror/pin preload: %.3f ms (excluded from token timing).\n",preload_ms);
    puts("Each token reads ALL weights (3360KiB); no CPU weight staging during tokens.");
    puts("KV cache stays FP16 in normal DDR/L2, never copied into TCM.");
    puts("Transfer N+1 is issued before computing from TCM chunk N (ping-pong overlap).");
    puts("The vendor driver may leak ~one page of coherent descriptor memory per DMA; limit repeats.");
    puts("Original fused DDR versus tiled DDR isolates tiling; tiled DDR versus AI-DMA isolates the transfer path.");

    for(int w=0;w<3;++w){
        for(int k=0;k<HIDDEN;++k)x[k]=random_float();
        int m=w%3;
        if(m==0)(void)run_layer_fused(&l,&ddr_fw,x,kc[m],vc[m],context,y[m],NULL,NULL);
        if(m==1)(void)rt_run_layer(&l,&ddr_fw,&ddr,x,kc[m],vc[m],context,y[m],NULL,NULL);
        if(m==2)(void)rt_run_layer(&l,&fw,&tcm,x,kc[m],vc[m],context,y[m],NULL,NULL);
    }
    /* Fresh input on each repetition, but identical input across all methods. */
    size_t request_start=dma.requests, byte_start=dma.bytes;
    double avg[3]={0};
    for(int rep=0;rep<repeats;++rep){
        for(int k=0;k<HIDDEN;++k)x[k]=random_float();
        for(int j=0;j<3;++j){
            int m=(rep+j)%3;
            double start=seconds_now();
            if(m==0)(void)run_layer_fused(&l,&ddr_fw,x,kc[m],vc[m],context,y[m],NULL,NULL);
            if(m==1)(void)rt_run_layer(&l,&ddr_fw,&ddr,x,kc[m],vc[m],context,y[m],NULL,NULL);
            if(m==2)(void)rt_run_layer(&l,&fw,&tcm,x,kc[m],vc[m],context,y[m],NULL,NULL);
            double elapsed=seconds_now()-start;
            times[m][rep]=elapsed;
            avg[m]+=elapsed;
        }
    }
    size_t measured_requests=dma.requests-request_start;
    size_t measured_bytes=dma.bytes-byte_start;
    printf("\nMeasured AI-DMA: %zu requests, %.2f MiB total (%.2f KiB/token).\n",
           measured_requests,(double)measured_bytes/(1024.0*1024.0),
           (double)measured_bytes/((double)repeats*1024.0));
    double med[3];
    puts("\nWALL-CLOCK LATENCY (all AI-DMA issue/waits counted):");
    printf(" %-22s %13s %13s\n","Method","median ms","mean ms");
    for(int m=0;m<3;++m){
        med[m]=rt_median(times[m],repeats);
        printf(" %-22s %13.4f %13.4f\n",names[m],med[m]*1e3,avg[m]/repeats*1e3);
    }
    printf("  Original/tiled-DDR: %.4fx; tiled-DDR/TCM: %.4fx; original/TCM: %.4fx\n",
           med[0]/med[1],med[1]/med[2],med[0]/med[2]);

    /* Independent, untimed validation, including the exact capture-free path
     * used in the timing loops and a second capture-enabled replay. */
    float *o0=malloc(HIDDEN*sizeof(float)),*o1=malloc(HIDDEN*sizeof(float));
    block_q8_0 *q0=malloc((HIDDEN/QK)*sizeof(block_q8_0));
    block_q8_0 *q1=malloc((HIDDEN/QK)*sizeof(block_q8_0));
    if(!o0||!o1||!q0||!q1)die("Out of memory validating TCM path");
    int tests[3]={1,context>=17?17:context,context};
    dma.verify=1; /* compare every staged chunk to its DDR source, untimed only */
    for(int test=0;test<3;++test){
        int n=tests[test];
        for(int k=0;k<HIDDEN;++k)x[k]=random_float();
        /* Same initial KV history for all paths, freshly generated. */
        memset(kc[0],0,(size_t)n*KVDIM*sizeof(uint16_t));
        memset(vc[0],0,(size_t)n*KVDIM*sizeof(uint16_t));
        fill_kv_history(kc[0],vc[0],n);
        memcpy(kc[1],kc[0],(size_t)n*KVDIM*sizeof(uint16_t));
        memcpy(vc[1],vc[0],(size_t)n*KVDIM*sizeof(uint16_t));
        memcpy(kc[2],kc[0],(size_t)n*KVDIM*sizeof(uint16_t));
        memcpy(vc[2],vc[0],(size_t)n*KVDIM*sizeof(uint16_t));
        (void)run_layer_fused(&l,&ddr_fw,x,kc[0],vc[0],n,y[0],o0,q0);
        (void)rt_run_layer(&l,&ddr_fw,&ddr,x,kc[1],vc[1],n,y[1],NULL,NULL);
        (void)rt_run_layer(&l,&fw,&tcm,x,kc[2],vc[2],n,y[2],o1,q1);
        printf("\nVALIDATION context=%d:\n",n);
        rt_check_f32("Fused DDR vs tiled DDR",y[0],y[1],HIDDEN);
        rt_check_f32("Fused DDR vs TCM Y",y[0],y[2],HIDDEN);
        rt_check_f32("Fused DDR vs TCM O",o0,o1,HIDDEN);
        rt_check_q8(q0,q1);
        rt_check_kv(kc[0],kc[1],n);rt_check_kv(vc[0],vc[1],n);
        rt_check_kv(kc[0],kc[2],n);rt_check_kv(vc[0],vc[2],n);
        /* Capture disabled: the exact fast path must return the same Y. */
        (void)rt_run_layer(&l,&fw,&tcm,x,kc[2],vc[2],n,y[1],NULL,NULL);
        rt_check_f32("TCM capture vs no capture",y[1],y[2],HIDDEN);
    }
    dma.verify=0;
    printf("\nAI-DMA total across warmup, timed runs and checks: %zu hardware requests (startup pin loads=%zu).\n",
           dma.requests,dma.preload_requests);
    puts("PASS: DDR and AI-DMA fused paths agree; all TCM weight chunks verified byte-exact.");
    free(o0);free(o1);free(q0);free(q1);
    for(int m=0;m<3;++m){free(y[m]);free(kc[m]);free(vc[m]);free(times[m]);}
    free(x);ra_close(&dma);fused_weights_free(&fw);free_layer(&l);
    return 0;
}
#else
int main(void){fputs("AI-DMA fusion benchmark requires RV64GCV/Zvfh\n",stderr);return 2;}
#endif
