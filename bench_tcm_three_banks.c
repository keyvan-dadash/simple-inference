/* Three 30KiB TCM-bank experiment; does NOT alter the existing fused path.
 * Include the existing single-TU model helpers without its CLI entry point.
 * Run pinned to the owning CPU: make tcm-three-banks ...
 */
#define main qwen_main_for_tcm_three_banks
#include "qwen_one_layer.c"
#undef main
#include "tcm_three_banks.h"

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
        fprintf(stderr,"Usage: %s weights [context=256] [repeat=30] [mode=1 (stream)|2 (pin)]\n",argv[0]);
        return 2;
    }
    const char *dir=argv[1];
    int context=argc>2?atoi(argv[2]):256;
    int repeats=argc>3?atoi(argv[3]):30;
    int mode=argc>4?atoi(argv[4]):RT_STREAM;
    int pin_pair=getenv("TCM_PIN_PAIR")?atoi(getenv("TCM_PIN_PAIR")):0;
    int block=getenv("TCM_BLOCK")?atoi(getenv("TCM_BLOCK")):0;
    if(context<1||context>32768||repeats<3||repeats>100000||
       (mode!=RT_STREAM&&mode!=RT_PIN)||pin_pair<0||pin_pair>=6)
        die("Invalid context, repetitions, mode or pinned pair");

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
    double preload_begin=seconds_now();
    rt_init(&tcm,&fw,mode,pin_pair);
    double preload_ms=(seconds_now()-preload_begin)*1e3;

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
    const char *names[3]={"Original fused DDR", "Tiled weights DDR", "Tiled weights TCM"};
    const char *mode_name=mode==RT_STREAM?"STREAM every tile":"PIN three subtiles";
    printf("THREE-BANK TCM WEIGHT EXPERIMENT | context=%d repeat=%d | %s\n",context,repeats,mode_name);
    printf("Banks: Q=30KiB, interleaved KV=30KiB, W_O=30KiB, total=90KiB.\n");
    puts("KV history remains FP16 in normal memory; NO KV-cache TCM copies.");
    if(mode==RT_STREAM)
        printf("STREAM copies %.1f KiB per token INCLUDING the complete Q, KV and W_O weights.\n",
               14.0*FUSED_TILE_BYTES/1024.0);
    else
        printf("PIN pair=%d, KV group=%d, first 30KiB of W_O y-tile=0; preload %.4f ms (excluded).\n",
               pin_pair,pin_pair/3,preload_ms);
    puts("Original fused DDR versus tiled DDR isolates tiling; tiled DDR versus TCM isolates placement/copies.");

    for(int w=0;w<6;++w){
        for(int k=0;k<HIDDEN;++k)x[k]=random_float();
        int m=w%3;
        if(m==0)(void)run_layer_fused(&l,&ddr_fw,x,kc[m],vc[m],context,y[m],NULL,NULL);
        if(m==1)(void)rt_run_layer(&l,&ddr_fw,&ddr,x,kc[m],vc[m],context,y[m],NULL,NULL);
        if(m==2)(void)rt_run_layer(&l,&fw,&tcm,x,kc[m],vc[m],context,y[m],NULL,NULL);
    }
    /* Fresh input on each repetition, but identical input across all methods. */
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
    double med[3];
    puts("\nWALL-CLOCK LATENCY (all TCM copies counted):");
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
    puts("PASS: TCM/DDR paths agree within stated tolerances.");
    free(o0);free(o1);free(q0);free(q1);
    for(int m=0;m<3;++m){free(y[m]);free(kc[m]);free(vc[m]);free(times[m]);}
    free(x);fused_weights_free(&fw);free_layer(&l);
    return 0;
}
#else
int main(void){fputs("TCM three-bank benchmark requires RV64GCV/Zvfh\n",stderr);return 2;}
#endif
