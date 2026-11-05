#pragma once
/* K3 A100 AI-DMA weight prefetch for the existing register-fused head-pair
 * arithmetic in tcm_three_banks.h. No CPU weight copies during a token.
 *
 * IMPORTANT: include qwen_one_layer.c, define RT_ENABLE_AIDMA=1, then include
 * tcm_three_banks.h BEFORE this header. Only the opt-in benchmark uses it.
 *
 * Hardware ABI: SpacemiT linux-6.18 (k3-br-v1.0.y):
 *   drivers/dma/ai_dma.h, ai_dma_dev.c, ai_dma_ctrl.c, udma.c.
 * /dev/udma supplies coherent SOURCE ALLOCATIONS ONLY; hardware transfers use
 * /dev/aidma_list + /dev/dma_msi. Persistent CPU memcpy at load time simply
 * mirrors repacked weights into DMA-addressable memory and is NOT per-token.
 *
 * One 384 KiB TCM block: two 30 KiB slots per Q/KV/W_O = 180 KiB; optional
 * additional pinned 30 KiB Q/KV/W_O = 270 KiB total. All remaining KV cache
 * data remains in ordinary DDR/L2. One in-flight DMA per weight family (3
 * total, safely below the driver's 8-controller capacity).
 *
 * WARNING: the published AI-DMA driver appears to leak DMA descriptors per
 * request. Keep measurements short and reboot before prolonged experiments.
 */
#if HAVE_RVV && defined(RT_ENABLE_AIDMA)
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define RA_REQ_COUNT 32
#define RA_RING_SIZE 4096
#define RA_BANK_BYTES RT_BYTES
#define RA_STREAM_BYTES (3u*2u*RA_BANK_BYTES)
#define RA_PIN_BYTES (3u*RA_BANK_BYTES)
#define RA_MAX_TCM_BYTES (RA_STREAM_BYTES+RA_PIN_BYTES)
#define RA_TIMEOUT_NS UINT64_C(2000000000)
_Static_assert(RA_BANK_BYTES == 30720u, "AI-DMA must use 30-KiB aligned chunks");
_Static_assert(RA_MAX_TCM_BYTES == 276480u, "AI-DMA TCM layout mismatch");

typedef struct {
    size_t k_size,m_size,kr_size,mr_size;
    _Bool pack,transpose;
    uint32_t pad_value;
    int32_t ele_size;
} ra_pack_t;
typedef struct {
    void *src,*dst;
    int32_t pid;
    _Bool is_sgdg;
    size_t size;
    ra_pack_t ai_param;
} ra_transfer_t;
typedef struct {
    ra_transfer_t params;
    uint32_t status;
} __attribute__((aligned(64))) ra_request_t;
enum { RA_FREE=0, RA_SUBMIT=1, RA_PROCESS=2, RA_DONE=3 };

_Static_assert(sizeof(void*)==8 && sizeof(size_t)==8,
               "AI-DMA requires 64-bit Linux pointers");
_Static_assert(sizeof(ra_pack_t)==48,"AI-DMA pack ABI mismatch");
_Static_assert(sizeof(ra_transfer_t)==80,"AI-DMA parameter ABI mismatch");
_Static_assert(offsetof(ra_transfer_t,size)==24,"AI-DMA size field ABI mismatch");
_Static_assert(offsetof(ra_request_t,status)==80,"AI-DMA status field ABI mismatch");
_Static_assert(sizeof(ra_request_t)==128,"AI-DMA request ABI mismatch");
_Static_assert(RA_REQ_COUNT*sizeof(ra_request_t)==RA_RING_SIZE,
               "AI-DMA ring ABI mismatch");

typedef struct {
    const uint8_t *reference;
    uint8_t *coherent;
    size_t size;
} ra_source_t;
typedef struct {
    uint8_t *slot[2],*pinned;
    const uint8_t *pending_src;
    size_t pending_bytes;
    int pending_slot, has_pending, last_slot;
    int pin_index;
} ra_lane_t;

typedef struct rt_ai_pipeline {
    int udma_fd, ring_fd, msi_fd;
    ra_request_t *ring;
    volatile uint32_t *msi_page;
    uint32_t msi_data;
    int32_t pid;
    ra_source_t src[3][6];
    ra_lane_t lane[3];
    int pin, pin_pair, verify;
    size_t requests, bytes, preload_requests;
} rt_ai_pipeline_t;

static void ra_die_errno(const char *where) {
    int e=errno;
    fprintf(stderr,"AI-DMA %s: %s (errno=%d)\n",where,strerror(e),e);
    die("AI-DMA kernel interface failed; no CPU fallback");
}
static uint64_t ra_ns(void) {
    struct timespec ts;
    if(clock_gettime(CLOCK_MONOTONIC_RAW,&ts))ra_die_errno("clock_gettime");
    return (uint64_t)ts.tv_sec*UINT64_C(1000000000)+(uint64_t)ts.tv_nsec;
}
static uint32_t ra_status(const ra_request_t *r) {
    return __atomic_load_n(&r->status,__ATOMIC_ACQUIRE);
}
static void ra_status_set(ra_request_t *r,uint32_t x) {
    __atomic_store_n(&r->status,x,__ATOMIC_RELEASE);
}

/* All source spans are within ONE independently DMA-coherent 240KiB mmap.
 * This is required because the vendor AI-DMA driver only translates the
 * first PTE and assumes physical contiguity for the whole request. */
static uint8_t *ra_source_ptr(rt_ai_pipeline_t *a,int kind,int index,
                              const uint8_t *src,size_t n) {
    int count=kind==RT_KIND_KV?2:6;
    if(kind<0 || kind>RT_KIND_WO || index<0 || index>=count)
        die("AI-DMA source kind/index out of range");
    ra_source_t *s=&a->src[kind][index];
    uintptr_t p=(uintptr_t)src,b=(uintptr_t)s->reference;
    if(p<b || p-b>s->size || n>s->size-(p-b))
        die("AI-DMA source outside the selected contiguous tile");
    return s->coherent+(p-b);
}

/* Nonblocking hardware submission. A distinct ring entry is assigned per
 * weight family; never reuse it while PROCESS/SUBMIT. Doorbell is MMIO. */
static void ra_submit(rt_ai_pipeline_t *a,int kind,const uint8_t *orig,
                      int index,uint8_t *dst,size_t n) {
    ra_request_t *r=&a->ring[kind];
    uint32_t old=ra_status(r);
    if(old!=RA_FREE && old!=RA_DONE)
        die("AI-DMA request still in flight; refusing to overwrite");
    if(!n || n>RA_BANK_BYTES || (n%FUSED_GROUP_BYTES)!=0)
        die("Invalid AI-DMA copy length");
    uint8_t *src=ra_source_ptr(a,kind,index,orig,n);
    r->params.src=src;
    r->params.dst=dst;
    r->params.pid=a->pid;        /* cached once: no getpid syscall per tile */
    r->params.is_sgdg=0;          /* raw copy: no transpose or packing */
    r->params.size=n;
    /* ai_param stays zero from the initial ring memset. */
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    ra_status_set(r,RA_SUBMIT);
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    a->msi_page[0]=a->msi_data;   /* actual K3 IMSIC doorbell */
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    a->requests++;
    a->bytes+=n;
}

/* The normal successful path only polls status, without a syscall inside
 * the live RVV arithmetic. A clock check is performed on the rare long-wait
 * path to prevent an infinite spin if hardware/driver fails. */
static void ra_wait(rt_ai_pipeline_t *a,int kind) {
    ra_request_t *r=&a->ring[kind];
    uint64_t deadline=0;
    uint32_t status;
    unsigned long spins=0;
    for(;;) {
        status=ra_status(r);
        if(status==RA_DONE)break;
        if(status!=RA_PROCESS && status!=RA_SUBMIT)
            die("Unexpected AI-DMA status; stopping (no fallback)");
        if((++spins & ((1u<<16)-1u))==0) {
            uint64_t now=ra_ns();
            if(!deadline)deadline=now+RA_TIMEOUT_NS;
            else if(now>=deadline) {
                fprintf(stderr,"AI-DMA timed out: family=%d status=%u\n",kind,status);
                die("DMA request did not complete; do not reuse this buffer");
            }
        }
    }
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

static void ra_close(rt_ai_pipeline_t *a) {
    for(int k=0;k<3;++k) {
        if(a->lane[k].has_pending)
            ra_wait(a,k);
    }
    if(a->ring && a->ring!=MAP_FAILED)munmap(a->ring,RA_RING_SIZE);
    if(a->msi_page && (void*)a->msi_page!=MAP_FAILED)munmap((void*)a->msi_page,4096);
    for(int kind=0;kind<3;++kind)for(int i=0;i<(kind==RT_KIND_KV?2:6);++i)
        if(a->src[kind][i].coherent && a->src[kind][i].coherent!=MAP_FAILED)
            munmap(a->src[kind][i].coherent,a->src[kind][i].size);
    if(a->udma_fd>=0)close(a->udma_fd);
    if(a->ring_fd>=0)close(a->ring_fd);
    if(a->msi_fd>=0)close(a->msi_fd);
}

/* Open the real AI-DMA engine and create per-projection contiguous sources.
 * Startup copies happen only ONCE, outside any token measurements. */
static void ra_open(rt_ai_pipeline_t *a,const fused_weights_t *fw,
                    int mode,int pinned_pair) {
    memset(a,0,sizeof(*a));
    a->udma_fd=a->ring_fd=a->msi_fd=-1;
    a->pin=(mode==RT_AIDMA_PIN);
    a->pin_pair=pinned_pair;
    a->pid=(int32_t)getpid();
    if(!fw->tcm_tile || fw->tcm_map_bytes==0)
        die("AI-DMA requires exclusive mapped TCM from fused_weights_load");
    /* fused_tcm_open validates local CPU affinity and block >= 240 KiB.
     * Re-check this specific 270-KiB layout independently. */
    fused_tcm_info_t ti={0};
    if(ioctl(fw->tcm_fd,FUSED_TCM_INFO_GET,&ti)!=0)
        ra_die_errno("TCM_INFO_GET workspace geometry");
    size_t needed=a->pin?RA_MAX_TCM_BYTES:RA_STREAM_BYTES;
    if(ti.block_num==0 || ti.block_size<needed || fw->tcm_block<0 ||
       (size_t)fw->tcm_block>=ti.block_num ||
       ti.block_size>SIZE_MAX/ti.block_num ||
       fw->tcm_map_bytes<ti.block_size*ti.block_num)
        die("Selected TCM block cannot hold AI-DMA ping-pong workspace");
    uint8_t *base=fw->tcm_tile;
    for(int k=0;k<3;++k) {
        a->lane[k].slot[0]=base+(size_t)(2*k)*RA_BANK_BYTES;
        a->lane[k].slot[1]=base+(size_t)(2*k+1)*RA_BANK_BYTES;
        a->lane[k].pinned=base+RA_STREAM_BYTES+(size_t)k*RA_BANK_BYTES;
        a->lane[k].last_slot=1;
        a->lane[k].pin_index=k==RT_KIND_KV?pinned_pair/3:pinned_pair;
    }
    a->udma_fd=open("/dev/udma",O_RDWR|O_SYNC|O_CLOEXEC);
    if(a->udma_fd<0)ra_die_errno("open(/dev/udma) coherent allocator");
    const uint8_t *base_src[3]={fw->q,fw->kv,fw->o};
    for(int kind=0;kind<3;++kind) {
        for(int index=0;index<(kind==RT_KIND_KV?2:6);++index) {
            ra_source_t *s=&a->src[kind][index];
            s->reference=base_src[kind]+(size_t)index*FUSED_TILE_BYTES;
            s->size=FUSED_TILE_BYTES;
            s->coherent=mmap(NULL,s->size,PROT_READ|PROT_WRITE,
                             MAP_SHARED,a->udma_fd,0);
            if(s->coherent==MAP_FAILED)ra_die_errno("mmap(/dev/udma) weight tile");
            memcpy(s->coherent,s->reference,s->size); /* ONE-TIME only */
        }
    }
    a->ring_fd=open("/dev/aidma_list",O_RDWR|O_CLOEXEC);
    if(a->ring_fd<0)ra_die_errno("open(/dev/aidma_list)");
    a->ring=mmap(NULL,RA_RING_SIZE,PROT_READ|PROT_WRITE,MAP_SHARED,a->ring_fd,0);
    if(a->ring==MAP_FAILED)ra_die_errno("mmap(/dev/aidma_list)");
    memset(a->ring,0,RA_RING_SIZE); /* dedicated ring for this process */
    a->msi_fd=open("/dev/dma_msi",O_RDWR|O_CLOEXEC);
    if(a->msi_fd<0)ra_die_errno("open(/dev/dma_msi)");
    if(ioctl(a->msi_fd,0UL,&a->msi_data)<0)ra_die_errno("MSI_GET_HWIRQ");
    a->msi_page=mmap(NULL,4096,PROT_READ|PROT_WRITE,MAP_SHARED,a->msi_fd,0);
    if(a->msi_page==MAP_FAILED)ra_die_errno("mmap(/dev/dma_msi)");
    if(a->pin) {
        /* Q[first 30KiB], KV[first 30KiB], WO[first 30KiB] stay in TCM
         * across repeated token invocations and are never overwritten. */
        for(int kind=0;kind<3;++kind) {
            int index=a->lane[kind].pin_index;
            ra_submit(a,kind,a->src[kind][index].reference,index,
                      a->lane[kind].pinned,RA_BANK_BYTES);
            ra_wait(a,kind);
            a->preload_requests++;
        }
    }
}

/* Called in a known-invocation order from the existing register-resident
 * projection/WO loops. Submit chunk N+1 to the opposite bank, return chunk N.
 * We never DMA into the buffer the vector kernel is reading. */
static const uint8_t *rt_ai_take(rt_workspace_t *rt,int kind,int index,
                                  int yt,int start,const uint8_t *src,int n) {
    rt_ai_pipeline_t *a=rt->ai;
    if(!a)die("AI-DMA workspace not initialized");
    if(n<=0 || n>RT_BLOCKS || start<0 || start>=48)
        die("AI-DMA invalid quant-group boundaries");
    ra_lane_t *l=&a->lane[kind];
    int abs_start=(kind==RT_KIND_WO)?yt*8+start:start;
    if(kind==RT_KIND_WO && (yt<0 || yt>=6 || start>=8 || start+n>8))
        die("AI-DMA invalid W_O subtile boundary");
    if(abs_start+n>48)die("AI-DMA tile overrun");
    size_t sz=(size_t)n*FUSED_GROUP_BYTES;
    int resident=a->pin && index==l->pin_index && abs_start==0 && n==RT_BLOCKS;
    const uint8_t *current;
    int current_slot=-1;
    if(resident) {
        if(l->has_pending)die("AI-DMA unexpected pending tile at pinned chunk");
        current=l->pinned;
    } else if(l->has_pending) {
        if(l->pending_src!=src || l->pending_bytes!=sz)
            die("AI-DMA prefetch order mismatch; refusing to use stale weight data");
        ra_wait(a,kind);
        current_slot=l->pending_slot;
        current=l->slot[current_slot];
        l->has_pending=0;
    } else {
        current_slot=l->last_slot^1;
        ra_submit(a,kind,src,index,l->slot[current_slot],sz);
        ra_wait(a,kind);
        current=l->slot[current_slot];
    }
    if(current_slot>=0)l->last_slot=current_slot;
    if(a->verify && memcmp(current,src,sz)!=0)
        die("AI-DMA weight tile differs from repacked DDR source");

    int next=abs_start+n;
    int next_index=index, next_n=0;
    const uint8_t *next_src=NULL;
    if(next<48) {
        /* W_O uses six groups then two groups per 8-group Y subtile;
         * Q and KV always use six groups per 48-group full tile. */
        next_n=kind==RT_KIND_WO ? (8-next%8) : RT_BLOCKS;
        if(next_n>RT_BLOCKS)next_n=RT_BLOCKS;
        if(next_n>48-next)next_n=48-next;
        next_src=src+sz;
    } else {
        /* Critically, also prefetch the FIRST chunk of the NEXT head pair
         * while the current head pair is still computing. This can remain
         * in the inactive bank through attention and W_O. For KV the next
         * group is prefetched while all the previous GQA heads execute.
         * Wrap to pair/group 0 across token boundaries in stream mode. */
        next_index=(index+1)%(kind==RT_KIND_KV?2:6);
        if(!(a->pin && next_index==l->pin_index)) {
            next_src=a->src[kind][next_index].reference;
            next_n=RT_BLOCKS;
        }
    }
    if(next_src) {
        int dst_slot=(current_slot<0)?(l->last_slot^1):(current_slot^1);
        ra_submit(a,kind,next_src,next_index,l->slot[dst_slot],
                  (size_t)next_n*FUSED_GROUP_BYTES);
        l->pending_src=next_src;
        l->pending_bytes=(size_t)next_n*FUSED_GROUP_BYTES;
        l->pending_slot=dst_slot;
        l->has_pending=1;
    }
    return current;
}
#endif /* HAVE_RVV && RT_ENABLE_AIDMA */
