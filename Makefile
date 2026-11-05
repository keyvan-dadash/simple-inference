# Qwen2.5-Coder-1.5B single-layer experiments on SpacemiT K3 A100.
# All generated executables, assembly and perf logs belong in build/.
CC         = gcc
VECTOR    ?= 1
BUILD_DIR ?= build
RVV_ARCH  ?= rv64gcv_zvfh_zfh_zicbop
CFLAGS    ?= -O2 -std=c11 -Wall -Wextra -Wshadow -ffp-contract=off
LDFLAGS   ?= -lm
GGUF      ?=
LAYER     ?= 0
WEIGHTS   ?= weights
CONTEXT   ?= 256
REPEAT    ?= 30
ATTENTION ?= fused-compare
FUSED_TCM ?= 0
TCM_BLOCK ?= 0

ifeq ($(VECTOR),1)
CFLAGS += -march=$(RVV_ARCH) -mabi=lp64d -DUSE_RVV=1
else
CFLAGS += -DUSE_RVV=0
endif

COMMON = qwen_one_layer.c attention_kernels.h fused_layer.h register_fusion.h soa_staged.h defines.h
BIN = $(BUILD_DIR)/qwen_one_layer
AUDIT = $(BUILD_DIR)/audit_fusion_timing
VERIFY = $(BUILD_DIR)/verify_fused_projections
LLAMA = $(BUILD_DIR)/bench_llama_fusion

.PHONY: all extract repack run a100 audit three-way reference soa-staged soa-fused verify llama-baseline reg-asm test clean
all: $(BIN)

$(BUILD_DIR):
	mkdir -p "$@"

$(BIN): $(COMMON) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o "$@" qwen_one_layer.c $(LDFLAGS)

$(AUDIT): audit_fusion_timing.c $(COMMON) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o "$@" audit_fusion_timing.c $(LDFLAGS)

$(VERIFY): verify_fused_projections.c $(COMMON) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o "$@" verify_fused_projections.c $(LDFLAGS)

$(LLAMA): bench_llama_fusion.c $(COMMON) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o "$@" bench_llama_fusion.c $(LDFLAGS)

extract:
	@test -n "$(GGUF)" || (echo 'Usage: make extract GGUF=/path/to/qwen-q4_0.gguf' >&2; exit 1)
	python3 extract_weights.py "$(GGUF)" --layer "$(LAYER)" --out "$(WEIGHTS)"

repack:
	python3 repack_weights.py --weights "$(WEIGHTS)"

run: $(BIN)
	FUSED_TCM="$(FUSED_TCM)" TCM_BLOCK="$(TCM_BLOCK)" "$(BIN)" "$(WEIGHTS)" "$(CONTEXT)" "$(REPEAT)" "$(ATTENTION)"

# Run on pinned A100 core (also sets the A100 execution mode via the driver).
a100: $(BIN)
	FUSED_TCM="$(FUSED_TCM)" TCM_BLOCK="$(TCM_BLOCK)" sh -c 'echo $$$$ > /proc/set_ai_thread || exit 1; exec taskset -c 8 "$$1" "$$2" "$$3" "$$4" "$$5"' sh "$(BIN)" "$(WEIGHTS)" "$(CONTEXT)" "$(REPEAT)" "$(ATTENTION)"

# Official three-way baseline: (A) generic llama.cpp-style RVV row-major,
# (B) matched staged using exactly the same rf_* kernels as (C) register fusion.
# Timing differences isolate full Q/O/Q8 materialization and compiler effects.
three-way: audit

# Individual modes use the same qwen_one_layer driver for stable inputs.
reference: ATTENTION = reference
reference: a100
soa-staged: ATTENTION = soa-staged
soa-staged: a100
soa-fused: ATTENTION = soa-fused
soa-fused: a100

# Matched staged/fused both use Zvfh, the same exp approximation and W_O kernel.
audit: $(AUDIT)
	FUSED_TCM="$(FUSED_TCM)" TCM_BLOCK="$(TCM_BLOCK)" sh -c 'echo $$$$ > /proc/set_ai_thread || exit 1; exec taskset -c 8 "$$1" "$$2" "$$3" "$$4"' sh "$(AUDIT)" "$(WEIGHTS)" "$(CONTEXT)" "$(REPEAT)"

# New matched control: original ggml-like Q4_0 RVV dot in both paths.
llama-baseline: $(LLAMA)
	sh -c 'echo $$$$ > /proc/set_ai_thread || exit 1; exec taskset -c 8 "$$1" "$$2" "$$3" "$$4"' sh "$(LLAMA)" "$(WEIGHTS)" "$(CONTEXT)" "$(REPEAT)"

# Review compiler-emitted vector saves/spills in the new fused path.
reg-asm: $(COMMON) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -S -o "$(BUILD_DIR)/register_fusion.s" qwen_one_layer.c

verify: $(VERIFY)
	sh -c 'echo $$$$ > /proc/set_ai_thread || exit 1; exec taskset -c 8 "$$1" "$$2"' sh "$(VERIFY)" "$(WEIGHTS)"

test: verify

# Doesn't remove or modify your extracted weights, sources, or old root binaries.
clean:
	rm -rf "$(BUILD_DIR)"

# TCM experiment: three 30-KiB Q/KV/W_O banks; normal KV cache always DDR/L2.
# 1 = stream every chunk, 2 = permanently pin one chunk of each family
TCM_TILE_MODE ?= 1
TCM_PIN_PAIR ?= 0
TCM3 = $(BUILD_DIR)/bench_tcm_three_banks

$(TCM3): bench_tcm_three_banks.c tcm_three_banks.h $(COMMON) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o "$@" bench_tcm_three_banks.c $(LDFLAGS)

.PHONY: tcm-three-banks
tcm-three-banks: $(TCM3)
	TCM_BLOCK="$(TCM_BLOCK)" TCM_PIN_PAIR="$(TCM_PIN_PAIR)" sh -c 'echo $$$$ > /proc/set_ai_thread || exit 1; exec taskset -c 8 "$$1" "$$2" "$$3" "$$4" "$$5"' sh "$(TCM3)" "$(WEIGHTS)" "$(CONTEXT)" "$(REPEAT)" "$(TCM_TILE_MODE)"

# Real AI-DMA variant: asynchronous 30KiB Q/KV/W_O weight ping-pong +
# optional 90KiB cross-token pinned resident bank. Does not affect old modes.
# WARNING: vendor AI-DMA driver leaks descriptors per transfer. Limit tests.
AI_DMA_REPEAT ?= 5
AI_DMA_MODE ?= 2
AI_DMA_PIN_PAIR ?= 0
AIDMAF = $(BUILD_DIR)/bench_ai_dma_fusion

$(AIDMAF): bench_ai_dma_fusion.c ai_dma_pipeline.h tcm_three_banks.h $(COMMON) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o "$@" bench_ai_dma_fusion.c $(LDFLAGS)

.PHONY: ai-dma-fusion
ai-dma-fusion: $(AIDMAF)
	TCM_BLOCK="$(TCM_BLOCK)" TCM_PIN_PAIR="$(AI_DMA_PIN_PAIR)" sh -c 'echo $$$$ > /proc/set_ai_thread || exit 1; exec taskset -c 8 "$$1" "$$2" "$$3" "$$4" "$$5"' sh "$(AIDMAF)" "$(WEIGHTS)" "$(CONTEXT)" "$(AI_DMA_REPEAT)" "$(AI_DMA_MODE)"

# Causal weight-stationary PREFILL: one AI-DMA weight transfer per tile per
# prompt batch; tile is applied to ALL prompt tokens before it is replaced.
# This leaves the decode benchmarks untouched. /dev/udma allocation is root
# only on the Firefly, so run the compiled benchmark as root (not `sudo make`).
PREFILL_TOKENS ?= 100
PREFILL_PREFIX ?= 0
PREFILL_REPEAT ?= 1
PREFILL_VERIFY_TILES ?= 0
AIPREF = $(BUILD_DIR)/bench_ai_dma_prefill

$(AIPREF): bench_ai_dma_prefill.c prefill_weight_stationary.h ai_dma_pipeline.h tcm_three_banks.h $(COMMON) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o "$@" bench_ai_dma_prefill.c $(LDFLAGS)

.PHONY: ai-dma-prefill-build ai-dma-prefill
ai-dma-prefill-build: $(AIPREF)

ai-dma-prefill: $(AIPREF)
	sudo sh -c 'echo $$$$ > /proc/set_ai_thread || exit 1; export TCM_BLOCK="$$6" PREFILL_VERIFY_TILES="$$7"; exec taskset -c 8 "$$1" "$$2" "$$3" "$$4" "$$5"' sh "$(AIPREF)" "$(WEIGHTS)" "$(PREFILL_TOKENS)" "$(PREFILL_PREFIX)" "$(PREFILL_REPEAT)" "$(TCM_BLOCK)" "$(PREFILL_VERIFY_TILES)"
