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
# (B) old column-SoA RVV staged, (C) new register-fused Q->attn->W_O.
# This reuses the independent, wall-clock-tested audit harness.
three-way: audit

# Individual modes use the same qwen_one_layer driver for stable inputs.
reference: ATTENTION = reference
reference: a100
soa-staged: ATTENTION = soa-staged
soa-staged: a100
soa-fused: ATTENTION = soa-fused
soa-fused: a100

# NOTE: fused core changes Zvfh, exp approximation, and W_O; not a pure schedule comparison.
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
