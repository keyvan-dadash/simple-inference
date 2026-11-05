# Single-token Qwen2.5-Coder-1.5B fused benchmark for A100 RVV
CC       = gcc
VECTOR  ?= 1
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra -Wshadow -ffp-contract=off
LDFLAGS ?= -lm
GGUF    ?=
LAYER   ?= 0
WEIGHTS ?= weights
CONTEXT ?= 256
REPEAT  ?= 3
ATTENTION ?= fused-compare
FUSED_TCM ?= 0
TCM_BLOCK ?= 0

ifeq ($(VECTOR),1)
CFLAGS += -march=rv64gcv_zicbop -mabi=lp64d -DUSE_RVV=1
else
CFLAGS += -DUSE_RVV=0
endif

.PHONY: all extract repack run a100 test clean
all: qwen_one_layer

qwen_one_layer: qwen_one_layer.c attention_kernels.h fused_layer.h defines.h
	$(CC) $(CFLAGS) -o $@ qwen_one_layer.c $(LDFLAGS)

extract:
	@test -n "$(GGUF)" || (echo 'Usage: make extract GGUF=/path/to/qwen-q4_0.gguf' >&2; exit 1)
	python3 extract_weights.py "$(GGUF)" --layer "$(LAYER)" --out "$(WEIGHTS)"

repack:
	python3 repack_weights.py --weights "$(WEIGHTS)"

run: qwen_one_layer
	FUSED_TCM="$(FUSED_TCM)" TCM_BLOCK="$(TCM_BLOCK)" ./qwen_one_layer "$(WEIGHTS)" "$(CONTEXT)" "$(REPEAT)" "$(ATTENTION)"

a100: qwen_one_layer
	FUSED_TCM="$(FUSED_TCM)" TCM_BLOCK="$(TCM_BLOCK)" sh -c 'echo $$$$ > /proc/set_ai_thread || exit 1; exec taskset -c 8 ./qwen_one_layer "$$1" "$$2" "$$3" "$$4"' sh "$(WEIGHTS)" "$(CONTEXT)" "$(REPEAT)" "$(ATTENTION)"

test: qwen_one_layer
	python3 test_fused.py

clean:
	rm -f qwen_one_layer
