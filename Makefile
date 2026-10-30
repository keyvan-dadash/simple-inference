# On A100: make ; make extract GGUF=/path/model-q4_0.gguf ; make run
# On non-RISC-V machines for functional testing: make VECTOR=0
CC       = gcc
VECTOR  ?= 1
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra -Wshadow
LDFLAGS ?= -lm
GGUF    ?=
LAYER   ?= 0
WEIGHTS ?= weights
CONTEXT ?= 256
REPEAT  ?= 1

ifeq ($(VECTOR),1)
CFLAGS += -march=rv64gcv -mabi=lp64d -DUSE_RVV=1
else
CFLAGS += -DUSE_RVV=0
endif

.PHONY: all extract list run a100 clean
all: qwen_one_layer

qwen_one_layer: qwen_one_layer.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

extract:
	@test -n "$(GGUF)" || (echo 'Usage: make extract GGUF=/path/to/qwen-q4_0.gguf' >&2; exit 1)
	python3 extract_weights.py "$(GGUF)" --layer "$(LAYER)" --out "$(WEIGHTS)"

list:
	@test -n "$(GGUF)" || (echo 'Usage: make list GGUF=/path/to/qwen-q4_0.gguf' >&2; exit 1)
	python3 extract_weights.py "$(GGUF)" --layer "$(LAYER)" --list

run: qwen_one_layer
	./qwen_one_layer "$(WEIGHTS)" "$(CONTEXT)" "$(REPEAT)"

a100: qwen_one_layer
	sh -c 'echo $$$$ > /proc/set_ai_thread || exit 1; exec taskset -c 8 ./qwen_one_layer "$$1" "$$2" "$$3"' sh "$(WEIGHTS)" "$(CONTEXT)" "$(REPEAT)"

clean:
	rm -f qwen_one_layer
