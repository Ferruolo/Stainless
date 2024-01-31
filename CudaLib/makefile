CC = nvcc
CFLAGS = -std=c++17 -arch=sm_75
LDFLAGS = --shared
XCOMPILER_FLAGS = -Xcompiler="-fPIC"

SRC = library.cu
PROFILE_SRC = Profile.cu
LIB_TARGET = libCudaLib.so
DEBUG_TARGET = cudaProfiler_debug
MEMCHECK_TARGET = cudaProfiler_memcheck
PROFILE_TARGET = cudaProfiler_profile

.PHONY: all debug release memcheck profile clean

all: release

debug: CFLAGS += -g -G
debug: $(DEBUG_TARGET)

release: CFLAGS += -O3
release: $(LIB_TARGET)

memcheck: CFLAGS += -g -G
memcheck: LDFLAGS += -lineinfo
memcheck: $(MEMCHECK_TARGET)

profile: CFLAGS += -g -G
profile: LDFLAGS += -lineinfo
profile: SRC = $(PROFILE_SRC)
profile: $(PROFILE_TARGET)

$(LIB_TARGET): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) $(XCOMPILER_FLAGS) -o $@ $<

$(DEBUG_TARGET): $(PROFILE_SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) $(XCOMPILER_FLAGS) -o $@ $<

$(MEMCHECK_TARGET): $(PROFILE_SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) $(XCOMPILER_FLAGS) -o $@ $<

$(PROFILE_TARGET): $(PROFILE_SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) $(XCOMPILER_FLAGS) -o $@ $<

clean:
	rm -f $(LIB_TARGET) $(DEBUG_TARGET) $(MEMCHECK_TARGET) $(PROFILE_TARGET)
