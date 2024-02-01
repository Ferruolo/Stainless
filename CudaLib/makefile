# Compiler
NVCC = nvcc

# Flags
NVCCFLAGS = -std=c++11

# Directories
SRC_DIR = .
BUILD_DIR = build
LIB_DIR = lib
BIN_DIR = bin

# Files
LIB_SRCS = $(wildcard $(SRC_DIR)/*.cu)
LIB_HDRS = $(wildcard $(SRC_DIR)/*.cuh) $(wildcard $(SRC_DIR)/*.h)
LIB_OBJS = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(LIB_SRCS))
LIB_NAME = $(LIB_DIR)/liblibrary.so

PROFILE_SRC = profile.cu
PROFILE_OBJ = $(BUILD_DIR)/profile.o
PROFILE_BIN = $(BIN_DIR)/profile

# Libraries
LIBS = -lcublas

# Architecture
ARCH = sm_75

# Targets
.PHONY: all clean release memcheck debug

all: memcheck debug

release: NVCCFLAGS += -O3 -arch $(ARCH) -fPIC
release: $(LIB_NAME)

memcheck debug: NVCCFLAGS += -g -G -arch $(ARCH)
memcheck debug: $(PROFILE_BIN)

$(LIB_NAME): $(LIB_OBJS)
	$(NVCC) -shared -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(LIB_HDRS)
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

$(PROFILE_BIN): $(PROFILE_OBJ) $(LIB_OBJS)
	$(NVCC) -o $@ $^ $(LIBS)

clean:
	rm -rf $(BUILD_DIR)/* $(LIB_NAME) $(PROFILE_BIN)
