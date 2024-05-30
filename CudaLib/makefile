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
SRC_FILES = $(wildcard $(SRC_DIR)/*.cu)
LIB_HEADERS = $(wildcard $(SRC_DIR)/*.cuh) $(wildcard $(SRC_DIR)/*.h)
LIB_OBJS = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(SRC_FILES))
LIB_NAME = $(LIB_DIR)/libCuda.so

PROFILE_SRC = profile/profile.cu
PROFILE_OBJ = $(BUILD_DIR)/profile.o
PROFILE_BIN = $(BIN_DIR)/profile

# Libraries
LIBS = -lcublas

# Architecture
ARCH = sm_75

# Targets
.PHONY: all clean release memcheck debug

all: memcheck debug

release: NVCCFLAGS += -O3 -arch $(ARCH) -Xcompiler -fPIC
release: $(LIB_NAME)

memcheck debug: NVCCFLAGS += -g -lineinfo -arch $(ARCH)
memcheck debug: $(PROFILE_BIN)

$(LIB_NAME): $(LIB_OBJS)
	$(NVCC) -shared -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(LIB_HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

$(PROFILE_OBJ): $(PROFILE_SRC)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

$(PROFILE_BIN): $(PROFILE_OBJ) $(LIB_OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) -o $@ $^ $(LIBS)

clean:
	rm -rf $(BUILD_DIR) $(LIB_NAME) $(PROFILE_BIN)
