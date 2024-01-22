# Makefile for CudaLib

CC = nvcc
CFLAGS = -std=c++17 -arch=sm_75 -O3
LDFLAGS = --shared
XCOMPILER_FLAGS = -Xcompiler="-fPIC"

SRC = library.cu
TARGET = libCudaLib.so

.PHONY: all debug release memcheck clean

all: release

debug: CFLAGS += -g -G
debug: $(TARGET)

release: $(TARGET)

memcheck: CFLAGS += -g -G
memcheck: LDFLAGS += -lineinfo
memcheck: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) $(XCOMPILER_FLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
