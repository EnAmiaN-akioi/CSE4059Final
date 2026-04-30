NVCC ?= nvcc
TARGET = canny
SRC = main.cu

# Change sm_70 to match your GPU if needed, for example sm_75, sm_80, sm_86, sm_89.
ARCH ?= sm_70

NVCCFLAGS = -O3 -std=c++17 -arch=$(ARCH) -lineinfo

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET) *.o
