// main.cu
// CUDA Canny Edge Detection Project
// CPU baseline + Dense CUDA + Optimized CUDA + Sparse CUDA with stream compaction
// Input/output format: binary PGM (P5) grayscale images.
// Compile: make
// Run examples:
//   ./canny input.pgm out_cpu.pgm --mode cpu
//   ./canny input.pgm out_dense.pgm --mode dense
//   ./canny input.pgm out_opt.pgm --mode opt
//   ./canny input.pgm out_sparse.pgm --mode sparse
//   ./canny input.pgm out_sparse.pgm --mode sparse --low 40 --high 100 --iters 4 --repeat 10

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/fill.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t err__ = (call);                                                  \
    if (err__ != cudaSuccess) {                                                  \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__            \
                  << " -> " << cudaGetErrorString(err__) << std::endl;          \
        std::exit(EXIT_FAILURE);                                                 \
    }                                                                            \
} while (0)

static constexpr unsigned char NON_EDGE = 0;
static constexpr unsigned char WEAK_EDGE = 128;
static constexpr unsigned char STRONG_EDGE = 255;

// 5-tap separable Gaussian roughly matching sigma around 1.0: [1 4 6 4 1] / 16
__constant__ float c_gauss5[5];
__constant__ int c_sobelX[9];
__constant__ int c_sobelY[9];

struct ImageU8 {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> data;
};

struct Timings {
    float h2d = 0.0f;
    float gaussian = 0.0f;
    float sobel = 0.0f;
    float nms = 0.0f;
    float threshold = 0.0f;
    float scan = 0.0f;
    float scatter = 0.0f;
    float hysteresis = 0.0f;
    float d2h = 0.0f;
    float total = 0.0f;
    int candidates = 0;
};

static inline int idx2(int x, int y, int w) { return y * w + x; }
static inline int clampi(int v, int lo, int hi) { return std::max(lo, std::min(v, hi)); }

static bool readTokenSkippingComments(std::istream& in, std::string& token) {
    token.clear();
    while (in >> token) {
        if (!token.empty() && token[0] == '#') {
            std::string dummy;
            std::getline(in, dummy);
            continue;
        }
        return true;
    }
    return false;
}

ImageU8 readPGM(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open input image: " + path);
    }

    std::string magic, sw, sh, smax;
    if (!readTokenSkippingComments(in, magic) || magic != "P5") {
        throw std::runtime_error("Only binary PGM P5 input is supported. Convert your image to .pgm first.");
    }
    if (!readTokenSkippingComments(in, sw) || !readTokenSkippingComments(in, sh) || !readTokenSkippingComments(in, smax)) {
        throw std::runtime_error("Invalid PGM header.");
    }

    ImageU8 img;
    img.width = std::stoi(sw);
    img.height = std::stoi(sh);
    int maxv = std::stoi(smax);
    if (img.width <= 0 || img.height <= 0 || maxv != 255) {
        throw std::runtime_error("PGM must have positive dimensions and max value 255.");
    }

    in.get(); // consume one whitespace byte after max value
    img.data.resize((size_t)img.width * img.height);
    in.read(reinterpret_cast<char*>(img.data.data()), img.data.size());
    if (!in) {
        throw std::runtime_error("Could not read complete PGM pixel data.");
    }
    return img;
}

void writePGM(const std::string& path, const ImageU8& img) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Could not open output image: " + path);
    }
    out << "P5\n" << img.width << " " << img.height << "\n255\n";
    out.write(reinterpret_cast<const char*>(img.data.data()), img.data.size());
}

// ---------------------------- CPU reference ----------------------------

void cpuGaussianSeparable(const unsigned char* input, float* tmp, float* blur, int w, int h) {
    const float g[5] = {1.f/16.f, 4.f/16.f, 6.f/16.f, 4.f/16.f, 1.f/16.f};
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float sum = 0.0f;
            for (int k = -2; k <= 2; ++k) {
                int xx = clampi(x + k, 0, w - 1);
                sum += g[k + 2] * input[idx2(xx, y, w)];
            }
            tmp[idx2(x, y, w)] = sum;
        }
    }
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float sum = 0.0f;
            for (int k = -2; k <= 2; ++k) {
                int yy = clampi(y + k, 0, h - 1);
                sum += g[k + 2] * tmp[idx2(x, yy, w)];
            }
            blur[idx2(x, y, w)] = sum;
        }
    }
}

void cpuSobel(const float* blur, float* mag, unsigned char* dir, int w, int h) {
    const int sx[9] = {-1,0,1,-2,0,2,-1,0,1};
    const int sy[9] = {-1,-2,-1,0,0,0,1,2,1};

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (x == 0 || y == 0 || x == w - 1 || y == h - 1) {
                mag[idx2(x,y,w)] = 0.0f;
                dir[idx2(x,y,w)] = 0;
                continue;
            }
            float gx = 0.0f, gy = 0.0f;
            int p = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    float v = blur[idx2(x + dx, y + dy, w)];
                    gx += sx[p] * v;
                    gy += sy[p] * v;
                    ++p;
                }
            }
            float m = std::sqrt(gx * gx + gy * gy);
            float angle = std::atan2(gy, gx) * 180.0f / 3.14159265f;
            if (angle < 0) angle += 180.0f;
            unsigned char qdir;
            if ((angle >= 0 && angle < 22.5f) || (angle >= 157.5f && angle <= 180.0f)) qdir = 0;
            else if (angle >= 22.5f && angle < 67.5f) qdir = 1;
            else if (angle >= 67.5f && angle < 112.5f) qdir = 2;
            else qdir = 3;
            mag[idx2(x,y,w)] = m;
            dir[idx2(x,y,w)] = qdir;
        }
    }
}

void cpuNMS(const float* mag, const unsigned char* dir, float* nms, int w, int h) {
    std::fill(nms, nms + (size_t)w * h, 0.0f);
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            int id = idx2(x,y,w);
            float m = mag[id];
            float a = 0.0f, b = 0.0f;
            switch (dir[id]) {
                case 0: a = mag[idx2(x-1,y,w)]; b = mag[idx2(x+1,y,w)]; break;
                case 1: a = mag[idx2(x-1,y-1,w)]; b = mag[idx2(x+1,y+1,w)]; break;
                case 2: a = mag[idx2(x,y-1,w)]; b = mag[idx2(x,y+1,w)]; break;
                case 3: a = mag[idx2(x+1,y-1,w)]; b = mag[idx2(x-1,y+1,w)]; break;
            }
            nms[id] = (m >= a && m >= b) ? m : 0.0f;
        }
    }
}

void cpuThreshold(const float* nms, unsigned char* thr, int w, int h, float low, float high) {
    for (int i = 0; i < w * h; ++i) {
        float v = nms[i];
        if (v >= high) thr[i] = STRONG_EDGE;
        else if (v >= low) thr[i] = WEAK_EDGE;
        else thr[i] = NON_EDGE;
    }
}

void cpuHysteresis(const unsigned char* thr, unsigned char* out, int w, int h, int iterations) {
    std::vector<unsigned char> cur(thr, thr + (size_t)w * h);
    std::vector<unsigned char> next = cur;

    for (int it = 0; it < iterations; ++it) {
        next = cur;
        for (int y = 1; y < h - 1; ++y) {
            for (int x = 1; x < w - 1; ++x) {
                int id = idx2(x,y,w);
                if (cur[id] != WEAK_EDGE) continue;
                bool connected = false;
                for (int dy = -1; dy <= 1 && !connected; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        if (cur[idx2(x+dx,y+dy,w)] == STRONG_EDGE) {
                            connected = true;
                            break;
                        }
                    }
                }
                if (connected) next[id] = STRONG_EDGE;
            }
        }
        cur.swap(next);
    }

    for (int i = 0; i < w * h; ++i) {
        out[i] = (cur[i] == STRONG_EDGE) ? 255 : 0;
    }
}

ImageU8 runCPU(const ImageU8& input, float low, float high, int hysteresisIters, double& cpuMs) {
    int w = input.width, h = input.height;
    size_t n = (size_t)w * h;
    std::vector<float> tmp(n), blur(n), mag(n), nms(n);
    std::vector<unsigned char> dir(n), thr(n), out(n);

    auto t0 = std::chrono::high_resolution_clock::now();
    cpuGaussianSeparable(input.data.data(), tmp.data(), blur.data(), w, h);
    cpuSobel(blur.data(), mag.data(), dir.data(), w, h);
    cpuNMS(mag.data(), dir.data(), nms.data(), w, h);
    cpuThreshold(nms.data(), thr.data(), w, h, low, high);
    cpuHysteresis(thr.data(), out.data(), w, h, hysteresisIters);
    auto t1 = std::chrono::high_resolution_clock::now();
    cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    ImageU8 result;
    result.width = w;
    result.height = h;
    result.data = std::move(out);
    return result;
}

// ---------------------------- CUDA kernels ----------------------------

__global__ void gaussian2DNaiveKernel(const unsigned char* in, float* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const float g2[25] = {
        1.f/256, 4.f/256, 6.f/256, 4.f/256, 1.f/256,
        4.f/256,16.f/256,24.f/256,16.f/256,4.f/256,
        6.f/256,24.f/256,36.f/256,24.f/256,6.f/256,
        4.f/256,16.f/256,24.f/256,16.f/256,4.f/256,
        1.f/256, 4.f/256, 6.f/256, 4.f/256, 1.f/256
    };
    float sum = 0.0f;
    for (int dy = -2; dy <= 2; ++dy) {
        int yy = min(max(y + dy, 0), h - 1);
        for (int dx = -2; dx <= 2; ++dx) {
            int xx = min(max(x + dx, 0), w - 1);
            sum += g2[(dy+2)*5 + (dx+2)] * in[yy*w + xx];
        }
    }
    out[y*w + x] = sum;
}

__global__ void gaussianHorizontalKernel(const unsigned char* in, float* tmp, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float sum = 0.0f;
    for (int k = -2; k <= 2; ++k) {
        int xx = min(max(x + k, 0), w - 1);
        sum += c_gauss5[k + 2] * in[y*w + xx];
    }
    tmp[y*w + x] = sum;
}

__global__ void gaussianVerticalKernel(const float* tmp, float* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float sum = 0.0f;
    for (int k = -2; k <= 2; ++k) {
        int yy = min(max(y + k, 0), h - 1);
        sum += c_gauss5[k + 2] * tmp[yy*w + x];
    }
    out[y*w + x] = sum;
}

__global__ void sobelKernel(const float* blur, float* mag, unsigned char* dir, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int id = y*w + x;
    if (x == 0 || y == 0 || x == w - 1 || y == h - 1) {
        mag[id] = 0.0f;
        dir[id] = 0;
        return;
    }
    float gx = 0.0f, gy = 0.0f;
    int p = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            float v = blur[(y+dy)*w + (x+dx)];
            gx += c_sobelX[p] * v;
            gy += c_sobelY[p] * v;
            ++p;
        }
    }
    float m = sqrtf(gx*gx + gy*gy);
    float angle = atan2f(gy, gx) * 180.0f / 3.14159265f;
    if (angle < 0) angle += 180.0f;
    unsigned char qdir;
    if ((angle >= 0 && angle < 22.5f) || (angle >= 157.5f && angle <= 180.0f)) qdir = 0;
    else if (angle >= 22.5f && angle < 67.5f) qdir = 1;
    else if (angle >= 67.5f && angle < 112.5f) qdir = 2;
    else qdir = 3;
    mag[id] = m;
    dir[id] = qdir;
}

__global__ void nmsKernel(const float* mag, const unsigned char* dir, float* nms, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int id = y*w + x;
    if (x == 0 || y == 0 || x == w - 1 || y == h - 1) {
        nms[id] = 0.0f;
        return;
    }
    float m = mag[id];
    float a = 0.0f, b = 0.0f;
    switch (dir[id]) {
        case 0: a = mag[y*w + (x-1)]; b = mag[y*w + (x+1)]; break;
        case 1: a = mag[(y-1)*w + (x-1)]; b = mag[(y+1)*w + (x+1)]; break;
        case 2: a = mag[(y-1)*w + x]; b = mag[(y+1)*w + x]; break;
        case 3: a = mag[(y-1)*w + (x+1)]; b = mag[(y+1)*w + (x-1)]; break;
    }
    nms[id] = (m >= a && m >= b) ? m : 0.0f;
}

__global__ void thresholdKernel(const float* nms, unsigned char* thr, int* mask, int n, float low, float high) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = nms[i];
    unsigned char t = NON_EDGE;
    if (v >= high) t = STRONG_EDGE;
    else if (v >= low) t = WEAK_EDGE;
    thr[i] = t;
    if (mask) mask[i] = (t != NON_EDGE) ? 1 : 0;
}

__global__ void hysteresisDenseKernel(const unsigned char* in, unsigned char* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int id = y*w + x;
    unsigned char v = in[id];
    if (v == STRONG_EDGE) {
        out[id] = STRONG_EDGE;
        return;
    }
    if (v != WEAK_EDGE || x == 0 || y == 0 || x == w - 1 || y == h - 1) {
        out[id] = v;
        return;
    }
    bool connected = false;
    for (int dy = -1; dy <= 1 && !connected; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            if (in[(y+dy)*w + (x+dx)] == STRONG_EDGE) {
                connected = true;
                break;
            }
        }
    }
    out[id] = connected ? STRONG_EDGE : WEAK_EDGE;
}

__global__ void finalizeEdgesKernel(const unsigned char* in, unsigned char* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = (in[i] == STRONG_EDGE) ? 255 : 0;
}

__global__ void scatterCandidatesKernel(const int* mask, const int* scan, int* candidates, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (mask[i]) {
        int pos = scan[i];
        candidates[pos] = i;
    }
}

__global__ void hysteresisSparseKernel(const unsigned char* in, unsigned char* out, const int* candidates, int numCandidates, int w, int h) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numCandidates) return;
    int id = candidates[tid];
    int x = id % w;
    int y = id / w;
    unsigned char v = in[id];
    if (v == STRONG_EDGE) {
        out[id] = STRONG_EDGE;
        return;
    }
    if (v != WEAK_EDGE || x == 0 || y == 0 || x == w - 1 || y == h - 1) {
        out[id] = v;
        return;
    }
    bool connected = false;
    for (int dy = -1; dy <= 1 && !connected; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            if (in[(y+dy)*w + (x+dx)] == STRONG_EDGE) {
                connected = true;
                break;
            }
        }
    }
    out[id] = connected ? STRONG_EDGE : WEAK_EDGE;
}

// ---------------------------- GPU driver ----------------------------

float timeEventMs(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms;
}

enum class Mode { Dense, Opt, Sparse };

ImageU8 runGPU(const ImageU8& input, Mode mode, float low, float high, int hysteresisIters, Timings& t) {
    int w = input.width, h = input.height;
    int n = w * h;
    size_t nU8 = (size_t)n * sizeof(unsigned char);
    size_t nF = (size_t)n * sizeof(float);
    size_t nI = (size_t)n * sizeof(int);

    cudaEvent_t e0,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2)); CUDA_CHECK(cudaEventCreate(&e3));
    CUDA_CHECK(cudaEventCreate(&e4)); CUDA_CHECK(cudaEventCreate(&e5));
    CUDA_CHECK(cudaEventCreate(&e6)); CUDA_CHECK(cudaEventCreate(&e7));
    CUDA_CHECK(cudaEventCreate(&e8)); CUDA_CHECK(cudaEventCreate(&e9));
    CUDA_CHECK(cudaEventCreate(&e10)); CUDA_CHECK(cudaEventCreate(&e11)); CUDA_CHECK(cudaEventCreate(&e12));

    unsigned char *d_in=nullptr, *d_dir=nullptr, *d_thr=nullptr, *d_hystA=nullptr, *d_hystB=nullptr, *d_out=nullptr;
    float *d_tmp=nullptr, *d_blur=nullptr, *d_mag=nullptr, *d_nms=nullptr;
    int *d_mask=nullptr, *d_scan=nullptr, *d_candidates=nullptr;

    CUDA_CHECK(cudaMalloc(&d_in, nU8));
    CUDA_CHECK(cudaMalloc(&d_tmp, nF));
    CUDA_CHECK(cudaMalloc(&d_blur, nF));
    CUDA_CHECK(cudaMalloc(&d_mag, nF));
    CUDA_CHECK(cudaMalloc(&d_dir, nU8));
    CUDA_CHECK(cudaMalloc(&d_nms, nF));
    CUDA_CHECK(cudaMalloc(&d_thr, nU8));
    CUDA_CHECK(cudaMalloc(&d_hystA, nU8));
    CUDA_CHECK(cudaMalloc(&d_hystB, nU8));
    CUDA_CHECK(cudaMalloc(&d_out, nU8));

    if (mode == Mode::Sparse) {
        CUDA_CHECK(cudaMalloc(&d_mask, nI));
        CUDA_CHECK(cudaMalloc(&d_scan, nI));
        CUDA_CHECK(cudaMalloc(&d_candidates, nI));
    }

    dim3 block2(16,16);
    dim3 grid2((w + block2.x - 1) / block2.x, (h + block2.y - 1) / block2.y);
    int block1 = 256;
    int grid1 = (n + block1 - 1) / block1;

    CUDA_CHECK(cudaEventRecord(e0));
    CUDA_CHECK(cudaMemcpy(d_in, input.data.data(), nU8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e1));

    if (mode == Mode::Dense) {
        gaussian2DNaiveKernel<<<grid2, block2>>>(d_in, d_blur, w, h);
    } else {
        gaussianHorizontalKernel<<<grid2, block2>>>(d_in, d_tmp, w, h);
        gaussianVerticalKernel<<<grid2, block2>>>(d_tmp, d_blur, w, h);
    }
    CUDA_CHECK(cudaEventRecord(e2));

    sobelKernel<<<grid2, block2>>>(d_blur, d_mag, d_dir, w, h);
    CUDA_CHECK(cudaEventRecord(e3));

    nmsKernel<<<grid2, block2>>>(d_mag, d_dir, d_nms, w, h);
    CUDA_CHECK(cudaEventRecord(e4));

    thresholdKernel<<<grid1, block1>>>(d_nms, d_thr, d_mask, n, low, high);
    CUDA_CHECK(cudaEventRecord(e5));

    if (mode == Mode::Sparse) {
        thrust::device_ptr<int> maskPtr(d_mask);
        thrust::device_ptr<int> scanPtr(d_scan);
        thrust::exclusive_scan(maskPtr, maskPtr + n, scanPtr);
        int lastMask = 0, lastScan = 0;
        CUDA_CHECK(cudaMemcpy(&lastMask, d_mask + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&lastScan, d_scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
        t.candidates = lastMask + lastScan;
        CUDA_CHECK(cudaEventRecord(e6));

        scatterCandidatesKernel<<<grid1, block1>>>(d_mask, d_scan, d_candidates, n);
        CUDA_CHECK(cudaEventRecord(e7));

        CUDA_CHECK(cudaMemcpy(d_hystA, d_thr, nU8, cudaMemcpyDeviceToDevice));
        for (int it = 0; it < hysteresisIters; ++it) {
            CUDA_CHECK(cudaMemcpy(d_hystB, d_hystA, nU8, cudaMemcpyDeviceToDevice));
            int sparseGrid = (std::max(t.candidates, 1) + block1 - 1) / block1;
            hysteresisSparseKernel<<<sparseGrid, block1>>>(d_hystA, d_hystB, d_candidates, t.candidates, w, h);
            std::swap(d_hystA, d_hystB);
        }
        CUDA_CHECK(cudaEventRecord(e8));
    } else {
        CUDA_CHECK(cudaMemcpy(d_hystA, d_thr, nU8, cudaMemcpyDeviceToDevice));
        for (int it = 0; it < hysteresisIters; ++it) {
            hysteresisDenseKernel<<<grid2, block2>>>(d_hystA, d_hystB, w, h);
            std::swap(d_hystA, d_hystB);
        }
        CUDA_CHECK(cudaEventRecord(e8));
    }

    finalizeEdgesKernel<<<grid1, block1>>>(d_hystA, d_out, n);
    CUDA_CHECK(cudaEventRecord(e9));

    ImageU8 result;
    result.width = w;
    result.height = h;
    result.data.resize(n);
    CUDA_CHECK(cudaMemcpy(result.data.data(), d_out, nU8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(e10));
    CUDA_CHECK(cudaEventSynchronize(e10));

    t.h2d = timeEventMs(e0, e1);
    t.gaussian = timeEventMs(e1, e2);
    t.sobel = timeEventMs(e2, e3);
    t.nms = timeEventMs(e3, e4);
    t.threshold = timeEventMs(e4, e5);
    if (mode == Mode::Sparse) {
        t.scan = timeEventMs(e5, e6);
        t.scatter = timeEventMs(e6, e7);
        t.hysteresis = timeEventMs(e7, e8);
    } else {
        t.hysteresis = timeEventMs(e5, e8);
    }
    t.d2h = timeEventMs(e9, e10);
    t.total = timeEventMs(e0, e10);

    CUDA_CHECK(cudaGetLastError());

    cudaFree(d_in); cudaFree(d_tmp); cudaFree(d_blur); cudaFree(d_mag); cudaFree(d_dir);
    cudaFree(d_nms); cudaFree(d_thr); cudaFree(d_hystA); cudaFree(d_hystB); cudaFree(d_out);
    if (d_mask) cudaFree(d_mask);
    if (d_scan) cudaFree(d_scan);
    if (d_candidates) cudaFree(d_candidates);

    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
    cudaEventDestroy(e4); cudaEventDestroy(e5); cudaEventDestroy(e6); cudaEventDestroy(e7);
    cudaEventDestroy(e8); cudaEventDestroy(e9); cudaEventDestroy(e10); cudaEventDestroy(e11); cudaEventDestroy(e12);

    return result;
}

void uploadConstants() {
    float h_g[5] = {1.f/16.f, 4.f/16.f, 6.f/16.f, 4.f/16.f, 1.f/16.f};
    int h_sx[9] = {-1,0,1,-2,0,2,-1,0,1};
    int h_sy[9] = {-1,-2,-1,0,0,0,1,2,1};
    CUDA_CHECK(cudaMemcpyToSymbol(c_gauss5, h_g, sizeof(h_g)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_sobelX, h_sx, sizeof(h_sx)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_sobelY, h_sy, sizeof(h_sy)));
}

struct Args {
    std::string input;
    std::string output;
    std::string mode = "sparse";
    float low = 50.0f;
    float high = 100.0f;
    int iters = 4;
    int repeat = 1;
    bool compareCPU = false;
};

void printUsage() {
    std::cerr << "Usage:\n"
              << "  ./canny input.pgm output.pgm --mode cpu|dense|opt|sparse [--low 50] [--high 100] [--iters 4] [--repeat 1] [--compare-cpu]\n";
}

Args parseArgs(int argc, char** argv) {
    if (argc < 3) {
        printUsage();
        std::exit(EXIT_FAILURE);
    }
    Args a;
    a.input = argv[1];
    a.output = argv[2];
    for (int i = 3; i < argc; ++i) {
        std::string s = argv[i];
        auto need = [&](const char* name) {
            if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value after ") + name);
            return std::string(argv[++i]);
        };
        if (s == "--mode") a.mode = need("--mode");
        else if (s == "--low") a.low = std::stof(need("--low"));
        else if (s == "--high") a.high = std::stof(need("--high"));
        else if (s == "--iters") a.iters = std::stoi(need("--iters"));
        else if (s == "--repeat") a.repeat = std::stoi(need("--repeat"));
        else if (s == "--compare-cpu") a.compareCPU = true;
        else throw std::runtime_error("Unknown argument: " + s);
    }
    if (a.low < 0 || a.high < 0 || a.low > a.high) throw std::runtime_error("Require 0 <= low <= high.");
    if (a.iters < 1) throw std::runtime_error("--iters must be >= 1.");
    if (a.repeat < 1) throw std::runtime_error("--repeat must be >= 1.");
    return a;
}

void printTimings(const Timings& t, int repeat) {
    std::cout << "\nAverage GPU timings over " << repeat << " run(s), milliseconds:\n";
    std::cout << "  H2D transfer : " << t.h2d << "\n";
    std::cout << "  Gaussian     : " << t.gaussian << "\n";
    std::cout << "  Sobel        : " << t.sobel << "\n";
    std::cout << "  NMS          : " << t.nms << "\n";
    std::cout << "  Threshold    : " << t.threshold << "\n";
    if (t.scan > 0 || t.scatter > 0) {
        std::cout << "  Scan         : " << t.scan << "\n";
        std::cout << "  Scatter      : " << t.scatter << "\n";
        std::cout << "  Candidates   : " << t.candidates << "\n";
    }
    std::cout << "  Hysteresis   : " << t.hysteresis << "\n";
    std::cout << "  D2H transfer : " << t.d2h << "\n";
    std::cout << "  Total        : " << t.total << "\n";
}

void compareImages(const ImageU8& a, const ImageU8& b) {
    if (a.width != b.width || a.height != b.height || a.data.size() != b.data.size()) {
        std::cout << "Cannot compare: different dimensions.\n";
        return;
    }
    long long diff = 0;
    long long absSum = 0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        int d = std::abs((int)a.data[i] - (int)b.data[i]);
        if (d != 0) diff++;
        absSum += d;
    }
    double total = (double)a.data.size();
    double match = 100.0 * (1.0 - diff / total);
    double mae = absSum / total;
    std::cout << "\nCPU/GPU correctness comparison:\n";
    std::cout << "  Different pixels : " << diff << " / " << a.data.size() << "\n";
    std::cout << "  Match rate       : " << match << "%\n";
    std::cout << "  Mean abs error   : " << mae << "\n";
}

int main(int argc, char** argv) {
    try {
        Args args = parseArgs(argc, argv);
        ImageU8 input = readPGM(args.input);
        std::cout << "Loaded " << args.input << " (" << input.width << " x " << input.height << ")\n";
        std::cout << "Mode=" << args.mode << ", low=" << args.low << ", high=" << args.high
                  << ", hysteresis iterations=" << args.iters << "\n";

        if (args.mode == "cpu") {
            double ms = 0.0;
            ImageU8 out = runCPU(input, args.low, args.high, args.iters, ms);
            writePGM(args.output, out);
            std::cout << "CPU time: " << ms << " ms\n";
            std::cout << "Wrote " << args.output << "\n";
            return 0;
        }

        uploadConstants();
        Mode mode;
        if (args.mode == "dense") mode = Mode::Dense;
        else if (args.mode == "opt") mode = Mode::Opt;
        else if (args.mode == "sparse") mode = Mode::Sparse;
        else throw std::runtime_error("Unknown mode. Use cpu, dense, opt, or sparse.");

        // Warmup once to remove first-run CUDA overhead from measurement.
        Timings warm;
        ImageU8 warmOut = runGPU(input, mode, args.low, args.high, args.iters, warm);

        Timings avg;
        ImageU8 out;
        for (int r = 0; r < args.repeat; ++r) {
            Timings cur;
            out = runGPU(input, mode, args.low, args.high, args.iters, cur);
            avg.h2d += cur.h2d;
            avg.gaussian += cur.gaussian;
            avg.sobel += cur.sobel;
            avg.nms += cur.nms;
            avg.threshold += cur.threshold;
            avg.scan += cur.scan;
            avg.scatter += cur.scatter;
            avg.hysteresis += cur.hysteresis;
            avg.d2h += cur.d2h;
            avg.total += cur.total;
            avg.candidates = cur.candidates;
        }
        float inv = 1.0f / args.repeat;
        avg.h2d *= inv; avg.gaussian *= inv; avg.sobel *= inv; avg.nms *= inv;
        avg.threshold *= inv; avg.scan *= inv; avg.scatter *= inv; avg.hysteresis *= inv;
        avg.d2h *= inv; avg.total *= inv;

        writePGM(args.output, out);
        printTimings(avg, args.repeat);
        std::cout << "Wrote " << args.output << "\n";

        if (args.compareCPU) {
            double cpuMs = 0.0;
            ImageU8 cpuOut = runCPU(input, args.low, args.high, args.iters, cpuMs);
            std::cout << "CPU reference time: " << cpuMs << " ms\n";
            std::cout << "Speedup vs CPU total: " << (cpuMs / avg.total) << "x\n";
            compareImages(cpuOut, out);
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        printUsage();
        return EXIT_FAILURE;
    }
}
