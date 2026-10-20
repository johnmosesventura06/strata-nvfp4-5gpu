// q8mma_bench: the tensor-core Q8_0 GEMV (STRATA_MMA_Q8) against the dp4a kernels.
//   q8mma_bench [iters] [cfg]
// per shape: (1) every column of an n-column MMA call bitwise equal to that column called alone (the multi_exact
// contract), (2) the MMA outputs against dp4a's (relative to the output scale), (3) time per call, dp4a vs MMA.
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
using namespace strata::kernels;
static uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}
int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
    const int cfg = argc > 2 ? std::atoi(argv[2]) : 0;
    const int shapes[][2] = {{2560, 2560}, {2560, 10240}, {10240, 2560}, {2560, 6144}, {6144, 2560}, {2560, 12288},
                             {2560, 640}, {2560, 512}, {4096, 2560}};
    cudaStream_t s; cudaStreamCreate(&s);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    long long bad_cols = 0;
    double worst_rel = 0;
    for (auto& sh : shapes) {
        const int n_in = sh[0], n_out = sh[1];
        std::mt19937 rng(77u + n_in * 3u + n_out);
        const size_t nblk = (size_t) n_out * (n_in / 32);
        std::vector<uint8_t> w(nblk * 34);
        for (auto& b : w) b = (uint8_t) (rng() & 0xff);
        for (size_t k = 0; k < nblk; ++k) { const uint16_t h = sane_half(rng); std::memcpy(&w[k * 34], &h, 2); }
        std::vector<float> x((size_t) 8 * n_in);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : x) v = nd(rng);
        void* dw; float* dx; void* xq; float *y8, *y1, *yr;
        cudaMalloc(&dw, w.size()); cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
        cudaMalloc(&dx, x.size() * 4); cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        const size_t qb = native_q8_1_bytes(n_in, 1);
        cudaMalloc(&xq, qb * 8);
        native_quantize_q8_1(dx, xq, n_in, 8, s);
        cudaMalloc(&y8, (size_t) 8 * n_out * 4); cudaMalloc(&y1, (size_t) 8 * n_out * 4); cudaMalloc(&yr, (size_t) 8 * n_out * 4);
        std::printf("n_in %5d n_out %5d |", n_in, n_out);
        for (int nc : {1, 2, 4, 6, 8}) {
            // dp4a reference and timing
            native_q8_0_set_mma(false, cfg);
            native_q8_0_mmvq(dw, xq, yr, n_in, n_out, nc, s);
            cudaEventRecord(e0, s);
            for (int i = 0; i < iters; ++i) native_q8_0_mmvq(dw, xq, yr, n_in, n_out, nc, s);
            cudaEventRecord(e1, s); cudaEventSynchronize(e1);
            float t_old; cudaEventElapsedTime(&t_old, e0, e1);
            native_q8_0_set_mma(true, cfg);
            native_q8_0_mmvq(dw, xq, y8, n_in, n_out, nc, s);
            cudaEventRecord(e0, s);
            for (int i = 0; i < iters; ++i) native_q8_0_mmvq(dw, xq, y8, n_in, n_out, nc, s);
            cudaEventRecord(e1, s); cudaEventSynchronize(e1);
            float t_new; cudaEventElapsedTime(&t_new, e0, e1);
            // every column alone
            for (int j = 0; j < nc; ++j)
                native_q8_0_mmvq((const void*) dw, (const char*) xq + qb * j, y1 + (size_t) j * n_out, n_in, n_out, 1, s);
            cudaStreamSynchronize(s);
            std::vector<float> a((size_t) nc * n_out), b((size_t) nc * n_out), r((size_t) nc * n_out);
            cudaMemcpy(a.data(), y8, a.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(b.data(), y1, b.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(r.data(), yr, r.size() * 4, cudaMemcpyDeviceToHost);
            if (std::memcmp(a.data(), b.data(), a.size() * 4) != 0) ++bad_cols;
            double mx = 0, md = 0;
            for (size_t i = 0; i < a.size(); ++i) { mx = std::fmax(mx, std::fabs(r[i])); md = std::fmax(md, std::fabs((double) a[i] - r[i])); if (!std::isfinite(a[i])) md = 1e30; }
            worst_rel = std::fmax(worst_rel, md / mx);
            std::printf(" c%d %5.1f->%5.1f us%s", nc, 1000 * t_old / iters, 1000 * t_new / iters,
                        std::memcmp(a.data(), b.data(), a.size() * 4) ? "!" : "");
        }
        std::printf("\n");
        cudaFree(dw); cudaFree(dx); cudaFree(xq); cudaFree(y8); cudaFree(y1); cudaFree(yr);
    }
    std::printf("column-independence failures: %lld; worst |mma - dp4a| / max|y|: %.3e\n", bad_cols, worst_rel);
    return bad_cols ? 1 : 0;
}
