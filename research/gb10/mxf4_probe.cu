// Standalone probe: does sm_121 (compute_120f-family, GB10) accept and correctly
// execute mma.sync.aligned.kind::mxf4.block_scale... m16n8k64 (MXFP4 tensor-core MMA)?
//
// Strategy: feed uniform "1.0" operands (E2M1 nibble 0x2 = table value 2, *0.5 = 1.0;
// E8M0 scale byte 127 = 2^0 = 1.0) into every A/B register slot, for every lane. Because
// every element is exactly 1.0 regardless of the (undocumented-here) per-lane element
// permutation, D must come out as exactly 64.0f (sum of 64 k-products of 1*1, scale 1*1)
// in every one of the 4 floats each of the 32 lanes holds -- a layout-independent
// correctness check.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void mxf4_probe_kernel(float * out) {
    // Operand registers exactly as ggml's mma_block_scaled_fp4<GGML_TYPE_MXFP4> uses.
    unsigned int Axi[4];
    unsigned int Bxi[2];
    float        Dxi[4] = {0.f, 0.f, 0.f, 0.f};

    // Every nibble = 0x2 (E2M1 table index 2 -> value 2 -> *0.5 = 1.0).
    for (int i = 0; i < 4; ++i) Axi[i] = 0x22222222u;
    for (int i = 0; i < 2; ++i) Bxi[i] = 0x22222222u;

    // E8M0 scale byte 127 -> 2^(127-127) = 1.0. scale_vec::2X consumes TWO scale
    // bytes per operand row (one per k32 sub-block of the k64 tile); pack 127 into
    // every byte of the 32-bit scale register (not just byte 0) so the result is
    // correct regardless of which byte-id/thread-id the hardware actually reads --
    // the original bug left byte 1 (and up) as 0x00 (~2^-127), collapsing the second
    // k32 sub-block's contribution to ~0 and halving D from 64.0 to 32.0.
    unsigned int a_scale = 0x7F7F7F7Fu;
    unsigned int b_scale = 0x7F7F7F7Fu;

    asm volatile(
        "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};"
        : "+f"(Dxi[0]), "+f"(Dxi[1]), "+f"(Dxi[2]), "+f"(Dxi[3])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]),
          "r"(a_scale), "r"(b_scale));

    int lane = threadIdx.x;
    for (int i = 0; i < 4; ++i) out[lane * 4 + i] = Dxi[i];
}

int main() {
    float * d_out;
    cudaError_t e = cudaMalloc(&d_out, 32 * 4 * sizeof(float));
    if (e != cudaSuccess) { fprintf(stderr, "cudaMalloc: %s\n", cudaGetErrorString(e)); return 1; }

    mxf4_probe_kernel<<<1, 32>>>(d_out);
    e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "launch: %s\n", cudaGetErrorString(e)); return 1; }
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { fprintf(stderr, "sync: %s\n", cudaGetErrorString(e)); return 1; }

    float h_out[32 * 4];
    cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);

    int nonzero = 0, correct = 0, bad = 0;
    for (int i = 0; i < 32 * 4; ++i) {
        if (h_out[i] != 0.f) nonzero++;
        if (h_out[i] == 64.f) correct++;
        else bad++;
    }
    printf("nonzero=%d/128 correct(==64.0)=%d/128 bad=%d\n", nonzero, correct, bad);
    printf("sample D[0..3] (lane0)=%.3f %.3f %.3f %.3f\n", h_out[0], h_out[1], h_out[2], h_out[3]);
    printf("sample D lane16 =%.3f %.3f %.3f %.3f\n", h_out[64], h_out[65], h_out[66], h_out[67]);

    if (correct == 128) {
        printf("PROBE VERDICT: PASS -- mxf4 mma.sync produced correct result on sm_121\n");
        return 0;
    } else {
        printf("PROBE VERDICT: FAIL -- incorrect numerical result\n");
        return 2;
    }
}
