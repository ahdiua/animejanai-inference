#include <cstdio>
#include <cuda_runtime.h>

__global__ void saxpy(int n, float a, const float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

int main()
{
    int dev = 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        printf("FAIL: no CUDA device\n");
        return 1;
    }
    printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    const int n = 1 << 20;
    float *x, *y;
    cudaMallocManaged(&x, n * sizeof(float));
    cudaMallocManaged(&y, n * sizeof(float));
    for (int i = 0; i < n; i++) { x[i] = 1.0f; y[i] = 2.0f; }

    saxpy<<<(n + 255) / 256, 256>>>(n, 2.0f, x, y);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        printf("FAIL: kernel launch/sync\n");
        return 1;
    }

    float maxerr = 0.0f;
    for (int i = 0; i < n; i++)
        maxerr = maxerr > fabsf(y[i] - 4.0f) ? maxerr : fabsf(y[i] - 4.0f);
    printf(maxerr == 0.0f ? "SAXPY OK\n" : "FAIL: maxerr=%f\n", maxerr);
    cudaFree(x); cudaFree(y);
    return maxerr == 0.0f ? 0 : 1;
}
