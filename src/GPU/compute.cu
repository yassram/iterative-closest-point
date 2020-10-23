#include "compute.hh"

__global__ void test() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

void call_test(){
    test<<<10, 10>>>();
    cudaDeviceSynchronize();
}
