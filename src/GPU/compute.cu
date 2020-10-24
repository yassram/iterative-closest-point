#include "compute.hh"

// #define cudaCheckError() {                                              \
//         cudaError_t e=cudaGetLastError();                               \
//         if(e!=cudaSuccess) {                                            \
//             printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
//             exit(EXIT_FAILURE);                                                 \
//         }                                                               \
//     }

namespace GPU {
    void Matrix::fromGpu(double *gpu_rep, unsigned row, unsigned col, size_t pitch) {
        MatrixXd tmp{row, col};
        double *h_d = (double*) std::malloc(sizeof(double) * col * row);

        cudaMemcpy2D(h_d, sizeof(double)*col, gpu_rep, pitch, sizeof(double)*col,
                     row, cudaMemcpyDeviceToHost);

        for(unsigned i = 0; i < row; ++i)
            for (unsigned j = 0; j < col; ++j)
                tmp(i,j) = h_d[col*i + j];
        Matrix new_matrix{tmp};
        *this = new_matrix;
        std::free(h_d);
    }

    double *Matrix::toGpu(size_t *pitch) {
        unsigned r = this->rows();
        unsigned c = this->cols();

        double *d_x;
        cudaMallocPitch((void **) &d_x, pitch, sizeof(double) * c, r);
        // cudaCheckError();

        Matrix tmp {this->transpose()};
        double *h_d = tmp.data();
        cudaMemcpy2D(d_x, *pitch, h_d, c*sizeof(double), sizeof(double)*c,
                     r, cudaMemcpyHostToDevice);
        return (double*)d_x;
    }
}

void computeDim(unsigned width, unsigned height,
                dim3 *block, dim3 *grid) {
    int devId = 0; // There may be more devices!
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, devId);

    // int xMaxBlocks = deviceProp.maxGridSize[0];
    // int yMaxBlocks = deviceProp.maxGridSize[1];

    int xThreads = 32; // deviceProp.maxThreadsDim[0];
    int yThreads = 32; // deviceProp.maxThreadsDim[1];

    // int maxThreadPB = deviceProp.maxThreadsPerBlock;

    *block = dim3(xThreads, yThreads, 1);

    int xBlocks = (int) ceil(width / xThreads);
    int yBlocks = (int) ceil(height / yThreads);

    *grid = dim3(xBlocks, yBlocks, 1);
}


__global__ void compute_distance(double *m, double *pi, double *distance, unsigned int size){
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int bxDim = blockDim.x;
    unsigned int byDim = blockDim.y;

    unsigned int i = (bx+ty)*bxDim + by*byDim + tx;

    if (i >= size)
        return;

    double x, y, z;
    x = pi[0] - m[i];
    y = pi[1] - m[i + size];
    z = pi[2] - m[i + size*2];

    distance[i] = x*x + y*y + z*z;
}

__global__ void find_min_distance(double *distance, int *minIdx,  unsigned int size) {
    *minIdx = 0;
    for (unsigned i = 1; i < size; i++)
        if (distance[*minIdx] > distance[i])
            *minIdx = i;
 }

int compute_distance_w(GPU::Matrix m, GPU::Matrix pi){
    size_t m_p, pi_p;
    double *m_gpu = m.toGpu(&m_p);
    double *pi_gpu = pi.toGpu(&pi_p);

    dim3 distBlk, distGrd;
    computeDim(m.cols(), 1, &distBlk, &distGrd);

    double *distance;
    cudaMalloc((void **) &distance, sizeof(double)*m.cols());
    compute_distance<<<distGrd, distBlk>>>(m_gpu, pi_gpu, distance, m.cols());
    cudaDeviceSynchronize();

    cudaFree(m_gpu);
    cudaFree(pi_gpu);

    int *minIdx;
    cudaMalloc((void **) &minIdx, sizeof(int));
    find_min_distance<<<1, 1>>>(distance, minIdx, m.cols());
    cudaDeviceSynchronize();

    cudaFree(distance);

    int h_minIdx = 0;
    cudaMemcpy(&h_minIdx, minIdx, sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(minIdx);

    return h_minIdx;
}
