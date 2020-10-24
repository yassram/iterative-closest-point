#include "compute.hh"


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
    int yThreads = 1; // deviceProp.maxThreadsDim[1];

    // int maxThreadPB = deviceProp.maxThreadsPerBlock;

    *block = dim3(xThreads, yThreads, 1);

    int xBlocks = (int) std::ceil(((double)width) / xThreads);
    int yBlocks = (int) std::ceil(((double)height) / yThreads);
    *grid = dim3(xBlocks, yBlocks, 1);
}


__global__ void compute_distance(double *m, double *pi, double *distance, int size, size_t pitch){
    int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    int tidy = blockIdx.y*blockDim.y + threadIdx.y;

    printf("\ntidx: %d, tidy: %d, pitch: %lu, size: %d", tidx, tidy, pitch, size);
    int i = tidx + tidy;

    if (tidx >= size)
        return;

    char *mm = (char*)m;
    double mx = 0;//*( (double*) (((char*)m) + tidx) );
    double my = 0; // *( (double*) (((char*)m) + i + pitch) );
    double mz = 0; // *( (double*) (((char*)m) + i + 2*pitch) );

    printf("[size: %d / i: %d](%f, %f, %f) \n", size, i, mx, my, mz);

    double x = pi[0] - mx;
    double y = pi[1] - my;
    double z = pi[2] - mz;

    distance[i] = x*x + y*y + z*z;
}

__global__ void find_min_distance(double *distance, int *minIdx, int size) {
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
    compute_distance<<<distGrd, distBlk>>>(m_gpu, pi_gpu, distance, m.cols(), m_p);
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
