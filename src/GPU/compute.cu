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
    int yThreads = 32; // deviceProp.maxThreadsDim[1];

    // int maxThreadPB = deviceProp.maxThreadsPerBlock;

    *block = dim3(xThreads, yThreads, 1);

    int xBlocks = (int) std::ceil(((double)width) / xThreads);
    int yBlocks = (int) std::ceil(((double)height) / yThreads);
    *grid = dim3(xBlocks, yBlocks, 1);
}


__global__ void compute_distance(double *m, size_t m_p, double *p, size_t p_p,
                                 double *distance, size_t distance_p,  int xSize, int ySize){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i >= xSize || j >= ySize)
        return;

    m_p = m_p/sizeof(double);
    double mx = m[i];
    double my = m[i + m_p];
    double mz = m[i + 2*m_p];

    p_p = p_p/sizeof(double);
    double x = p[j] - mx;
    double y = p[j + p_p] - my;
    double z = p[j + 2*p_p] - mz;

    distance_p = distance_p/sizeof(double);
    distance[i + j * distance_p] = x*x + y*y + z*z;

    printf("> (%d,%d) : dist = %lf (%lf, %lf, %lf) (%lf, %lf, %lf)\n",
           i, j, distance[i + j * distance_p], p[j], p[j + p_p], p[j + 2*p_p], mx, my, mz);
}

__global__ void find_Y(double *distance, size_t distance_p,
                                  double *m, size_t m_p, double *Y, size_t Y_p,
                                  int xSize, int ySize) {
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (j >= ySize)
        return;

    distance_p = distance_p/sizeof(double);
    Y_p = Y_p / sizeof(double);

    int minIdx = 0;
    for (int i = 1; i < xSize; i++)
        if (distance[minIdx + j*distance_p] > distance[i + j+distance_p])
            minIdx = i;

     printf("> (%d) : dist = %lf \n",
            j, distance[minIdx + j * distance_p]);

    Y[j] = m[minIdx];
    Y[j+ Y_p] = m[minIdx + m_p];
    Y[j+ 2*Y_p] = m[minIdx + 2*m_p];
}

GPU::Matrix compute_Y_w(GPU::Matrix m, GPU::Matrix p, GPU::Matrix Y){
    size_t m_p, p_p, Y_p;
    double *m_gpu = m.toGpu(&m_p);
    double *p_gpu = p.toGpu(&p_p);
    double *Y_gpu = Y.toGpu(&Y_p);

    dim3 distBlk, distGrd;
    computeDim(m.cols(), 1, &distBlk, &distGrd);

    double *distance;
    size_t distance_p;

    cudaMallocPitch((void **) &distance, &distance_p, sizeof(double) * m.cols(), p.cols());

    compute_distance<<<distGrd, distBlk>>>(m_gpu, m_p, p_gpu, p_p, distance, distance_p, m.cols(), p.cols());
    cudaDeviceSynchronize();

    cudaFree(m_gpu);
    cudaFree(p_gpu);

    dim3 YBlk, YGrd;
    YBlk = dim3(1, 32, 1);

    int xBlocks = (int) std::ceil(1.0 / xThreads);
    int yBlocks = (int) std::ceil(((double) p.cols()) / yThreads);
    YGrd = dim3(xBlocks, yBlocks, 1);

    // computeDim(1, p.cols(), &YBlk, &YGrd);
    find_Y<<<YGrd, YBlk>>>(distance, distance_p, m_gpu, m_p, Y_gpu, Y_p, m.cols(), p.cols());
    cudaDeviceSynchronize();

    cudaFree(distance);

    Y.fromGpu(Y_gpu, Y.rows(), Y.cols(), Y_p);
    std::cout << "cu: \n" << Y << "\n"<< std::endl;

    cudaFree(Y_gpu);
    return Y;
}
