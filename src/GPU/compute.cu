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

    double *Matrix::toGpu(size_t *pitch) const {
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
    distance[i + j*distance_p] = x*x + y*y + z*z;

}



__global__ void find_Y(double *distance, size_t distance_p,
                       double *m, size_t m_p, double *Y, size_t Y_p,
                       int xSize, int ySize) {

    int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (j >= ySize)
        return;

    distance_p = distance_p/sizeof(double);
    Y_p = Y_p / sizeof(double);
    m_p = m_p/ sizeof(double);

    int minIdx = 0;
    for (int i = 1; i < xSize; i++) {
        if (distance[minIdx + j*distance_p] > distance[i + j*distance_p]){
            minIdx = i;
        }
    }

    double mx = m[minIdx];
    double my = m[minIdx + m_p];
    double mz = m[minIdx + 2*m_p];

    Y[j] = mx;
    Y[j+ Y_p] = my;
    Y[j+ 2*Y_p] = mz;
}

void compute_Y_w(const GPU::Matrix &m, const GPU::Matrix &p, GPU::Matrix &Y){
    size_t m_p, p_p, Y_p;
    double *m_gpu = m.toGpu(&m_p);
    double *p_gpu = p.toGpu(&p_p);
    double *Y_gpu = Y.toGpu(&Y_p);


    double *distance;
    size_t distance_p;
    cudaMallocPitch((void **) &distance, &distance_p, sizeof(double) * m.cols(), p.cols());

    dim3 distBlk, distGrd;
    computeDim(m.cols(), p.cols(), &distBlk, &distGrd);
    compute_distance<<<distGrd, distBlk>>>(m_gpu, m_p, p_gpu, p_p, distance, distance_p, m.cols(), p.cols());
    cudaDeviceSynchronize();

    cudaFree(p_gpu);

    dim3 YBlk, YGrd;
    YBlk = dim3(1, 32, 1);
    int xBlocks = 1;
    int yBlocks = (int) std::ceil(((double) p.cols()) / 32);
    YGrd = dim3(xBlocks, yBlocks, 1);
    find_Y<<<YGrd, YBlk>>>(distance, distance_p, m_gpu, m_p, Y_gpu, Y_p, m.cols(), p.cols());
    cudaDeviceSynchronize();

    cudaFree(m_gpu);
    cudaFree(distance);

    Y.fromGpu(Y_gpu, Y.rows(), Y.cols(), Y_p);

    cudaFree(Y_gpu);
}

__global__ void compute_err(double *Y_gpu, double *p_gpu, double *sr_gpu, double
                            *t_gpu, double *err, size_t Y_p, size_t p_p, size_t sr_p,
                            size_t t_p, size_t err_p, unsigned int size)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    Y_p = Y_p / sizeof(double);
    p_p = p_p / sizeof(double);
    sr_p = sr_p / sizeof(double);
    t_p = t_p / sizeof(double);
    err_p = err_p / sizeof(double);

    double px = sr_gpu[0] * p_gpu[i] + sr_gpu[1] * p_gpu[i + p_p] + sr_gpu[2] * p_gpu[i + 2* p_p];
    double py = sr_gpu[sr_p] * p_gpu[i] + sr_gpu[1 + sr_p] * p_gpu[i + p_p] + sr_gpu[2 + sr_p] * p_gpu[i + 2* p_p];
    double pz = sr_gpu[2 * sr_p] * p_gpu[i] + sr_gpu[1 + 2*sr_p] * p_gpu[i + p_p] + sr_gpu[2 + 2*sr_p] * p_gpu[i + \
                                                                                                               2 * p_p];

    p_gpu[i] = px + t_gpu[0];
    p_gpu[i + p_p] = py + t_gpu[t_p];
    p_gpu[i + 2*p_p] = pz + t_gpu[2*t_p];

    Y_gpu[i] = Y_gpu[i] - p_gpu[i];
    Y_gpu[i + Y_p] = Y_gpu[i + Y_p] - p_gpu[i + p_p];
    Y_gpu[i + 2*Y_p] = Y_gpu[i + 2*Y_p] - p_gpu[i + 2*p_p];
    err[i] = Y_gpu[i] * Y_gpu[i] + Y_gpu[i + Y_p] * Y_gpu[i + Y_p] + Y_gpu[i + 2*Y_p] * Y_gpu[i + 2*Y_p];
}


double compute_err_w(const GPU::Matrix &Y, GPU::Matrix &p, bool in_place,
                     const GPU::Matrix &sr, const GPU::Matrix &t)
{
    size_t p_p, sr_p, t_p, Y_p;
    double *p_gpu = p.toGpu(&p_p);
    double *sr_gpu = sr.toGpu(&sr_p);
    double *t_gpu = t.toGpu(&t_p);
    double *Y_gpu =Y.toGpu(&Y_p);

    size_t err_p;
    GPU::Matrix tmp {MatrixXd{1,Y.cols()}};
    double *err = tmp.toGpu(&err_p);

    dim3 PBlk, PGrd;
    PBlk = dim3(32, 1, 1);
    int xBlocks = (int) std::ceil(((double) p.cols()) / 32);
    int yBlocks = 1;
    PGrd = dim3(xBlocks, yBlocks, 1);
    compute_err<<<PGrd,PBlk>>>(Y_gpu, p_gpu, sr_gpu, t_gpu, err, Y_p, p_p,
                               sr_p, t_p, err_p, p.cols());
    cudaDeviceSynchronize();

    if (!in_place)
        p.fromGpu(p_gpu, p.rows(), p.cols(), p_p);

    cudaFree(p_gpu);
    cudaFree(sr_gpu);
    cudaFree(t_gpu);

    tmp.fromGpu(err, 1, tmp.cols(), err_p);
    cudaFree(Y_gpu);
    cudaFree(err);

    return tmp.sum();
}
