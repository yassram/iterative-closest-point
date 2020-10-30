#include "compute.hh"

namespace GPU
{
    void Matrix::fromGpu(double *gpu_rep, unsigned row, unsigned col, size_t pitch)
    {
        MatrixXd tmp{row, col};
        double *h_d = (double *)std::malloc(sizeof(double) * col * row);

        cudaMemcpy2D(h_d, sizeof(double) * col, gpu_rep, pitch, sizeof(double) * col,
                     row, cudaMemcpyDeviceToHost);

        for (unsigned i = 0; i < row; ++i)
            for (unsigned j = 0; j < col; ++j)
                tmp(i, j) = h_d[col * i + j];
        Matrix new_matrix{tmp};
        *this = new_matrix;
        std::free(h_d);
    }

    double *Matrix::toGpu(size_t *pitch) const
    {
        unsigned r = this->rows();
        unsigned c = this->cols();

        double *d_x;
        cudaMallocPitch((void **)&d_x, pitch, sizeof(double) * c, r);

        Matrix tmp{this->transpose()};
        double *h_d = tmp.data();
        cudaMemcpy2D(d_x, *pitch, h_d, c * sizeof(double), sizeof(double) * c,
                     r, cudaMemcpyHostToDevice);
        return (double *)d_x;
    }
} // namespace GPU

void computeDim(unsigned width, unsigned height,
                dim3 *block, dim3 *grid)
{
    int devId = 0; // There may be more devices!
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, devId);

    // int xMaxBlocks = deviceProp.maxGridSize[0];
    // int yMaxBlocks = deviceProp.maxGridSize[1];

    int xThreads = 32; // deviceProp.maxThreadsDim[0];
    int yThreads = 32; // deviceProp.maxThreadsDim[1];

    // int maxThreadPB = deviceProp.maxThreadsPerBlock;

    *block = dim3(xThreads, yThreads, 1);

    int xBlocks = (int)std::ceil(((double)width) / xThreads);
    int yBlocks = (int)std::ceil(((double)height) / yThreads);
    *grid = dim3(xBlocks, yBlocks, 1);
}

__global__ void compute_distance(double *m, size_t m_p, double *p, size_t p_p,
                                 double *distance, size_t distance_p, int xSize, int ySize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= xSize || j >= ySize)
        return;

    m_p = m_p / sizeof(double);
    double mx = m[i];
    double my = m[i + m_p];
    double mz = m[i + 2 * m_p];

    p_p = p_p / sizeof(double);
    double x = p[j] - mx;
    double y = p[j + p_p] - my;
    double z = p[j + 2 * p_p] - mz;

    distance_p = distance_p / sizeof(double);
    distance[i + j * distance_p] = x * x + y * y + z * z;
}

__global__ void find_Y(double *distance, size_t distance_p,
                       double *m, size_t m_p, double *Y, size_t Y_p,
                       int xSize, int ySize)
{

    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= ySize)
        return;

    distance_p = distance_p / sizeof(double);
    Y_p = Y_p / sizeof(double);
    m_p = m_p / sizeof(double);

    int minIdx = 0;
    for (int i = 1; i < xSize; i++)
    {
        if (distance[minIdx + j * distance_p] > distance[i + j * distance_p])
        {
            minIdx = i;
        }
    }

    double mx = m[minIdx];
    double my = m[minIdx + m_p];
    double mz = m[minIdx + 2 * m_p];

    Y[j] = mx;
    Y[j + Y_p] = my;
    Y[j + 2 * Y_p] = mz;
}

void compute_Y_w(const GPU::Matrix &m, const GPU::Matrix &p, GPU::Matrix &Y)
{
    size_t m_p, p_p, Y_p;
    double *m_gpu = m.toGpu(&m_p);
    double *p_gpu = p.toGpu(&p_p);
    double *Y_gpu = Y.toGpu(&Y_p);

    double *distance;
    size_t distance_p;
    cudaMallocPitch((void **)&distance, &distance_p, sizeof(double) * m.cols(), p.cols());

    dim3 distBlk, distGrd;
    computeDim(m.cols(), p.cols(), &distBlk, &distGrd);
    compute_distance<<<distGrd, distBlk>>>(m_gpu, m_p, p_gpu, p_p, distance, distance_p, m.cols(), p.cols());
    cudaDeviceSynchronize();

    cudaFree(p_gpu);

    dim3 YBlk, YGrd;
    YBlk = dim3(1, 32, 1);
    int xBlocks = 1;
    int yBlocks = (int)std::ceil(((double)p.cols()) / 32);
    YGrd = dim3(xBlocks, yBlocks, 1);
    find_Y<<<YGrd, YBlk>>>(distance, distance_p, m_gpu, m_p, Y_gpu, Y_p, m.cols(), p.cols());
    cudaDeviceSynchronize();

    cudaFree(m_gpu);
    cudaFree(distance);

    Y.fromGpu(Y_gpu, Y.rows(), Y.cols(), Y_p);

    cudaFree(Y_gpu);
}

__global__ void compute_err(double *Y_gpu, double *p_gpu, double *sr_gpu, double *t_gpu, double *err, size_t Y_p, size_t p_p, size_t sr_p,
                            size_t t_p, size_t err_p, unsigned int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    Y_p = Y_p / sizeof(double);
    p_p = p_p / sizeof(double);
    sr_p = sr_p / sizeof(double);
    t_p = t_p / sizeof(double);
    err_p = err_p / sizeof(double);

    double px = sr_gpu[0] * p_gpu[i] + sr_gpu[1] * p_gpu[i + p_p] + sr_gpu[2] * p_gpu[i + 2 * p_p];
    double py = sr_gpu[sr_p] * p_gpu[i] + sr_gpu[1 + sr_p] * p_gpu[i + p_p] + sr_gpu[2 + sr_p] * p_gpu[i + 2 * p_p];
    double pz = sr_gpu[2 * sr_p] * p_gpu[i] + sr_gpu[1 + 2 * sr_p] * p_gpu[i + p_p] + sr_gpu[2 + 2 * sr_p] * p_gpu[i + 2 * p_p];

    p_gpu[i] = px + t_gpu[0];
    p_gpu[i + p_p] = py + t_gpu[t_p];
    p_gpu[i + 2 * p_p] = pz + t_gpu[2 * t_p];

    Y_gpu[i] = Y_gpu[i] - p_gpu[i];
    Y_gpu[i + Y_p] = Y_gpu[i + Y_p] - p_gpu[i + p_p];
    Y_gpu[i + 2 * Y_p] = Y_gpu[i + 2 * Y_p] - p_gpu[i + 2 * p_p];
    err[i] = Y_gpu[i] * Y_gpu[i] + Y_gpu[i + Y_p] * Y_gpu[i + Y_p] + Y_gpu[i + 2 * Y_p] * Y_gpu[i + 2 * Y_p];
}

double compute_err_w(const GPU::Matrix &Y, GPU::Matrix &p, bool in_place,
                     const GPU::Matrix &sr, const GPU::Matrix &t)
{
    size_t p_p, sr_p, t_p, Y_p;
    double *p_gpu = p.toGpu(&p_p);
    double *sr_gpu = sr.toGpu(&sr_p);
    double *t_gpu = t.toGpu(&t_p);
    double *Y_gpu = Y.toGpu(&Y_p);

    size_t err_p;
    GPU::Matrix tmp{MatrixXd{1, Y.cols()}};
    double *err = tmp.toGpu(&err_p);

    dim3 PBlk, PGrd;
    PBlk = dim3(32, 1, 1);
    int xBlocks = (int)std::ceil(((double)p.cols()) / 32);
    int yBlocks = 1;
    PGrd = dim3(xBlocks, yBlocks, 1);
    compute_err<<<PGrd, PBlk>>>(Y_gpu, p_gpu, sr_gpu, t_gpu, err, Y_p, p_p,
                                sr_p, t_p, err_p, p.cols());
    cudaDeviceSynchronize();

    if (in_place)
        p.fromGpu(p_gpu, p.rows(), p.cols(), p_p);

    cudaFree(p_gpu);
    cudaFree(sr_gpu);
    cudaFree(t_gpu);

    tmp.fromGpu(err, 1, tmp.cols(), err_p);
    cudaFree(Y_gpu);
    cudaFree(err);

    return tmp.sum();
}

__global__ void substract_col(double *M_gpu, double *m_gpu, size_t M_p, size_t m_p,
                              int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;
    M_p = M_p / sizeof(double);
    m_p = m_p / sizeof(double);

    double mx = M_gpu[i] - m_gpu[0];
    double my = M_gpu[i + M_p] - m_gpu[m_p];
    double mz = M_gpu[i + 2 * M_p] - m_gpu[2 * m_p];

    M_gpu[i] = mx;
    M_gpu[i + M_p] = my;
    M_gpu[i + 2 * M_p] = mz;
}

GPU::Matrix substract_col_w(const GPU::Matrix &M, const GPU::Matrix &m)
{
    size_t M_p, m_p;
    double *m_gpu = m.toGpu(&m_p);
    double *M_gpu = M.toGpu(&M_p);

    dim3 PBlk, PGrd;
    PBlk = dim3(32, 1, 1);
    int xBlocks = (int)std::ceil(((double)M.cols()) / 32);
    int yBlocks = 1;
    PGrd = dim3(xBlocks, yBlocks, 1);
    substract_col<<<PGrd, PBlk>>>(M_gpu, m_gpu, M_p, m_p, M.cols());
    cudaDeviceSynchronize();
    cudaFree(m_gpu);

    GPU::Matrix tmp{MatrixXd{M.rows(), M.cols()}};
    tmp.fromGpu(M_gpu, M.rows(), M.cols(), M_p);
    cudaFree(M_gpu);
    return tmp;
}

__global__ void y_p_norm(const double *y, const double *p,
                         double *out_p, double *out_y,
                         const unsigned int y_p_,
                         const unsigned int p_p_,
                         const unsigned int size_arr)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int y_p = y_p_ / sizeof(double);
    unsigned int p_p = p_p_ / sizeof(double);

    double col[3];
    if (i < size_arr)
    {
        for (unsigned int j = 0; j < 3; j++)
            col[j] = y[i + j * y_p];
        out_y[i] = col[0] * col[0] + col[1] * col[1] + col[2] * col[2];
    }
    if (i < size_arr)
    {
        for (unsigned int j = 0; j < 3; j++)
            col[j] = p[i + j * p_p];
        out_p[i] = col[0] * col[0] + col[1] * col[1] + col[2] * col[2];
    }
}

unsigned int powerizer(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

void y_p_norm_wrapper(const GPU::Matrix &y, const GPU::Matrix &p, unsigned int size_arr, double &d_caps, double &sp)
{
    size_t y_p, p_p;
    double *y_gpu = y.toGpu(&y_p);
    double *p_gpu = p.toGpu(&p_p);

    unsigned int block_sz = ((size_arr) < 512) ? powerizer(size_arr) : 512;
    unsigned int grid_sz = std::ceil((double)(size_arr) / (double)block_sz);
    unsigned int smem_sz = sizeof(double) * block_sz;

    double *out_p;
    cudaMalloc(&out_p, sizeof(double) * size_arr);

    double *out_y;
    cudaMalloc(&out_y, sizeof(double) * size_arr);

    y_p_norm<<<grid_sz, block_sz, smem_sz>>>(y_gpu, p_gpu, out_p, out_y, y_p, p_p, size_arr);
    cudaFree(p_gpu);
    cudaFree(y_gpu);

    double *r_p;
    r_p = (double *)malloc(sizeof(double) * size_arr);
    double *r_y;
    r_y = (double *)malloc(sizeof(double) * size_arr);
    cudaMemcpy(r_p, out_p, sizeof(double) * size_arr, cudaMemcpyDeviceToHost);
    cudaMemcpy(r_y, out_y, sizeof(double) * size_arr, cudaMemcpyDeviceToHost);
    cudaFree(out_p);
    cudaFree(out_y);

    for (unsigned int i = 0; i < size_arr; i++)
    {
        d_caps += r_y[i];
        sp += r_p[i];
    }
    free(r_p);
    free(r_y);
}

// __global__ void dist_1(const double *m, const double *pi,
//                        double *out, const unsigned int nm)
// {
//     unsigned int tid = threadIdx.x;
//     unsigned int i = blockIdx.x * blockDim.x + tid;

//     if (i > nm)
//         return;

//     for (unsigned j = 0; j < 3; j++)
//     {
//         double tmp_val = (pi[j] - m[j * nm + i]);
//         out[i] += tmp_val * tmp_val;
//     }
// }

// void dist_1_wrapper(const double *m, const double *pi,
//                     double *out, const unsigned int nm)
// {
//     unsigned int block_sz = (nm < 512) ? powerizer(nm) : 512;
//     unsigned int grid_sz = std::ceil((double)nm / (double)block_sz);
//     unsigned int smem_sz = sizeof(double) * block_sz;
//     dist_1<<<grid_sz, block_sz, smem_sz>>>(m, pi, out, nm);
// }

// void loopdist_1(const GPU::Matrix &p, const GPU::Matrix &m, unsigned int nm, GPU::Matrix &Y, unsigned int j, cublasHandle_t hdl)
// {
//     size_t m_p, p_p;
//     double *p_gpu = p.toGpu(&p_p);
//     double *m_gpu = m.toGpu(&m_p);

//     double *d;
//     cudaMalloc(&d, sizeof(double) * nm);
//     cudaMemset(d, 0., sizeof(double) * nm);

//     dist_1_wrapper(m_gpu, p_gpu, d, nm);
//     cudaDeviceSynchronize();

//     cudaFree(p_gpu);
//     cudaFree(m_gpu);

//     int amin{};
//     // cublasIdamin(hdl, nm, d, 1, &amin);
//     // cudaDeviceSynchronize();
//     cudaFree(d);

//     for (unsigned i = 0; i < 3; i++)
//         Y[j + i * np] = m[(amin - 1) + nm * i];
// }