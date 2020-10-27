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

__global__ void y_p_norm(const double *y_gpu, const double *p_gpu,
                         double *d_gpu, double *sp_gpu,
                         size_t y_p,
                         size_t p_p,
                         size_t d_p,
                         size_t sp_p,
                         const int size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    y_p = y_p / sizeof(double);
    p_p = p_p / sizeof(double);
    d_p = d_p / sizeof(double);
    sp_p = sp_p / sizeof(double);

    d_gpu[i] = y_gpu[i] * y_gpu[i] + y_gpu[i + y_p] * y_gpu[i + y_p]
                                        + y_gpu[i + 2*y_p] * y_gpu[i + 2*y_p];
    sp_gpu[i] = p_gpu[i] * p_gpu[i] + p_gpu[i + p_p] * p_gpu[i + p_p]
                                        + p_gpu[i + 2*p_p] * p_gpu[i + 2*p_p];
}
/*
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
*/
void y_p_norm_w(const GPU::Matrix &y, const GPU::Matrix &p, size_t size_arr,
                                                    double &d_caps, double &sp)
{
    size_t y_p, p_p, out_sp_p, out_d_p;
    double *y_gpu = y.toGpu(&y_p);
    double *p_gpu = p.toGpu(&p_p);

    GPU::Matrix out_d{MatrixXd{1, size_arr}};
    GPU::Matrix out_sp{MatrixXd{1, size_arr}};

    double *d_gpu = out_d.toGpu(&out_d_p);
    double *sp_gpu = out_sp.toGpu(&out_sp_p);

    dim3 PBlk, PGrd;
    PBlk = dim3(32, 1, 1);
    int xBlocks = (int)std::ceil(((double) size_arr) / 32);
    int yBlocks = 1;
    PGrd = dim3(xBlocks, yBlocks, 1);
    y_p_norm<<<PGrd, PBlk>>>(y_gpu, p_gpu, d_gpu, sp_gpu, y_p, p_p, out_d_p,
                                                            out_sp_p, size_arr);
    cudaFree(p_gpu);
    cudaFree(y_gpu);

    out_d.fromGpu(d_gpu, 1, out_d.cols(), out_d_p);
    out_sp.fromGpu(sp_gpu, 1, out_sp.cols(), out_sp_p);
    cudaFree(d_gpu);
    cudaFree(sp_gpu);

    d_caps = out_d.sum();
    sp = out_sp.sum();
}
/*
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
}*/
