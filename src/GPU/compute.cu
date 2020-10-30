#include "compute.hh"

#define MAX_THREADS_PER_BLOCK 256
#define SHARED_THREADS_PER_BLOCK 32
#define BATCH_SIZE 1280

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
        auto err = cudaMallocPitch((void **)&d_x, pitch, sizeof(double) * c, r);
        Matrix tmp{this->transpose()};
        double *h_d = tmp.data();
        cudaMemcpy2D(d_x, *pitch, h_d, c * sizeof(double), sizeof(double) * c,
                     r, cudaMemcpyHostToDevice);
        return (double *)d_x;
    }

    int Matrix::toGpu(double** p, size_t *pitch, size_t offset, size_t batch_size, bool iscol) const
    {
        unsigned r = this->rows();
        unsigned c = this->cols();
        int err;
        if (iscol)
            err = cudaMallocPitch((void **)p, pitch, sizeof(double) * batch_size, r);
        else
            err = cudaMallocPitch((void **)p, pitch, sizeof(double) * c, batch_size);
        if (err != 0)
            return err;
        Matrix tmp{this->transpose()};
        double *h_d = tmp.data();
        if (iscol)
            cudaMemcpy2D(*p, *pitch, h_d + offset, c * sizeof(double), sizeof(double) * batch_size,
                         r, cudaMemcpyHostToDevice);
        else
            cudaMemcpy2D(*p, *pitch, h_d + offset, c * sizeof(double), sizeof(double) * c,
                         batch_size, cudaMemcpyHostToDevice);
        return err;
    }
} // namespace GPU

void computeDim(unsigned width, unsigned height,
                dim3 *block, dim3 *grid)
{
    int xThreads;
    int yThreads;

    if (width == 1){
        xThreads = 1;
        yThreads = MAX_THREADS_PER_BLOCK;
    }
    else if (height == 1){
        xThreads = MAX_THREADS_PER_BLOCK;
        yThreads = 1;
    }
    else{
        xThreads = SHARED_THREADS_PER_BLOCK;
        yThreads = SHARED_THREADS_PER_BLOCK;
    }

    *block = dim3(xThreads, yThreads, 1);

    int xBlocks = (int)std::ceil(((double)width) / xThreads);
    int yBlocks = (int)std::ceil(((double)height) / yThreads);
    *grid = dim3(xBlocks, yBlocks, 1);
}

__global__ void compute_distance(double *m, size_t m_p, double *p, size_t p_p,
                                 double *distance, size_t distance_p, int xSize,
                                 int ySize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= xSize)
        return;
    if (j >= ySize)
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
    size_t batch_size = BATCH_SIZE;
    size_t n = (p.cols() / batch_size) + 5;

    size_t m_p;
    double *m_gpu = m.toGpu(&m_p);

    double *distance[n];
    size_t distance_p[n];
    double *distance_cpu = (double*)std::malloc(sizeof(double) * p.cols() * m.cols());

    double *d_prime;
    double *p_prime;

    double *p_gpu[n];
    size_t p_p[n];

    double *y_cpu = (double*)std::malloc(sizeof(double) * p.cols() * 3);

    size_t current_batch_size = 0;
    size_t last_offset = 0;
    for (size_t offset = 0; offset < p.cols();)
    {
        int i= 0;

        while(true && offset < p.cols()){
            if (offset + batch_size >= p.cols())
                current_batch_size = p.cols() - offset;
            else
                current_batch_size = batch_size;

            auto err1 = cudaMallocPitch((void **) &d_prime, &distance_p[i],
                                        sizeof(double) * m.cols(), current_batch_size);

            auto err2 = p.toGpu(&p_prime, &p_p[i], offset, current_batch_size, true);
            p_gpu[i] = p_prime;

            if (err2 != 0 || err1 != 0)
                break;

            dim3 distBlk, distGrd;
            computeDim(m.cols(), current_batch_size, &distBlk, &distGrd);
            compute_distance<<<distGrd, distBlk>>>(m_gpu, m_p, p_gpu[i], p_p[i], d_prime, distance_p[i],
                                                   m.cols(), current_batch_size);

            distance[i] = d_prime;
            i++;
            offset += current_batch_size;
        }
        cudaDeviceSynchronize();

        for (size_t j = 0; j < i; j++){
            if (last_offset + batch_size >= p.cols())
                current_batch_size = p.cols() - last_offset;
            else
                current_batch_size = batch_size;

            cudaFree(p_gpu[j]);

            d_prime = distance[j];

            double *Y_prime;
            size_t Y_prime_p;
            auto err1 = cudaMallocPitch((void **) &Y_prime, &Y_prime_p,
                                        sizeof(double) * current_batch_size, p.rows());

            dim3 YBlk, YGrd;
            computeDim(1,current_batch_size, &YBlk, &YGrd);
            find_Y<<<YGrd, YBlk>>>(d_prime, distance_p[j], m_gpu, m_p, Y_prime,
                                   Y_prime_p, m.cols(), current_batch_size);
            cudaDeviceSynchronize();

            auto err_cpy = cudaMemcpy2D(y_cpu + last_offset, sizeof(double)*p.cols(),
                                        Y_prime, Y_prime_p, sizeof(double) * current_batch_size,
                                        p.rows(), cudaMemcpyDeviceToHost);

            cudaFree(Y_prime);
            cudaFree(d_prime);
            last_offset += batch_size;
        }
    }
    cudaFree(m_gpu);

    for (int i = 0; i < Y.rows(); i++){
        for (int j = 0; j < Y.cols(); j++){
            Y(i,j) = y_cpu[i * Y.cols() + j];
        }
    }
    std::free(y_cpu);
    std::free(distance_cpu);
}

__global__ void compute_err(double *Y_gpu, double *p_gpu, double *sr_gpu, double *t_gpu,
                            double *err, size_t Y_p, size_t p_p, size_t sr_p,
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
    computeDim(p.cols(), 1, &PBlk, &PGrd);
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
    computeDim(M.cols(), 1, &PBlk, &PGrd);
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
    computeDim(size_arr, 1, &PBlk, &PGrd);
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
