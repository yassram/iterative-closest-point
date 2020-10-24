#include "compute.hh"


namespace GPU {
    void Matrix::fromGpu(double *gpu_rep, unsigned row, unsigned col, size_t pitch) {
        MatrixXd tmp{row, col};
        double *h_d = (double*) std::malloc(sizeof(double) * col * row);

        cudaMemcpy2D(h_d, sizeof(double)*col, gpu_rep, pitch, col*sizeof(double),
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

__global__
void test() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}



void call_test(){
    GPU::Matrix a{MatrixXd{2,2}};
    a << 1, 2,
        3, 4;
    std::cout << a << std::endl;

    size_t s;
    double *m = a.toGpu(&s);
    std::cout << "size: " << s << std::endl;

    GPU::Matrix b{MatrixXd{2,2}};
    b.fromGpu(m, 2, 2, s);

    std::cout << b << std::endl;
    cudaDeviceSynchronize();
}
