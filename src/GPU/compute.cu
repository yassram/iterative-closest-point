#include "compute.hh"


namespace GPU {
    void Matrix::fromGpu(double *gpu_rep, unsigned row, unsigned col) {
        MatrixXd tmp{row, col};
        double *h_d = (double*) std::malloc(sizeof(double) * col * row);

        cudaMemcpy(h_d, gpu_rep, row*col*sizeof(double), cudaMemcpyDeviceToHost);

        for(unsigned i = 0; i < row; ++i)
            for (unsigned j = 0; j < col; ++j)
                tmp(i,j) = h_d[col*i + j];
        Matrix new_matrix{tmp};
        *this = new_matrix;
        std::free(h_d);
    }

    double *Matrix::toGpu() {
        unsigned r = this->rows();
        unsigned c = this->cols();

        double *d_x;
        cudaMalloc((void **) &d_x,  sizeof(double) * r*c);
        Matrix tmp {this->transpose()};
        double *h_d = tmp.data();
        cudaMemcpy(d_x, h_d, r*c*sizeof(double), cudaMemcpyHostToDevice);
        return (double*)d_x;
    }
}

__global__ void test() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

void call_test(){
    GPU::Matrix a{MatrixXd{2,2}};
    a << 1, 2,
        3, 4;
    std::cout << a << std::endl;

    double *m = a.toGpu();

    GPU::Matrix b{MatrixXd{2,2}};
    b.fromGpu(m, 2, 2);

    std::cout << b << std::endl;
    cudaDeviceSynchronize();
}
