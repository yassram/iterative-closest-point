#include "compute.hh"


namespace GPU {

    void Matrix::fromGpu(double *gpu_rep, unsigned row, unsigned col) {
        MatrixXd tmp{row, col};
        double data[row*col];

        cudaMemcpy(data, gpu_rep, r*c*sizeof(double), cudaMemcpyDeviceToHost);

        for(unsigned i = 0; i < row; ++i)
            for (unsigned j = 0; j < col; ++j)
                tmp(i,j) = data[col*i + j];
        Matrix new_matrix{tmp};
        *this = new_matrix;
    }

    double *Matrix::toGpu() {
        unsigned r = this->rows();
        unsigned c = this->cols();

        double *p;
        cudaMalloc(&p, sizeof(double)*c*r;

        double *data = this->data();
        cudaMemcpy(p, data, r*c*sizeof(double), cudaMemcpyHostToDevice);
        return p;
    }
}

__global__ void test() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

void call_test(){
    GPU::Matrix a{MatrixXd{2,2}}
    a << 1, 2 , 3, 4;
    std::cout << a << std::endl;

    double *m = a.toGpu();

    GPU::Matrix b{MatixXD{2,2}}
    b.fromGpu(m, 2, 2);

    std::cout << b << std::endl;
    cudaDeviceSynchronize();
}
