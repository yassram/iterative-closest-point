#pragma once
#include <iostream>
#include "gpu.hh"

namespace GPU
{
    class Matrix: public MatrixXd
    {
    public:
        void fromGpu(double *gpu_rep, unsigned row, unsigned col);
        double *toGpu();
    };
}

void call_test();
